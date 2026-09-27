#!/usr/bin/env python3
"""Judge every greedy divergence point against a fixed-shape target-logprob oracle.

Why this is the right oracle: greedy output on this stack is NOT invariant to the
number of verification rows per step (DSpark K=5 runs six rows, K=0 runs one), so
"two configurations disagree at c1" cannot by itself indict either one. What is
decidable is how large the target's logit margin is at the position where they
disagree: an exact tie or a sub-0.05-nat margin is numerics, while a token the
target scores far below its argmax is a genuine correctness fault.

Every oracle query is pinned to one canonical shape: concurrency 1, max_tokens=1,
teacher-forced prefix, and DSpark switched off through the hot control file so
the step holds a single row. Each prefix is queried twice to prove the
fixed-shape answer is stable before any margin is trusted.

usage: greedy-divergence-audit.py <A/run0.json> <B/run0.json> [--out audit.json]
       [--label-a NAME] [--label-b NAME]
env:   VLLM_API_KEY, BASE_URL (default http://127.0.0.1:18000)
"""
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18000").rstrip("/")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
TOPK = 20   # vLLM caps top_logprobs at 20
TIE, SMALL, CLEAR = 0.05, 0.5, 2.0      # nats


def post(path, body, timeout=240):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps({"model": MODEL, **body}).encode(),
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from {path}: {e.read().decode()[:400]}")


def load(path):
    d = json.load(open(path))
    return {r["prompt"]: r["token_ids"] for r in d["results"]}, d.get("label", path)


def tok_text(token_id):
    return post("/detokenize", {"tokens": [token_id]})["prompt"]


def main():
    args = sys.argv[1:]
    a_path, b_path = args[0], args[1]
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path("audit.json")
    A, la = load(a_path)
    B, lb = load(b_path)
    keys = sorted(set(A) & set(B))
    print(f"auditing {len(keys)} prompts  A={la}  B={lb}")
    rows = []
    for n, k in enumerate(keys, 1):
        x, y = A[k], B[k]
        i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if i is None:
            rows.append({"prompt": k[:56], "identical": True})
            continue
        prefix_ids = post("/tokenize", {"messages": [{"role": "user", "content": k}],
                                        "add_generation_prompt": True})["tokens"] + x[:i]
        q = {"prompt": prefix_ids, "max_tokens": 1, "temperature": 0, "top_p": 1,
             "return_token_ids": True, "logprobs": TOPK, "cache_salt": str(uuid.uuid4())}
        o1 = post("/v1/completions", q)["choices"][0]
        o2 = post("/v1/completions", q)["choices"][0]
        tl = o1["logprobs"]["top_logprobs"][0]
        top_lp = max(tl.values())
        sx, sy = tok_text(x[i]), tok_text(y[i])
        lx, ly = tl.get(sx), tl.get(sy)
        # how many candidates are tied with the top score to float precision
        ties = sum(1 for v in tl.values() if v == top_lp)
        mids = None if lx is None or ly is None else (min(lx, ly) - top_lp)
        rows.append({
            "prompt": k[:56], "identical": False, "divergence_at": i,
            "A_token": x[i], "B_token": y[i], "A_str": sx, "B_str": sy,
            "oracle_token": o1["token_ids"][0], "oracle_str": o1["logprobs"]["tokens"][0],
            "oracle_stable": o1["token_ids"][0] == o2["token_ids"][0],
            "oracle_top_lp": top_lp, "A_lp": lx, "B_lp": ly,
            "n_exact_ties_at_top": ties, "margin_nats": mids,
            "oracle_pick": ("A" if o1["token_ids"][0] == x[i] else "B" if o1["token_ids"][0] == y[i] else "neither"),
        })
        r = rows[-1]
        print(f"  {n:>2} diverge@{r['divergence_at']:>3}  margin="
              f"{'n/a' if mids is None else round(mids, 4)}  exact_ties={ties}  "
              f"oracle->{r['oracle_pick']} stable={r['oracle_stable']}  "
              f"A={sx!r} B={sy!r}")
    both = [r for r in rows if not r.get("identical")]
    def bucket(r):
        if r["n_exact_ties_at_top"] > 1:
            return "exact_tie"
        if r["margin_nats"] is None:
            return "beyond_topk"
        if abs(r["margin_nats"]) < TIE:
            return "near_tie"
        if r["margin_nats"] > -CLEAR:
            return "small_margin"
        return "clear_violation"
    counts = {}
    for r in both:
        counts[bucket(r)] = counts.get(bucket(r), 0) + 1
    unstable = sum(1 for r in both if not r["oracle_stable"])
    margins = sorted(abs(r["margin_nats"]) for r in both if r["margin_nats"] is not None)
    summary = {
        "A": la, "B": lb, "prompts": len(keys),
        "identical": sum(1 for r in rows if r.get("identical")),
        "divergent": len(both),
        "oracle_unstable_at_fixed_shape": unstable,
        "classification": counts,
        "median_abs_margin_nats": (margins[len(margins) // 2] if margins else None),
        "max_abs_margin_nats": (margins[-1] if margins else None),
    }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    print("\nREAD: 'exact_tie'/'near_tie' = numerics can legitimately flip this; "
          "'clear_violation' = a token sits >%.1f nats below the target argmax and needs a bug hunt. "
          "oracle_unstable_at_fixed_shape must be 0 or the whole audit is void." % CLEAR)


if __name__ == "__main__":
    main()
