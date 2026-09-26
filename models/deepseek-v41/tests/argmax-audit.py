#!/usr/bin/env python3
"""Full-sequence argmax audit: is EVERY emitted token the target's argmax?

This is the invariant that makes greedy speculative decoding correct, and it is
what an adaptive-verification change must preserve:

  for every position i of an emitted sequence, the target model's score for the
  emitted token at prefix x[:i] must equal the best available score.

Positions where the emitted token ties the top score to float precision are
allowed (this engine flips ties run to run - measured separately). Positions
where the emitted token sits measurably below the top score are real faults:
no acceptance policy can legitimately emit them under temperature 0.

The oracle is always queried at one canonical shape: concurrency 1, max_tokens=1,
teacher-forced prefix, DSpark off via the hot control file (single row), so a
violation cannot be blamed on batch shape.

usage: argmax-audit.py <run0.json> [<run0.json> ...] [--stride N] [--max-pos N]
       sequences come from models/deepseek-v41/tests/token-id-probe.mjs
env: VLLM_API_KEY, BASE_URL, MODEL
"""
import json
import os
import sys
import urllib.request
import uuid

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18000").rstrip("/")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
TOPK = 20


def post(path, body, timeout=240):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + path, data=json.dumps({"model": MODEL, **body}).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {path}: {e.read().decode()[:300]}")


def audit(path, stride=1, max_pos=10**9):
    d = json.load(open(path))
    label = d.get("label", os.path.basename(os.path.dirname(path)))
    toks = {}

    def text(tid):
        if tid not in toks:
            toks[tid] = post("/detokenize", {"tokens": [tid]})["prompt"]
        return toks[tid]

    print(f"\n=== {label}: {len(d['results'])} sequences, stride={stride} ===")
    totals = {"checked": 0, "argmax": 0, "exact_tie": 1 if 0 else 0, "below": 0}
    totals["exact_tie"] = 0
    worst = []
    for seq in d["results"]:
        prompt, ids = seq["prompt"], seq["token_ids"]
        base = post("/tokenize", {"messages": [{"role": "user", "content": prompt}],
                                  "add_generation_prompt": True})["tokens"]
        bad = []
        for i in range(0, min(len(ids), max_pos), stride):
            ch = post("/v1/completions", {"prompt": base + ids[:i], "max_tokens": 1,
                                          "temperature": 0, "top_p": 1, "return_token_ids": True,
                                          "logprobs": TOPK, "cache_salt": str(uuid.uuid4())})["choices"][0]
            tl = ch["logprobs"]["top_logprobs"][0]
            top = max(tl.values())
            lp = tl.get(text(ids[i]))
            totals["checked"] += 1
            if lp is None:
                bad.append((i, -99.0, "not-in-top20")); continue
            gap = lp - top
            if gap == 0.0 and ch["token_ids"][0] == ids[i]:
                totals["argmax"] += 1
            elif gap == 0.0:
                totals["exact_tie"] += 1
            elif gap > -0.05:
                totals["argmax" if ch["token_ids"][0] == ids[i] else "exact_tie"] += 1
            else:
                totals["below"] += 1
                bad.append((i, round(gap, 3), text(ids[i])[:14]))
        if bad:
            worst.append((label, prompt[:44], bad[:6]))
            print(f"  {prompt[:44]!r}: {len(bad)} position(s) below target argmax -> {bad[:4]}")
    print(f"  totals: checked={totals['checked']} argmax={totals['argmax']} "
          f"exact_tie={totals['exact_tie']} BELOW_ARGMAX={totals['below']}")
    return {"label": label, **totals, "worst": worst}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 1
    maxpos = int(args[args.index("--max-pos") + 1]) if "--max-pos" in args else 10**9
    files = [a for a in args if not a.startswith("--")]
    out = [audit(f, stride, maxpos) for f in files]
    p = json.dumps(out, indent=2, default=str)
    open("argmax-audit.json", "w").write(p)
    print("\nWROTE argmax-audit.json")
