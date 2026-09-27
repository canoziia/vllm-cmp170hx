#!/usr/bin/env python3
"""Align PP debug JSONL by global step and *actual* request cohort.

CPU sections are enclosed/enqueue durations, not GPU execution durations.
CUDA Event spans from distinct streams or devices must not be added together.
Use with real API usage/SSE measurements and an uninstrumented A/B run.
"""
import argparse
import bisect
import json
from pathlib import Path


def quantiles(values):
    values = sorted(values)
    if not values:
        return "-"
    return "/".join(f"{values[min(len(values)-1, int(q*(len(values)-1)))]:.2f}" for q in (.1, .5, .9))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", help="trace path prefix before -rankN.jsonl")
    p.add_argument("--cohort", type=int, required=True, help="actual meta.num_reqs, not client concurrency")
    p.add_argument("--scheduled", type=int, help="meta.scheduled_tokens; filters prefill/other cohorts")
    p.add_argument("--spec", type=int, help="meta.spec_reqs")
    p.add_argument("--trim", type=int, default=5, help="remove this many aligned boundary steps per side")
    p.add_argument("--max-skew-ms", type=float, default=30.0, help="max rank0-to-rankN CPU-start gap")
    args = p.parse_args()
    ranks = []
    for rank in range(6):
        path = Path(f"{args.prefix}-rank{rank}.jsonl")
        records = [json.loads(s) for s in path.read_text().splitlines() if s]
        records = {x["step"]: x for x in records if
                   x["meta"].get("num_reqs") == args.cohort and
                   (args.scheduled is None or x["meta"].get("scheduled_tokens") == args.scheduled) and
                   (args.spec is None or x["meta"].get("spec_reqs") == args.spec)}
        ranks.append(records)
    # Worker step counters can differ by the pipeline depth (rank5 was six
    # behind rank0 in a real c1 trace). Do NOT equate their step IDs. Match
    # nearest CPU starts, allow a few microseconds of negative scheduling skew,
    # and reject reused or equidistant/ambiguous rank-local records. Validate
    # the resulting cohort and local-step progression before interpreting it.
    ordered = [sorted(r.values(), key=lambda x: x['wall_start_ns']) for r in ranks]
    starts = [[x['wall_start_ns'] for x in o] for o in ordered]
    aligned = []
    last_used = [-1]*6
    for x in ordered[0]:
        row = [x]
        indices = [bisect.bisect_left(starts[0], x['wall_start_ns'])]
        for rank in range(1, 6):
            j = bisect.bisect_left(starts[rank], x['wall_start_ns'])
            candidates = sorted((abs(starts[rank][i]-x['wall_start_ns']), i)
                                for i in (j-1,j) if 0 <= i < len(starts[rank]) and
                                i > last_used[rank] and
                                abs(starts[rank][i]-x['wall_start_ns'])/1e6 <= args.max_skew_ms)
            if not candidates or (len(candidates)>1 and candidates[1][0] == candidates[0][0]):
                break
            i = candidates[0][1]
            indices.append(i)
            row.append(ordered[rank][i])
        if len(row) == 6:
            aligned.append(row)
            last_used = indices
    # A PP stage can be one pipeline cycle behind the next. Keep only a
    # consistent step-offset tuple, otherwise a nearest-time match can silently
    # join different requests after a stall or transition.
    from collections import Counter
    offsets = Counter(tuple(row[i]['step']-row[0]['step'] for i in range(1, 6))
                      for row in aligned)
    if offsets:
        dominant, _ = offsets.most_common(1)[0]
        aligned = [row for row in aligned if tuple(row[i]['step']-row[0]['step'] for i in range(1, 6)) == dominant]
    if args.trim:
        aligned = aligned[args.trim:len(aligned)-args.trim]
    if not aligned:
        raise SystemExit("No unambiguous wall-aligned steps: inspect actual cohort, scheduled tokens, skew and rank-local counters")
    print(f"dominant worker step offsets={dominant}; rejected other offsets={sum(offsets.values())-offsets[dominant]}")
    print(f"aligned={len(aligned)} rank-local first={[(x['rank'],x['step']) for x in aligned[0]]} "
          f"last={[(x['rank'],x['step']) for x in aligned[-1]]} cohort={args.cohort}")
    print("rank  CPU execute p10/p50/p90   metadata p10/p50/p90  forward p10/p50/p90  "
          "recv-enqueue p10/p50/p90  target GPU p10/p50/p90")
    for i, records in enumerate(ranks):
        selected = [row[i] for row in aligned]
        def col(mapping, key):
            return quantiles([x.get(mapping, {}).get(key, 0) for x in selected])
        print(f"{i:4d}  {col('cpu_ms','execute_model'):>21s}  "
              f"{col('cpu_ms','metadata_and_model_inputs'):>21s}  "
              f"{col('cpu_ms','forward_dispatch'):>21s}  "
              f"{col('cpu_ms','pp_irecv_enqueue'):>24s}  "
              f"{col('gpu_ms','target_forward'):>21s}")
    # Worker step IDs include idle PP callbacks and cannot be the denominator
    # for decode rate. Only consecutive sampled *decode* starts are comparable;
    # use SSE for actual output rate when the trace is sparse.
    gaps = [(b[0]["wall_start_ns"] - a[0]["wall_start_ns"]) / 1e6
            for a,b in zip(aligned, aligned[1:])]
    print(f"PP0 sampled decode-start gap ms p10/p50/p90={quantiles(gaps)} (n={len(gaps)}; sparse sampling may skip decodes)")
    # Same host CLOCK_REALTIME per process; cross-host/GPU Event clock alignment is
    # not implied. Wall intervals may overlap asynchronous device work.
    spans = [(row[5]["wall_end_ns"]-row[0]["wall_start_ns"])/1e6 for row in aligned]
    print(f"PP0 start to PP5 CPU finish ms p10/p50/p90={quantiles(spans)} (n={len(spans)})")
    for r in [5]:
        selected = [row[r] for row in aligned]
        keys = sorted({k for x in selected for k in x.get('gpu_ms',{}) if k != 'target_forward'})
        print(f"rank{r} GPU event spans (not additive): " + ", ".join(
            f"{k}={quantiles([x.get('gpu_ms',{}).get(k,0) for x in selected])}"
            for k in keys))


if __name__ == "__main__":
    main()
