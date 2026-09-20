#!/usr/bin/env python3
"""Summarize hot perf-debug JSONL files without external dependencies."""

import argparse
import glob
import json
import statistics
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs="+", help="JSONL files or glob patterns")
args = parser.parse_args()

paths = sorted({p for pattern in args.paths for p in glob.glob(pattern)})
if not paths:
    raise SystemExit("no matching files")

records = []
for path in paths:
    with open(path) as source:
        records.extend(json.loads(line) for line in source if line.strip())

by_rank = defaultdict(list)
for record in records:
    by_rank[record["rank"]].append(record)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    return values[round((len(values) - 1) * p)]


for rank, samples in sorted(by_rank.items()):
    print(f"rank={rank} samples={len(samples)}")
    modes = defaultdict(int)
    for sample in samples:
        modes[sample.get("meta", {}).get("graph_mode", "unknown")] += 1
    print("  graph_modes=" + json.dumps(modes, sort_keys=True))

    metrics = defaultdict(list)
    for sample in samples:
        metrics["cpu_total_ms"].append(sample["cpu_total_ms"])
        for name, value in sample.get("cpu_ms", {}).items():
            metrics[f"cpu.{name}"].append(value)
        for name, value in sample.get("gpu_ms", {}).items():
            metrics[f"gpu.{name}"].append(value)
        acceptance = sample.get("acceptance")
        if acceptance:
            metrics["accept.sampled_mean"].append(acceptance["sampled_mean"])

    for name, values in sorted(metrics.items()):
        print(
            f"  {name}: mean={statistics.fmean(values):.4f} "
            f"p50={percentile(values, 0.50):.4f} "
            f"p95={percentile(values, 0.95):.4f} "
            f"max={max(values):.4f} n={len(values)}"
        )
