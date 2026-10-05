#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-call kernel time split by what ran beside it (CPU only, no torch).

    python3 trace_overlap_split.py OURS_TRACE.json[.gz] [MM_TRACE.json[.gz]] \
        [--kernels _thin_gemm_kernel,moe_dec_gemm,_post_update_num_computed]

For each kernel family (substring match on the name) and each trace, every
instance is classified by the other-stream kernels that overlap it on the same
device: ``alone``, ``nccl`` (any overlapping kernel name containing "nccl"),
``other`` (only non-NCCL kernels on other streams, e.g. shared experts on the
aux stream, the 0021 draft tail). Prints count / mean / median us per class
and the time-weighted overlap fraction.

Reading: if the ``alone`` medians of ours and MM's match while ours has more
instances (or more time) in ``nccl`` / ``other``, the "same kernel is 3-5 %
slower" gap is concurrency, not the kernel or its inputs (STEP-GAP-2.md).
If ``alone`` already differs, look at the binary (Triton version, PTX hash
from test_thin_gemm_select_gpu.py) and clocks.
"""

import argparse
import gzip
import json
import statistics
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    out = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "Kernel"):
            continue
        a = e.get("args", {})
        out.append((a.get("device", e.get("pid")), a.get("stream", e.get("tid")),
                    float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"]))
    return out


def classify(kernels, families):
    by_dev = defaultdict(list)
    for k in kernels:
        by_dev[k[0]].append(k)
    stats = {f: defaultdict(list) for f in families}
    frac = {f: defaultdict(float) for f in families}
    for dev, ks in by_dev.items():
        ks.sort(key=lambda k: k[2])
        starts = [k[2] for k in ks]
        max_dur = max((k[3] - k[2] for k in ks), default=0.0)
        import bisect
        for k in ks:
            fam = next((f for f in families if f in k[4]), None)
            if fam is None:
                continue
            lo = bisect.bisect_left(starts, k[2] - max_dur)
            hi = bisect.bisect_right(starts, k[3])
            cls, ov_n, ov_o = "alone", 0.0, 0.0
            for o in ks[lo:hi]:
                if o is k or o[1] == k[1]:
                    continue
                ov = min(k[3], o[3]) - max(k[2], o[2])
                if ov <= 0:
                    continue
                if "nccl" in o[4].lower():
                    ov_n += ov
                else:
                    ov_o += ov
            if ov_n > 0:
                cls = "nccl"
            elif ov_o > 0:
                cls = "other"
            dur = k[3] - k[2]
            stats[fam][cls].append(dur)
            frac[fam]["nccl"] += min(ov_n, dur)
            frac[fam]["other"] += min(ov_o, dur)
            frac[fam]["total"] += dur
    return stats, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--kernels",
                    default="_thin_gemm_kernel,moe_dec_gemm,_post_update_num_computed")
    args = ap.parse_args()
    families = [f for f in args.kernels.split(",") if f]
    for path in args.traces:
        stats, frac = classify(load(path), families)
        print(f"== {path}")
        for fam in families:
            tot = frac[fam]["total"]
            if not tot:
                print(f"  {fam}: none")
                continue
            print(f"  {fam}: overlapped by NCCL {frac[fam]['nccl'] / tot:6.1%}, "
                  f"by other streams {frac[fam]['other'] / tot:6.1%} of its time")
            for cls in ("alone", "nccl", "other"):
                d = stats[fam][cls]
                if d:
                    print(f"    {cls:6s} n={len(d):6d} mean {statistics.fmean(d):9.2f} us "
                          f"median {statistics.median(d):9.2f} us total {sum(d) / 1e3:9.2f} ms")


if __name__ == "__main__":
    main()
