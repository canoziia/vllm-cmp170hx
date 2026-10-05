#!/usr/bin/env python3
"""Patch 0031 (dev 0029) GPU test: two processes, two GPUs, real NCCL (+ Gloo metadata).

Usage (main session, e.g. GPU5/6):
  CUDA_VISIBLE_DEVICES=5,6 python3 test_pp_pack_hop_gpu.py --no-bench [--tree DIR]
      [--iters 300] [--rows 1,8,16,32,64,128,256,512]

1. correctness: 120 steps, real isend_tensor_dict/irecv_tensor_dict (AST-
   extracted from the patched tree, as in the Gloo test), off and on; the
   received tensors (bf16 [rows,4,4096] mHC state, fp32 [rows,4096] fc
   partial, int32 verification lengths, odd fp16/bool) must be bit-identical
   on device, with key order unchanged. In packed mode the sender overwrites
   its sources right after isend (the wire buffer is a copy); fire-and-forget
   with the worker's top-of-step wait, like gpu_worker.execute_model.
2. microbenchmark per hop, both ranks timing the same loop:
   a. raw NCCL: 2 x isend (bf16 + fp32) vs pack copy + 1 x isend;
   b. full isend_tensor_dict/irecv_tensor_dict path (incl. Gloo metadata),
      off vs on.
   Shapes: hidden [rows,4096] bf16 and mHC [rows,4,4096] bf16, + [rows,4096]
   fp32 partial, rows = padded batch rows.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
import time

if "--tree" in sys.argv:  # before importing G (also in spawned children)
    os.environ["PP_PACK_TREE"] = os.path.abspath(sys.argv[sys.argv.index("--tree") + 1])

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_pack_hop_gloo as G  # noqa: E402


def coord(methods, rank, packed, cpu_group):
    c = G.make_coord(methods, rank, packed)
    c.cpu_group = cpu_group
    c.device_group = dist.group.WORLD
    return c


def payload(step, dev, mhc=4):
    g = torch.Generator(device=dev).manual_seed(77 + step)
    rows = (8, 16, 64, 8, 3, 128)[step % 6]
    hshape = (rows, mhc, 4096) if mhc else (rows, 4096)
    d = {
        "hidden_states": torch.randn(hshape, generator=g, device=dev).bfloat16(),
        "aux_fc_partial": torch.randn(rows, 4096, generator=g, device=dev),
        "__verification_budget": step % 5,
    }
    if step % 3 == 0:
        d["__verification_lengths"] = torch.randint(0, 9, (rows,), generator=g,
                                                    device=dev, dtype=torch.int32)
        d["odd"] = torch.randn(rows, 3, generator=g, device=dev).half()
        d["flags"] = torch.rand(rows, 5, generator=g, device=dev) > 0.5
    return d


def correctness(rank, methods, packed, cpu_group, dev):
    c = coord(methods, rank, packed, cpu_group)
    prev = []
    for step in range(120):
        exp = payload(step, dev)
        if rank == 0:
            for h in prev:
                h.wait()
            src = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in exp.items()}
            prev = c.isend_tensor_dict(src, dst=1, all_gather_group=G.AG(1))[1:]
            if packed:
                for v in src.values():
                    if isinstance(v, torch.Tensor) and v.dtype != torch.bool:
                        v.fill_(float("nan") if v.is_floating_point() else -1)
            del src
        else:
            td, hs, post = c.irecv_tensor_dict(src=0, all_gather_group=G.AG(1))
            for h in hs:
                h.wait()
            for fn in post:
                fn()
            G.same({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in exp.items()},
                   {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in td.items()})
    for h in prev:
        h.wait()
    torch.cuda.synchronize()


def bench(rank, methods, cpu_group, dev, rows_list, iters):
    results = []
    for mhc in (0, 4):
        for rows in rows_list:
            h = torch.randn((rows, mhc, 4096) if mhc else (rows, 4096), device=dev).bfloat16()
            p = torch.randn(rows, 4096, device=dev)
            row = {"mhc": mhc, "rows": rows,
                   "bytes": h.numel() * 2 + p.numel() * 4}
            # a. raw NCCL
            for name in ("raw_sep", "raw_pack"):
                def once():
                    if name == "raw_sep":
                        if rank == 0:
                            ws = [dist.isend(h, 1), dist.isend(p, 1)]
                        else:
                            ws = [dist.irecv(h, 0), dist.irecv(p, 0)]
                    else:
                        nb = (h.numel() * 2 + 15) // 16 * 16
                        buf = torch.empty(nb + p.numel() * 4, dtype=torch.uint8, device=dev)
                        if rank == 0:
                            buf[: h.numel() * 2].view(torch.bfloat16).view(h.shape).copy_(h)
                            buf[nb:].view(torch.float32).view(p.shape).copy_(p)
                            ws = [dist.isend(buf, 1)]
                        else:
                            ws = [dist.irecv(buf, 0)]
                    for w in ws:
                        w.wait()
                row[name] = timed(once, iters)
            # b. full tensor-dict path
            for packed in (False, True):
                c = coord(methods, rank, packed, cpu_group)
                def once():
                    if rank == 0:
                        for w in c.isend_tensor_dict(
                                {"hidden_states": h, "aux_fc_partial": p,
                                 "__verification_budget": 3}, dst=1,
                                all_gather_group=G.AG(1))[1:]:
                            w.wait()
                    else:
                        _, hs, post = c.irecv_tensor_dict(src=0, all_gather_group=G.AG(1))
                        for w in hs:
                            w.wait()
                row["dict_on" if packed else "dict_off"] = timed(once, iters)
            results.append(row)
    return results


def timed(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def worker(rank, path, args):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method="file://" + path, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120), device_id=dev)
    cpu_group = dist.new_group([0, 1], backend="gloo")
    pack, _, methods = G.load()
    for packed in (False, True):
        correctness(rank, methods, packed, cpu_group, dev)
        if rank == 1:
            print(f"PASS correctness packed={packed}: 120 steps bit-identical", flush=True)
    if args.no_bench:
        dist.barrier()
        dist.destroy_process_group()
        return
    res = bench(rank, methods, cpu_group, dev, args.rows, args.iters)
    if rank == 0:
        print("mhc rows   MB   raw_sep  raw_pack  dict_off  dict_on   (us/hop, rank0 wall)")
        for r in res:
            print(f"{r['mhc']:3d} {r['rows']:5d} {r['bytes']/2**20:5.2f} "
                  f"{r['raw_sep']:8.1f} {r['raw_pack']:8.1f} {r['dict_off']:9.1f} {r['dict_on']:8.1f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rows", default="1,8,16,32,64,128,256,512")
    # The raw-NCCL micro-benchmark hung on node2 (two unbatched isends on the
    # default group); the correctness phase does not depend on it.
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()
    a.rows = [int(x) for x in a.rows.split(",")]
    if a.tree:  # spawned children re-import G, which reads PP_PACK_TREE
        os.environ["PP_PACK_TREE"] = os.path.abspath(a.tree)
    assert torch.cuda.device_count() >= 2, "needs two visible GPUs"
    with tempfile.TemporaryDirectory(prefix="pp0029-gpu-") as d:
        mp.spawn(worker, args=(d + "/init", a), nprocs=2, join=True)
