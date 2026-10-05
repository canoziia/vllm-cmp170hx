#!/usr/bin/env python3
"""Patch 0031 (dev 0029, VLLM_PP_PACK_TENSORS_0029): two CPU processes, real Gloo.

The *real* ``isend_tensor_dict`` / ``irecv_tensor_dict`` / ``isend_object`` /
``recv_object`` / ``_reap_completed_isends`` / ``_should_use_all_gather`` are
AST-extracted from the patched ``parallel_state.py`` and bound to a stand-in
coordinator; ``pp_pack_0029.py`` is imported from the patched tree.
vLLM itself need not be importable (no GPU, no node2).

Checks
  1. off and on: every received tensor is bit-identical to the sent one
     (bf16/fp32/fp16/int32/int64/bool, odd sizes for alignment, empty
     tensors, non-tensor fields, the verification lengths tensor), and the
     dict key order is unchanged;
  2. on: one tensor P2P op per hop when >= 2 tensors are packable; with a
     TP all-gather group of size 2 the all-gathered key stays separate and
     its postprocess still lands in the returned dict;
  3. 200 ordered steps with varying padded row counts, fire-and-forget
     sends (retention FIFO bounded), sender overwrites its sources right
     after isend (the pack copy has already been taken);
  4. configure(): agreement passes, a mismatch raises on both ranks.

Run: python3 test_pp_pack_hop_gloo.py [--tree DIR]   (CPU torch; tree defaults to
$GLM_DFLASH2_TREE, else the image site-packages)
"""

from __future__ import annotations

import ast
import datetime
import importlib.util
import os
import pickle
import sys
import tempfile
import textwrap
from collections import deque, namedtuple
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("PP_PACK_TREE", os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages"))
if "--tree" in sys.argv:
    TREE = sys.argv[sys.argv.index("--tree") + 1]
PS = os.path.join(TREE, "vllm", "distributed", "parallel_state.py")
PK = os.path.join(TREE, "vllm", "distributed", "pp_pack_0029.py")

METHODS = ("isend_tensor_dict", "irecv_tensor_dict", "isend_object", "recv_object",
           "_reap_completed_isends", "_should_use_all_gather")


def load():
    spec = importlib.util.spec_from_file_location("vllm.distributed.pp_pack_0029", PK)
    pack = importlib.util.module_from_spec(spec)
    sys.modules["vllm.distributed.pp_pack_0029"] = pack  # for the method-local import
    import types
    for name in ("vllm", "vllm.distributed"):
        m = sys.modules.setdefault(name, types.ModuleType(name))
        m.__path__ = []
    sys.modules["vllm.distributed"].pp_pack_0029 = pack
    spec.loader.exec_module(pack)

    src = open(PS).read()
    tree = ast.parse(src)
    ns: dict[str, Any] = dict(torch=torch, deque=deque, namedtuple=namedtuple,
                              Any=Any, Callable=Callable, Protocol=Protocol,
                              pickle=pickle)
    ns["GroupCoordinator"] = object
    wanted_top = {"_RetainedHandle", "_split_tensor_dict", "Handle"}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted_top:
            exec(compile(ast.Module([node], []), PS, "exec"), ns)
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "TensorMetadata"):
            exec(compile(ast.Module([node], []), PS, "exec"), ns)
    psmod = types.ModuleType("vllm.distributed.parallel_state")
    ns["TensorMetadata"].__module__ = psmod.__name__
    psmod.TensorMetadata = ns["TensorMetadata"]
    sys.modules[psmod.__name__] = psmod
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator")
    methods = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            code = textwrap.dedent(ast.get_source_segment(src, node))
            exec(code, ns)
            methods[node.name] = ns[node.name]
    assert set(methods) == set(METHODS), set(METHODS) - set(methods)
    return pack, ns, methods


class AG:
    """Stand-in TP group: all_gather duplicates the received slice."""
    def __init__(self, world_size): self.world_size = world_size; self.rank_in_group = 0
    def all_gather(self, t, dim=0):
        return t if self.world_size == 1 else torch.cat([t] * self.world_size, dim=dim)


def make_coord(methods, rank, packed):
    Coord = type("Coord", (), dict(methods))
    c = Coord()
    c.world_size = 2; c.rank_in_group = rank; c.ranks = [0, 1]
    c.cpu_group = dist.group.WORLD; c.device_group = dist.group.WORLD
    c.use_cpu_custom_send_recv = False; c.device_communicator = None
    c._pending_isends = deque(); c._pp_pack_0029 = packed
    return c


def payload(step: int, kind: int):
    g = torch.Generator().manual_seed(1000 + step)
    rows = (8, 16, 8, 64, 3, 1)[step % 6]
    d: dict[str, Any] = {}
    d["hidden_states"] = torch.randn(rows, 4, 64, generator=g).to(torch.bfloat16)
    if kind >= 1:
        d["aux_fc_partial"] = torch.randn(rows, 64, generator=g)
    d["__verification_budget"] = step % 7
    if kind >= 2:
        d["odd_fp16"] = torch.randn(rows, 3, generator=g).half()      # 6*rows bytes
        d["flags"] = torch.rand(rows, 5, generator=g) > 0.5           # bool, 5*rows
        d["empty"] = torch.empty(0, 4)
        d["__verification_lengths"] = torch.randint(0, 9, (rows,), generator=g, dtype=torch.int32)
        d["ids"] = torch.randint(-2**62, 2**62, (rows, 1), generator=g, dtype=torch.int64)
    if kind == 3:
        # Replicated tensor all-gathered over a TP group of 2: halves equal.
        half = torch.randn(rows, 32, generator=g)
        d["ag_rep"] = torch.cat([half, half], dim=1).contiguous().view(2, rows, 32)
    return d


def same(a, b):
    assert list(a) == list(b), (list(a), list(b))
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape, (k, x.dtype, y.dtype, x.shape, y.shape)
            assert torch.equal(x.contiguous().view(-1).view(torch.uint8) if x.numel() else x,
                               y.contiguous().view(-1).view(torch.uint8) if y.numel() else y), k
        else:
            assert x == y, k


def run_steps(rank, methods, packed, steps, counts):
    c = make_coord(methods, rank, packed)
    real_isend, real_irecv = dist.isend, dist.irecv
    prev = []
    for step in range(steps):
        kind = step % 4
        if kind == 3:
            agg, agt = AG(2), {"ag_rep": True, "hidden_states": False,
                               "aux_fc_partial": False, "odd_fp16": False, "flags": False,
                               "__verification_lengths": False, "ids": False, "empty": False}
        else:
            agg, agt = AG(1), None
        expect = payload(step, kind)
        if kind == 3:
            # Sender ships slice 0 of the flattened tensor: make it the same
            # shape the receiver will reconstruct.
            expect["ag_rep"] = expect["ag_rep"].reshape(-1).view(2, -1)[0].repeat(2).view(expect["ag_rep"].shape)
        n = {"ops": 0}
        def cnt_isend(t, *a, **k):
            n["ops"] += 1
            return real_isend(t, *a, **k)
        def cnt_irecv(t, *a, **k):
            n["ops"] += 1
            return real_irecv(t, *a, **k)
        if rank == 0:
            src = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in expect.items()}
            dist.isend = cnt_isend
            try:
                # Worker pattern: top-of-step wait on the previous device sends.
                for h in prev: h.wait()
                prev = c.isend_tensor_dict(src, dst=1, all_gather_group=agg, all_gather_tensors=agt)
            finally:
                dist.isend = real_isend
            prev = prev[1:]
            # Packed: the wire buffer is a copy, so sources may be reused at
            # once (unpacked sends still alias their sources: not here).
            for v in (src.values() if packed and kind in (1, 2) else ()):
                if isinstance(v, torch.Tensor) and v.numel() and v.dtype != torch.bool:
                    v.fill_(0)
            counts.append(n["ops"] - 2)  # minus metadata size + object
            assert len(c._pending_isends) <= 64
        else:
            dist.irecv = cnt_irecv
            try:
                td, hs, post = c.irecv_tensor_dict(src=0, all_gather_group=agg, all_gather_tensors=agt)
            finally:
                dist.irecv = real_irecv
            for h in hs: h.wait()
            for fn in post: fn()
            same(expect, td)
            counts.append(n["ops"])
    for h, _ in list(c._pending_isends):
        for x in h: x.wait()


def worker(rank, path, mode, out):
    os.environ["VLLM_PP_PACK_TENSORS_0029"] = mode[rank] if isinstance(mode, tuple) else mode
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    pack, ns, methods = load()
    coord = make_coord(methods, rank, False)
    if isinstance(mode, tuple):
        try:
            pack.configure(coord)
        except RuntimeError as e:
            assert "identically" in str(e)
            out[rank] = "raised"
        else:
            out[rank] = "no-raise"
        dist.destroy_process_group()
        return
    flag = pack.configure(coord)
    assert flag == (mode == "1")
    counts: list[int] = []
    run_steps(rank, methods, flag, 200, counts)
    out[rank] = counts
    dist.barrier(); dist.destroy_process_group()


def spawn(mode):
    with tempfile.TemporaryDirectory(prefix="pp0029-") as d:
        mgr = mp.Manager(); out = mgr.dict()
        mp.spawn(worker, args=(d + "/init", mode, out), nprocs=2, join=True)
        return dict(out)


def check_layout():
    pack, _, _ = load()
    ents = [("h", "cuda", torch.bfloat16, (5, 4, 4096)), ("p", "cuda", torch.float32, (5, 4096)),
            ("b", "cuda", torch.bool, (7,)), ("e", "cuda", torch.float32, (0, 4)),
            ("i", "cuda", torch.int64, (3,)), ("c", "cpu", torch.int32, (2,))]
    segs, total = pack.plan(ents, lambda k, n: True)
    assert [s[0] for s in segs] == [0, 1, 2, 4], segs     # empty + other device left out
    assert all(off % 16 == 0 for _, off, _ in segs) and total % 16 == 0, segs
    assert segs[1][1] == 5 * 4 * 4096 * 2 and segs[3][1] == segs[2][1] + 16
    assert pack.plan(ents[:1], lambda k, n: True) is None
    print("PASS layout: 16-byte aligned segments, total", total)


if __name__ == "__main__":
    check_layout()
    off = spawn("0"); on = spawn("1")
    # tensors per kind: 0 ->1, 1 ->2, 2 ->6 non-empty, 3 ->7 (1 all-gathered)
    exp_off = [(1, 2, 6, 7)[s % 4] for s in range(200)]
    exp_on = [(1, 1, 1, 2)[s % 4] for s in range(200)]
    for r in (0, 1):
        assert off[r] == exp_off, (r, off[r][:8])
        assert on[r] == exp_on, (r, on[r][:8])
    print("PASS off: bit-identical, P2P ops per hop", sorted(set(exp_off)))
    print("PASS on : bit-identical, key order kept, P2P ops per hop", sorted(set(exp_on)))
    mm = spawn(("1", "0"))
    assert mm == {0: "raised", 1: "raised"}, mm
    print("PASS mismatch: both ranks raise")
