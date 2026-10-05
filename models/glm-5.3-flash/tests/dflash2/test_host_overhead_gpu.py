# SPDX-License-Identifier: Apache-2.0
"""GPU tests for patches 0018-0020 (host-overhead series).

Needs: CUDA and the patched vLLM importable (the GLM DFlash2 image, or a tree
with the series applied on PYTHONPATH).

For every patch: flag off vs flag on on the same inputs, every model input /
metadata tensor compared bit for bit (dtype, shape, values), and the number of
``Memcpy HtoD`` device events counted with torch.profiler.

  python -m pytest -q -s test_host_overhead_gpu.py
"""

from __future__ import annotations

import random
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda:0")


def _count_copies(fn):
    """Run fn under the profiler; return (result, {'HtoD':n, 'DtoD':n, 'kernels':n})."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    c = {"HtoD": 0, "DtoD": 0, "kernels": 0, "sync": 0}
    for e in prof.events():
        name = e.name
        if "Memcpy HtoD" in name:
            c["HtoD"] += 1
        elif "Memcpy DtoD" in name:
            c["DtoD"] += 1
        elif e.device_type.name == "CUDA" and "Memcpy" not in name and "Memset" not in name:
            c["kernels"] += 1
        if name in ("cudaStreamSynchronize", "cudaDeviceSynchronize"):
            c["sync"] += 1
    return out, c


_BASE = None


def _baseline():
    """Counts of an empty window: _count_copies itself calls
    torch.cuda.synchronize() inside the profiled region (cudaDeviceSynchronize,
    and on some builds a cudaStreamSynchronize too). Subtract these."""
    global _BASE
    if _BASE is None:
        _count_copies(lambda: None)  # profiler warm-up
        _BASE = _count_copies(lambda: None)[1]
    return _BASE


def _net(c):
    b = _baseline()
    return {k: c[k] - b[k] for k in c}


def _eq(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert torch.equal(a, b), what


# --------------------------------------------------------------------------- #
# 0018: GDN builder
# --------------------------------------------------------------------------- #
def _gdn_builder(num_spec, max_num_seqs, mamba_block):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    b = object.__new__(GDNAttentionMetadataBuilder)
    b.num_spec = num_spec
    b.use_spec_decode = True
    b.use_full_cuda_graph = True
    b.decode_cudagraph_max_bs = max_num_seqs * (num_spec + 1)
    spec = object.__new__(MambaSpec)
    object.__setattr__(spec, "block_size", mamba_block)
    object.__setattr__(spec, "num_speculative_blocks", num_spec)
    b.kv_cache_spec = spec
    b.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    bs, k = b.decode_cudagraph_max_bs, num_spec + 1
    g = torch.Generator(device=DEV).manual_seed(3)
    r = lambda *s: torch.randint(-999, 999, s, generator=g, device=DEV, dtype=torch.int32)  # noqa: E731
    b.spec_state_indices_tensor = r(bs, k)
    b.non_spec_state_indices_tensor = r(bs)
    b.spec_sequence_masks = torch.randint(0, 2, (bs,), generator=g, device=DEV).bool()
    b.spec_token_indx = r(bs * k)
    b.non_spec_token_indx = r(bs * k)
    b.spec_query_start_loc = r(bs + 1)
    b.non_spec_query_start_loc = r(bs + 1)
    b.num_accepted_tokens = r(bs)
    return b


_BUF = (
    "spec_state_indices_tensor", "non_spec_state_indices_tensor", "spec_sequence_masks",
    "spec_token_indx", "non_spec_token_indx", "spec_query_start_loc",
    "non_spec_query_start_loc", "num_accepted_tokens",
)


def _gdn_batch(rng, num_spec, max_num_seqs, mamba_block):
    real = rng.randint(1, max_num_seqs)
    pad = rng.randint(0, max_num_seqs - real)
    n = real + pad
    drafts = [rng.randint(1, num_spec) for _ in range(real)]
    qlens = [d + 1 for d in drafts] + [0] * pad
    qsl = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(qlens, out=qsl[1:])
    ncols = 64
    seq = [rng.randint(q, (ncols - num_spec - 1) * mamba_block) for q in qlens[:real]] + [0] * pad
    m = SimpleNamespace(
        query_start_loc=torch.from_numpy(qsl).to(DEV),
        query_start_loc_cpu=torch.from_numpy(qsl.copy()),
        seq_lens=torch.tensor(seq, dtype=torch.int32, device=DEV),
        block_table_tensor=torch.randint(1, 10_000, (n, ncols), dtype=torch.int32, device=DEV),
        num_reqs=n,
        num_actual_tokens=int(qsl[-1]),
    )
    acc = torch.ones(n, dtype=torch.int32, device=DEV)
    acc[:real] = torch.tensor([rng.randint(1, num_spec + 1) for _ in range(real)], dtype=torch.int32)
    dd = torch.tensor(drafts + [-1] * pad, dtype=torch.int32)
    return m, acc, dd


def _set_fuse(on):
    from vllm.v1.worker.gpu import prologue_fuse as pf

    pf.settings = lambda: pf.PrologueFuseSettings(enabled=on, gdn=True)


def test_0018_gdn_bitwise_and_copies():
    rng = random.Random(0)
    tot = {"off": None, "on": None}
    for it in range(200):
        ns = rng.choice([3, 7])
        mns = rng.choice([1, 4, 8])
        mb = rng.choice([16, 4608])
        m, acc, dd = _gdn_batch(rng, ns, mns, mb)
        a, b = _gdn_builder(ns, mns, mb), _gdn_builder(ns, mns, mb)
        if it == 0:
            # JIT-compile / module-load both paths outside the measured window
            wa, wb = _gdn_builder(ns, mns, mb), _gdn_builder(ns, mns, mb)
            _set_fuse(False)
            wa.build(0, m, acc.clone(), dd.clone())
            _set_fuse(True)
            wb.build(0, m, acc.clone(), dd.clone())
        _set_fuse(False)
        ra, ca = _count_copies(lambda: a.build(0, m, acc.clone(), dd.clone()))
        _set_fuse(True)
        rb, cb = _count_copies(lambda: b.build(0, m, acc.clone(), dd.clone()))
        ca, cb = _net(ca), _net(cb)
        for f in fields(ra):
            x, y = getattr(ra, f.name), getattr(rb, f.name)
            if isinstance(x, torch.Tensor):
                _eq(x, y, f.name)
            else:
                assert x == y, f.name
        for n in _BUF:
            _eq(getattr(a, n), getattr(b, n), n)
        if it == 0:
            tot = {"off": ca, "on": cb}
    print("\n0018 per GDN build (one KDA group):", tot)
    # The two CPU-mask indexings are counted as HtoD too (pageable).
    assert tot["on"]["HtoD"] == 0 and tot["off"]["HtoD"] >= 1
    assert tot["on"]["kernels"] < tot["off"]["kernels"]
    # Net of the window's own torch.cuda.synchronize(): the fused path adds no
    # host sync; the upstream path's two CPU-mask indexings each add one.
    assert tot["on"]["sync"] == 0, tot
    print("upstream extra host syncs per GDN build:", tot["off"]["sync"])


# --------------------------------------------------------------------------- #
# 0019: prepare_inputs + prepare_attn with the real runner methods
# --------------------------------------------------------------------------- #
def _runner_fixture(num_reqs_live, num_spec, seed):
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.states import RequestState

    rng = np.random.default_rng(seed)
    max_reqs, max_tokens = 8, 256
    rs = RequestState(max_reqs, 4096, max_tokens, num_spec, 1000, DEV)
    bt = BlockTables([16, 64], max_reqs, max_tokens, [256, 64], DEV, [16, 64])
    req_ids = []
    for i in range(num_reqs_live):
        rid = f"r{i}"
        plen = int(rng.integers(5, 600))
        rs.add_request(rid, plen, list(rng.integers(0, 1000, plen)), plen, 64)
        idx = rs.req_id_to_index[rid]
        nb16 = (plen + 64) // 16 + 1
        nb64 = (plen + 64) // 64 + 1
        bt.append_block_ids(
            idx,
            (list(rng.permutation(4000)[:nb16]), list(rng.permutation(1000)[:nb64])),
            overwrite=True,
        )
        req_ids.append(rid)
    rs.apply_staged_writes()
    bt.apply_staged_writes()
    rs.last_sampled_tokens.copy_(torch.randint(0, 1000, rs.last_sampled_tokens.shape, device=DEV))
    rs.draft_tokens.copy_(torch.randint(0, 1000, rs.draft_tokens.shape, device=DEV))
    fake = SimpleNamespace(
        input_buffers=InputBuffers(max_reqs, max_tokens, DEV),
        req_states=rs,
        block_tables=bt,
        model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
        adaptive_verification=None,
        max_num_reqs=max_reqs,
        device=DEV,
        decode_query_len=num_spec + 1,
        model_config=SimpleNamespace(rswa_window=None),
        pcp_manager=None,
    )
    return fake, req_ids, rng


def _run_prepare(fake, req_ids, drafts, padded_reqs, padded_tokens):
    from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner

    rs = fake.req_states
    idx_np = np.array([rs.req_id_to_index[r] for r in req_ids], dtype=np.intp)
    nst = np.array([1 + len(drafts.get(r, ())) for r in req_ids], dtype=np.int32)
    brs = BatchReqState(
        req_ids=req_ids,
        num_scheduled_tokens=nst,
        num_tokens=int(nst.sum()),
        idx_mapping_np=idx_np,
        prefill_len_np=rs.prefill_len.np[idx_np],
        num_computed_prefill_tokens_np=rs.num_computed_prefill_tokens[idx_np],
        is_prefilling_np=np.zeros(len(req_ids), dtype=bool),
        has_prefill=False,
    )
    so = SimpleNamespace(scheduled_spec_decode_tokens=drafts, has_structured_output_requests=False)
    desc = SimpleNamespace(num_tokens=padded_tokens, num_reqs=padded_reqs)
    ib = GPUModelRunner.prepare_inputs(fake, so, brs, desc)
    bts, sms = GPUModelRunner.prepare_attn(fake, ib)
    snap = {
        "idx_mapping": ib.idx_mapping, "expanded_idx_mapping": ib.expanded_idx_mapping,
        "expanded_local_pos": ib.expanded_local_pos, "cu_num_logits": ib.cu_num_logits,
        "query_start_loc": ib.query_start_loc, "seq_lens": ib.seq_lens,
        "positions": ib.positions, "input_ids": ib.input_ids,
        "logits_indices": ib.logits_indices, "slot_mappings": sms,
        "qsl_buffer": fake.input_buffers.query_start_loc,
        "seq_lens_buffer": fake.input_buffers.seq_lens,
    }
    for i, t in enumerate(bts):
        snap[f"block_table_{i}"] = t
    return {k: v.clone() for k, v in snap.items()}, ib


def _set_packed(on):
    from vllm.v1.worker.gpu import prologue_h2d

    prologue_h2d.enabled = lambda: on


@pytest.mark.parametrize("with_drafts", [True, False])
def test_0019_prepare_inputs_bitwise(with_drafts):
    num_spec = 7
    for seed in range(20):
        n = 1 + seed % 8
        fake, req_ids, rng = _runner_fixture(n, num_spec, seed)
        drafts = (
            {r: list(range(int(rng.integers(1, num_spec + 1)))) for r in req_ids}
            if with_drafts else {}
        )
        ntok = sum(1 + len(drafts.get(r, ())) for r in req_ids)
        padded_reqs, padded_tokens = 8, max(ntok, 64)
        qsl_ptr = fake.input_buffers.query_start_loc.data_ptr()

        def scribble():
            # prepare_pos_seq_lens / combine_sampled_and_draft_tokens write
            # only rows < num_tokens; rows [num_tokens, num_tokens_after_padding)
            # keep whatever the persistent buffer held (input_batch.py:327-360).
            # Give both runs the same stale contents.
            for t in (fake.input_buffers.query_start_loc, fake.input_buffers.positions,
                      fake.input_buffers.seq_lens, fake.input_buffers.input_ids):
                t.fill_(-3)

        # warm-up both paths (Triton JIT) outside the measured window
        for on in (False, True):
            _set_packed(on)
            scribble()
            _run_prepare(fake, req_ids, drafts, padded_reqs, padded_tokens)
        _set_packed(False)
        scribble()
        (a, _), ca = _count_copies(lambda: _run_prepare(fake, req_ids, drafts, padded_reqs, padded_tokens))
        _set_packed(True)
        scribble()
        (b, ib), cb = _count_copies(lambda: _run_prepare(fake, req_ids, drafts, padded_reqs, padded_tokens))
        ca, cb = _net(ca), _net(cb)
        assert fake.input_buffers.query_start_loc.data_ptr() == qsl_ptr
        assert set(a) == set(b)
        for k in a:
            _eq(a[k], b[k], k)
        # real rows are produced, padding rows are untouched stale (-3)
        assert (b["positions"][ntok:] == -3).all() and (b["positions"][:ntok] >= 0).all()
        if seed == 0:
            print(f"\n0019 prepare_inputs+prepare_attn drafts={with_drafts}: off={ca} on={cb}")
            # off: idx_mapping (+ cu_num_logits) + query_start_loc; on: one.
            saved = 2 if with_drafts else 1
            assert cb["HtoD"] == ca["HtoD"] - saved, (ca, cb)
            assert cb["sync"] <= ca["sync"], (ca, cb)
    _set_packed(False)


# --------------------------------------------------------------------------- #
# 0020: sparse-MLA req_id_per_token
# --------------------------------------------------------------------------- #
def test_0020_req_ids_bitwise():
    from vllm.utils.torch_utils import np_to_pinned_tensor
    from vllm.v1.attention.backends.mla.req_id_per_token import fill_req_id_per_token

    rng = np.random.default_rng(4)
    L = 2312
    for it in range(300):
        real = int(rng.integers(0, 9))
        pad = int(rng.integers(0, 4))
        lens = list(rng.integers(0, 64, real)) + [0] * pad
        qsl = np.zeros(real + pad + 1, dtype=np.int32)
        np.cumsum(lens, out=qsl[1:])
        qsl_cpu = torch.from_numpy(qsl)
        qsl_gpu = qsl_cpu.to(DEV)
        a = torch.full((L,), 77, dtype=torch.int32, device=DEV)
        b = a.clone()

        def upstream():
            starts = np.asarray(qsl_cpu, dtype=np.int32)
            seg = np.diff(starts)
            ids = np.repeat(np.arange(seg.shape[0], dtype=np.int32), seg)
            a.fill_(0)
            a[: ids.shape[0]].copy_(np_to_pinned_tensor(ids), non_blocking=True)

        _, ca = _count_copies(upstream)
        _, cb = _count_copies(lambda: fill_req_id_per_token(qsl_gpu, real + pad, b))
        ca, cb = _net(ca), _net(cb)
        _eq(a, b, "req_id_per_token")
        if it == 0:
            print("\n0020 per sparse-MLA build:", ca, cb)
        if int(qsl[-1]) > 0:
            assert ca["HtoD"] == 1 and cb["HtoD"] == 0, (ca, cb)
