# SPDX-License-Identifier: Apache-2.0
"""Real NVIDIA kernels, no CPU addressing mirror. See ../ROLLBACK-TEST.md."""
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

POOL, DIM, PAGE = 4, 128, 64
CANDIDATE = Path(os.environ.get(
    "KPOOL_CANDIDATE", os.path.join(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages"),
                                    "vllm/models/glm5next/nvidia/ops/kpool_compress.py")
)).resolve()


@pytest.fixture(scope="module")
def ops():
    # Deliberately fail, not silently skip, when the GPU validation environment
    # is missing. Collection and py_compile do not initialize CUDA.
    assert torch.cuda.is_available(), "GPU harness requires CUDA; use --collect-only locally"
    assert torch.version.hip is None, "NVIDIA-only evidence"
    spec = importlib.util.spec_from_file_location("rollback_candidate_kpool", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == CANDIDATE
    print(f"\nkpool candidate={CANDIDATE} sha256={hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()}")
    print(f"torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
    return module


class Harness:
    def __init__(self, ops, ring, start, seed_mode):
        self.ops, self.ring = ops, ring
        g = torch.Generator(device="cuda").manual_seed(427)
        self.key = torch.randn(128, DIM, generator=g, device="cuda").bfloat16()
        self.score = torch.randn(128, DIM, generator=g, device="cuda").bfloat16()
        self.bad_key = torch.randn(128, DIM, generator=g, device="cuda").bfloat16() * 8
        self.bad_score = torch.randn(128, DIM, generator=g, device="cuda").bfloat16()
        self.ape = torch.randn(POOL, DIM, generator=g, device="cuda")
        self.kv = torch.full((4, PAGE, DIM + 4), 0xA5, device="cuda", dtype=torch.uint8)
        # Actual alias-style layout: K and score halves and request blocks have
        # gaps in a shared backing allocation, but each ring row is contiguous.
        self.backing = torch.full((5, 2, ring + 12, DIM), -123,
                                  device="cuda", dtype=torch.bfloat16)
        self.tail = self.backing[:, :, :ring, :]
        assert self.tail.stride(1) != ring * DIM
        assert self.tail.stride(0) != 2 * ring * DIM
        self.block = 2
        self.launch(0, start)
        if seed_mode != "decode":
            self.tail.fill_(-123)
            # Contiguous prefill inputs are required by the seed ABI. Chunked
            # mode uses absolute positions and two independently seeded chunks.
            chunks = [(0, start)] if seed_mode == "prefill" else [(0, start - 4), (start - 4, start)]
            for lo, hi in chunks:
                p = torch.arange(lo, hi, device="cuda", dtype=torch.int64)
                slots = self.block * ring + p % ring
                ops.kpool_seed_tail_cache(self.tail, self.key[lo:hi],
                                          self.score[lo:hi], slots, POOL, DIM)
                for pos in range(max(lo, hi - POOL), hi):
                    assert torch.equal(self.tail[self.block, 0, pos % ring], self.key[pos])
                    assert torch.equal(self.tail[self.block, 1, pos % ring], self.score[pos])
        self.guard = self.backing.clone()

    def launch(self, lo, hi, reject=None):
        if hi == lo:
            return
        p = torch.arange(lo, hi, device="cuda", dtype=torch.int32)[None, :]
        # Token-granular tail mapping; pool mapping is valid ONLY on completion,
        # as in the production metadata (unlike the older small test).
        loc = torch.where(p % POOL == POOL - 1, PAGE + p // POOL, -1)
        slots = self.block * self.ring + p % self.ring
        k, s = self.key[lo:hi], self.score[lo:hi]
        if reject is not None:
            mask = (p[0] >= reject)[:, None]
            k = torch.where(mask, self.bad_key[lo:hi], k)
            s = torch.where(mask, self.bad_score[lo:hi], s)
        # Non-dense token stride, passed unchanged to the production wrapper.
        kb = torch.empty(1, hi - lo, DIM * 2, device="cuda", dtype=torch.bfloat16)
        sb = torch.empty_like(kb)
        kb[:, :, :DIM] = k
        sb[:, :, :DIM] = s
        self.ops.kpool_decode_update_and_maybe_write_cache_batched(
            self.kv, self.tail, slots, kb[:, :, :DIM], sb[:, :, :DIM],
            self.ape, loc, p, POOL, DIM)

    def snapshot(self, end):
        # Only incomplete current pool is live raw tail. Speculative future
        # slots are intentionally not compared, nor is the entire ring.
        live = list(range(end // POOL * POOL, end))
        tail = self.tail[self.block, :, [p % self.ring for p in live], :].clone()
        # Compare every committed compressed pool: FP8 bytes AND fp32 scale
        # bytes, respecting the packed page layout, not kv[row] indexing.
        page = self.kv[1].flatten()
        pools = end // POOL
        kv = torch.cat((page[:pools * DIM],
                        page[PAGE * DIM:PAGE * DIM + pools * 4])).clone()
        return tail, kv

    def check_guards(self):
        assert torch.equal(self.backing[:, :, self.ring:], self.guard[:, :, self.ring:])
        for block in (0, 1, 3, 4):
            assert torch.equal(self.backing[block], self.guard[block])


def equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def scenario(ops, k, phase, base, offset, seed_mode, ring=16, rounds=2):
    start = base + phase
    clean = Harness(ops, ring, start, seed_mode)
    redo = Harness(ops, ring, start, seed_mode)
    outcomes = []
    for _ in range(rounds):
        # Position start is the known target input; k following draft inputs.
        # Reject offset is zero-based among those k draft positions.
        reject = None if offset == "all" else start + 1 + offset
        redo.launch(start, start + k + 1, reject)
        if reject is None:
            end = start + k + 1
            clean.launch(start, end)
        else:
            end = reject + 1
            clean.launch(start, end)
            redo.launch(reject, end)  # recompute rejected position with truth
        outcomes.append(equal(clean.snapshot(end), redo.snapshot(end)))
        # Check immediately, then each new committed token through the next
        # completion. This exposes contamination of an unfinished current pool.
        continuation_end = ((end + POOL - 1) // POOL + 1) * POOL
        for pos in range(end, continuation_end):
            clean.launch(pos, pos + 1)
            redo.launch(pos, pos + 1)
            outcomes.append(equal(clean.snapshot(pos + 1), redo.snapshot(pos + 1)))
        start = continuation_end
    clean.check_guards()
    redo.check_guards()
    torch.cuda.synchronize()
    return outcomes


CASES = [(k, phase, base, offset, seed)
         for k in (3, 7) for phase in range(4) for base in (12, 16)
         for offset in (*range(k), "all")
         for seed in ("decode", "prefill", "chunked")]


@pytest.mark.parametrize("k,phase,base,offset,seed", CASES)
def test_patched_rollback_matrix(ops, k, phase, base, offset, seed):
    # Ring16 is the maximum-k=7 allocation, also used at adaptive k=3.
    assert all(scenario(ops, k, phase, base, offset, seed))


def test_old_ring_negative_control_must_disagree(ops):
    # start=14, first rejected draft=15 completes pool; successors overwrite
    # positions 12..14 with ring4. Assertion of clean equality MUST fail.
    results = scenario(ops, 7, 2, 12, 0, "prefill", ring=4, rounds=1)
    assert not all(results), "negative control failed to expose old-ring corruption"
    with pytest.raises(AssertionError):
        assert all(results), "old ring differs from clean committed state"


@pytest.mark.parametrize("phase", range(4))
def test_k3_minimum_ring8(ops, phase):
    assert all(scenario(ops, 3, phase, 16, 0, "chunked", ring=8))
