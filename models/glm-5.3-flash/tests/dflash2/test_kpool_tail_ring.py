# SPDX-License-Identifier: Apache-2.0
"""Patch 0002: the GLM-5 kpool indexer tail ring is sized for the draft depth.

Verify tokens are stashed into the per-request tail ring before acceptance.
With a one-pool ring (ring == index_kpool, our original code) the drafts behind
a rejected pool-completing draft overwrite the pool's earlier keys, so the redo
of the completing position compresses wrong keys into the indexer cache.

* CPU: a mirror of the kernel addressing reproduces the bug at ring == kpool
  and shows the enlarged ring fixes it (adapted from Morrowmake
  tests/v1/attention/test_kpool_tail_slot_mapping.py::
  test_rejected_completing_draft_needs_ring_slots), plus the ring sizes.
* CUDA: the real Triton kernel (kpool_decode_update_and_maybe_write_cache_batched)
  on the same scenario: the pool written by the redo must be byte-identical to
  the pool written without any rejected draft. Skipped without a GPU.
"""

import pytest
import torch

KPOOL = 4  # Glm5NextConfig.index_kpool default


class TailRingMirror:
    """Mirror of the tail-ring addressing in _kpool_tail_seed_kernel and
    _kpool_decode_update_batched_kernel (after patch 0002)."""

    def __init__(self, num_blocks, kpool=KPOOL, ring=None):
        self.kpool = kpool
        self.ring = ring or kpool
        self.k = torch.full((num_blocks, self.ring, 3), float("nan"))
        self.s = torch.full((num_blocks, self.ring, 3), float("nan"))

    def stash(self, tail_slot, pos, k, s):
        blk, off = tail_slot // self.ring, pos % self.ring
        self.k[blk, off] = k
        self.s[blk, off] = s

    def complete(self, tail_slot, pos, k, s):
        blk = tail_slot // self.ring
        start = pos - (self.kpool - 1)
        kk = torch.stack([self.k[blk, (start + i) % self.ring] for i in range(self.kpool)])
        ss = torch.stack([self.s[blk, (start + i) % self.ring] for i in range(self.kpool)])
        kk[-1], ss[-1] = k, s  # is_current for the completing token
        w = torch.softmax(ss, dim=0)
        return (kk * w).sum(0)


def token_kv(req, pos):
    k = torch.tensor([pos + 100.0 * req, pos + 0.5, 2.0 * pos + 0.25])
    s = torch.tensor([0.1 * (pos + 1) + req, 0.2 * pos, 0.05 * pos])
    return k, s


def _ring_size(num_spec):
    from vllm.models.glm5next.nvidia.attention import Glm5NextTailCache

    return Glm5NextTailCache.tail_ring_size(KPOOL, num_spec)


@pytest.mark.parametrize(
    "num_spec,ring", [(0, 4), (1, 8), (3, 8), (4, 8), (5, 16), (7, 16), (13, 32)]
)
def test_ring_size(num_spec, ring, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    got = _ring_size(num_spec)
    assert got == ring
    assert got >= KPOOL + num_spec and got % KPOOL == 0
    # power-of-two pools -> divides any attention block that is a multiple
    # of 128 (e.g. the recipe's 4608 PP4 block)
    assert 4608 % got == 0


@pytest.mark.parametrize("num_spec", [1, 3, 7])
def test_rejected_completing_draft_mirror(num_spec, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    for ring_size, expect_ok in ((KPOOL, False), (_ring_size(num_spec), True)):
        truth = TailRingMirror(2, ring=ring_size)
        ring = TailRingMirror(2, ring=ring_size)
        block = 1

        def slot(pos):
            return block * ring_size + pos % ring_size

        for pos in range(4, 7):
            truth.stash(slot(pos), pos, *token_kv(0, pos))
            ring.stash(slot(pos), pos, *token_kv(0, pos))
        expected = truth.complete(slot(7), 7, *token_kv(0, 7))
        # verify step: draft at 7 completes the pool and is rejected; drafts
        # 8 .. 7+num_spec are stashed in the same launch
        for pos in range(7, 8 + num_spec):
            k, s = token_kv(9, pos)
            if pos % KPOOL == KPOOL - 1:
                ring.complete(slot(pos), pos, k, s)
            ring.stash(slot(pos), pos, k, s)
        redo = ring.complete(slot(7), 7, *token_kv(0, 7))
        if expect_ok:
            torch.testing.assert_close(redo, expected)
        else:
            assert not torch.allclose(redo, expected)


# ----------------------------------------------------------------------------
# CUDA: the real kernel
# ----------------------------------------------------------------------------

HEAD_DIM = 128
PAGE = 64


def _launch(kv, tail, block, ring, positions, keys, scores, ape):
    from vllm.models.glm5next.nvidia.ops import kpool_compress as ops

    n = len(positions)
    pos = torch.tensor([positions], dtype=torch.int32, device="cuda")
    tail_slot = (block * ring + pos % ring).to(torch.int32)
    slot_mapping = (pos // KPOOL).to(torch.int32)  # pool-granular cache loc
    ops.kpool_decode_update_and_maybe_write_cache_batched(
        kv,
        tail,
        tail_slot,
        keys.view(1, n, HEAD_DIM),
        scores.view(1, n, HEAD_DIM),
        ape,
        slot_mapping,
        pos,
        KPOOL,
        HEAD_DIM,
    )


def _scenario(ring, with_rejected_draft):
    g = torch.Generator(device="cuda").manual_seed(0)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    true_k, true_s = rnd(16, HEAD_DIM), rnd(16, HEAD_DIM)
    draft_k, draft_s = rnd(16, HEAD_DIM), rnd(16, HEAD_DIM)
    ape = torch.randn(KPOOL, HEAD_DIM, generator=g, device="cuda")
    kv = torch.zeros(2, PAGE, HEAD_DIM + 4, dtype=torch.uint8, device="cuda")
    tail = torch.zeros(4, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    block = 2
    # positions 4,5,6 accepted (pool 1 in progress)
    _launch(kv, tail, block, ring, [4, 5, 6], true_k[4:7], true_s[4:7], ape)
    if with_rejected_draft:
        # verify 7..10 with drafts; 7 completes pool 1 and is then rejected
        _launch(kv, tail, block, ring, [7, 8, 9, 10], draft_k[7:11], draft_s[7:11], ape)
    # (re)compute position 7 with the accepted token
    _launch(kv, tail, block, ring, [7], true_k[7:8], true_s[7:8], ape)
    # pool 1 -> page 0, row 1. Page layout (kernel): PAGE rows of HEAD_DIM fp8
    # K bytes, then PAGE fp32 scales.
    page = kv[0].view(-1)
    k_bytes = page[1 * HEAD_DIM : 2 * HEAD_DIM].clone()
    scale = page[PAGE * HEAD_DIM + 4 * 1 : PAGE * HEAD_DIM + 4 * 2].clone()
    assert k_bytes.any(), "pool 1 was never written"
    return k_bytes, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_kernel_redo_matches_clean_with_enlarged_ring():
    ring = 8  # tail_ring_size(4, 3)
    clean = _scenario(ring, with_rejected_draft=False)
    redo = _scenario(ring, with_rejected_draft=True)
    assert torch.equal(clean[0], redo[0])
    assert torch.equal(clean[1], redo[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_kernel_one_pool_ring_reproduces_the_bug():
    """Documents the pre-patch behaviour (ring == kpool): the redo differs."""
    clean = _scenario(KPOOL, with_rejected_draft=False)
    redo = _scenario(KPOOL, with_rejected_draft=True)
    assert not (torch.equal(clean[0], redo[0]) and torch.equal(clean[1], redo[1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("ring", [KPOOL, 8])
def test_seed_kernel_writes_last_pool_at_pos_mod_ring(ring):
    from vllm.models.glm5next.nvidia.ops import kpool_compress as ops

    n = 10  # prompt of 10 tokens: incomplete tail = positions 8, 9
    block = 3
    pos = torch.arange(n, device="cuda")
    tslot = (block * ring + pos % ring).to(torch.int64)
    key = torch.randn(n, HEAD_DIM, device="cuda").to(torch.bfloat16)
    score = torch.randn(n, HEAD_DIM, device="cuda").to(torch.bfloat16)
    tail = torch.zeros(5, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ops.kpool_seed_tail_cache(tail, key, score, tslot, KPOOL, HEAD_DIM)
    # The seed writes the last KPOOL tokens of the request (tokens whose
    # KPOOL-ahead neighbour is in a different block / past the batch).
    for p in range(n - KPOOL, n):
        assert torch.equal(tail[block, 0, p % ring], key[p])
        assert torch.equal(tail[block, 1, p % ring], score[p])
