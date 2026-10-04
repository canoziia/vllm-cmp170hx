"""Patch 0016 (VLLM_GLM5_INDEXER_GATHER_CLAMP). Runs inside the patched
container; needs `import vllm` but no GPU. Not executed by the author session.

    python -m pytest -q -s tests/test_indexer_gather_clamp.py

Checks:
  * default (env unset) returns the legacy heuristic exactly;
  * clamp = min(legacy, max_num_seqs * cdiv(max_model_len, kpool)), and the
    saving in bytes at the deployment shape;
  * _split_indexer_prefill_chunks forms identical chunks with the legacy and
    the clamped workspace for random batches of <= max_num_seqs prefills at
    any length <= max_model_len (so outputs are bitwise unchanged), and every
    chunk's gathered rows fit the clamped workspace.
"""
import random
from types import SimpleNamespace

import pytest
import torch

MAX_LEN, SEQS, KPOOL, MNBT = 262144, 8, 4, 2312


def cfg():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=SEQS),
    )


def test_default_is_legacy(monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_INDEXER_GATHER_CLAMP", raising=False)
    from vllm.v1.attention.backends.mla import indexer

    assert indexer.get_indexer_gather_workspace_size(cfg(), KPOOL) == (
        indexer.get_max_prefill_buffer_size(cfg())
    ) == 131072 * 40


def test_clamp_value(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_INDEXER_GATHER_CLAMP", "1")
    from vllm.v1.attention.backends.mla import indexer

    got = indexer.get_indexer_gather_workspace_size(cfg(), KPOOL)
    assert got == SEQS * (MAX_LEN // KPOOL) == 524288
    saved = (131072 * 40 - got) * (128 + 4)
    print(f"\nworkspace rows {131072*40} -> {got}; saves {saved/2**20:.1f} MiB/card")


@pytest.mark.parametrize("seed", range(20))
def test_chunks_identical(seed, monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_INDEXER_GATHER_CLAMP", "1")
    from vllm.v1.attention.backends.mla import indexer

    split = indexer.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks
    legacy = indexer.get_max_prefill_buffer_size(cfg())
    clamped = indexer.get_indexer_gather_workspace_size(cfg(), KPOOL)
    rnd = random.Random(seed)
    for logits_mb in (128, 512):
        n = rnd.randint(1, SEQS)
        budget = MNBT
        q_lens, s_lens = [], []
        for _ in range(n):
            q = max(1, min(budget, rnd.randint(1, MNBT)))
            budget = max(1, budget - q)
            s = rnd.choice([q, rnd.randint(q, MAX_LEN), MAX_LEN])
            q_lens.append(q)
            s_lens.append(s // KPOOL)  # builder: seq_lens // compress_ratio
        ql, sl = torch.tensor(q_lens), torch.tensor(s_lens)
        a = split(sl, ql, legacy, logits_mb << 20)
        b = split(sl, ql, clamped, logits_mb << 20)
        assert a == b
        for req, _ in b:
            assert int(sl[req].sum()) <= clamped
