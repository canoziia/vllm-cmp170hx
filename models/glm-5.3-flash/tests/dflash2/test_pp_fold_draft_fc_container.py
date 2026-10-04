# SPDX-License-Identifier: Apache-2.0
"""Patch 0023 (VLLM_GLM5_PP_FOLD_DRAFT_FC), CPU only.

Needs torch + safetensors + an importable vllm with 0001..0023 applied
(the production container). Run from this directory:

    python -m pytest -q tests/test_pp_fold_draft_fc_container.py

Adapted from Morrowmake tests/models/glm5next/test_pp_fold_draft_fc.py,
re-targeted at vllm.models.glm5next.nvidia.model through the stub stages of
test_aux_hidden_states_pp.py.

Key property: a simulated N-stage run with the fold hands the drafter one
fp32 tensor equal to fc(cat(aux)) of a single-stage run up to the fp32
summation order; after the drafter's one rounding to bf16 the two agree to
within one bf16 ulp. Also: the receive buffer names exactly the keys the
sender sends (the runner indexes every buffer key in the received dict), the
relay is switched off, off/unsupported is a no-op, checkpoint name aliases,
the drafter's folded combine path, the last-stage weight cross-check.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_glm5next_aux_hidden_states_pp import (  # noqa: E402
    DFLASH2_AUX,
    HIDDEN,
    TOKENS,
    VOCAB,
    _build_stage,
    _set_pp,
)

import vllm.models.glm5next.nvidia.model as glm_model  # noqa: E402
from vllm.model_executor.models.qwen3_dflash import (  # noqa: E402
    DFlashQwen3ForCausalLM,
)
from vllm.sequence import IntermediateTensors  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.eagle import (  # noqa: E402
    aux_fc_fold,
    eagle3_utils,
)
from vllm.v1.worker.gpu.spec_decode.eagle.aux_fc_fold import (  # noqa: E402
    AUX_FC_PARTIAL_KEY,
    load_fc_column_blocks,
    maybe_configure_aux_fc_fold,
)


def _fc_file(tmp_path, num_aux, name="fc.weight", dtype=torch.bfloat16):
    torch.manual_seed(1)
    w = (torch.randn(HIDDEN, num_aux * HIDDEN) / HIDDEN**0.5).to(dtype)
    path = tmp_path / "model.safetensors"
    save_file({name: w, "norm.weight": torch.ones(HIDDEN)}, str(path))
    return w, [str(path)]


def _embed():
    torch.manual_seed(0)
    return nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)


def _single_stage(monkeypatch, embed, num_layers, aux_layers, ids, pos):
    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, num_layers, 0, num_layers, aux_layers)
    with torch.no_grad():
        return m(ids, pos, None)


def _folded_pipeline(
    monkeypatch, embed, num_layers, partition, aux_layers, ids, pos, files
):
    monkeypatch.setenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "1")
    spec = SimpleNamespace(method="dflash")
    world = len(partition)
    bounds = [sum(partition[:r]) for r in range(world + 1)]
    received = None
    for rank in range(world):
        _set_pp(monkeypatch, rank, world)
        m = _build_stage(embed, num_layers, bounds[rank], bounds[rank + 1], aux_layers)
        m.device = "cpu"
        wrapper = SimpleNamespace(
            model=m, make_empty_intermediate_tensors=m.make_empty_intermediate_tensors
        )
        # Runner order (model_runner.load_model): reserve aux slots
        # (set_eagle3_aux_hidden_state_layers), configure the relay, then fold.
        eagle3_utils.reserve_aux_intermediate_tensor_slots(wrapper)
        handler = SimpleNamespace(
            aux_hidden_state_relay_keys=eagle3_utils.aux_hidden_state_relay_keys(
                wrapper
            )
        )
        assert maybe_configure_aux_fc_fold(
            wrapper, spec, handler, files=files, dtype=torch.bfloat16
        )
        assert handler.aux_hidden_state_relay_keys == ()
        if rank > 0:
            buf = wrapper.make_empty_intermediate_tensors(TOKENS, torch.bfloat16, "cpu")
            # Only the streams and the fp32 partial sum travel; the reserved
            # aux slots are gone.
            assert set(buf.tensors) == {"hidden_states", AUX_FC_PARTIAL_KEY}
            assert buf[AUX_FC_PARTIAL_KEY].dtype == torch.float32
            assert set(received.tensors) == set(buf.tensors)
            for k, v in buf.tensors.items():
                assert v.shape == received[k].shape, k
        with torch.no_grad():
            out = m(ids, pos, received)
        if rank == world - 1:
            return out
        assert isinstance(out, IntermediateTensors)
        received = out
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "num_layers,partition,aux_layers",
    [
        (45, [13, 11, 11, 10], DFLASH2_AUX),  # our recipe split
        (45, [13, 11, 12, 9], DFLASH2_AUX),
        (45, [12, 11, 11, 11], DFLASH2_AUX),
        (10, [3, 2, 3, 2], (2, 3, 5, 7, 8)),  # aux on boundaries; none last
        (12, [4, 4, 2, 2], (1, 2, 11)),  # middle stages without aux
    ],
)
def test_folded_fc_equals_single_stage_fc(
    monkeypatch, tmp_path, num_layers, partition, aux_layers
):
    embed = _embed()
    ids = torch.randint(0, VOCAB, (TOKENS,))
    pos = torch.arange(TOKENS)
    w, files = _fc_file(tmp_path, len(aux_layers))
    _, aux = _single_stage(monkeypatch, embed, num_layers, aux_layers, ids, pos)
    # Unfolded fc: one GEMM over K = n * hidden, fp32 accumulation.
    ref = torch.mm(torch.cat(aux, dim=-1).float(), w.float().t())
    _, out = _folded_pipeline(
        monkeypatch, embed, num_layers, partition, aux_layers, ids, pos, files
    )
    assert len(out) == 1
    folded = out[0]
    assert folded.dtype == torch.float32 and folded.shape == (TOKENS, HIDDEN)
    torch.testing.assert_close(folded, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        folded.to(torch.bfloat16).float(),
        ref.to(torch.bfloat16).float(),
        rtol=2**-7,
        atol=1e-6,
    )


@pytest.mark.parametrize("name", ["fc.weight", "model.fc.weight", "encoder.fc.weight"])
def test_column_blocks_are_the_fc_slices(tmp_path, name):
    w, files = _fc_file(tmp_path, 5, name=name)
    blocks = load_fc_column_blocks(files, [0, 3, 4], HIDDEN, 5)
    for i, b in blocks.items():
        assert torch.equal(b, w[:, i * HIDDEN : (i + 1) * HIDDEN])
    with pytest.raises(ValueError):
        load_fc_column_blocks(files, [0], HIDDEN, 4)


def test_missing_fc_raises(tmp_path):
    path = tmp_path / "x.safetensors"
    save_file({"norm.weight": torch.ones(HIDDEN)}, str(path))
    with pytest.raises(FileNotFoundError):
        load_fc_column_blocks([str(path)], [0], HIDDEN, 1)


def test_fold_off_or_unsupported_is_a_no_op(monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", raising=False)
    assert not maybe_configure_aux_fc_fold(SimpleNamespace(), None, None)
    monkeypatch.setenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "1")
    _set_pp(monkeypatch, 0, 1)
    assert not maybe_configure_aux_fc_fold(
        SimpleNamespace(model=SimpleNamespace()), SimpleNamespace(method="dflash"), None
    )
    _set_pp(monkeypatch, 0, 4)
    m = _build_stage(_embed(), 45, 0, 13, DFLASH2_AUX)
    w = SimpleNamespace(model=m, make_empty_intermediate_tensors=None)
    assert not maybe_configure_aux_fc_fold(w, SimpleNamespace(method="eagle3"), None)
    assert m.aux_fc_blocks is None


def test_fold_off_leaves_forward_unchanged(monkeypatch):
    """Off: the boundary carries aux_hidden_states_<slot> as before 0023."""
    monkeypatch.delenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", raising=False)
    _set_pp(monkeypatch, 0, 4)
    m = _build_stage(_embed(), 45, 0, 13, DFLASH2_AUX)
    with torch.no_grad():
        out = m(torch.randint(0, VOCAB, (TOKENS,)), torch.arange(TOKENS), None)
    assert set(out.tensors) == {"hidden_states", "aux_hidden_states_0"}


def test_drafter_takes_the_folded_fc_output():
    d = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(d)
    fc = SimpleNamespace(output_size=HIDDEN, weight=torch.zeros(1, dtype=torch.bfloat16))
    d.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    d.aux_fc_folded = True
    x = torch.randn(TOKENS, HIDDEN)
    out = d.combine_hidden_states(x)
    assert out.dtype == torch.bfloat16 and torch.equal(out, x.to(torch.bfloat16))
    with pytest.raises(AssertionError):
        d.combine_hidden_states(torch.randn(TOKENS, 5 * HIDDEN))


def test_mark_drafter_cross_checks_blocks():
    torch.manual_seed(3)
    W = torch.randn(HIDDEN, 5 * HIDDEN).to(torch.bfloat16)
    inner = SimpleNamespace(aux_fc_blocks={4: W[:, 4 * HIDDEN :].clone()})
    target = SimpleNamespace(model=inner)
    spec = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(fc=SimpleNamespace(weight=W)))
    )
    aux_fc_fold.mark_drafter_aux_fc_folded(spec, target)
    assert spec.model.aux_fc_folded
    inner.aux_fc_blocks = {4: W[:, :HIDDEN].clone()}  # wrong block
    spec2 = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(fc=SimpleNamespace(weight=W)))
    )
    with pytest.raises(RuntimeError):
        aux_fc_fold.mark_drafter_aux_fc_folded(spec2, target)


def test_mm_fp32_matches_upcast():
    x = torch.randn(7, HIDDEN).to(torch.bfloat16)
    w = torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16)
    out = glm_model._mm_fp32(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.float() @ w.float().t())
