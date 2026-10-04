# SPDX-License-Identifier: Apache-2.0
"""Glm5Next (our nvidia/model.py, patch 0001) relays EAGLE3/DFlash auxiliary
hidden states across pipeline stages.

Key property: a simulated N-stage run (stage-by-stage forward, the runner's
relay of upstream keys on middle stages, the receiver's reserved slots) hands
the drafter the same aux tensors, in the same order, as a single-stage run.

Adapted from Morrowmake/vllm-cmp170hx
tests/models/glm5next/test_aux_hidden_states_pp.py, re-targeted at
vllm.models.glm5next.nvidia.model (our layout: make_empty_intermediate_tensors
is a method driven by config.mhc, the stage boundary materialises via
layer.hc_post).

CPU only, needs torch + an importable vllm with the patches applied.
The decoder layers are stubs with the same deferred-mHC contract as
Glm5NextDecoderLayer: a layer returns (branch, streams, post, comb) and leaves
its hc_post to the next layer, the first layer on a later stage takes
materialised streams, and the model's last layer contracts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.distributed.parallel_state as ps
import vllm.models.glm5next.nvidia.model as glm_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils

N_STREAMS = 4
HIDDEN = 16
TOKENS = 5
VOCAB = 32

# incoai/GLM-5.3-Flash-DFlash2 @ bf582e4e config.json
DFLASH2_TARGET_LAYER_IDS = [5, 14, 24, 33, 42]
DFLASH2_AUX = tuple(i + 1 for i in DFLASH2_TARGET_LAYER_IDS)  # (6,15,25,34,43)


class _StubLayer(nn.Module):
    def __init__(self, layer_idx: int, is_last: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_last = is_last

    @staticmethod
    def hc_post(branch, residual, post, comb):
        return residual * comb + branch[:, None, :] * post

    def forward(self, positions, hidden_states, residual, post, comb):
        if residual is None:
            if hidden_states.dim() == 2:  # layer 0: expand the embedding
                streams = hidden_states[:, None, :].expand(-1, N_STREAMS, -1)
                streams = streams.contiguous()
            else:  # first layer of a later stage: materialised streams
                streams = hidden_states
        else:  # fused path: apply the previous layer's deferred hc_post
            streams = self.hc_post(hidden_states, residual, post, comb)
        i = self.layer_idx
        branch = torch.tanh(streams.mean(1) * (1.0 + 0.1 * i)) + 0.01 * i
        p = torch.tensor(1.0 + 0.05 * i)
        c = torch.tensor(0.9 - 0.01 * i)
        if self.is_last:
            out = glm_model.hc_contract(self.hc_post(branch, streams, p, c), N_STREAMS)
            return out, None, None, None
        return branch, streams, p, c


class _FakePP:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1


def _set_pp(monkeypatch, rank: int, world_size: int) -> None:
    fake = _FakePP(rank, world_size)
    monkeypatch.setattr(glm_model, "get_pp_group", lambda: fake)
    monkeypatch.setattr(ps, "get_pp_group", lambda: fake)
    monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: True)


def _build_stage(embed, num_layers, start, end, aux_layers, mode="stream_mean"):
    m = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(m)
    m.config = SimpleNamespace(
        hidden_size=HIDDEN, mhc=True, mhc_num_residual_streams=N_STREAMS
    )
    m.mhc = True
    m.mhc_num_residual_streams = N_STREAMS
    m.is_sequence_parallel = False
    m.aux_hidden_state_mode = mode
    m.embed_tokens = embed
    m.norm = nn.Identity()
    m.start_layer, m.end_layer = start, end
    m._active_layers = [
        _StubLayer(i, is_last=(i == num_layers - 1)) for i in range(start, end)
    ]
    m._set_aux_hidden_state_layers(aux_layers)
    return m


def _wrapper(m):
    # Stand-in for the top-level model the runner holds: eagle3_utils unwraps
    # `.model` and replaces `make_empty_intermediate_tensors` on it.
    return SimpleNamespace(
        model=m, make_empty_intermediate_tensors=m.make_empty_intermediate_tensors
    )


def _run_single(monkeypatch, embed, num_layers, aux_layers, input_ids, positions):
    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, num_layers, 0, num_layers, aux_layers)
    with torch.no_grad():
        return m(input_ids, positions, None)


def _run_pipeline(
    monkeypatch, embed, num_layers, partition, aux_layers, input_ids, positions
):
    world = len(partition)
    bounds = [sum(partition[:r]) for r in range(world + 1)]
    received = None
    for rank in range(world):
        _set_pp(monkeypatch, rank, world)
        m = _build_stage(embed, num_layers, bounds[rank], bounds[rank + 1], aux_layers)
        wrapper = _wrapper(m)
        if rank > 0:
            # The receiver's persistent buffer must name exactly the keys the
            # sender produced (the runner copies every buffer key out of the
            # received tensors), with the same shapes.
            eagle3_utils.reserve_aux_intermediate_tensor_slots(wrapper)
            buf = wrapper.make_empty_intermediate_tensors(TOKENS, torch.float32, "cpu")
            assert set(buf.tensors) == set(received.tensors)
            for k, v in buf.tensors.items():
                assert v.shape == received[k].shape, k
        relay_keys = eagle3_utils.aux_hidden_state_relay_keys(wrapper)
        with torch.no_grad():
            out = m(input_ids, positions, received)
        if rank == world - 1:
            return out
        assert isinstance(out, IntermediateTensors)
        # PPHandler.relay_aux_hidden_states on middle stages
        if relay_keys:
            out = IntermediateTensors(out.tensors | {k: received[k] for k in relay_keys})
        received = out
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "num_layers,partition,aux_layers",
    [
        # aux ids on a stage boundary, several on one stage, and a last stage
        # with none of its own
        (10, [3, 2, 3, 2], (2, 3, 5, 7, 8)),
        # GLM-5.3-Flash + DFlash2 (45 layers), recipe PP4 split and variants
        (45, [13, 11, 11, 10], DFLASH2_AUX),
        (45, [12, 11, 11, 11], DFLASH2_AUX),
        (45, [14, 12, 12, 7], DFLASH2_AUX),
        # a middle stage with no aux layer of its own
        (12, [4, 4, 2, 2], (1, 2, 11)),
        (8, [4, 4], (4, 7)),
    ],
)
@pytest.mark.parametrize("mode", ["stream_mean", "branch"])
def test_aux_hidden_states_match_single_stage(
    monkeypatch, num_layers, partition, aux_layers, mode
):
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (TOKENS,))
    positions = torch.arange(TOKENS)

    # Every stage (and the single-stage reference) is built in `mode`.
    orig_build = _build_stage

    def build(*a, **k):
        return orig_build(*a, mode=mode, **k)

    monkeypatch.setitem(globals(), "_build_stage", build)

    ref_hidden, ref_aux = _run_single(
        monkeypatch, embed, num_layers, aux_layers, input_ids, positions
    )
    pp_hidden, pp_aux = _run_pipeline(
        monkeypatch, embed, num_layers, partition, aux_layers, input_ids, positions
    )

    assert len(ref_aux) == len(aux_layers)
    assert len(pp_aux) == len(ref_aux)
    torch.testing.assert_close(pp_hidden, ref_hidden, rtol=0, atol=1e-6)
    for got, want in zip(pp_aux, ref_aux):
        assert got.shape == (TOKENS, HIDDEN)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_stream_mean_is_contracted_materialised_state(monkeypatch):
    """stream_mean aux for layer i == mean over streams of the state after
    layer i (what the next layer would see), computed independently."""
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (TOKENS,))
    positions = torch.arange(TOKENS)
    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, 6, 0, 6, (3,))  # capture after layer 2
    with torch.no_grad():
        _, aux = m(input_ids, positions, None)
        # Reference: run layers 0..2 by hand and materialise.
        h, r, p, c = embed(input_ids), None, None, None
        for layer in m._active_layers[:3]:
            h, r, p, c = layer(positions, h, r, p, c)
        want = _StubLayer.hc_post(h, r, p, c).mean(1)
    torch.testing.assert_close(aux[0], want, rtol=0, atol=0)


def test_glm5next_declares_aux_relay_support():
    inner = Glm5NextModel.__new__(Glm5NextModel)
    eagle3_utils.verify_supports_aux_hidden_states_over_pp(
        SimpleNamespace(model=inner), "dflash"
    )


def test_outer_classes_expose_eagle3_interface():
    """The original failure: `Model does not support EAGLE3 interface`."""
    from vllm.model_executor.models.interfaces import SupportsEagle3

    for cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert issubclass(cls, SupportsEagle3), cls
        assert getattr(cls, "supports_eagle3", False) is True


def test_set_aux_layers_routes_to_inner_model(monkeypatch):
    """SupportsEagle3.set_aux_hidden_state_layers on the multimodal wrapper
    reaches Glm5NextModel through get_language_model().model."""
    _set_pp(monkeypatch, 0, 1)
    inner = _build_stage(nn.Embedding(VOCAB, HIDDEN), 4, 0, 4, ())
    lm = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(lm)
    lm.model = inner
    top = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(top)
    top.language_model = lm
    monkeypatch.setattr(
        Glm5NextForConditionalGeneration,
        "get_language_model",
        lambda self: self.language_model,
        raising=False,
    )
    top.set_aux_hidden_state_layers(DFLASH2_AUX)
    assert inner.aux_hidden_state_layers == DFLASH2_AUX


def test_dflash2_config_maps_to_capture_after_ids():
    hf = SimpleNamespace(
        dflash_config={"target_layer_ids": DFLASH2_TARGET_LAYER_IDS},
    )
    spec = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf))
    assert eagle3_utils.get_eagle3_aux_layers_from_config(spec) == DFLASH2_AUX


def test_recipe_partition_gives_every_stage_its_slots(monkeypatch):
    """13,11,11,10 over 45 layers: per-stage slot base and relay keys."""
    partition = [13, 11, 11, 10]
    bounds = [sum(partition[:r]) for r in range(5)]
    expected_base = [0, 1, 2, 4]  # aux 6 | 15 | 25,34 | 43
    for rank in range(4):
        _set_pp(monkeypatch, rank, 4)
        m = _build_stage(
            nn.Embedding(VOCAB, HIDDEN), 45, bounds[rank], bounds[rank + 1],
            DFLASH2_AUX,
        )
        assert m._aux_slot_base_cached == expected_base[rank]
        keys = eagle3_utils.aux_hidden_state_relay_keys(_wrapper(m))
        if rank in (1, 2):
            assert keys == tuple(
                f"aux_hidden_states_{i}" for i in range(expected_base[rank])
            )
        else:
            assert keys == ()


def test_no_aux_layers_leaves_boundary_unchanged(monkeypatch):
    """Without a drafter the boundary carries only the mHC streams, and the
    last stage returns a bare tensor (runner asserts on the type)."""
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    _set_pp(monkeypatch, 0, 2)
    m = _build_stage(embed, 4, 0, 2, ())
    with torch.no_grad():
        out = m(torch.randint(0, VOCAB, (TOKENS,)), torch.arange(TOKENS), None)
    assert isinstance(out, IntermediateTensors)
    assert set(out.tensors) == {"hidden_states"}
    assert out["hidden_states"].shape == (TOKENS, N_STREAMS, HIDDEN)

    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, 4, 0, 4, ())
    with torch.no_grad():
        out = m(torch.randint(0, VOCAB, (TOKENS,)), torch.arange(TOKENS), None)
    assert isinstance(out, torch.Tensor)


def test_last_stage_returns_tuple_even_without_local_aux(monkeypatch):
    """Last stage owning no aux layer still returns (hidden, remote_aux)."""
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (TOKENS,))
    positions = torch.arange(TOKENS)
    hidden, aux = _run_pipeline(
        monkeypatch, embed, 10, [3, 2, 3, 2], (2, 3, 5, 7, 8), input_ids, positions
    )
    assert isinstance(hidden, torch.Tensor) and len(aux) == 5


def test_target_embed_kept_on_last_rank_for_dflash(monkeypatch):
    from vllm.model_executor.models import utils as model_utils

    fake = _FakePP(3, 4)
    monkeypatch.setattr(ps, "get_pp_group", lambda: fake)
    cfg = SimpleNamespace(speculative_config=SimpleNamespace(method="dflash"))
    assert model_utils.spec_decode_needs_target_embed(cfg)
    fake = _FakePP(1, 4)
    monkeypatch.setattr(ps, "get_pp_group", lambda: fake)
    assert not model_utils.spec_decode_needs_target_embed(cfg)
