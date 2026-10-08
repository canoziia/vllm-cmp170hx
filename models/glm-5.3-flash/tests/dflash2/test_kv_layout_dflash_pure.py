#!/usr/bin/env python3
"""Pure-Python (no torch) reproduction of the PP4 + DFlash2 KV-init failure and
check of patch 0003.

The real grouping functions are extracted from `kv_cache_utils.py` with `ast`
(pristine `/tmp/ours` for "before", `../patched` for "after"). They are executed
against lightweight stand-ins for the spec classes. The stand-ins reproduce the
page arithmetic of `vllm/v1/kv_cache_interface.py`:
`num_heads * (block // tokens_per_state) * (head + head_v) * dtype_bytes`,
`page_size_padded` overrides it, and Mamba pages come from state shapes.

Spec numbers (see ANALYSIS.md section 6):
* MLA latent: kv_lora_rank 512 + rope 0, bf16. That is 1024 B/token.
* kpool indexer k_cache: 132 B uint8 per pool entry, tokens_per_state = 4.
  That is 33 B/token.
* kpool tail: ring 8 (patch 0002, k = 3), 2 x 128 bf16.
* KDA state, TP1, k = 3: 64*128*128*4 + 6*24576*2 = 4,489,216 B. This page is
  padded to the MLA page by the platform.
* DFlash2 SWA layer: 8 KV heads x (128 + 128) x bf16 = 4096 B/token. Its block
  equals the cache block (FlashAttention advertises MultipleOf(16)).

Run: python3 tests/test_kv_layout_dflash_pure.py   (or with pytest)
"""

import os
import ast
import dataclasses
import math
import pathlib
import re
import sys
import types
from dataclasses import dataclass, replace
from typing import cast

HERE = pathlib.Path(__file__).resolve().parent
# GLM_DFLASH2_PRISTINE_TREE: tree before the dflash2 series (DeepSeek image source);
# GLM_DFLASH2_TREE: tree after it. Both are directories that hold vllm/.
PRISTINE = pathlib.Path(os.environ["GLM_DFLASH2_PRISTINE_TREE"]) / "vllm/v1/core/kv_cache_utils.py"
PATCHED = pathlib.Path(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages")) / "vllm/v1/core/kv_cache_utils.py"

KPOOL = 4
NUM_LAYERS = 45
PARTITION = [13, 11, 11, 10]
MLA_LAYERS = [i for i in range(NUM_LAYERS) if i % 4 == 3]  # inference: 3,7,..,43
MAMBA_REAL_PAGE = 64 * 128 * 128 * 4 + (3 + 3) * (3 * 64 * 128) * 2  # 4,489,216


def cdiv(a, b):
    return -(-a // b)


# ---- spec stand-ins -------------------------------------------------------------
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int = 1
    head_size: int = 0
    head_size_v: int = 0
    dtype_bytes: int = 2
    tokens_per_state: int = 1
    page_size_padded: int | None = None

    @property
    def unpadded_page_size_bytes(self):
        return (self.num_kv_heads * (self.block_size // self.tokens_per_state)
                * (self.head_size + self.head_size_v) * self.dtype_bytes)

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.unpadded_page_size_bytes


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    pass


@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):
    pass


@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int = 0


@dataclass(frozen=True)
class KpoolTailSpec(SlidingWindowSpec):
    pass


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    real_page_size_bytes: int = 0
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.real_page_size_bytes


class UniformTypeKVCacheSpecs:
    def __init__(self, block_size, kv_cache_specs):
        self.block_size = block_size
        self.kv_cache_specs = kv_cache_specs

    @classmethod
    def from_specs(cls, specs):
        types_ = {type(s) for s in specs.values()}
        if len(types_) != 1:
            return None
        return cls(next(iter(specs.values())).block_size, dict(specs))


@dataclass
class KVCacheGroupSpec:
    layer_names: list
    kv_cache_spec: object
    is_eagle_group: bool = False


def create_kv_cache_group_specs(specs, grouped_names):
    return [KVCacheGroupSpec(names, specs[names[0]]) for names in grouped_names]


class _Log:
    def __init__(self):
        self.records = []

    def _rec(self, msg, *args):
        self.records.append(msg % args if args else msg)

    warning = info = warning_once = info_once = debug = _rec


# vllm.distributed.utils / vllm.model_executor.models.utils stubs (function-local
# imports inside the extracted code)
def _get_pp_indices(total, rank, size):
    b = [sum(PARTITION[:r]) for r in range(len(PARTITION) + 1)]
    return b[rank], b[rank + 1]


def _extract_layer_index(name):
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


for mod, attrs in {
    "vllm": {},
    "vllm.distributed": {},
    "vllm.distributed.utils": {"get_pp_indices": _get_pp_indices},
    "vllm.model_executor": {},
    "vllm.model_executor.models": {},
    "vllm.model_executor.models.utils": {"extract_layer_index": _extract_layer_index},
}.items():
    m = sys.modules.setdefault(mod, types.ModuleType(mod))
    for k, v in attrs.items():
        setattr(m, k, v)

FUNCS = [
    "unify_kv_cache_spec_page_size",
    "_pp_balanced_mamba_group_count",
    "_get_kv_cache_groups_glm5_next",
    "_glm5_next_draft_mla_slots",
    "_glm5_next_draft_groups",
]


def load(path):
    tree = ast.parse(path.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    for f in fns:
        f.returns = None
        for a in f.args.args + f.args.kwonlyargs:
            a.annotation = None
    log = _Log()
    ns = dict(
        cdiv=cdiv, math=math, replace=replace, cast=cast, logger=log,
        KVCacheSpec=KVCacheSpec, AttentionSpec=AttentionSpec,
        MLAAttentionSpec=MLAAttentionSpec, SlidingWindowSpec=SlidingWindowSpec,
        KpoolTailSpec=KpoolTailSpec, MambaSpec=MambaSpec,
        UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
        KVCacheGroupSpec=KVCacheGroupSpec,
        create_kv_cache_group_specs=create_kv_cache_group_specs,
    )
    exec(compile(ast.Module(fns, []), str(path), "exec"), ns)
    ns["_log"] = log
    return ns


def vllm_config(spec=True):
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=len(PARTITION)),
        model_config=types.SimpleNamespace(get_total_num_hidden_layers=lambda: NUM_LAYERS),
        speculative_config=(
            types.SimpleNamespace(use_eagle_block_drop=lambda: True) if spec else None
        ),
    )


def merged_specs(block, with_drafter, ring=8):
    """Merged (all PP workers) spec dict, as get_kv_cache_configs builds it."""
    mla_page = block * 1024
    specs = {}
    for i in range(NUM_LAYERS):
        p = f"language_model.model.layers.{i}.self_attn"
        if i in MLA_LAYERS:
            specs[f"{p}.attn"] = MLAAttentionSpec(block, num_kv_heads=1, head_size=512)
            specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(
                block, num_kv_heads=1, head_size=132, dtype_bytes=1,
                tokens_per_state=KPOOL)
            specs[f"{p}.indexer.tail_cache"] = KpoolTailSpec(
                ring, num_kv_heads=2, head_size=128, sliding_window=ring)
        else:
            specs[f"{p}.attn"] = MambaSpec(
                block, real_page_size_bytes=MAMBA_REAL_PAGE, page_size_padded=mla_page)
    if with_drafter:
        for j in range(5):
            specs[f"draft.model.layers.{j}.self_attn.attn"] = SlidingWindowSpec(
                block, num_kv_heads=8, head_size=128, head_size_v=128,
                sliding_window=2048)
    return specs


def generic_path(ns, specs):
    """get_kv_cache_groups after the GLM path declined: unify the pages."""
    return ns["unify_kv_cache_spec_page_size"](specs)


def summary(groups):
    return [
        (tuple(g.layer_names), type(g.kv_cache_spec).__name__,
         g.kv_cache_spec.block_size, g.is_eagle_group)
        for g in groups
    ]


# ---- tests ----------------------------------------------------------------------
def test_page_sizes_table():
    for block in (4480, 4608):
        s = merged_specs(block, True)
        pages = {
            "mla": s["language_model.model.layers.3.self_attn.attn"].page_size_bytes,
            "indexer": s["language_model.model.layers.3.self_attn.indexer.k_cache"].page_size_bytes,
            "tail": s["language_model.model.layers.3.self_attn.indexer.tail_cache"].page_size_bytes,
            "mamba": s["language_model.model.layers.0.self_attn.attn"].page_size_bytes,
            "drafter": s["draft.model.layers.0.self_attn.attn"].page_size_bytes,
        }
        print(block, pages)
        assert pages["mla"] == block * 1024
        assert pages["indexer"] == block // 4 * 132
        assert pages["drafter"] == 4 * pages["mla"]  # the max page
        assert pages["drafter"] % pages["indexer"] != 0
    # 4480 is the platform's choice for this mamba page (k = 3):
    assert 128 * cdiv(MAMBA_REAL_PAGE, 128 * 1024) == 4480


def _raises(fn, *a):
    try:
        fn(*a)
    except NotImplementedError as e:
        return str(e)
    return None


def test_before_reproduces_error():
    ns = load(PRISTINE)
    for block in (4480, 4608):
        specs = merged_specs(block, True)
        assert ns["_get_kv_cache_groups_glm5_next"](vllm_config(), specs) is None
        msg = _raises(generic_path, ns, specs)
        assert msg and "indexer.k_cache: page size is not divisible" in msg, msg


def test_root_cause_is_indexer_vs_mla_not_drafter_size():
    """Even a drafter page smaller than the MLA page fails in the generic path:
    the indexer page (33 B/token) never divides the MLA page (1024 B/token)."""
    ns = load(PRISTINE)
    specs = merged_specs(4608, True)
    for k, v in list(specs.items()):
        if k.startswith("draft."):
            specs[k] = replace(v, block_size=64)  # 262,144 B < MLA page
    assert ns["_get_kv_cache_groups_glm5_next"](vllm_config(), specs) is None
    assert _raises(generic_path, ns, specs)


def test_after_5120_uses_shared_layout():
    ns = load(PATCHED)
    block = 5120
    groups = ns["_get_kv_cache_groups_glm5_next"](vllm_config(), merged_specs(block, True))
    assert groups is not None
    draft = [g for g in groups if g.layer_names[0].startswith("draft.")]
    # last stage 35..44 owns MLA layers 35, 39, 43 -> 3 slots -> 2 groups [3, 2]
    slots = sum(35 <= i < 45 for i in MLA_LAYERS)
    assert [len(g.layer_names) for g in draft] == (
        [3, 2] if slots == 3 else [len(g.layer_names) for g in draft])
    for g in draft:
        assert g.is_eagle_group
        for spec in g.kv_cache_spec.kv_cache_specs.values():
            assert spec.block_size == 1024
            assert spec.page_size_bytes == block * 1024
            assert spec.page_size_padded == block * 1024
            assert spec.unpadded_page_size_bytes == 4 * 1024**2
    # existing groups keep their positions and specs
    assert type(groups[0].kv_cache_spec).__name__ == "UniformTypeKVCacheSpecs"
    assert groups[-len(draft):] == draft


def test_after_no_256_requirement():
    ns = load(PATCHED)
    specs = merged_specs(4480, True)
    # K3 stand-in fits this page; actual K7 requires >=4608. No geometric
    # B/4 or 256-multiple restriction remains in the sharing helper.
    groups = ns["_get_kv_cache_groups_glm5_next"](vllm_config(), specs)
    assert groups is not None
    for g in groups:
        if g.layer_names[0].startswith('draft.'):
            assert g.kv_cache_spec.block_size == 1024


def test_no_drafter_behaviour_unchanged():
    before, after = load(PRISTINE), load(PATCHED)
    for block in (4480, 4608, 8960):
        for spec in (True, False):
            specs = merged_specs(block, False)
            g0 = before["_get_kv_cache_groups_glm5_next"](vllm_config(spec), specs)
            g1 = after["_get_kv_cache_groups_glm5_next"](vllm_config(spec), specs)
            assert summary(g0) == summary(g1)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
