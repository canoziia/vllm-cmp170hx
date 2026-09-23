#!/usr/bin/env python3
"""Regression: Qwen MTP marks only its dedicated full-attention KV group."""

import ast
import sys
import types
from pathlib import Path

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/local/lib/python3.12/dist-packages")
path = source_root / "vllm/v1/core/kv_cache_utils.py"
tree = ast.parse(path.read_text())
names = {"_is_qwen4_exp_mtp", "_annotate_eagle_groups"}
funcs = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in names
]
assert {func.name for func in funcs} == names
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *funcs,
    ],
    type_ignores=[],
)

class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args)

class MambaSpec:
    pass


logger = Logger()
namespace = {
    "logger": logger,
    "MambaSpec": MambaSpec,
    "iter_layer_specs": lambda spec: (spec,),
}
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
annotate = namespace["_annotate_eagle_groups"]

speculative = types.SimpleNamespace(
    method="mtp",
    draft_model_config=types.SimpleNamespace(
        hf_config=types.SimpleNamespace(model_type="qwen4_exp_mtp")
    ),
    use_eagle_block_drop=lambda: True,
)
config = types.SimpleNamespace(speculative_config=speculative)
groups = [
    types.SimpleNamespace(
        layer_names=["language_model.model.layers.0.linear_attn"],
        kv_cache_spec=types.SimpleNamespace(non_causal_multi_token_decode=False),
        is_eagle_group=False,
    ),
    types.SimpleNamespace(
        layer_names=["language_model.model.layers.7.self_attn.attn"],
        kv_cache_spec=types.SimpleNamespace(non_causal_multi_token_decode=False),
        is_eagle_group=False,
    ),
    types.SimpleNamespace(
        layer_names=["mtp.layers.48.self_attn.indexer.raw_key_cache"],
        kv_cache_spec=types.SimpleNamespace(non_causal_multi_token_decode=False),
        is_eagle_group=False,
    ),
    types.SimpleNamespace(
        layer_names=[
            "mtp.layers.48.self_attn.indexer.compressed_key_cache",
            "mtp.layers.48.self_attn.attn",
        ],
        kv_cache_spec=types.SimpleNamespace(non_causal_multi_token_decode=False),
        is_eagle_group=False,
    ),
]
annotate(config, {}, groups)
assert [group.is_eagle_group for group in groups] == [False, False, True, True]
assert logger.messages and "2 Qwen4Exp" in logger.messages[-1]

try:
    annotate(config, {}, groups[:2])
except ValueError as error:
    assert "did not resolve" in str(error)
else:
    raise AssertionError("missing Qwen MTP group did not fail closed")

mamba_group = types.SimpleNamespace(
    layer_names=["mtp.layers.48.linear_attn"],
    kv_cache_spec=MambaSpec(),
    is_eagle_group=False,
)
try:
    annotate(config, {}, [mamba_group])
except ValueError as error:
    assert "Mamba" in str(error)
else:
    raise AssertionError("Qwen MTP group containing Mamba did not fail closed")

print("QWEN_MTP_GROUP_ANNOTATION PASS")
