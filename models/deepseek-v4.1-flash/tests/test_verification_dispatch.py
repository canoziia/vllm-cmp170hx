#!/usr/bin/env python3
"""CPU-only exact candidate test, run with applied pinned SOURCE_TREE argument."""
import ast, bisect, os, sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import groupby, product
from pathlib import Path
from types import SimpleNamespace as NS


class CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Mode:
    def decode_mode(self):
        return CUDAGraphMode.FULL

    def mixed_mode(self):
        return CUDAGraphMode.PIECEWISE

    def separate_routine(self):
        return True


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: object
    num_tokens: int
    num_reqs: object
    uniform_token_count: object = None
    max_query_len: object = None
    num_active_loras: int = 0
    num_ubatches: int = 1


class ModelCudaGraphManager:
    def _maybe_ubatch_twin(self, desc):
        return None


def load(path, flag, adaptive):
    os.environ["VLLM_DSV41_VERIFICATION"] = str(flag)
    os.environ["VLLM_DSV41_SUBWIDTH_GRAPHS"] = "1,2,3,4,5,6,7,8:1-6;16,24,32:2-6"
    comp_tree = ast.parse((path.parents[3] / "config/compilation.py").read_text())
    helpers = [
        x
        for x in comp_tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name
        in ("decode_cudagraph_capture_shapes", "decode_cudagraph_capture_max_tokens")
    ]
    tree = ast.parse(path.read_text())
    cls = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "CudaGraphManager"
    )
    f = next(
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef) and x.name == "_init_candidates"
    )
    ns = dict(globals(), round_up=lambda x, y: (x + y - 1) // y * y)
    exec(compile(ast.Module([*helpers, f], type_ignores=[]), str(path), "exec"), ns)
    s = ModelCudaGraphManager()
    s.compilation_config = NS(
        cudagraph_capture_sizes=sorted(
            set(
                [1, 2, 4, 6, 8, 12, 16]
                + list(range(24, 257, 8))
                + list(range(272, 385, 16))
            )
        ),
        max_cudagraph_capture_size=384,
        prefill_cudagraph_capture_token_sizes=[8, 16, 32, 64, 128, 256],
        decode_cudagraph_capture_request_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32],
        max_decode_cudagraph_capture_tokens=None,
    )
    s.cudagraph_mode = Mode()
    s.max_num_reqs = 32
    s.decode_request_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32]
    s.decode_max_reqs = {32: 6}
    s.decode_query_len = 6
    s.varlen_decode = adaptive
    s.lora_capture_cases = [0]
    s.vllm_config = NS(
        speculative_config=None,
        model_config=NS(
            architectures=["DeepseekV41ForCausalLM"], max_model_len=1048576
        ),
        cache_config=NS(use_kda_recoverssm=False),
        parallel_config=NS(use_ubatching=False),
        get_varlen_decode_cudagraph_token_sizes=lambda q: list(range(1, 49)),
    )
    s._capture_descs = {}
    s._decode_descs = {}
    s._candidates = {}
    ns["_init_candidates"](s)
    return s


path = Path(sys.argv[1]) / "vllm/v1/worker/gpu/cudagraph_utils.py"
for adaptive in (False, True):
    s = load(path, 1, adaptive)
    for reqs in range(1, 9):
        for width in range(1, 7):
            rows = reqs * width
            if not adaptive:
                d = s._decode_descs[(width, 0, 1)][reqs]
                assert d.cg_mode == CUDAGraphMode.FULL and d.num_tokens == rows and d.num_reqs == reqs
            else:
                ds = [
                    d
                    for d in s._candidates[rows, 0]
                    if d.cg_mode == CUDAGraphMode.FULL
                    and d.num_tokens >= rows
                    and (d.uniform_token_count is None or d.uniform_token_count == width)
                    and d.num_reqs >= reqs
                ]
                assert ds[0].num_tokens == rows, (adaptive, reqs, width, ds)
    for ds in s._capture_descs.values():
        assert len(ds) == len(set(ds))
    if adaptive:
        for rows in range(1, 49):
            assert any(
                d.cg_mode == CUDAGraphMode.FULL
                and d.num_tokens == rows
                and d.uniform_token_count is None
                for d in s._candidates[rows, 0]
            )
print(
    "PASS uniform widths1..6 cohorts1..8 exact tokens/requests, ragged rows1..48, no duplicate captures"
)
