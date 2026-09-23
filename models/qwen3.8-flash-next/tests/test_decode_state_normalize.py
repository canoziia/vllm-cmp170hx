#!/usr/bin/env python3
"""GPU byte-equality regression for generated-history Mamba checkpoints."""

import torch

from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel

cases = 0
for dtype in (torch.bfloat16, torch.float32):
    for dim_first in (False, True):
        for accepted in (1, 2, 3, 4):
            width, dim = 7, 128
            shape = (8, dim, width) if dim_first else (8, width, dim)
            conv = torch.arange(
                8 * width * dim, device="cuda", dtype=torch.float32
            ).reshape(shape).to(dtype)
            temporal = torch.arange(
                8 * 128 * 128, device="cuda", dtype=torch.float32
            ).reshape(8, 128, 128).to(dtype)
            original_conv = conv.clone()
            original_temporal = temporal.clone()
            block_table = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
            i64 = lambda values: torch.tensor(values, device="cuda", dtype=torch.int64)
            i32 = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
            accepted_gpu = i32([accepted])
            accepted_out = accepted_gpu.clone()
            state_idx = i32([2])
            computed = i32([2176])  # 17 * 128, not aligned to 832.
            mapping = i32([0])
            postprocess_mamba_fused_kernel[(1, 2, 1)](
                accepted_gpu,
                state_idx,
                None,
                computed,
                None,
                i64([block_table.data_ptr()]),
                8,
                i64([conv.data_ptr(), temporal.data_ptr()]),
                i64([
                    conv.stride(0) * conv.element_size(),
                    temporal.stride(0) * temporal.element_size(),
                ]),
                i32([conv.element_size(), temporal.element_size()]),
                i64([dim, 128 * 128]),
                i32([width, 0]),
                i32([0, 0]),
                i32([dim if dim_first else 0, 0]),
                i64([width * conv.element_size() if dim_first else 0, 0]),
                accepted_out,
                mapping,
                1,
                block_size=832,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=dim_first,
                HAS_IDX_MAPPING=True,
                PRECOMPUTED_NEW_COMPUTED=True,
                DECODE_CHECKPOINT_UNIT=128,
                TEMPORAL_TILES=1,
            )
            torch.cuda.synchronize()
            bias = accepted - 1
            assert torch.equal(temporal[2], original_temporal[2 + bias])
            if dim_first:
                assert torch.equal(
                    conv[2, :, : width - bias], original_conv[2, :, bias:]
                )
            else:
                assert torch.equal(
                    conv[2, : width - bias], original_conv[2, bias:]
                )
            assert accepted_out.item() == 1
            cases += 1
print(f"DECODE_NORMALIZE_BYTE_EQUAL cases={cases} PASS")
