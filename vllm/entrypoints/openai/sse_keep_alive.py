# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility re-export for ``vllm.entrypoints.serve.utils.sse_keep_alive``."""

from vllm.entrypoints.serve.utils.sse_keep_alive import (
    SSE_KEEP_ALIVE_COMMENT,
    with_sse_keep_alive,
)

__all__ = ["SSE_KEEP_ALIVE_COMMENT", "with_sse_keep_alive"]
