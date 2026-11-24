# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 request-local verification width (``vllm_xargs.spec_k``).

``vllm_xargs`` already reaches ``SamplingParams.extra_args`` unchanged for every
model. Only ``DeepseekV41ForCausalLM`` consults this module (two
architecture-gated call sites: input validation and scheduler admission);
every other model ignores the key exactly like the unpatched engine.
"""
import os

from vllm.exceptions import VLLMValidationError


def _widths_enabled(spec) -> bool:
    # Explicit verification widths, or the public adaptive switch selecting
    # DeepSeek's CPU history policy (which honours manual spec_k first).
    return (os.environ.get("VLLM_DSV41_VERIFICATION", "0") == "1"
            or spec.uses_history_verification())


def validate_request(extra_args, speculative_config) -> None:
    """Front-end validation; failures surface as HTTP 400."""
    if not extra_args or "spec_k" not in extra_args:
        return
    value = extra_args["spec_k"]
    if type(value) is not int or value < 0:
        raise VLLMValidationError(
            "vllm_xargs.spec_k must be a nonnegative integer",
            parameter="vllm_xargs.spec_k",
        )
    if (speculative_config is None
            or speculative_config.method != "dspark"
            or not _widths_enabled(speculative_config)):
        raise VLLMValidationError(
            "spec_k requires VLLM_DSV41_VERIFICATION=1 and DSpark",
            parameter="vllm_xargs.spec_k",
        )
    if not 0 <= value <= speculative_config.num_speculative_tokens:
        raise VLLMValidationError(
            "spec_k must be an integer between 0 and configured draft count",
            parameter="vllm_xargs.spec_k",
        )


def draft_limit(request, spec, num_spec_tokens: int,
                use_v2_model_runner: bool) -> int | None:
    """Scheduler admission: the request's verified-draft limit, or None."""
    params = request.sampling_params
    extra = params.extra_args if params is not None else None
    limit = extra.get("spec_k") if extra else None
    if limit is None:
        return None
    if (spec is None or spec.method != "dspark"
            or not use_v2_model_runner
            or not _widths_enabled(spec)):
        raise ValueError("spec_k requires enabled DSpark verification widths")
    if type(limit) is not int or not 0 <= limit <= num_spec_tokens:
        raise ValueError("spec_k must be an integer in the configured draft range")
    return limit
