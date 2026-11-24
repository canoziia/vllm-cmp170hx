# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 / V4.1 request-level ``thinking`` switch (DeepSeek API format).

DeepSeek's API takes ``"thinking": {"type": "enabled" | "disabled"}`` at the
top level of a chat request. vLLM keeps unknown top-level fields in
``model_extra`` and otherwise ignores them, so the DeepSeek tokenizers and
output parsers (which read ``chat_template_kwargs``) fell back to thinking mode.

One shared implementation for every DeepSeek architecture listed in
``DEEPSEEK_THINKING_ARCHITECTURES`` (the V4-Flash image stacks its series on
the V4.1 series, so both models use this file). The switch is copied into
``chat_template_kwargs["thinking"]`` once, before the parser and the renderer
read the kwargs, so both see the same value (and a PD prefill/decode pair
renders the same prompt). An explicit ``chat_template_kwargs.thinking`` /
``enable_thinking`` wins.
"""

from typing import Any

# V4: Vision-Exp checkpoints declare DeepseekV4ForCausalLM and resolve to
# DeepseekV4ForConditionalGeneration. V4.1: DeepseekV41ForCausalLM (declared
# and resolved).
DEEPSEEK_THINKING_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForConditionalGeneration",
        "DeepseekV41ForCausalLM",
    }
)
_TYPES = {"enabled": True, "disabled": False}


def is_deepseek_thinking_model(architectures: Any) -> bool:
    return any(a in DEEPSEEK_THINKING_ARCHITECTURES for a in (architectures or ()))


def apply_deepseek_thinking(request: Any, architectures: Any) -> str | None:
    """Fold a top-level ``thinking`` into ``chat_template_kwargs``.

    Returns an error message for an invalid value, otherwise None. Requests
    for other architectures, or without a top-level ``thinking``, are not
    modified.
    """
    if not is_deepseek_thinking_model(architectures):
        return None
    extra = getattr(request, "model_extra", None) or {}
    if "thinking" not in extra:
        return None
    value = extra["thinking"]
    kind = value.get("type") if isinstance(value, dict) else None
    if kind not in _TYPES:
        return (
            "'thinking' must be an object {\"type\": \"enabled\"} or "
            f"{{\"type\": \"disabled\"}}; got {value!r}"
        )
    kwargs = dict(request.chat_template_kwargs or {})
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        # both keys: build_chat_params derives enable_thinking from
        # reasoning_effort unless it is present, and the tokenizers OR them
        kwargs["thinking"] = kwargs["enable_thinking"] = _TYPES[kind]
        request.chat_template_kwargs = kwargs
    return None
