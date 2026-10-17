#!/usr/bin/env python3
"""Patch 0016: DeepSeek V4 / V4.1 top-level ``thinking: {type}`` request switch.

usage: test_deepseek_thinking.py SOURCE_TREE
Shared by the V4.1 and V4-Flash images (the V4-Flash series is applied on top
of the V4.1 series). Loads the helper from the patched tree; uses vLLM's real
ChatCompletionRequest when it is importable (inside the image), otherwise a
minimal stand-in.
"""
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
chat_dir = root / "vllm/entrypoints/openai/chat_completion"
assert not (chat_dir / "deepseek_v4_thinking.py").exists(), "one shared implementation only"
spec = importlib.util.spec_from_file_location("deepseek_thinking", chat_dir / "deepseek_thinking.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
apply = mod.apply_deepseek_thinking

serving = (chat_dir / "serving.py").read_text()
assert serving.count("apply_deepseek_thinking(") == 1, "exactly one call site"
assert 'self.model_config, "architectures"' in serving, "gate on the resolved architecture too"
assert serving.index("apply_deepseek_thinking(") < serving.index(
    "chat_template_kwargs = self._effective_chat_template_kwargs(request)"
), "must run before the parser/renderer read chat_template_kwargs"

try:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    def req(**kw):
        return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kw)
    real = True
except Exception:  # outside the image
    class _Req:
        def __init__(self, chat_template_kwargs=None, **extra):
            self.chat_template_kwargs = chat_template_kwargs
            self.model_extra = extra

    def req(**kw):
        return _Req(**kw)
    real = False

V4, V41 = ["DeepseekV4ForCausalLM"], ["DeepseekV41ForCausalLM"]
assert mod.DEEPSEEK_THINKING_ARCHITECTURES == {
    "DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration", "DeepseekV41ForCausalLM"}

# resolved + declared architecture lists as serving passes them
for arch in (["DeepseekV4ForConditionalGeneration"],
             ["DeepseekV4ForConditionalGeneration", "DeepseekV4ForCausalLM"],
             V41, V41 + V41):
    r = req(thinking={"type": "disabled"})
    assert apply(r, arch) is None and r.chat_template_kwargs["thinking"] is False, arch

for family in (V4, V41):
    # mapping
    for kind, val in (("disabled", False), ("enabled", True)):
        r = req(thinking={"type": kind})
        assert apply(r, family) is None
        assert r.chat_template_kwargs == {"thinking": val, "enable_thinking": val}, r.chat_template_kwargs
        if real:
            kw = r.build_chat_params(None, "auto").chat_template_kwargs
            assert kw["thinking"] is val and kw["enable_thinking"] is val, kw
            # reasoning_effort must not re-enable a disabled request
            r2 = req(thinking={"type": kind}, reasoning_effort="high")
            apply(r2, family)
            kw = r2.build_chat_params(None, "auto").chat_template_kwargs
            assert kw["enable_thinking"] is val and kw["thinking"] is val, kw

    # explicit chat_template_kwargs win
    for explicit in ({"thinking": True}, {"enable_thinking": True}, {"thinking": False}):
        r = req(thinking={"type": "disabled"}, chat_template_kwargs=dict(explicit, x=1))
        assert apply(r, family) is None
        assert r.chat_template_kwargs == dict(explicit, x=1), r.chat_template_kwargs

    # invalid values
    for bad in ({"type": "auto"}, {"type": None}, {}, "disabled", True, None):
        r = req(thinking=bad)
        err = apply(r, family)
        assert err and "thinking" in err, (bad, err)
        assert not r.chat_template_kwargs

    r = req(chat_template_kwargs={"a": 1})
    assert apply(r, family) is None and r.chat_template_kwargs == {"a": 1}

# other models are untouched
for arch in (["DeepseekV3ForCausalLM"], ["Qwen3ForCausalLM"], None, []):
    r = req(thinking={"type": "bogus"})
    assert apply(r, arch) is None and not r.chat_template_kwargs

# V4.1 tokenizer reads chat_template_kwargs.thinking / enable_thinking
tok = (root / "vllm/tokenizers/deepseek_v41.py").read_text()
assert 'kwargs.get("thinking") or kwargs.get("enable_thinking")' in tok

print(f"PASS deepseek V4/V4.1 top-level thinking ({'real' if real else 'stand-in'} request model)")
