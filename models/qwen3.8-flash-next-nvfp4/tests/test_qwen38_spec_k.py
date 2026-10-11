#!/usr/bin/env python3
"""CPU check for vllm/v1/core/sched/qwen38_spec_k.py.

  python3 test_qwen38_spec_k.py [site-packages]
"""
import os
import sys
import types

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched import qwen38_spec_k as m

ok = True


def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok &= bool(cond)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


spec = types.SimpleNamespace(method="mtp", num_speculative_tokens=5)
check("architecture gate", m.is_qwen38(["Qwen4ExpForConditionalGeneration"])
      and not m.is_qwen38(["DeepseekV41ForCausalLM"]) and not m.is_qwen38(None))
os.environ["VLLM_QWEN38_VERIFICATION_WIDTHS"] = "1"
m.validate_request(None, spec)
m.validate_request({"other": 1}, spec)
for v in (0, 3, 5):
    m.validate_request({"spec_k": v}, spec)
check("valid values accepted", True)
check("out of range -> 400", raises(lambda: m.validate_request({"spec_k": 6}, spec), VLLMValidationError))
for bad in (-1, 1.5, "2", [1]):
    check(f"invalid {bad!r} -> 400", raises(lambda: m.validate_request({"spec_k": bad}, spec), VLLMValidationError))
check("non-mtp -> 400", raises(lambda: m.validate_request({"spec_k": 1}, types.SimpleNamespace(
    method="ngram", num_speculative_tokens=5)), VLLMValidationError))
req = types.SimpleNamespace(sampling_params=types.SimpleNamespace(extra_args={"spec_k": 2}))
check("draft_limit", m.draft_limit(req, spec, 5, True) == 2)
check("draft_limit none", m.draft_limit(types.SimpleNamespace(sampling_params=None), spec, 5, True) is None)
os.environ["VLLM_QWEN38_VERIFICATION_WIDTHS"] = "0"
check("feature off -> 400", raises(lambda: m.validate_request({"spec_k": 1}, spec), VLLMValidationError))
check("feature off -> scheduler error", raises(lambda: m.draft_limit(req, spec, 5, True), ValueError))
print("QWEN38_SPEC_K", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
