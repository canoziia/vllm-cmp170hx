"""Differential test: patched vLLM deepseek_v41_encoding vs the 1017 reference.

Usage: python3 test_dsv41_encoding_1017.py SOURCE_TREE [REFERENCE_ENCODING_DIR]

REFERENCE_ENCODING_DIR defaults to the vendored copy of the checkpoint's
encoding/ directory (deepseek-ai/DeepSeek-V4.1-Flash-1017 revision
1bae0e1088993b9b82de08f59de5be5a34e42170, MIT licence) under
fixtures/deepseek-v4.1-flash-1017-encoding.

Every comparison is character-for-character. Golden fixtures from the reference
checkpoint (encoding/tests/test_input_N.json -> test_output_N.txt) are rendered
through the vLLM encoder after the same image->placeholder normalisation the
vLLM tokenizer wrapper performs, then compared with the checked-in outputs.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = load(
    "vllm_dsv41_encoding",
    Path(sys.argv[1]) / "vllm/tokenizers/deepseek_v41_encoding.py",
)
REF_DIR = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else Path(__file__).resolve().parent / "fixtures/deepseek-v4.1-flash-1017-encoding"
)
sys.path.insert(0, str(REF_DIR))
R = load("ref_encoding", REF_DIR / "encoding.py")

failures = []
checks = 0


def check(name, got, want):
    global checks
    checks += 1
    if got != want:
        failures.append(name)
        print(f"FAIL {name}")
        if not (isinstance(got, str) and isinstance(want, str)):
            print("  got", repr(got), "want", repr(want))
            return
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                print("  first diff at", i, repr(got[max(0, i - 60):i + 60]), "!=",
                      repr(want[max(0, i - 60):i + 60]))
                break
        else:
            print("  length", len(got), "!=", len(want))


def normalize(messages):
    """Mirror vllm/tokenizers/deepseek_v41.py::_normalize_messages."""
    out = [dict(m) for m in copy.deepcopy(messages)]
    for m in out:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for b in c:
                t = b.get("type")
                if t in ("text", "input_text", "output_text"):
                    parts.append(b.get("text", ""))
                elif t in ("image_url", "input_image", "image_pil", "image"):
                    parts.append(V.IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(t)
            m["content"] = "\n\n".join(parts)
    return out


def both(name, messages, **kw):
    ref = R.encode_messages(copy.deepcopy(messages), **kw)
    got = V.encode_messages(normalize(messages), **kw)
    check(name, got, ref)
    return got


# 1. Golden fixtures shipped with the checkpoint.
for inp in sorted((REF_DIR / "tests").glob("test_input_*.json")):
    case_id = inp.stem.split("_")[-1]
    want = (REF_DIR / "tests" / f"test_output_{case_id}.txt").read_text()
    case = R.load_cases(str(inp))[0]
    ref_prompt, _ = R.encode_case(case, thinking_mode="chat")
    check(f"reference-self golden {case_id}", ref_prompt, want)
    got = V.encode_messages(
        normalize(case["messages"]),
        thinking_mode=case.get("thinking_mode") or "chat",
        context=normalize(case["context"]) if case.get("context") else None,
        reasoning_effort=case.get("reasoning_effort"),
    )
    check(f"vllm golden {case_id}", got, want)

# 2. Reasoning effort mapping and default.
for effort, budget in [(None, 75), ("low", 50), ("high", 75), ("max", 100), (1, 1), (42, 42), (100, 100)]:
    p = both(f"effort {effort}", [{"role": "user", "content": "hi"}],
             thinking_mode="thinking", reasoning_effort=effort)
    check(f"effort {effort} budget", f"Reasoning Effort: {budget} " in p, True)
check("xhigh removed", "xhigh" in V.REASONING_EFFORT_MAPPINGS, False)


def tool(name="lookup", **extra):
    t = {"type": "function", "function": {
        "name": name, "description": "Look up a value",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}}}}
    t.update(extra)
    return t


def tc(cid, name="lookup", args='{"query":"value","limit":2}', **extra):
    c = {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
    c.update(extra)
    return c


# 3. Tool-result-only user messages do not start a new turn (drop_thinking path).
agent = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "reasoning_content": "r2", "tool_calls": [tc("c1")]},
    {"role": "tool", "tool_call_id": "c1", "content": "v1"},
    {"role": "assistant", "reasoning_content": "r3", "tool_calls": [tc("c2"), tc("c3")]},
    {"role": "tool", "tool_call_id": "c3", "content": "v3"},
    {"role": "tool", "tool_call_id": "c2", "content": "v2"},
]
for mode in ("thinking", "chat"):
    for drop in (True, False):
        p = both(f"agent loop {mode} drop={drop}", agent, thinking_mode=mode, drop_thinking=drop)
        if mode == "thinking":
            check(f"agent loop keeps r2/r3 {drop}", "<think>" not in p.split("q2")[1][:0] and "r2</think>" in p and "r3</think>" in p, True)
            check(f"agent loop drops r1 {drop}", ("r1" in p) == (not drop), True)
            check(f"agent loop ends <think> {drop}", p.endswith("<｜Assistant｜><think>"), True)
both("agent next turn", agent + [{"role": "user", "content": "q3"}], thinking_mode="thinking")
both("agent with tools declared", [dict(agent[0], tools=[tool()])] + agent[1:] + [{"role": "user", "content": "q3"}],
     thinking_mode="thinking")
both("agent with context", agent[3:], thinking_mode="thinking", context=agent[:3])
both("mid system", [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a", "reasoning_content": "r"},
                    {"role": "system", "content": "update"}], thinking_mode="thinking")

# 4. Namespaces.
for loc in ("tool", "function"):
    for ns in ("search", {"name": "search", "description": "Search tools."}):
        t = tool()
        (t if loc == "tool" else t["function"])["namespace"] = ns
        p = both(f"ns schema {loc} {type(ns).__name__}",
                 [{"role": "system", "content": "system", "tools": [t]}, {"role": "user", "content": "q"}],
                 thinking_mode="chat")
        check(f"ns schema name {loc}", '"name": "search::lookup"' in p and '"namespace":' not in p, True)
for loc in ("tool", "function", "qualified_name"):
    c = tc("c1")
    if loc == "qualified_name":
        c["function"]["name"] = "search::lookup"
    else:
        (c if loc == "tool" else c["function"])["namespace"] = "search"
    for mode in ("thinking", "chat"):
        p = both(f"ns call {loc} {mode}", [
            {"role": "user", "content": "question"},
            {"role": "assistant", "reasoning_content": "  reason  ", "content": "summary", "tool_calls": [c]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}], thinking_mode=mode)
        check(f"ns call name {loc} {mode}", '<｜DSML｜ invoke name="search::lookup">' in p, True)
tools, calls = [], []
for ns in (None, "search", "files"):
    t = tool()
    c = tc(f"id{ns}", args='{"query":"value"}')
    if ns:
        t["namespace"] = {"name": ns}
        c["namespace"] = ns
    tools.append(t)
    calls.append(c)
both("same name separate ns", [{"role": "system", "content": "system", "tools": tools},
                               {"role": "user", "content": "question"},
                               {"role": "assistant", "content": "summary", "tool_calls": calls}],
     thinking_mode="chat")
for name, ns, err in [("search::lookup", "files", "Conflicting tool namespaces"),
                      ("search::nested::lookup", None, "Tool name must not contain"),
                      ("lookup", "search::nested", "Tool namespace must not contain")]:
    t = tool(name)
    t["namespace"] = ns
    try:
        V.tools_from_openai_format([t])
        check(f"reject {err}", "no error", err)
    except AssertionError as e:
        check(f"reject {err}", err in str(e), True)

# 5. Structured content for tool/user messages (normalised to text by vLLM).
both("tool list content", [
    {"role": "user", "content": "q"},
    {"role": "assistant", "tool_calls": [tc("a"), tc("b")]},
    {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "rb"}]},
    {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "ra"}]}],
     thinking_mode="thinking")

print(f"{checks - len(failures)}/{checks} checks passed")
sys.exit(1 if failures else 0)
