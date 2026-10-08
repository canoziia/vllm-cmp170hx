#!/usr/bin/env python3
"""CPU: the generation prompt of models/glm-5.3-flash/chat_template.jinja.

enable_thinking=false (and reasoning_effort=none, which vLLM turns into
enable_thinking=false) must end in an empty <think></think>; every other
request keeps the original open <think>. Rendered with transformers' own
Jinja environment, as vLLM does.
usage: test_chat_template.py CHAT_TEMPLATE
"""
import sys
import unittest

from transformers.utils.chat_template_utils import _compile_jinja_template

TEMPLATE = _compile_jinja_template(open(sys.argv.pop(1)).read())
MESSAGES = [{"role": "user", "content": "hi"}]


def render(**kwargs):
    return TEMPLATE.render(messages=MESSAGES, add_generation_prompt=True, **kwargs)


class Tests(unittest.TestCase):
    def test_default_and_enabled_open_think(self):
        for kwargs in ({}, {"enable_thinking": True}, {"reasoning_effort": "low"},
                       {"reasoning_effort": "high", "enable_thinking": True}):
            out = render(**kwargs)
            self.assertTrue(out.endswith("<|assistant|><think>"), (kwargs, out[-40:]))

    def test_disabled_empty_think(self):
        for kwargs in ({"enable_thinking": False},
                       {"reasoning_effort": "none", "enable_thinking": False}):
            out = render(**kwargs)
            self.assertTrue(out.endswith("<|assistant|><think></think>"), (kwargs, out[-40:]))

    def test_prefix_unchanged(self):
        on, off = render(), render(enable_thinking=False)
        self.assertEqual(on, off[: -len("</think>")])
        self.assertTrue(on.startswith("[gMASK]<sop>"), on[:20])


if __name__ == "__main__":
    unittest.main()
