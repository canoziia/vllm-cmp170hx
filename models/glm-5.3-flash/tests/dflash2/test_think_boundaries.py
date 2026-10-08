#!/usr/bin/env python3
"""Run in the candidate vLLM image, CPU only; tests the actual parser engine."""
import os
import unittest
from vllm.parser.glm47_moe import Glm47MoeParser

class Tokenizer:
    def get_vocab(self):
        return {'<think>': 1, '</think>': 2, '<tool_call>': 3, '</tool_call>': 4}

class Tests(unittest.TestCase):
    def parser(self, enabled, **flags):
        os.environ['VLLM_GLM53_FORCE_THINK_BOUNDARIES'] = str(int(enabled))
        return Glm47MoeParser(Tokenizer(), chat_template_kwargs=flags)

    def test_default_unchanged(self):
        for flags in ({}, {'enable_thinking': True}, {'reasoning_effort': 'low'}, {'reasoning_effort': 'high'}):
            a = self.parser(False, **flags).extract_reasoning('work</think>391', None)
            b = self.parser(True, **flags).extract_reasoning('work</think>391', None)
            self.assertEqual(a, b)
            self.assertEqual(b, ('work', '391'))

    def test_false_opt_in(self):
        for flags in ({'enable_thinking': False}, {'thinking': False}, {'enable_thinking': False, 'reasoning_effort': 'none'}):
            self.assertEqual(self.parser(False, **flags).extract_reasoning('work</think>391', None), (None, 'work</think>391'))
            p = self.parser(True, **flags)
            self.assertEqual(p.extract_reasoning('work</think>391', None), ('work', '391'))
            self.assertFalse(p.is_reasoning_end([1, 9]))
            self.assertTrue(p.is_reasoning_end([1, 9, 2, 10]))
            self.assertEqual(p.extract_content_ids([9, 2, 10]), [10])

    def test_empty_and_truncated(self):
        p = self.parser(True, enable_thinking=False)
        self.assertEqual(p.extract_reasoning('</think>391', None), (None, '391'))
        self.assertEqual(p.extract_reasoning('unfinished', None), ('unfinished', None))
        self.assertEqual(p.extract_reasoning('a</think>x<think>b</think>y', None), ('ab', 'xy'))

    def test_stream_split_boundaries(self):
        for pieces in (['work', '</th', 'ink>', '391'], list('work</think>391')):
            p = self.parser(True, enable_thinking=False)
            previous = ''; reasoning = ''; content = ''
            for text in pieces:
                current = previous + text
                delta = p.extract_reasoning_streaming(previous, current, text, [], [], [])
                previous = current
                if delta:
                    reasoning += delta.reasoning or ''
                    content += delta.content or ''
            delta = p.finish_streaming()
            if delta:
                reasoning += delta.reasoning or ''
                content += delta.content or ''
            self.assertEqual((reasoning, content), ('work', '391'))

if __name__ == '__main__':
    unittest.main()
