#!/usr/bin/env python3
"""Offline checks for patches/router/, run inside the built router image."""

import json
import unittest

from vllm_router.services.request_service.request import _UsageMergeState


def _usage(prompt, cached, completion):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": cached,
            "created_cache_tokens": 0,
        },
    }


class UsageMergeTests(unittest.TestCase):
    def test_cold_request(self):
        """No pre-existing cache: handoff must not be billed as a cached read."""
        out = _UsageMergeState._merge(
            _usage(51056, 48000, 16), _usage(51056, 0, 1), "cold"
        )
        det = out["prompt_tokens_details"]
        self.assertEqual(out["prompt_tokens"], 51056)
        self.assertEqual(out["completion_tokens"], 16)
        self.assertEqual(det["cached_tokens"], 0)
        self.assertEqual(det["created_cache_tokens"], 51056 + 3056)
        self.assertEqual(out["router_hops"]["cached_read"], 0)
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], 54112)

    def test_warm_request(self):
        """Warm: min() caps at the hit that existed before the request."""
        out = _UsageMergeState._merge(
            _usage(51056, 49600, 16), _usage(51056, 48000, 1), "warm"
        )
        det = out["prompt_tokens_details"]
        self.assertEqual(det["cached_tokens"], 48000)
        self.assertEqual(det["created_cache_tokens"], 3056 + 1456)

    def test_missing_prefill_usage_is_passthrough(self):
        out = _UsageMergeState._merge(_usage(10, 0, 2), None, "nop")
        self.assertEqual(out["prompt_tokens_details"]["cached_tokens"], 0)

    def test_prompt_mismatch_keeps_decode(self):
        out = _UsageMergeState._merge(
            _usage(100, 0, 2), _usage(99, 0, 1), "mismatch"
        )
        self.assertEqual(out["prompt_tokens"], 100)


class StreamRewriteTests(unittest.TestCase):
    def test_stream_is_rewritten_and_preserved(self):
        head = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        usage = (
            b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":4,'
            b'"total_tokens":104,"prompt_tokens_details":{"cached_tokens":64,'
            b'"created_cache_tokens":0}}}\n\n'
        )
        done = b"data: [DONE]\n\n"
        raw = head + usage + done
        state = _UsageMergeState(_usage(100, 0, 1), "stream")
        out = b""
        for i in range(0, len(raw), 7):  # awkward splits across events
            out += b"".join(state.feed(raw[i : i + 7]))
        out += state.flush()
        self.assertIn(head, out)
        self.assertTrue(out.endswith(done))
        rewritten = [e for e in out.split(b"\n\n") if b'"usage"' in e][0]
        payload = json.loads(rewritten.split(b"data: ", 1)[1])["usage"]
        self.assertEqual(payload["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(payload["prompt_tokens_details"]["created_cache_tokens"], 136)
        self.assertEqual(payload["completion_tokens"], 4)
        self.assertIn("router_hops", payload)

    def test_non_usage_events_untouched(self):
        event = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        state = _UsageMergeState(_usage(5, 0, 1), "keep")
        self.assertEqual(b"".join(state.feed(event)), event)


class PatchLandedTests(unittest.TestCase):
    def test_merge_is_shipped(self):
        import inspect

        import vllm_router.services.request_service.request as mod

        src = inspect.getsource(mod)
        self.assertIn("router_hops", src)
        self.assertIn("cache_write_prefill_compute", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
