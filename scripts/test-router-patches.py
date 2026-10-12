#!/usr/bin/env python3
"""Offline checks for patches/router/, run inside the built router image."""

import json
import unittest

from vllm_router.services.request_service.request import _UsageMergeState


def _usage(prompt, cached, completion):
    """A hop's usage as the engines report it (standard fields only)."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


class UsageMergeTests(unittest.TestCase):
    def test_cold_request(self):
        """No pre-existing cache: the handoff must not become a cached read."""
        out = _UsageMergeState._merge(
            _usage(51056, 48000, 16), _usage(51056, 0, 1), "cold"
        )
        self.assertEqual(out["prompt_tokens"], 51056)
        self.assertEqual(out["completion_tokens"], 16)
        self.assertEqual(out["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(out["router_hops"]["input"], 51056)
        self.assertEqual(out["router_hops"]["cached_read"], 0)
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], 51056 + 3056)

    def test_warm_request(self):
        """Warm: min() caps at the hit that existed before the request."""
        out = _UsageMergeState._merge(
            _usage(51056, 49600, 16), _usage(51056, 48000, 1), "warm"
        )
        self.assertEqual(out["prompt_tokens_details"]["cached_tokens"], 48000)
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], 3056 + 1456)

    def test_standard_fields_stay_standard(self):
        """No fork-specific keys in the standard usage that a client sees."""
        decode = _usage(100, 64, 4)
        decode["prompt_tokens_details"]["created_cache_tokens"] = 36  # engine-reported
        out = _UsageMergeState._merge(decode, _usage(100, 0, 1), "clean")
        details = out["prompt_tokens_details"]
        self.assertNotIn("created_cache_tokens", details)
        self.assertNotIn("cache_write_tokens", details)
        self.assertEqual(sorted(details), ["cached_tokens"])
        # the detail is still available, just namespaced
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], 100 + 36)

    def test_missing_prefill_usage_is_passthrough(self):
        out = _UsageMergeState._merge(_usage(10, 0, 2), None, "nop")
        self.assertEqual(out["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], None)

    def test_prompt_mismatch_keeps_decode(self):
        out = _UsageMergeState._merge(_usage(100, 0, 2), _usage(99, 0, 1), "mismatch")
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
        self.assertEqual(payload["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertEqual(payload["completion_tokens"], 4)
        self.assertIn("router_hops", payload)

    def test_non_usage_events_untouched(self):
        event = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        state = _UsageMergeState(_usage(5, 0, 1), "keep")
        self.assertEqual(b"".join(state.feed(event)), event)


class ResponsesShapeTests(unittest.TestCase):
    """The Responses API names the fields differently and nests the usage."""

    def test_responses_usage_is_merged(self):
        decode = {
            "input_tokens": 100,
            "output_tokens": 4,
            "total_tokens": 104,
            "input_tokens_details": {"cached_tokens": 64},
        }
        prefill = {
            "input_tokens": 100,
            "output_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
        }
        out = _UsageMergeState._merge(decode, prefill, "resp")
        self.assertEqual(out["input_tokens_details"], {"cached_tokens": 0})
        self.assertEqual(out["output_tokens"], 4)
        self.assertEqual(out["router_hops"]["cache_write_prefill_compute"], 100 + 36)

    def test_responses_event_is_rewritten(self):
        event = (
            b'event: response.completed\ndata: {"response":{"id":"r1","usage":'
            b'{"input_tokens":100,"output_tokens":4,"input_tokens_details":'
            b'{"cached_tokens":64,"created_cache_tokens":0}}}}\n\n'
        )
        state = _UsageMergeState(
            {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}}, "rid"
        )
        out = b"".join(state.feed(event)) + state.flush()
        self.assertIn(b"event: response.completed", out)
        usage = json.loads(out.split(b"data: ", 1)[1])["response"]["usage"]
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 0)
        self.assertNotIn("created_cache_tokens", usage["input_tokens_details"])
        self.assertIn("router_hops", usage)

    def test_unrelated_events_pass_through(self):
        event = b'event: response.output_text.delta\ndata: {"delta":"hi"}\n\n'
        state = _UsageMergeState({"input_tokens": 1}, "x")
        self.assertEqual(b"".join(state.feed(event)), event)


class NonStreamingBodyTests(unittest.TestCase):
    """Non-streaming clients get a single JSON body instead of SSE events."""

    def test_chat_completions_body_is_merged(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 4,
                "total_tokens": 104,
                "prompt_tokens_details": {"cached_tokens": 64, "created_cache_tokens": 0},
            },
        }
        usage = _UsageMergeState._find_usage(payload)
        self.assertIsNotNone(usage)
        _UsageMergeState._merge(
            usage, {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}, "ns"
        )
        self.assertEqual(payload["usage"]["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertIn("router_hops", payload["usage"])

    def test_responses_body_is_merged(self):
        payload = {
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 64, "created_cache_tokens": 0},
                }
            }
        }
        usage = _UsageMergeState._find_usage(payload)
        self.assertIsNotNone(usage)
        _UsageMergeState._merge(
            usage, {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}}, "ns2"
        )
        details = payload["response"]["usage"]["input_tokens_details"]
        self.assertEqual(details, {"cached_tokens": 0})
        self.assertIn("router_hops", payload["response"]["usage"])

    def test_find_usage_returns_none_without_usage(self):
        self.assertIsNone(_UsageMergeState._find_usage({"choices": []}))
        self.assertIsNone(_UsageMergeState._find_usage("not a dict"))


class PatchLandedTests(unittest.TestCase):
    def test_merge_is_shipped(self):
        import inspect

        import vllm_router.services.request_service.request as mod

        src = inspect.getsource(mod)
        self.assertIn("router_hops", src)
        self.assertIn("cache_write_prefill_compute", src)


class ClientAuthTests(unittest.TestCase):
    """patches/router/0002: VLLM_API_KEY gates every router except /health."""

    def setUp(self):
        import os

        self._old = os.environ.get("VLLM_API_KEY")
        os.environ["VLLM_API_KEY"] = "k-one, k-two"

    def tearDown(self):
        import os

        if self._old is None:
            os.environ.pop("VLLM_API_KEY", None)
        else:
            os.environ["VLLM_API_KEY"] = self._old

    def _status(self, path, auth=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_router.auth import verify_api_key
        from fastapi import Depends

        app = FastAPI()

        @app.get("/health")
        @app.get("/v1/models")
        @app.get("/metrics")
        def _ok():
            return {}

        app.router.dependencies.append(Depends(verify_api_key))
        headers = {"Authorization": auth} if auth else {}
        return TestClient(app).get(path, headers=headers).status_code

    def test_dependency(self):
        try:
            import httpx  # noqa: F401  (TestClient)
        except ImportError:
            self.skipTest("httpx missing")
        self.assertEqual(self._status("/health"), 200)
        self.assertEqual(self._status("/v1/models"), 401)
        self.assertEqual(self._status("/metrics"), 401)
        self.assertEqual(self._status("/v1/models", "Bearer nope"), 401)
        self.assertEqual(self._status("/v1/models", "Basic k-one"), 401)
        self.assertEqual(self._status("/v1/models", "Bearer k-one"), 200)
        self.assertEqual(self._status("/v1/models", "Bearer k-two"), 200)

    def test_parse_and_disable(self):
        import os

        from vllm_router.auth import get_allowed_api_keys

        self.assertEqual(get_allowed_api_keys(), frozenset({"k-one", "k-two"}))
        os.environ["VLLM_API_KEY"] = ""
        self.assertEqual(get_allowed_api_keys(), frozenset())

    def test_static_wiring(self):
        import inspect

        import importlib.util

        import vllm_router.auth as auth_mod

        src = inspect.getsource(auth_mod)
        self.assertIn("hmac.compare_digest", src)
        self.assertNotIn("logger", src)  # never log keys
        with open(importlib.util.find_spec("vllm_router.app").origin) as f:
            app_src = f.read()
        for name in ("main_router", "files_router", "batches_router", "metrics_router"):
            self.assertIn(f"app.include_router({name}, dependencies=_auth)", app_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
