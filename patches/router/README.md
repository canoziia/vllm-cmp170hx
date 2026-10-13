# vllm-router patch series

Applied in `series` order to `src/` of the production-stack release pinned in
`manifests/router.env`, by `scripts/build-router-image.sh`; checked inside the
image by `scripts/test-router-patches.py`.

| Patch | What |
| --- | --- |
| `0001-merge-hop-usage.patch` | caps the prefill hop and merges both PD hops' usage into the standard fields (`router_hops` for the raw numbers) |
| `0002-client-api-key-auth.patch` | client auth, ported from upstream PR #937 (router part only): `VLLM_API_KEY` (comma-separated allowed) makes every endpoint except `/health` require `Authorization: Bearer <key>`, compared with `hmac.compare_digest`; unset = no auth. Backends are still called with `OPENAI_API_KEY` |
| `0003-client-disconnect-cancel.patch` | client disconnect aborts the backend request, ported from upstream PR #1040 (open; issue #634): the backend call is raced against the ASGI `http.disconnect` message (as `vllm.entrypoints.utils.with_cancellation`, no `is_disconnected()` polling) and cancelled, closing the backend socket so vLLM aborts; extended to the PD prefill hop and the non-streaming decode hop; logs request id (CR/LF stripped), phase and waited seconds; returns 499 |
| `0004-stream-prefill-keep-alive.patch` | streaming PD clients get a streamed prefill hop (`max_tokens=1`, `stream_options.include_usage`): the prefill HTTP status/body is still returned as-is before the stream starts; afterwards the prefill engine's SSE comments (`: keep-alive`, vLLM `--sse-keep-alive-interval`, vllm-project/vllm#51034) are forwarded verbatim, its token/`[DONE]` dropped and only its usage kept, then the decode stream follows. The router has no keep-alive timer of its own; engines at interval 0 give byte-identical output. Disconnect closes the prefill/decode stream |

Compose files set both `OPENAI_API_KEY` (router -> engines) and `VLLM_API_KEY`
(clients -> router) from the same `.env` value. Prometheus scraping a router's
`/metrics` therefore needs the key as well.
