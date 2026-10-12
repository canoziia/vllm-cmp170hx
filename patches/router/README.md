# vllm-router patch series

Applied in `series` order to `src/` of the production-stack release pinned in
`manifests/router.env`, by `scripts/build-router-image.sh`; checked inside the
image by `scripts/test-router-patches.py`.

| Patch | What |
| --- | --- |
| `0001-merge-hop-usage.patch` | caps the prefill hop and merges both PD hops' usage into the standard fields (`router_hops` for the raw numbers) |
| `0002-client-api-key-auth.patch` | client auth, ported from upstream PR #937 (router part only): `VLLM_API_KEY` (comma-separated allowed) makes every endpoint except `/health` require `Authorization: Bearer <key>`, compared with `hmac.compare_digest`; unset = no auth. Backends are still called with `OPENAI_API_KEY` |

Compose files set both `OPENAI_API_KEY` (router -> engines) and `VLLM_API_KEY`
(clients -> router) from the same `.env` value. Prometheus scraping a router's
`/metrics` therefore needs the key as well.
