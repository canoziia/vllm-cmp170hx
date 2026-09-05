# DGX Spark patch series

The patches are applied in lexical order to files extracted from the ARM64 variant of `docker.io/vllm/vllm-openai:qwen38-flash-next`.

| Patch | Purpose |
| --- | --- |
| `0003-ple-nvme-pread.patch` | Keep the RadixArk checkpoint's 47.68-GiB FP8 PLE table in its original safetensors files, load only metadata/scales, gather selected rows with native parallel `pread` (or `mmap`/Python fallback), and apply bounded connector backpressure. |
| `0007-ple-next-chunk-prefetch.patch` | Predict one bounded future PLE batch from V2 request state, overlap its SSD gather with the current GPU forward, validate request IDs, positions, token IDs and n-gram context before reuse, and report eligible-prefill coverage. |
| `0008-mamba-scheduler-block-alignment.patch` | Align hybrid Mamba/QSA prefill checkpoint stops to the scheduler's LCM block size. |
| `0009-mtp-safe-partial-tail-checkpoint.patch` | Preserve fine-grained recurrent partial-tail correctness when MTP drops the final prefix-hash unit. |
| `0010-uniproc-ple-lifecycle.patch` | Spawn and join the dedicated PLE process from `UniProcExecutor`; the upstream lifecycle hook existed only in the multiprocess executor, leaving TP1×PP1 without a PLE server. |
| `0011-gb10-kernel-fixes.patch` | Enable the 99-KiB FLA large-tile path, pin the Blackwell-safe `chunk_delta_h` warp count, and route QSA selection to the deterministic SM121 top-k extension from vLLM PR #55122. |
| `0012-ple-dedup-telemetry.patch` | Deduplicate and sort large PLE row gathers, restore exact row order through an inverse map, and report content-independent row/read timing telemetry. |

| `0013-mamba-overlap-copy.patch` | Port the upstream overlapping Mamba state-copy race fix (PR #50729), without silently skipping invalid copies. |
| `0014-mamba-worker-block-size.patch` | Use the Mamba block size when restoring V2 worker recurrent state. |
| `0015-indexed-weight-loading.patch` | Index expert mapping candidates and filter the local Qwen MTP weight index before reading unrelated files. |
| `0016-cpu-stage-nvfp4-experts.patch` | Bounded per-module CPU staging of original packed NVFP4 tensors/scales, using the original loader, then bulk transfer; TP1 and opt-out supported. |
| `0017-cache-ple-headers.patch` | Avoid reparsing safetensors headers on cache hits. |

Twelve patch files total. `0003/0007/0008/0009` were inherited from the CMP170HX branch; `0010`–`0017` were added during the Spark port. Native PLE and deterministic top-k source accompanies the patches. The vendored top-k is from an Apache-2.0 upstream commit; see `vendor/kernel-det/SOURCE.txt`.

PP4 ownership, PP capture barriers and PP speculative-feedback patches are deliberately omitted. The Spark target is TP1 x PP1.

## Integrity

`manifests/base-qwen38-nvfp4-spark.sha256` pins every extracted source file before patching. `manifests/final-qwen38-nvfp4-spark.sha256` verifies every output file. The build fails on any source or output mismatch.
