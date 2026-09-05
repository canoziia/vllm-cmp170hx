# DGX Spark patch series

The patches are applied in lexical order to files extracted from the ARM64 variant of `docker.io/vllm/vllm-openai:qwen38-flash-next`.

| Patch | Purpose |
| --- | --- |
| `0003-ple-nvme-pread.patch` | Keep the RadixArk checkpoint's 47.68-GiB FP8 PLE table in its original safetensors files, load only metadata/scales, gather selected rows with native parallel `pread` (or `mmap`/Python fallback), and apply bounded connector backpressure. |
| `0007-ple-next-chunk-prefetch.patch` | Predict one bounded future PLE batch from V2 request state, overlap its SSD gather with the current GPU forward, validate request IDs, positions, token IDs and n-gram context before reuse, and report eligible-prefill coverage. |
| `0008-mamba-scheduler-block-alignment.patch` | Align hybrid Mamba/QSA prefill checkpoint stops to the scheduler's LCM block size. |
| `0009-mtp-safe-partial-tail-checkpoint.patch` | Preserve fine-grained recurrent partial-tail correctness when MTP drops the final prefix-hash unit. |
| `0010-uniproc-ple-lifecycle.patch` | Spawn and join the dedicated PLE process from `UniProcExecutor`; the upstream lifecycle hook existed only in the multiprocess executor, leaving TP1×PP1 without a PLE server. |

PP4 ownership, PP capture barriers and PP speculative-feedback patches are deliberately omitted. The Spark target is TP1 x PP1.

## Integrity

`manifests/base-qwen38-nvfp4-spark.sha256` pins every extracted source file before patching. `manifests/final-qwen38-nvfp4-spark.sha256` verifies every output file. The build fails on any source or output mismatch.
