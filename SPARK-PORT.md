# DGX Spark NVFP4 port plan

Target checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4`

Target platform:

- One DGX Spark
- GB10 / SM121
- ARM64
- TP1 x PP1
- 128 GiB unified memory
- PLE kept in the checkpoint's original safetensors files and gathered from local NVMe

## Checkpoint metadata verified

The Hugging Face metadata currently declares:

```text
architecture=Qwen4ExpForConditionalGeneration
model_type=qwen4_exp
quant_method=modelopt
quant_algo=NVFP4
total_size=135,195,303,851 bytes (125.91 GiB)
ple_embedding_dtype=float8_e4m3fn
split_ngram_parts=128
```

The weight map contains exactly 128 tensors matching:

```text
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<N>.weight
```

The PLE remains FP8 E4M3 rather than NVFP4. Its approximately 47.68 GiB can therefore use the existing byte-exact row-reader design. Excluding PLE leaves approximately 78.23 GiB of checkpoint data before runtime packing and allocator overhead.

## Patch disposition

| Existing patch | Spark action | Reason |
| --- | --- | --- |
| `0001-qwen38-pp4-model.patch` | Omit | PP ownership of embeddings, final mixer, LM head, and stage-filtered loading is unnecessary under PP1. |
| `0002-ple-pp4-ownership.patch` | Omit, retaining only any independently required NVMe validation when rebasing | Its purpose is placing PLE exclusively on PP0 and suppressing clients on PP1-PP3. Under PP1 the only rank already owns PLE and raw input IDs. |
| `0003-ple-nvme-pread.patch` | Port first | This is the essential Spark feature: leave the FP8 PLE table in safetensors, load only metadata/scales, and gather selected rows with `pread` or `mmap`. The RadixArk tensor suffix, dtype, row width, and 128-shard layout match its manifest discovery. |
| `0004-v2-mtp-local-speculator.patch` | Omit initially; port only a minimal config-isolation subset if reproduced | The standalone last-PP-rank workaround is specific to PP4. Under PP1 the target rank is both first and last. Native PP1 MTP should be tested before carrying this patch. |
| `0005-v2-mtp-capture-order.patch` | Omit | The CPU/Gloo barrier exists only to keep four PP ranks ordered while PP3 alone captures MTP. |
| `0006-v2-pp-speculative-feedback.patch` | Omit | PP1 has no cross-stage sampled-token or draft-token feedback channel. |
| `0007-ple-next-chunk-prefetch.patch` | Port after the synchronous NVMe reader passes | V2 future-prompt prediction and bounded one-batch PLE prefetch are not inherently PP-specific. PP1 still benefits from overlapping SSD gather with target GPU work. Revalidate on unified memory and the Spark NVMe. |
| `0008-mamba-scheduler-block-alignment.patch` | Port when enabling hybrid prefix caching/fine matching | This is scheduler correctness for hybrid Mamba/QSA block geometry, not CMP- or PP-specific. It is unnecessary for the first no-MTP/no-fine-prefix smoke test. |
| `0009-mtp-safe-partial-tail-checkpoint.patch` | Port when combining MTP with fine-grained prefix matching | This preserves recurrent-state correctness when MTP drops the final fine hash unit. It is portable but should not be part of the minimal first boot. |
| `0010-uniproc-ple-lifecycle.patch` | Required for single-Spark TP1×PP1 | Testing proved that upstream only called `spawn_ple_offload()` and `wait_ple_offload_ready()` from the multiprocess executor. `UniProcExecutor` registered the GPU connector but never started its PLE server until this lifecycle patch was added. |

## Native helper portability

`native/ple_pread.c` uses Linux `pread`, fixed-width integers, and OpenMP only. It has no x86 intrinsics and is source-portable to ARM64. The `.so` itself is not portable: it must be compiled inside the Spark ARM64 image with GCC and OpenMP available.

## Implemented and validated stages

The following stages were completed on the ARM64 DGX Spark:

1. Selected and pinned the official multi-architecture Qwen image with `qwen4_exp` and ModelOpt NVFP4 support.
2. Regenerated ARM64 source and final SHA256 manifests.
3. Ported `0003`, `0007`, `0008`, and `0009` together as requested.
4. Added `0010` after real TP1×PP1 startup exposed the missing UniProc PLE lifecycle.
5. Verified all 128 PLE shards, F8_E4M3 metadata, 320,001,536 rows, and 128 random rows byte-for-byte.
6. Verified the SM121 NVFP4 MoE backend is `FLASHINFER_CUTLASS`, not emulation.
7. Started native 262K context, MTP3, BF16 cache, and `prefix-match-unit=32` together.
8. Verified health, deterministic generation, fine prefix hits, prefetch telemetry, and prefill/decode throughput.

Multimodal remains intentionally disabled in the initial single-Spark profile.

## Configuration changes from CMP deployment

The Spark profile must remove:

```text
--pipeline-parallel-size=4
VLLM_PP_LAYER_PARTITION
all PP-specific runtime assumptions
```

It should use:

```text
--tensor-parallel-size=1
--pipeline-parallel-size=1 (or omit)
VLLM_PLE_CPU_OFFLOAD=1
VLLM_PLE_NVME_PATH=<RadixArk snapshot>
```

Model IDs, cache mount paths, snapshot metadata, image architecture, backend selection, context budget, and all manifests must be Spark/NVFP4-specific rather than inherited from the FP8 CMP profile.
