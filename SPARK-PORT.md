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

## Native helper portability

`native/ple_pread.c` uses Linux `pread`, fixed-width integers, and OpenMP only. It has no x86 intrinsics and is source-portable to ARM64. The `.so` itself is not portable: it must be compiled inside the Spark ARM64 image with GCC and OpenMP available.

## Recommended implementation stages

1. Select and pin an ARM64/SM121 vLLM base image with `qwen4_exp` and serialized ModelOpt NVFP4 support.
2. Regenerate source and final SHA256 manifests for that exact image. Do not reuse the CMP image manifests.
3. Start TP1 x PP1, text-only, no MTP, 32K-64K context.
4. Port only the synchronous portion of `0003`; verify all 128 PLE shards, F8_E4M3 metadata, total rows, and random-row byte equality.
5. Verify the selected NVFP4 Linear and MoE backends and end-to-end output correctness.
6. Port `0007`; run byte verification and uncached long-prefill A/B.
7. Raise context toward native 262K according to measured remaining unified memory.
8. Enable MTP; carry only the subset of `0004` that an actual PP1 failure proves necessary.
9. If using `--prefix-match-unit`, port and validate `0008` and `0009` together.
10. Enable multimodal only after text, PLE, prefix cache, and MTP are independently stable.

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
