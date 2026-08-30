# Patch series

The patches are applied in lexical order to files extracted from
`docker.io/vllm/vllm-openai:qwen38-flash-next`.

| Patch | Purpose |
| --- | --- |
| `0001-qwen38-pp4-model.patch` | Allocate embedding only on PP0 and LM head/final mixer only on PP3; filter checkpoint parameters by PP ownership. |
| `0002-ple-pp4-ownership.patch` | Require Qwen PLE layers to live on PP0 and prevent PP1–PP3 from creating PLE state/offload clients. |
| `0003-ple-nvme-pread.patch` | Add safetensors-backed PLE row access (`mmap` or parallel `pread`), isolate the PLE subprocess from the PP partition, load only small PLE metadata tensors, and apply bounded backpressure to the single-slot PLE transport under concurrent PP requests. |
| `0004-v2-mtp-local-speculator.patch` | Treat the complete MTP model on PP3 as a local standalone V2 speculator; inherit target 1M length and shared cache geometry. |
| `0005-v2-mtp-capture-order.patch` | Synchronize PP control flow with the CPU/Gloo group after the PP3-only MTP CUDA Graph capture. |
| `0006-v2-pp-speculative-feedback.patch` | Pad sampled-token broadcasts to a fixed width and propagate PP3's real draft IDs to the other PP target stages through the existing V2 PPHandler channel. |
| `0007-ple-next-chunk-prefetch.patch` | Predict one bounded future PLE batch from V2 request state, overlap its native disk gather with the current GPU forward, validate request IDs, absolute positions, token IDs, and n-gram context before reuse, and report candidate accuracy plus eligible-prefill step/token coverage without counting decode as a miss. |
| `0008-mamba-scheduler-block-alignment.patch` | Align Mamba prefill checkpoint stops to the scheduler's hybrid-group LCM block size instead of the minimum cache-group block size, preserving reusable recurrent-state checkpoints with fine-grained prefix matching and MTP. |

## Integrity

`manifests/base-qwen38-flash-next.sha256` pins every extracted source file before
patching. `manifests/final-qwen38-flash-next.sha256` verifies the final patched output.
The build fails if the base image changes any pinned file or if a patch produces
unexpected output.
