# Deterministic QSA top-k extension

Upstream implementation by jschmied and vLLM contributors, from vLLM PR #55122.
See SOURCE.txt for the pinned Apache-2.0 source revision and local adaptations.
This wrapper registers an independent `_C_det.persistent_topk` operator so the
base image's bundled `_C` library does not need a full rebuild.

Build inside the ARM64 CUDA image with `DET_ARCH=121a`; the runtime switch is
`VLLM_QSA_DET_TOPK=1`. The model still selects exactly its top-k candidates; this
is not quantization. Kernel determinism alone is not a claim of whole-model
bitwise determinism.
