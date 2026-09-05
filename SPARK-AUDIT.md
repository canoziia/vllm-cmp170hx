# Spark audit and clean-final release boundary

## Included source and evidence

Twelve patches: 0003/0007/0008/0009 inherited from CMP; 0010–0017 added on Spark.
- 0010 fixes missing UniProc PLE lifecycle.
- 0011 ports GB10 FLA settings and deterministic QSA top-k. Kernel source now comes
  from Apache-2.0 vLLM PR55122 commit c056c2d937f9997fd14ed67ca039c8a78f3d5ef8;
  license and source provenance are in vendor/kernel-det. Standalone SM121 test:
  FAILS0. Deterministic selection does not prove whole-model equivalence.
- 0012 deduplicates/sorts PLE reads and restores original row order. Warm byte
  checks passed; original cold timing was order-confounded and is not a speed claim.
- 0013 ports overlapping Mamba copy repair without silent bounds skipping.
- 0014 repairs the worker block-size seed as well as scheduler fix0008. Historical
  2459/4949/8909-token tests returned identical cached/uncached token IDs, text and
  first-token logprobs. Positive cached-token counts alone are not sufficient.
- 0015 indexes expert mapping candidates and prefilters standalone local Qwen MTP
  loads (33 tensors including shared head/embedding in 3 files, rather than206).
- 0016 stages original packed uint8/FP8/FP32 expert values on CPU per module while
  calling the original weight_loader, preserving Parameter identity and padding.
  Real ModelWeightParameter/PerTensorScaleParameter fixture verifies bytes.
- 0017 fixes eager header parsing through dict.setdefault.

Combined loading time ~805s → ~474s; this is not an inference throughput claim.
The new deployment uses batch8192/native262144 per user request. Historical
clean-final medians were batch4096; see README. Final manifests identify patched
Python files; native sources and OCI Git revision identify accompanying build.

## Findings that must not be overstated

- GB10 was stuck at507MHz. User power cycle restored ~2.4GHz; the resulting major
  inference speed gain is hardware recovery, not our patches.
- Independent PLE worker timing overlaps GPU work. Connector reuse-event waits and
  CUDA Command Buffer Full often reflect GPU backpressure; timings cannot be added.
- First calls can page in weights/PLE and compile kernels. Track cache state and
  swap deltas; no benchmark uses the server's sliding-window average as request speed.
- Historical and newly compiled AOT caches produced differing model logprobs. Cause
  is unresolved; cross-cache comparisons cannot establish GEMV error or equivalence.
  Use a new isolated cache for the committed deployment and rerun regressions.

## Excluded experiments

GEMV (whole-model/head/draft-head), B12x, forced vLLM CUTLASS NVFP4, reference mmap,
shared-expert overlap changes, enlarged prefill graph capture, and diagnostic
profilers are not in this release. GEMV benefited predictable inputs but lacked
stable general gain/correctness evidence. B12x failed model warmup; first debug
call had all expert IDs=-1, unlike legal-route microbench. It is not accepted.
PDL/finalize and MTP BF16 MoE tests did not expose the missing major performance gain.

## Reference comparison

blazux/qwen3.8-Flash-DGX89b852f advertises raw NVFP4 prefill2400–2900; its specific
2904 measurement is described as hybrid/MTP2. Neither fact proves raw2900
unattainable. Exact published32K prompt was not located.
A separate raw-path reconstruction on the same official ARM64 base, with our
MTP3/util.90/BF16 constraints, reference batch8192/PIECEWISE/mmap32, reached warm
uncached single-token rates2363/2377 at10665/41568tokens. Current clean-final
control reached2267/2266 on those same prompts (~4–5% difference, not isolated
mmap benefit). These EOS-only responses have no visible SSE TTFT.
Reference fixed real-document decode was slower. Its runtime block832 also differs
from README's1600; published MTP2/util.85/seqs8 remain unmatched. Do not claim a full
published-configuration reproduction or completed2900/45 target.

## Operations

Original model and PLE table remain read-only. No extra lossy quantization.
16GiB swap is enabled; persistent NixOS config needs an authorized rebuild.
Linger/user restart service enabled, complete host reboot test not performed.
No changes to the AMD64 CMP host are part of this release.
