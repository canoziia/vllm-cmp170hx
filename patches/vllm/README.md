# Shared vLLM patches

These patches target the pinned author source revision
`d63af5a472dc76b12d7a73d50a5af142844c15d1` and are applied to every model
build that uses that runtime.

1. `0001-async-pp-mamba-state-reclaim.patch`
   - avoids self-preempting a lone request while asynchronous pipeline work is
     still in flight;
   - retains every pending Mamba source-state index until its processed token
     boundary makes reclamation safe.

2. `0002-mamba-resolved-cache-geometry.patch`
   - binds the resolved `KVCacheConfig` before requests are admitted;
   - derives align-mode resumed-state columns from `MambaSpec.block_size`
     instead of the smallest global cache group.

3. `0003-balance-async-pp-decode-batches.patch`
   - port of vllm-project/vllm#57433: one scheduler step takes at most
     `ceil(max_num_seqs / pp_size)` established decode requests, so MRV2 + PP +
     async spreads the running set over the `pp_size` in-flight pipeline batches
     instead of letting one cadence group burst and the rest starve;
   - applied to the RUNNING loop, to established decodes resumed from WAITING,
     and the cohort slot is returned on preemption; prefill is never capped and
     PP1 / synchronous scheduling / MRV1 keep `max_num_seqs`;
   - one deviation: upstream enables it unconditionally, here it is gated behind
     `VLLM_PP_DECODE_COHORT_BALANCE` (default `0`) so one image serves an A/B,
     rollback is an env var, and the Qwen build of this shared series is not
     changed before it has been measured there;
   - concurrency 32 output rate +89% on prose and +184% on counting on the
     six-GPU SM80 PP6 deployment; see
     `models/deepseek-v41/docs/PP-DECODE-COHORT-BALANCE.md`.
   - upstream behavioural test ported into `tests/v1/core/test_async_scheduler.py`
     (CPU only), plus one test for the opt-in deviation.

Both patches were validated in the DeepSeek default/debug series and in the
Qwen PP2/MTP3 series. Model-specific code such as Qwen's process-isolated PLE
NVMe backend remains under `models/qwen3.8-flash-next/patches/`.
