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
     `VLLM_PP_DECODE_COHORT_BALANCE`, which **defaults to `1`** because the PP6
     A/B measured it as a large, repeatable win. Setting it to `0` restores
     upstream behaviour, so a rollback is one env var plus a restart. Note that
     the Qwen PP2 deployment runs with it enabled without having been A/B'd
     (its battery was measured in the on state only), so the default rests on
     the DeepSeek evidence, not on a Qwen measurement;
   - concurrency 32 output rate +89% on prose and +184% on counting on the
     six-GPU SM80 PP6 deployment; see
     `models/deepseek-v41/docs/PP-DECODE-COHORT-BALANCE.md`.
   - upstream behavioural test ported into `tests/v1/core/test_async_scheduler.py`
     (CPU only), plus one test for the opt-in deviation.

4. `0004-grammar-fail-closed-draft-rows.patch`
   - port of vllm-project/vllm#54442, the fail-closed half of #54437: a
     structured-output request could accept drafts that were verified against an
     all-permissive bitmask row. `grammar_bitmask` reads a `-1` placeholder in
     the scheduled draft window as "the grammar rejected this draft" and fills
     every later row with `_full_mask`, which only holds when that `-1` came from
     `validate_tokens`. With async scheduling + PP the drafts are filled in by
     the worker, so a request whose drafts never arrive keeps its placeholders
     while the worker verifies the real drafts from its own device copy;
     acceptance then walks into the permissive rows and the step emits up to
     `num_speculative_tokens` tokens with no grammar constraint — silently
     poisoning the request, or killing it with `Failed to advance FSM` /
     `grammar rejected tokens` when those tokens are not grammar-legal;
   - the scheduler now reports how many leading drafts the bitmask actually
     constrained (`GrammarOutput.num_acceptable_drafts`, derived from the
     upstream `strip_speculative_padding` helper), the model runner turns that
     into a mask over logit rows, and the rejection sampler verifies against
     drafts with those rows pinned invalid (`draft_sampled >= 0`), so acceptance
     cannot reach them;
   - one deviation: upstream keys the row limit by absolute flattened positions,
     which is wrong under adaptive verification because `cu_num_logits_np` keeps
     the pre-compaction layout; here local positions come from the device
     `cu_num_logits`, as in the rebased upstream revision;
   - upstream's unit test ported into
     `tests/v1/worker/test_grammar_invalid_drafts.py` (CPU only, a
     `SimpleNamespace` batch), including the adaptive-verification layout case;
   - the root cause is still open upstream (the draft hand-off carries no request
     identity), so this patch does not restore lost drafts; it only stops the
     request from sampling without the grammar. Warmup and the legacy
     `gpu_model_runner` path pass no row limit and keep their behaviour.

5. `0005-wait-for-structured-draft-backfill.patch`
   - async scheduling with PP can schedule worker-side `-1` draft placeholders
     after a prior result has fully settled (`num_output_placeholders == 0`).
     The upstream pending-grammar gate only checked outstanding output tokens,
     so it sometimes sampled before fetching the real draft IDs. With 0004's
     fail-closed check, this produces a zero-acceptance loop: every three-draft
     window has zero grammar-constrained positions and is rejected;
   - defer structured-output sampling when the *scheduled* draft window itself
     contains `-1`, even with no outstanding output tokens. The existing
     `take_draft_token_ids` / grammar validation path then fills the actual
     request's window before bitmask construction. If the older-result queue is
     empty, the engine backfills directly before queuing the current sample;
     otherwise the deferred path would pop an empty queue. Retain the old
     outstanding-output gate, and do not defer normal requests or already-real
     drafts;
   - CPU regressions test the zero-placeholder boundary and non-structured
     control, plus the empty-queue draft/backfill/grammar/sample order.

The first four patches were validated in the DeepSeek default/debug series and
in the Qwen PP2/MTP3 series. Patch 0005 needs a new-image runtime acceptance
A/B before claiming a measured gain. Model-specific code such as Qwen's
process-isolated PLE NVMe backend remains under
`models/qwen3.8-flash-next/patches/`.
