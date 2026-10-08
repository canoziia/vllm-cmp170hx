#!/usr/bin/env python3
"""Static regression for the shared vLLM patch series (patches/vllm/series).

Usage: test-vllm-common-patches.py <SOURCE_TREE>
Exits non-zero if a shared fix silently disappears from the patched source.
"""
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' :: ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


# ---- 0003 balance async PP decode batches (port of upstream #57433) ----
envs_src = (root / "vllm/envs.py").read_text()
sched_path = root / "vllm/v1/core/sched/scheduler.py"
sched_src = sched_path.read_text()
async_src = (root / "vllm/v1/core/sched/async_scheduler.py").read_text()
test_src = (root / "tests/v1/core/test_async_scheduler.py").read_text()
sched = ast.parse(sched_src)

print("0003-balance-async-pp-decode-batches:")
check("env var declared", "VLLM_PP_DECODE_COHORT_BALANCE: bool = True" in envs_src)
# The default is ON on purpose: the PP6 same-image A/B measured it as a large
# win (prose +19.3%/+77.8%, counting +48.9%/+78.7%, acceptance unchanged). The
# switch still exists so a deployment can restore upstream behaviour with
# VLLM_PP_DECODE_COHORT_BALANCE=0, and that path stays asserted below.
check("env var defaults to on", 'os.getenv("VLLM_PP_DECODE_COHORT_BALANCE", "1")' in envs_src)
check("the off path is still wired (not a removed gate)",
      "envs.VLLM_PP_DECODE_COHORT_BALANCE" in async_src)
check(
    "policy hook defined on the base Scheduler (PP1/sync/MRV1 unchanged)",
    "def _get_max_num_scheduled_decodes(self) -> int:\n        return self.max_num_running_reqs"
    in sched_src,
)
check(
    "AsyncScheduler overrides it behind the env var",
    "envs.VLLM_PP_DECODE_COHORT_BALANCE" in async_src
    and "def _get_max_num_scheduled_decodes" in async_src,
)
check(
    "async override keeps MRV2 + PP gating from upstream",
    "not self.use_v2_model_runner" in async_src and "not self.use_pp" in async_src,
)
check(
    "share is ceil(max_num_seqs / pp_size)",
    "(self.max_num_running_reqs + self.pp_size - 1) // self.pp_size" in async_src,
)
check(
    "cohort set is local to schedule(), not scheduler state",
    "scheduled_decode_req_ids: set[str] = set()" in sched_src
    and "_pp_scheduled_decode_ids" not in sched_src,
)
check(
    "established decode defined as upstream",
    sched_src.count("request.num_computed_tokens >= request.num_prompt_tokens") >= 1,
)
check(
    "cap sits before KV block allocation in the RUNNING loop",
    sched_src.find("len(scheduled_decode_req_ids) >= max_num_scheduled_decodes")
    < sched_src.find("# Schedule newly needed KV blocks for the request."),
)
check(
    "established decodes resumed from WAITING are capped too",
    "step_skipped_waiting.prepend_request(request)" in sched_src
    and sched_src.count("scheduled_decode_req_ids.add(request_id)") == 2,
)
check(
    "preemption returns the cohort slot",
    "scheduled_decode_req_ids.discard(preempted_req_id)" in sched_src,
)
check(
    "per-step invariant asserted at the end of schedule()",
    "assert len(scheduled_decode_req_ids) <= max_num_scheduled_decodes" in sched_src,
)
check(
    "prefill/admission uncapped: WAITING gate requires not load_kv_async",
    "not load_kv_async\n                    and num_computed_tokens >= request.num_prompt_tokens"
    in sched_src,
)
check(
    "upstream behavioural test ported",
    "def test_async_pp_balances_decode_batches_without_throttling_prefills" in test_src,
)
check(
    "opt-in deviation also covered",
    "def test_async_pp_balance_is_off_by_default" in test_src,
)
check(
    "no stale identifiers from the pre-port variant",
    not any(
        t in sched_src + async_src
        for t in ("pp_decode_cohort_target", "is_established_decode", "_pp_scheduled_decode")
    ),
)

# ---- 0001 / 0002 stay present (they are load-bearing for async PP) ----
print("0001/0002 async-PP Mamba fixes:")
check(
    "0001 async-PP lone-request drain guard present",
    "num_in_flight_tokens" in sched_src,
)
kv = (root / "vllm/v1/core/single_type_kv_cache_manager.py").read_text()
check("0001 deferred Mamba reclaim queue present", "_decode_tail_snapshots" in kv or "reclaim" in kv)
mr = (root / "vllm/v1/worker/gpu/model_runner.py").read_text()
check(
    "0002 resolved cache geometry is bound before admission",
    "set_kv_cache_config" in mr,
)

# ---- 0004 stop structured output from accepting drafts on permissive rows ----
print("0004-grammar-fail-closed-draft-rows:")
o4_out = (root / "vllm/v1/core/sched/output.py").read_text()
o4_mr = (root / "vllm/v1/worker/gpu/model_runner.py").read_text()
o4_rs = (root / "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py").read_text()
o4_shard = (root / "vllm/v1/worker/gpu/sample/batch_shard.py").read_text()
o4_test = (root / "tests/v1/worker/test_grammar_invalid_drafts.py").read_text()
check(
    "GrammarOutput carries the constrained draft prefix",
    "num_acceptable_drafts: list[int] | None = None" in o4_out,
)
check(
    "scheduler derives it from the scheduled draft window",
    "len(strip_speculative_padding(spec_tokens.get(req_id, [])))" in sched_src,
)
check(
    "row limit is keyed on device local positions",
    "input_batch.expanded_local_pos" in o4_mr
    and "cu_num_logits[1:] - cu_num_logits[:-1]" in o4_mr
    and "local_pos > row_limit" in o4_mr,
)
check(
    "absent prefix invalidates the whole draft window",
    "num_acceptable_drafts[i] if num_acceptable_drafts is not None else 0" in o4_mr,
)
check(
    "row limit reaches the rejection sampler",
    "invalid_drafts = grammar_invalid_drafts(" in o4_mr
    and "self.speculator.draft_logits,\n                invalid_drafts,\n" in o4_mr,
)
check(
    "sampler pins those drafts invalid before verification",
    "verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)" in o4_rs
    and "rejection_sample(\n            processed_logits,\n            draft_logits,\n            verify_draft_sampled,"
    in o4_rs,
)
check(
    "sharded TP batches keep the prefix aligned",
    "num_acceptable_drafts=num_acceptable," in o4_shard,
)
check(
    "upstream unit test ported",
    "def test_follow_the_device_layout_under_adaptive_verification" in o4_test,
)

# ---- 0005 backfill worker-side drafts before masking structured outputs ----
print("0005-wait-for-structured-draft-backfill:")
o5_test = (root / "tests/v1/core/test_async_structured_draft_handoff.py").read_text()
check(
    "structured requests defer when scheduled drafts are worker-side placeholders",
    "scheduled_drafts = spec_decode_tokens.get(req_id, ())" in async_src
    and "or -1 in scheduled_drafts" in async_src
    and "request.num_output_placeholders > 0" in async_src,
)
check(
    "CPU regression covers zero outstanding outputs and non-structured control",
    "def test_async_structured_draft_handoff_without_prior_output" in o5_test
    and "(True, 0, [-1, -1, -1], True)" in o5_test
    and "(False, 0, [-1, -1, -1], False)" in o5_test,
)
engine_src = (root / "vllm/v1/engine/core.py").read_text()
check(
    "empty queue backfills worker drafts before grammar and sampling",
    "if deferred_scheduler_output and not batch_queue:" in engine_src
    and "self.scheduler.update_draft_token_ids_in_output(\n"
    in engine_src
    and "return None, model_executed" in engine_src,
)
check(
    "CPU regression covers empty-queue backfill order",
    "def test_async_structured_backfill_with_empty_batch_queue" in o5_test
    and 'events == ["take_drafts", "backfill", "grammar", "sample"]' in o5_test,
)

# ---- 0006 retrieve draft IDs by request, not last batch ----
print("0006-key-structured-drafts-by-request:")
check(
    "engine requests exact scheduled draft request IDs on both deferred paths",
    engine_src.count("list(deferred_scheduler_output.scheduled_spec_decode_tokens)") == 2,
)
check(
    "worker reads per-request persistent draft state",
    "self.req_states.draft_tokens[slots].tolist()" in mr
    and "req_id in self.req_states.req_id_to_index" in mr,
)
check(
    "GPU regression covers unrelated batch and removed request",
    "def test_request_keyed_drafts_survive_interleaved_batches" in o5_test
    and 'runner.take_draft_token_ids(["B", "gone"])' in o5_test,
)

# ---- 0007 NVFP4 Marlin scale factor without the boolean gather ----
fp4_path = root / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
fp4_src = fp4_path.read_text()
fp4_fn = next(
    (n for n in ast.parse(fp4_src).body
     if isinstance(n, ast.FunctionDef) and n.name == "_nvfp4_compute_scale_factor"),
    None,
)
fp4_fn_src = ast.get_source_segment(fp4_src, fp4_fn) if fp4_fn else ""
print("0007-nvfp4-marlin-scale-factor-amax:")
check("scale factor function present", bool(fp4_fn_src))
check("max is an in-place amax()", "max_val = marlin_scales.amax().float() * (2**7)" in fp4_fn_src)
check(
    "no float copy, mask or boolean gather of the whole scale tensor",
    "marlin_scales.float()" not in fp4_fn_src
    and "nonzero_mask" not in fp4_fn_src
    and "[nonzero_mask]" not in fp4_fn_src,
)
check(
    "non-positive maxima still fall back to 1.0",
    "if max_val > 0 and max_val < 448 * (2**7):" in fp4_fn_src
    and fp4_fn_src.rstrip().endswith("return 1.0"),
)

# Execute the real scheduler-config merger without importing GPU dependencies.
from types import SimpleNamespace
import copy
utils_src = (root / "vllm/v1/core/kv_cache_utils.py").read_text()
utils_ast = ast.parse(utils_src)
merge = next(n for n in utils_ast.body if isinstance(n, ast.FunctionDef)
             and n.name == "generate_scheduler_kv_cache_config")
ns = dict(copy=copy, KVCacheConfig=SimpleNamespace,
          UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}))
exec(compile(ast.Module(body=[merge], type_ignores=[]), "<PP merger>", "exec"), ns)
def config(flags):
    return SimpleNamespace(num_blocks=32, kv_cache_groups=[
        SimpleNamespace(is_eagle_group=flag, kv_cache_spec=object()) for flag in flags])
a, b = config([False, False]), config([False, True])
merged = ns[merge.name]([a, b])
check("PP merger preserves owner-only draft flag", [g.is_eagle_group for g in merged.kv_cache_groups] == [False, True])
check("PP merger does not mutate workers", not a.kv_cache_groups[1].is_eagle_group)
check("empty PP group retains global identity", "is_eagle_group=group.is_eagle_group and bool(worker_layer_names)" not in utils_src)
manager_src = (root / "vllm/v1/core/single_type_kv_cache_manager.py").read_text()
check("Mamba partial gate uses common replay stop", "if self.drop_eagle_checkpoint_block:" in manager_src)
try:
    ns[merge.name]([a, config([True])])
except ValueError:
    check("PP count mismatch rejected", True)
else:
    check("PP count mismatch rejected", False)

print("VLLM_COMMON_PATCHES", "FAIL: " + ", ".join(failures) if failures else "PASS")
sys.exit(1 if failures else 0)
