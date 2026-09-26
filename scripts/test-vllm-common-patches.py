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

print("VLLM_COMMON_PATCHES", "FAIL: " + ", ".join(failures) if failures else "PASS")
sys.exit(1 if failures else 0)
