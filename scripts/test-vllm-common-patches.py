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


# ---- 0003 balance established decodes across PP batches ----
envs_src = (root / "vllm/envs.py").read_text()
sched_path = root / "vllm/v1/core/sched/scheduler.py"
sched_src = sched_path.read_text()
sched = ast.parse(sched_src)

print("0003-balance-pp-decode-cohorts:")
check("env var declared", "VLLM_PP_DECODE_COHORT_BALANCE: bool = False" in envs_src)
check(
    "env var defaults to off",
    'os.getenv("VLLM_PP_DECODE_COHORT_BALANCE", "0")' in envs_src,
)
check("gated on the env var", "envs.VLLM_PP_DECODE_COHORT_BALANCE" in sched_src)
check(
    "only when MRV2 + PP + async scheduling",
    all(
        s in sched_src
        for s in ("self.use_v2_model_runner", "self.use_pp", "scheduler_config.async_scheduling")
    ),
)
check(
    "target is ceil(max_num_seqs / pp_size)",
    "-(-self.max_num_running_reqs // pp_size)" in sched_src,
)
check(
    "limits established decodes only",
    "request.num_computed_tokens >= request.num_prompt_tokens" in sched_src,
)
check(
    "cap is evaluated after the existing cadence gate",
    sched_src.find("next_decode_eligible_step")
    < sched_src.find("is_established_decode"),
)
check(
    "counted at the schedule-accept point",
    "self._pp_scheduled_decode_ids.add(request_id)" in sched_src,
)
check(
    "count restored when a selected request is preempted",
    "self._pp_scheduled_decode_ids.discard(" in sched_src,
)
check(
    "per-step set cleared at the start of schedule()",
    "self._pp_scheduled_decode_ids.clear()" in sched_src,
)
# The cap must live in the RUNNING loop only: prefill/admission paths stay free.
schedule_fn = next(
    n for n in ast.walk(sched)
    if isinstance(n, ast.FunctionDef) and n.name == "schedule"
)
uses = sum(
    1
    for n in ast.walk(schedule_fn)
    if isinstance(n, ast.Name) and n.id == "is_established_decode"
)
# 3 = the assignment itself + the cap test + the accept-point increment.
check(
    "established-decode flag used exactly 3x (bind + cap + accept)",
    uses == 3,
    f"found {uses}",
)
check(
    "no new cap in the waiting/admission path",
    sched_src.count("pp_decode_cohort_target > 0") == 1,
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
