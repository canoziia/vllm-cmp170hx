#!/usr/bin/env python3
import ast,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
root=Path(sys.argv[1])/'vllm'
t=ast.parse((root/'v1/core/sched/async_scheduler.py').read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='AsyncScheduler');f=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='_update_after_schedule');f.body=f.body[1:]
ns=dict(os=os,SchedulerOutput=object);exec(compile(ast.Module([f],type_ignores=[]),'phase','exec'),ns)
os.environ['VLLM_DSV41_BALANCED_COHORTS']='1';os.environ['VLLM_DSV41_DECODE_TRACE']='0'
for n in range(1,33):
 capacity=(n+5)//6
 reqs={str(i):NS(is_prefill_chunk=False,num_output_placeholders=0,use_structured_output=False) for i in range(n)}
 s=NS(requests=reqs,running=list(reqs.values()),current_step=7,pp_size=6,use_pp=True,num_sampled_tokens_per_step=1,use_v2_model_runner=True,_spec_token_placeholders=[-1]*5,vllm_config=NS(model_config=NS(architectures=['DeepseekV41ForCausalLM'])),_get_max_num_scheduled_decodes=lambda:capacity)
 so=NS(scheduled_spec_decode_tokens={},num_spec_tokens_to_schedule=5,num_scheduled_tokens={k:8 for k in reqs},pending_structured_output_tokens=False)
 ns['_update_after_schedule'](s,so)
 counts=[sum(getattr(r,'decode_cohort_phase',None)==p for r in reqs.values()) for p in range(6)]
 assert max(counts)<=capacity
 assert sum(x>0 for x in counts)==(n+capacity-1)//capacity,(n,counts)
 assert all(13<=r.next_decode_eligible_step<=18 for r in reqs.values())
 for step in range(13,19):
  ids={k:6 for k,r in reqs.items() if r.next_decode_eligible_step==step};s.current_step=step;so.num_scheduled_tokens=ids
  ns['_update_after_schedule'](s,so)
  assert all(reqs[k].next_decode_eligible_step==step+6 for k in ids)
# Requests arriving gradually initially occupy all six one-request phases.
# Growing load changes capacity to two: the next safe decode of each partial
# phase should converge to four full phases, never advance a due timestamp.
reqs={str(i):NS(is_prefill_chunk=False,num_output_placeholders=0,use_structured_output=False,decode_cohort_phase=i%6,next_decode_eligible_step=12+i%6) for i in range(8)}
s.requests=reqs;s.running=list(reqs.values());s._get_max_num_scheduled_decodes=lambda:2
for step in range(12,60):
 ids={k:2 for k,r in reqs.items() if r.next_decode_eligible_step==step}
 s.current_step=step;so.num_scheduled_tokens=ids
 ns['_update_after_schedule'](s,so)
 assert all(step+6<=reqs[k].next_decode_eligible_step<=step+11 for k in ids)
counts=[sum(r.decode_cohort_phase==p for r in reqs.values()) for p in range(6)]
assert sorted(counts)==[0,0,2,2,2,2],counts
print('PASS C1..32 minimal phases; load-growth compaction C8; no early feedback; exact PP cadence when stable')
