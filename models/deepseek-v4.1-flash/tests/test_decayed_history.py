#!/usr/bin/env python3
import importlib.util,logging,sys,math
from pathlib import Path
from types import SimpleNamespace as NS
p=Path(sys.argv[1])/'vllm/v1/core/sched/dsv41_history_policy.py';s=importlib.util.spec_from_file_location('decay_candidate',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
h=m.History(decay=.9);h.observe(5,2);h.observe(2,2)
assert all(abs(a-b)<1e-12 for a,b in zip(h.risk,[1.9,1.9,.9,0,0]))
assert all(abs(a-b)<1e-12 for a,b in zip(h.success,[1.9,1.9,0,0,0]))
assert all(abs(a-b)<1e-12 for a,b in zip(h.risk_weight_sq,[1.81,1.81,.81,0,0]))
h=m.History(decay=0);h.observe(5,5);h.observe(5,0);assert h.risk==[1,0,0,0,0] and h.success==[0]*5
for invalid in (-.1,1,math.inf,math.nan,True):
 try:m.HistoryPolicy(logging.getLogger('silent'),decay=invalid)
 except ValueError:pass
 else:raise AssertionError(invalid)
r=NS(request_id='test',is_prefill_chunk=False,decode_cohort_phase=0)
pol=m.HistoryPolicy(logging.getLogger('silent'),decay=.9);plan=[]
for t in range(360):
 k=pol.choose([r],1,[r])[r.request_id];plan.append(k)
 actual=5 if t<100 or t>=220 else 2
 pol.observe(r.request_id,k,min(actual,k))
print('high initial k',sorted(set(plan[40:100])))
print('low first trim',next((i+100 for i,k in enumerate(plan[100:220]) if k<5),None),'tail k',plan[180:220])
print('high again first5',next((i+220 for i,k in enumerate(plan[220:]) if k==5),None),'last40',plan[-40:])
assert set(plan[40:100])=={5}
assert any(k<5 for k in plan[100:132])
assert sum(k==5 for k in plan[-40:])>=30,plan[-40:]
# Same-block choose calls must not satisfy two-feedback persistence.
p=m.HistoryPolicy(logging.getLogger('silent'),decay=.9)
for _ in range(20):p.observe('test',5,2)
p.rng.random=lambda:1
h=p.state('test');k=p.choose([r],1,[r])['test'];blocks=h.pending_blocks
for _ in range(5):p.choose([r],1,[r])
assert h.pending_blocks==blocks
r.verification_draft_limit=0;assert p.choose([r],1,[r])=={} and p.last_plan['test']==0
print('PASS configurable decay/risk censoring/weight ESS; high-low-high adaptation; fresh-feedback persistence; manual0')
