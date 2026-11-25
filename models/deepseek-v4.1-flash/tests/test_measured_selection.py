"""Production measured selector CPU safety and recovery without GPU imports."""
import runpy,sys,math
from pathlib import Path
ns=runpy.run_path(str(Path(sys.argv[1])/'vllm/v1/core/sched/dsv41_measured_policy.py'));P=ns['Policy'];ns['self_test']()
for c in (1,2,4,8,16,32):
 for start in range(1,6):
  p=P({});p.trimmed=start<5;k=start;states=[]
  for i in range(1600):
   a=k if i<300 or i>=900 else 0
   p.observe(dict(k=k,accepted=a,seconds=ns['costs'](c)[k-1],output_tokens=a+1,timing_valid=True,load_epoch=1))
   z=p.choose(dict(current_k=k,concurrency=c));assert 1<=z<=5 and z<=k+1;k=z
   assert all(math.isfinite(v) for key in ('score','gate','se') for v in p.diagnostics[key])
   if i in (299,899,1599):states.append(k)
  assert states==[5,1,5],(c,start,states)
OM=ns['OnlineCostModel']
om=OM(prior_table=ns['COST_MS_BASE'])
for _ in range(10):
 for c,row in ns['COST_MS_TP4'].items():
  for idx,y_ms in enumerate(row):om.observe(c,c*(idx+2),y_ms/1000.0)
for c in (1,2,4,8,16,32):
 for start in range(1,6):
  p=P({});p.trimmed=start<5;k=start;states=[]
  for i in range(1600):
   a=k if i<300 or i>=900 else 0
   dt=ns['COST_MS_TP4'][c][k-1]/1000.0
   om.observe(c,c*(k+1),dt)
   p.observe(dict(k=k,accepted=a,seconds=dt,output_tokens=a+1,timing_valid=True,load_epoch=1))
   z=p.choose(dict(current_k=k,concurrency=c,cost_model=om));assert 1<=z<=5 and z<=k+1;k=z
   if i in (299,899,1599):states.append(k)
  assert states==[5,1,5],('online',c,start,states)
print('PASS measured selection + online cost model all-width/six-concurrency recovery and finite scores')
