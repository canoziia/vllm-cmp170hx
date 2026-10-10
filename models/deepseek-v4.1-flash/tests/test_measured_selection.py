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
print('PASS measured selection all-width/six-concurrency recovery and finite scores')
