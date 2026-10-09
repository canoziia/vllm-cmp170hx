"""Document research behavior, including regressions; NOT algorithm acceptance."""
from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[1]/'scripts/policies'
for version in (9,10,11,12):
 ns=runpy.run_path(str(root/f'measured_v{version}.py'));ns['self_test']()
 for start in range(1,6):
  p=ns['Policy']({});p.trimmed=start<5;k=start;states=[]
  for i in range(1200):
   a=k if i<200 or i>=600 else 0
   p.observe(dict(k=k,accepted=a,seconds=ns['costs'](1)[k-1],output_tokens=1+a,timing_valid=True,load_epoch=1))
   z=p.choose(dict(current_k=k,concurrency=1));assert 1<=z<=5 and z<=k+1;k=z
   if i in (199,599,1199):states.append(k)
  assert states==[5,1,5],(version,start,states)
 print('PASS high-low-high version',version)
 # A permanently rejected next token still tempts repeated exploration once
 # old evidence decays. Preserve this negative result explicitly.
 p=ns['Policy']({});k=5;history=[]
 for _ in range(1200):
  a=min(k,2)
  p.observe(dict(k=k,accepted=a,seconds=ns['costs'](1)[k-1],output_tokens=1+a,timing_valid=True,load_epoch=1))
  k=p.choose(dict(current_k=k,concurrency=1));history.append(k)
 switches=sum(a!=b for a,b in zip(history[800:],history[801:]))
 print('NEGATIVE: stationary cap2 repeated probes version',version,'switches/400',switches)
 if version in (11,12):assert switches>0
