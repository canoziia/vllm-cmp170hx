"""Preserve NEGATIVE results; these experiments are not production defaults."""
from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[1]/'scripts/policies'
for version in range(3,9):
 ns=runpy.run_path(str(root/f'measured_v{version}.py'));ns['self_test']()
 p=ns['Policy']({});k=5;states=[]
 for i in range(1000):
  a=k if i<200 or i>=600 else 0
  p.observe(dict(k=k,accepted=a,seconds=ns['costs'](1)[k-1],output_tokens=a+1,timing_valid=True,load_epoch=1,concurrency=1))
  new=p.choose(dict(current_k=k,concurrency=1));assert 1<=new<=5 and new<=k+1;k=new
  if i in (199,599,999):states.append(k)
 expected=[5,1,3] if version in (3,4,5,6) else [5,1,5]
 assert states==expected,(version,states)
 print('documented regression v'+str(version),states)
