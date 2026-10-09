import runpy
from pathlib import Path
pfile=Path(__file__).resolve().parents[1]/'scripts/policies/measured_v2.py'
ns=runpy.run_path(str(pfile));ns['self_test']();P=ns['Policy']
def event(k,a,valid=True):return dict(k=k,accepted=a,output_tokens=1+a,seconds=.03,timing_valid=valid,load_epoch=1)
p=P({});p.observe(event(5,0));assert p.choose(dict(current_k=5,concurrency=1))==5,'one bad sample must not penalize only incumbent'
p.observe(event(5,0));assert p.choose(dict(current_k=3,concurrency=1))==3,'old feedback must not retrigger selection'
p.observe(event(3,0,False));assert p.choose(dict(current_k=3,concurrency=1))==3,'invalid timing not new-arm confirmation'
for start in (1,2,3,4,5):
 p=P({});k=start
 for _ in range(150):p.observe(event(k,k));new=p.choose(dict(current_k=k,concurrency=1));assert new<=k+1;k=new
 assert k==5
 for _ in range(300):p.observe(event(k,0));k=p.choose(dict(current_k=k,concurrency=1))
 assert k==1,k
 for _ in range(300):p.observe(event(k,k));k=p.choose(dict(current_k=k,concurrency=1))
 assert k==5,k
print('PASS calibrated one-sample guard, stale-width guard, valid-timing guard, synthetic high/low/high')
