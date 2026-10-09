from pathlib import Path
import runpy,math
ns=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/policies/measured_v9.py'));P=ns['Policy'];ns['self_test']()
def e(k,a,valid=True,cost=None):return dict(k=k,accepted=a,output_tokens=1+a,seconds=cost or ns['costs'](1)[k-1],timing_valid=valid,load_epoch=1)
for start in range(1,6):
 p=P({});p.trimmed=start<5;k=start
 states=[]
 for i in range(1200):
  a=k if i<200 or i>=600 else 0;p.observe(e(k,a));new=p.choose(dict(current_k=k,concurrency=1));assert 1<=new<=5 and new<=k+1;k=new
  if i in (199,599,1199):states.append(k)
 assert states==[5,1,5],(start,states)
p=P({});p.debt=.1;p.observe(e(5,0));assert math.isclose(p.debt,.095)
p.observe(e(5,0));assert p.choose(dict(current_k=3,concurrency=1))==3
p.observe(e(3,0,False));assert p.choose(dict(current_k=3,concurrency=1))==3
p=P({});p.debt=.1
for _ in range(1000):p.observe(e(5,5))
assert p.debt<1e-20
for config in ({'switch_margin':float('nan')},{'switch_margin':float('inf')},{'switch_margin':-1}):
 try:P(config)
 except ValueError:pass
 else:raise AssertionError(config)
print('PASS API self_test, max+1, high-low-high from all widths, stale/invalid guards, debt decay, invalid params')
