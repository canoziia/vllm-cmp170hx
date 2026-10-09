"""Uncertainty comparison / attribution / recovery tests (not GPU acceptance)."""
from pathlib import Path
import runpy,math,time
ns=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/policies/measured_v13.py'));P=ns['Policy'];ns['self_test']()
def event(k,a,valid=True,epoch=1,seconds=None):
 return dict(k=k,accepted=a,output_tokens=1+a,seconds=seconds or ns['costs'](1)[k-1],timing_valid=valid,load_epoch=epoch)
for c in (1,2,4,8,16,32):
 p=P({});k=5;states=[]
 for i in range(1600):
  a=k if i<300 or i>=900 else 0
  p.observe(event(k,a,seconds=ns['costs'](c)[k-1]));z=p.choose(dict(current_k=k,concurrency=c));assert 1<=z<=5 and z<=k+1;k=z
  if i in (299,899,1599):states.append(k)
  assert all(math.isfinite(v) for key in ('score','se','gate') for v in p.diagnostics[key])
 assert states==[5,1,5],(c,states)
p=P({});p.observe(event(5,0));assert p.choose(dict(current_k=3,concurrency=1))==3
p.observe(event(3,0,False));assert p.choose(dict(current_k=3,concurrency=1))==3
p=P({})
for _ in range(10):p.observe(event(2,1,seconds=.025))
old=p.n[1];oldrate=p.y[1]/p.t[1]
for _ in range(15):p.observe(event(3,2,seconds=3/90))
assert math.isclose(p.n[1],old*.95**15) and math.isclose(p.y[1]/p.t[1],oldrate)
p.observe(event(5,5,False,epoch=2));assert all(v==0 for field in ('n','y','t','q','y2','t2','yt') for v in getattr(p,field))
p=P({});k=5;history=[]
for _ in range(1200):
 p.observe(event(k,min(k,2)));k=p.choose(dict(current_k=k,concurrency=1));history.append(k)
switches=sum(a!=b for a,b in zip(history[800:],history[801:]));assert switches<=6,switches
print('PASS six-concurrency high/low/high, finite statistics, actual-width and invalid-timing gates, evidence-not-speed decay, load reset, stationary cap2 <=6 switches/400 (not zero)')
start=time.perf_counter()
for _ in range(10000):p.observe(event(3,2));p.choose(dict(current_k=3,concurrency=16))
print('CPU average us/observe+choose',round((time.perf_counter()-start)*1e6/10000,2))
