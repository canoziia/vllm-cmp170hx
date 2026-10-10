"""CPU-only attribution tests, no vLLM/GPU runtime required."""
import importlib.util,sys,tempfile,json,hashlib,logging
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
root=Path(sys.argv[1]);pkg=ModuleType('policy_test');pkg.__path__=[];sys.modules[pkg.__name__]=pkg
for name in ('dsv41_measured_policy','dsv41_fixed_policy'):
 spec=importlib.util.spec_from_file_location('policy_test.'+name,root/'vllm/v1/core/sched'/f'{name}.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
P=m.FixedHistoryPolicy
r=NS(request_id='a',verification_draft_limit=None,is_prefill_chunk=False)
p=P(logging.getLogger('test'));p.admit(r);e=p.active['a'];assert p.retained_limits([r])=={'a':5}
out=NS(scheduled_spec_decode_tokens={'a':[1]*5});p.scheduled(out,[r],1);p.feedback_time=out._dsv41_policy_stamps['a'][1]+.03;p.feedback(out,r,5,2,3,True)
assert e.plugin.latest['k']==5 and not e.plugin.latest['timing_valid']
# First choice has no comparable recurrence, no selection.
assert p.choose([r],1)=={'a':5}
out2=NS(scheduled_spec_decode_tokens={'a':[1]*5});p.scheduled(out2,[r],1);p.feedback_time=e.previous[0]+.03;p.feedback(out2,r,5,2,3,True)
assert e.plugin.latest['timing_valid'] and abs(e.plugin.latest['seconds']-.03)<1e-6
p.choose([r],1);before=e.plugin.n.copy();p.choose([r],1);assert before==e.plugin.n
# Stale request-id output must never train newly reused ID.
p.forget('a');p.admit(r);new=p.active['a'];p.feedback(out2,r,5,2,3,True);assert new.plugin.latest is None
r.verification_draft_limit=2;p.forget('a');p.admit(r);assert not p.active
r.verification_draft_limit=None;p.admit(r);entry=p.active['a'];entry.plugin.choose=lambda c:6;entry.fresh=True
assert p.choose([r],1)=={'a':5} and entry.failed
print('PASS fixed adapter actual-width/timing attribution, generation reuse, manual priority, no-fresh repeat, full5 failure')
# Optional debug class must not exist/import in production.
f=root/'vllm/v1/core/sched/dsv41_hot_policy_debug.py'
if f.exists():
 spec=importlib.util.spec_from_file_location('policy_test.dsv41_hot_policy_debug',f);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 import os
 with tempfile.TemporaryDirectory() as td:
  os.environ['VLLM_DSV41_POLICY_DIR']=td;d=Path(td);(d/'versions').mkdir()
  def publish(k):
   src=f'API_VERSION=1\nclass Policy:\n def __init__(self,c):pass\n def observe(self,e):pass\n def choose(self,c):return {k}\ndef self_test():pass\n'.encode();h=hashlib.sha256(src).hexdigest();(d/'versions'/f'{h}.py').write_bytes(src);(d/'control.json').write_text(json.dumps(dict(enabled=True,token='x'*32,sha256=h)));return h
  r.sampling_params=NS(extra_args={'spec_policy_token':'x'*32});p=m.HotHistoryPolicy(logging.getLogger('test'));h=publish(2);p.admit(r);e=p.active['a'];assert e.version==h
  publish(3);b=NS(request_id='b',sampling_params=r.sampling_params,verification_draft_limit=None);p.admit(b);assert e.version!=p.active['b'].version
  e.fresh=True;assert p.choose([r],1)=={'a':2}
  h=publish(6);p.forget('a');p.admit(r);p.active['a'].fresh=True;p.choose([r],1);assert h in p.disabled and not hasattr(p.active['a'],'version')
  print('PASS debug version pinning, trusted admission and illegal-plugin fixed fallback')
else:print('PASS production has no hot loader module')
