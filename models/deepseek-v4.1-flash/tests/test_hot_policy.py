"""CPU integration of actual hot facade. No GPU/service access."""
import hashlib,importlib.util,json,logging,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
root=Path(sys.argv[1]);pkg=ModuleType('hot_test');pkg.__path__=[];sys.modules['hot_test']=pkg
for name in ('dsv41_history_policy','dsv41_hot_policy'):
 spec=importlib.util.spec_from_file_location('hot_test.'+name,root/'vllm/v1/core/sched'/f'{name}.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
Hot=m.HotHistoryPolicy
with tempfile.TemporaryDirectory() as tmp:
 os.environ['VLLM_DSV41_POLICY_DIR']=tmp;d=Path(tmp);(d/'versions').mkdir()
 def publish(k):
  src=f'''API_VERSION=1
class Policy:
 def __init__(self,config):self.events=[]
 def observe(self,e):self.events.append(e)
 def choose(self,c):return {k}
def self_test():pass
'''.encode();h=hashlib.sha256(src).hexdigest();(d/'versions'/f'{h}.py').write_bytes(src);(d/'control.json').write_text(json.dumps(dict(enabled=True,token='a'*32,sha256=h)));return h
 def req(name,token='a'*32):return NS(request_id=name,sampling_params=NS(extra_args={'spec_policy_token':token}),verification_draft_limit=None,is_prefill_chunk=False)
 a=req('a');b=req('b');off=req('off','wrong');p=Hot(logging.getLogger('test'));v1=publish(2);p.admit(a);p.admit(off);assert 'off' not in p.active
 v2=publish(3);p.admit(b);assert p.active['a'].version==v1 and p.active['b'].version==v2
 # Repeated choose before feedback must not call plugin/change k.
 assert p.choose([a,b],2)=={'a':5,'b':5}
 out=NS(scheduled_spec_decode_tokens={'a':[1]*5,'b':[1]*5});p.scheduled(out,[a,b],2);p.begin_feedback()
 for r in (a,b):p.feedback(out,r,5,2,3,True)
 assert p.choose([a,b],2)=={'a':2,'b':3}
 assert p.active['a'].plugin.events[0]['k']==5
 assert not p.active['a'].plugin.events[0]['timing_valid']
 # Old output cannot update a replacement request with the same id.
 p.forget('a');p.admit(a);p.feedback(out,a,5,2,3,True);assert not p.active['a'].plugin.events
 # New timing epoch invalidates measurement; invalid feedback never trains.
 out2=NS(scheduled_spec_decode_tokens={'b':[1]*3});p.scheduled(out2,[b],1);p.begin_feedback();p.feedback(out2,b,3,1,2,True)
 assert not p.active['b'].plugin.events[-1]['timing_valid']
 before=len(p.active['b'].plugin.events);p.feedback(out2,b,3,0,1,False);assert len(p.active['b'].plugin.events)==before
 # Faulty return disables all requests of that revision, baseline remains.
 bad=publish(6);r=req('bad');p.admit(r);p.active['bad'].fresh=True;result=p.choose([r],1);assert bad in p.disabled and 1<=result['bad']<=5
 # Manual wins; corrupt publish can't destroy baseline or existing requests.
 manual=req('manual');manual.verification_draft_limit=2;p.admit(manual);assert 'manual' not in p.active
 (d/'control.json').write_text('{');p.admit(req('broken'));assert 'broken' not in p.active
 for name in list(p.active):p.forget(name)
 assert not p.active
print('PASS actual hot adapter admission, auth, version pinning, next-step, timing epochs, manual, stale identity, fallback, cleanup')
