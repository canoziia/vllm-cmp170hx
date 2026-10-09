"""Real CPU AsyncScheduler: preserve pinned widths under staggered PP feedback,
arrivals, manual overrides, short/long requests and synthetic stale gates.
Run in exact candidate image; no GPU or production access.
"""
exec(open('/ks/test-switch.py').read())
import hashlib
root=Path(tempfile.mkdtemp());os.environ['VLLM_DSV41_POLICY_DIR']=str(root);(root/'versions').mkdir()
src=b'''API_VERSION=1
class Policy:
 def __init__(self,c):self.k=int(c['width'])
 def observe(self,e):pass
 def choose(self,c):return self.k
def self_test():pass
''';h=hashlib.sha256(src).hexdigest();(root/'versions'/f'{h}.py').write_bytes(src)
def publish(k):
 (root/'control.json').write_text(json.dumps(dict(enabled=True,token='x'*32,sha256=h,config={'width':k})))
def add(s,index,k,manual=False):
 publish(k)
 r=create_requests(num_requests=1,num_tokens=8+(index%7),max_tokens=35+index%20,ignore_eos=True)[0]
 r.request_id=f'r{index}'
 r.sampling_params.extra_args={'spec_policy_token':'x'*32}
 if manual:r.sampling_params.extra_args['spec_k']=k
 s.add_request(r)
 return r
checks=0
for count in (1,2,4,8,16,32):
 s=make(True);expected={};queue=deque();reqs=[]
 for index in range(count):
  k=1+index%5;r=add(s,index,k,index%7==6);expected[r.request_id]=k;reqs.append(r)
 for tick in range(4000):
  # Stale-output gate injection excludes a resident from fresh choose only;
  # it must not lose its already selected/pinned width when later scheduled.
  # Restore after schedule; no stale output is delivered in this test.
  saved=[]
  if tick%11==0:
   for r in s.running:
    if r.request_id in s.dsv41_history_policy.active:
     saved.append((r,r.num_stale_output_tokens));r.num_stale_output_tokens=1
  selected={rid:e.k for rid,e in s.dsv41_history_policy.active.items()}
  so=s.schedule()
  for r,value in saved:r.num_stale_output_tokens=value
  for rid,ds in so.scheduled_spec_decode_tokens.items():
   entry=s.dsv41_history_policy.active.get(rid)
   # During prefill/first feedback k5 is expected for experiment admission.
   want=entry.k if entry is not None else expected[rid]
   assert len(ds)==want,(count,tick,rid,len(ds),want,selected.get(rid))
   checks+=1
  queue.append(so)
  if len(queue)>=6:
   old=queue.popleft();ids=list(old.num_scheduled_tokens)
   s.update_from_output(old,ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[]))
  if not s.has_requests():break
 else:raise AssertionError(('drain failed',count))
 assert not s.dsv41_history_policy.active
 print('PASS concurrency',count,'ticks',tick)
print('PASS actual scheduler pinned-width stress checks',checks)
