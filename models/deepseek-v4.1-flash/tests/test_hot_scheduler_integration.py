"""Actual AsyncScheduler opt-in and in-flight hot-version replacement on CPU."""
exec(open('/ks/test-switch.py').read())
import hashlib,time
root=Path(tempfile.mkdtemp());os.environ['VLLM_DSV41_POLICY_DIR']=str(root);(root/'versions').mkdir()
def publish(k):
 src=f'''API_VERSION=1
class Policy:
 def __init__(self,c):pass
 def observe(self,e):pass
 def choose(self,c):return {k}
def self_test():pass
'''.encode();h=hashlib.sha256(src).hexdigest();(root/'versions'/f'{h}.py').write_bytes(src);(root/'control.json').write_text(json.dumps(dict(enabled=True,token='x'*32,sha256=h)));return h
v1=publish(2);s=make(True);reqs=create_requests(num_requests=2,num_tokens=8,max_tokens=90,ignore_eos=True)
for r in reqs:r.sampling_params.extra_args={'spec_policy_token':'x'*32};s.add_request(r)
assert {e.version for e in s.dsv41_history_policy.active.values()}=={v1}
v2=publish(3)
q=deque();widths={r.request_id:[] for r in reqs}
for _ in range(900):
 so=s.schedule()
 for rid,ds in so.scheduled_spec_decode_tokens.items():widths[rid].append(len(ds))
 q.append(so)
 if len(q)>=6:
  old=q.popleft();ids=list(old.num_scheduled_tokens)
  s.update_from_output(old,ModelRunnerOutput(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[]))
 if not s.has_requests():break
else:raise AssertionError('failed drain')
assert all(2 in ds and set(ds)<={2,5} for ds in widths.values()),widths
trace=[json.loads(l) for l in (root/'trace.jsonl').read_text().splitlines()]
assert any(e['event']=='feedback' and e['timing_valid'] for e in trace)
assert not any(e['event']=='fallback' for e in trace)
assert not s.dsv41_history_policy.active
r=create_requests(num_requests=1,num_tokens=8,max_tokens=30,ignore_eos=True)[0];r.sampling_params.extra_args={'spec_policy_token':'x'*32};s.add_request(r)
assert s.dsv41_history_policy.active[r.request_id].version==v2
print('PASS actual AsyncScheduler CPU PP6 delayed feedback, opt-in k2, version pinning and next admission k3 revision')
