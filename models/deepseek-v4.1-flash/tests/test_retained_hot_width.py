"""CPU only real scheduler reproduction: skipped policy eligibility defaults full5."""
exec(open('/ks/test-switch.py').read())
import hashlib
root=Path(tempfile.mkdtemp());os.environ['VLLM_DSV41_POLICY_DIR']=str(root);(root/'versions').mkdir()
src=b'''API_VERSION=1
class Policy:
 def __init__(self,c):pass
 def observe(self,e):pass
 def choose(self,c):return 2
def self_test():pass
''';h=hashlib.sha256(src).hexdigest();(root/'versions'/f'{h}.py').write_bytes(src);(root/'control.json').write_text(json.dumps(dict(enabled=True,token='x'*32,sha256=h)))
s=make(True);r=create_requests(num_requests=1,num_tokens=8,max_tokens=300,ignore_eos=True)[0];r.sampling_params.extra_args={'spec_policy_token':'x'*32};s.add_request(r)
q=deque();seen=[]
for _ in range(300):
 so=s.schedule()
 if r.request_id in so.scheduled_spec_decode_tokens:
  k=len(so.scheduled_spec_decode_tokens[r.request_id]);seen.append(k)
  if k==2:break
 q.append(so)
 if len(q)>=6:
  old=q.popleft();ids=list(old.num_scheduled_tokens)
  s.update_from_output(old,ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[]))
else:raise AssertionError('never selected2')
assert s.dsv41_history_policy.active[r.request_id].k==2
# Isolate the policy eligibility predicate while leaving all scheduling gates open.
r.num_stale_output_tokens=1
r.next_decode_eligible_step=s.current_step
r.spec_token_ids=[-1]*5
so=s.schedule();actual=len(so.scheduled_spec_decode_tokens.get(r.request_id,[]))
print('REPRO selected',s.dsv41_history_policy.active[r.request_id].k,'actual',actual,'stale',r.num_stale_output_tokens)
assert actual==2,'pinned width lost on ineligible fresh choice'
policy=s.dsv41_history_policy
assert policy.retained_limits([r])=={r.request_id:2}
r.verification_draft_limit=3;assert not policy.retained_limits([r])
r.verification_draft_limit=None
entry=policy.active[r.request_id];entry.failed=True;assert not policy.retained_limits([r]);entry.failed=False
policy.disabled.add(entry.version);assert not policy.retained_limits([r]);policy.disabled.clear()
# Ordinary non-opt-in and absent IDs never get pinned limits.
other=create_requests(num_requests=1,num_tokens=8,max_tokens=30,ignore_eos=True)[0]
other.request_id='ordinary';assert not policy.retained_limits([other])
print('PASS retained width with stale feedback, manual priority, failed revision fallback, ordinary untouched')
