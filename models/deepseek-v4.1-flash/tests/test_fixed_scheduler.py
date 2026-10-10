"""Actual AsyncScheduler/config CPU regression for compiled measured policy.
Run in image with pinned tests.v1.core helpers under /ks/tests.
"""
from pathlib import Path
exec((Path('/ks/test-switch.py').read_text()).split('# Feedback low acceptance')[0])
import time
for enabled in (False,True):
 s=make(enabled)
 reqs=create_requests(num_requests=8,num_tokens=8,max_tokens=80,ignore_eos=True)
 for i,r in enumerate(reqs):
  if i==7:r.sampling_params.extra_args={'spec_k':2}
  s.add_request(r)
 q=deque();widths=[];checks=0
 for tick in range(1600):
  saved=[]
  if enabled and tick%11==0:
   for r in s.running:
    saved.append((r,r.num_stale_output_tokens));r.num_stale_output_tokens=1
  so=s.schedule()
  for r,v in saved:r.num_stale_output_tokens=v
  for rid,ds in so.scheduled_spec_decode_tokens.items():
   widths.append(len(ds))
   r=s.requests[rid]
   if getattr(r,'verification_draft_limit',None) is not None:assert len(ds)==2
   elif enabled:
    entry=s.dsv41_history_policy.active[rid];assert len(ds)==entry.k
   else:assert len(ds)==5
   checks+=1
  q.append(so)
  if len(q)>=6:
   old=q.popleft();ids=list(old.num_scheduled_tokens)
   # Ensure positive elapsed recurrence with no GPU timings or arbitrary policy wait.
   time.sleep(.0001)
   s.update_from_output(old,ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[]))
  if not s.has_requests():break
 else:raise AssertionError('drain failed')
 assert widths and (min(widths)<5 if enabled else True)
 if enabled:assert not s.dsv41_history_policy.active
 print('PASS fixed scheduler public',enabled,'widths',sorted(set(widths)),'checks',checks)
print('PASS actual PP6 scheduler retained limits, valid feedback, cleanup and manual')
