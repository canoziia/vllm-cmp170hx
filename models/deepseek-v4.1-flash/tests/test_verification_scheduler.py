#!/usr/bin/env python3
"""Run in candidate image with pinned-source tests on PYTHONPATH; CPU schedule only."""
import json,os,tempfile
from pathlib import Path
from collections import deque
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
from tests.v1.core.utils import create_scheduler,create_requests
from vllm.v1.outputs import ModelRunnerOutput
import vllm.envs as envs
os.environ['VLLM_DSV41_VERIFICATION']='1'
os.environ['VLLM_DSV41_BALANCED_COHORTS']='1'
# A local HF config avoids downloading OPT; no weights are loaded.
p=Path(tempfile.mkdtemp());(p/'config.json').write_text(json.dumps({'model_type':'opt','architectures':['OPTForCausalLM'],'hidden_size':64,'ffn_dim':256,'num_hidden_layers':2,'num_attention_heads':4,'vocab_size':1024,'max_position_embeddings':4096,'word_embed_proj_dim':64}))
def output(s):
 ids=list(s.num_scheduled_tokens)
 return ModelRunnerOutput(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[])
for n in (1,2,4,8,16,32):
 for k in range(6):
  s=create_scheduler(model=str(p),async_scheduling=True,max_num_seqs=32,max_num_batched_tokens=4096,num_speculative_tokens=5,speculative_method='ngram_gpu',use_v2_model_runner=False)
  s.pp_size=6;s.use_pp=True;s.use_v2_model_runner=True
  # Method gate is DSpark-only. Construct its scheduler state without loading
  # the giant model or invoking the actual drafter; target-step semantics same.
  s.vllm_config.model_config.model_arch_config.architectures=['DeepseekV41ForCausalLM']
  s.vllm_config.speculative_config.method='dspark'
  reqs=create_requests(num_requests=n,num_tokens=8,max_tokens=12,ignore_eos=True)
  for r in reqs:
   r.sampling_params.extra_args={'spec_k':k};s.add_request(r)
  q=deque();decode=0
  for t in range(250):
   so=s.schedule()
   if t == 0:
    capacity=(n+5)//6
    phases=[sum(getattr(r,'decode_cohort_phase',None)==phase for r in reqs) for phase in range(6)]
    assert sum(phases)==n and max(phases)<=capacity,(n,k,phases)
    assert sum(x>0 for x in phases)==(n+capacity-1)//capacity,(n,k,phases)
   for rid,count in so.num_scheduled_tokens.items():
    r=s.requests[rid]
    if rid in so.scheduled_spec_decode_tokens:assert len(so.scheduled_spec_decode_tokens[rid])<=k
    if not r.is_prefill_chunk and r.num_computed_tokens>r.num_prompt_tokens:
     assert count==1+k,(n,k,count)
     decode+=1
   q.append(so)
   if len(q)>=6:
    old=q.popleft();s.update_from_output(old,output(old))
   if not s.has_requests():break
  else:raise AssertionError(('no progress',n,k))
  assert decode>0
 print('PASS cohort',n,'k0..5: actual scheduler, placeholders, prefix rows, completion')
print('PASS ALL')
