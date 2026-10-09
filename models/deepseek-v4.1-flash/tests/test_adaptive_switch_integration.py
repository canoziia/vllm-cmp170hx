#!/usr/bin/env python3
"""Actual scheduler/config CPU test in image with updated Python files.

Pinned-source tests helpers must be on PYTHONPATH. No GPU/model load needed.
"""
import json,os,tempfile
from pathlib import Path
from collections import deque
from types import SimpleNamespace as NS
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,SpeculativeConfig,CUDAGraphMode
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.outputs import ModelRunnerOutput
from tests.v1.core.utils import create_scheduler,create_requests
os.environ['VLLM_DSV41_VERIFICATION']='1'
os.environ['VLLM_DSV41_BALANCED_COHORTS']='1'
p=Path(tempfile.mkdtemp());(p/'config.json').write_text(json.dumps({'model_type':'opt','architectures':['OPTForCausalLM'],'hidden_size':64,'ffn_dim':256,'num_hidden_layers':2,'num_attention_heads':4,'vocab_size':1024,'max_position_embeddings':4096,'word_embed_proj_dim':64}))
def make(enabled):
 import vllm.envs as envs
 envs.VLLM_USE_V2_MODEL_RUNNER=False
 os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
 seed=create_scheduler(model=str(p),async_scheduling=True,max_num_seqs=32,max_num_batched_tokens=4096,num_speculative_tokens=5,speculative_method='ngram_gpu',use_v2_model_runner=False)
 cfg=seed.vllm_config;spec=cfg.speculative_config
 cfg.model_config.model_arch_config.architectures=['DeepseekV41ForCausalLM']
 spec.method='dspark';spec.target_model_config=cfg.model_config;spec.enable_adaptive_verification=enabled
 os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
 import vllm.envs as envs
 envs.VLLM_USE_V2_MODEL_RUNNER=True
 s=AsyncScheduler(vllm_config=cfg,kv_cache_config=seed.kv_cache_manager.kv_cache_config,block_size=16,log_stats=True,structured_output_manager=StructuredOutputManager(cfg))
 s.pp_size=6;s.use_pp=True
 assert bool(s.dsv41_history_policy)==enabled
 return s
# Probe config validator's explicit supported-scope checks, with no CUDA init.
vllm.platforms._current_platform.is_device_capability_family=lambda family:family==80
spec=object.__new__(SpeculativeConfig);spec.enable_adaptive_verification=True;spec.method='dspark';spec.target_model_config=NS(architectures=['DeepseekV41ForCausalLM']);spec.num_speculative_tokens=5;spec.draft_sample_method='greedy'
cfg=NS(speculative_config=spec,lora_config=None,compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),parallel_config=NS(pipeline_parallel_size=6,tensor_parallel_size=1,data_parallel_size=1,prefill_context_parallel_size=1,decode_context_parallel_size=1,use_ubatching=False))
VllmConfig._validate_adaptive_verification(cfg)
spec.num_speculative_tokens=4
try:VllmConfig._validate_adaptive_verification(cfg)
except ValueError:pass
else:raise AssertionError('wrong calibrated depth accepted')
spec.enable_adaptive_verification=False;VllmConfig._validate_adaptive_verification(cfg)
# Feedback low acceptance yields real NEXT-step choices; false remains full.
for enabled in (False,True):
 s=make(enabled)
 reqs=create_requests(num_requests=4,num_tokens=8,max_tokens=30,ignore_eos=True)
 for r in reqs:s.add_request(r)
 if enabled:
  for _ in range(200):
   for r in reqs:s.dsv41_history_policy.observe(r.request_id,5,2)
  # Exercise actual scheduler integration with a pre-existing calibrated
  # confidence band; policy statistical conservatism is tested separately.
  for r in reqs:
   h=s.dsv41_history_policy.history[r.request_id]
   h.estimate=lambda:([1.,1.8,2.6,2.61,2.62,2.63],[0.]*6,[.8,1.,.0125,.5,.5])
 q=deque();widths=[]
 for _ in range(400):
  so=s.schedule()
  for ds in so.scheduled_spec_decode_tokens.values():widths.append(len(ds))
  q.append(so)
  if len(q)>=6:
   old=q.popleft();ids=list(old.num_scheduled_tokens)
   s.update_from_output(old,ModelRunnerOutput(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},sampled_token_ids=[[10] for _ in ids],logprobs=None,prompt_logprobs_dict={},pooler_output=[]))
  if not s.has_requests():break
 else:raise AssertionError('scheduler failed to drain')
 assert widths and all(1<=k<=5 for k in widths)
 assert min(widths)<5 if enabled else set(widths)=={5}
 print('PASS actual scheduler adaptive=',enabled,'observed widths=',sorted(set(widths)))
print('PASS public switch actual config/scheduler integration, fixed OFF, NEXT-step adaptive ON')
