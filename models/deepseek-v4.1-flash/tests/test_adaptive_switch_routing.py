#!/usr/bin/env python3
"""CPU-only public-switch and internal layout routing regression."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace as NS
root=Path(sys.argv[1])/'vllm'
t=ast.parse((root/'config/speculative.py').read_text())
c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SpeculativeConfig')
names=('use_dspark','uses_history_verification','uses_gpu_adaptive_verification')
methods=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in names]
cls=ast.ClassDef(name='Spec',bases=[],keywords=[],body=methods,decorator_list=[])
ns={};exec(compile(ast.fix_missing_locations(ast.Module([cls],type_ignores=[])),'switch','exec'),ns)
for enabled in (False,True):
 for arch in ('DeepseekV41ForCausalLM','OtherDSparkModel'):
  s=ns['Spec']();s.enable_adaptive_verification=enabled;s.method='dspark';s.target_model_config=NS(architectures=[arch])
  assert s.uses_history_verification()==(enabled and arch=='DeepseekV41ForCausalLM')
  assert s.uses_gpu_adaptive_verification()==(enabled and arch!='DeepseekV41ForCausalLM')
  assert not(s.uses_history_verification() and s.uses_gpu_adaptive_verification())
# All device-only consumers must route through the internal capability,
# never mistake CPU-known variable widths for a device-compacted layout.
for name in ('v1/worker/gpu/model_runner.py','v1/worker/gpu/spec_decode/dspark/speculator.py','v1/worker/gpu/spec_decode/rejection_sampler.py','v1/attention/selector.py','v1/attention/backends/mla/indexer.py','v1/attention/backends/flashinfer.py','models/deepseek_v4_1/ampere/ampere_sparse.py'):
 text=(root/name).read_text();ast.parse(text)
 assert '.enable_adaptive_verification' not in text or all(
  not isinstance(n,ast.Attribute) or not isinstance(n.value,ast.Name)
  or n.value.id not in ('spec','spec_config','speculative_config')
  or n.attr!='enable_adaptive_verification' for n in ast.walk(ast.parse(text))),name
 assert 'uses_gpu_adaptive_verification()' in text,name
scheduler=(root/'v1/core/sched/scheduler.py').read_text()
assert 'spec.uses_history_verification()' in scheduler
assert 'VLLM_DSV41_HISTORY_POLICY' not in scheduler
print('PASS public false=fixed, DeepSeek true=CPU history; other DSpark GPU path preserved; device consumers separated')
