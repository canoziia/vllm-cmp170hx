#!/usr/bin/env python3
"""Exercise actual load_dspark_model config derivation without weights/CUDA."""
import ast,copy,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
root=Path(sys.argv[1]);tree=ast.parse((root/'vllm/v1/worker/gpu/spec_decode/dspark/utils.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_dspark_model')
# Omit weight loader imports, stop immediately after actual config derivation.
body=[]
for stmt in fn.body:
 if isinstance(stmt,(ast.Import,ast.ImportFrom)):continue
 if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='quant_config' for t in stmt.targets):break
 body.append(stmt)
body.append(ast.Return(ast.Name('draft_vllm_config',ast.Load())))
fn.body=body;fn.returns=None
for arg in fn.args.args:arg.annotation=None
class Spec(NS):
 def uses_history_verification(self):return self.enable_adaptive_verification
for enabled in (False,True):
 spec=Spec(enable_adaptive_verification=enabled,draft_model_config=NS(hf_config=NS()),attention_backend=None,draft_parallel_config=NS(tensor_parallel_size=1),kv_cache_dtype=None)
 cfg=NS(speculative_config=spec,parallel_config=NS(pipeline_parallel_size=6,tensor_parallel_size=1),attention_config=NS(backend=None,use_non_causal=False),cache_config=NS())
 def replace(obj,**updates):
  result=copy.copy(obj)
  for k,v in updates.items():setattr(result,k,v)
  if hasattr(result,'parallel_config'):
   assert result.parallel_config.pipeline_parallel_size==1
   # This is what target-only PP6 validation checks during dataclass replace.
   assert not result.speculative_config.uses_history_verification()
  return result
 ns={'copy':copy,'replace':replace,'dflash_has_any_non_causal':lambda _:False,'_resolve_dspark_attention_backend':lambda *args:None}
 exec(compile(ast.fix_missing_locations(ast.Module([fn],type_ignores=[])),'draft-config','exec'),ns)
 draft=ns['load_dspark_model'](NS(),cfg)
 assert spec.enable_adaptive_verification==enabled
 assert not draft.speculative_config.enable_adaptive_verification
 assert draft.speculative_config.draft_model_config is spec.draft_model_config
print('PASS actual drafter config derivation: PP1 OFF, target PP6 unchanged, resolved draft metadata preserved')
