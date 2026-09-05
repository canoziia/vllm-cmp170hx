import ast,types,os,time,torch
from torch import nn
import vllm.model_executor.parameter as parameter_module
parameter_module.get_tensor_model_parallel_rank=lambda:0
parameter_module.get_tensor_model_parallel_world_size=lambda:1
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter

def extract(path,name):
 tree=ast.parse(open(path).read())
 # Select concrete method, ignoring overloads whose body is only Ellipsis.
 fs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name and not (len(n.body)==1 and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and n.body[0].value.value is Ellipsis)]
 f=fs[-1]; f.decorator_list=[]
 ns={'torch':torch,'nn':nn,'logger':__import__('logging').getLogger('test'),'__name__':'fixture', 'FusedMoeWeightScaleSupported':types.SimpleNamespace(BLOCK=types.SimpleNamespace(value='block'))}
 exec('from __future__ import annotations\n',ns)
 code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
 exec(compile(ast.fix_missing_locations(code),path,'exec'),ns);return ns[name]
base=os.environ['ORIGINAL_ROUTED_EXPERTS']
optimized=os.environ.get('OPTIMIZED_ROUTED_EXPERTS','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py')
assert open(base).read() != open(optimized).read(), 'Reference must be the unmodified loader'
class ModelOptNvFp4FusedMoE:
 use_global_sf=False
 def uses_weight_scale_2_pattern(self):return True
class Fixture(nn.Module):
 def __init__(self):
  super().__init__();self.layer_name='model.layers.0.mlp.experts';self.quant_method=ModelOptNvFp4FusedMoE();self.quant_config=types.SimpleNamespace(get_name=lambda:'modelopt_fp4');self.moe_config=types.SimpleNamespace(tp_rank=0,tp_size=1,is_act_and_mul=True,moe_parallel_config=types.SimpleNamespace(tp_size=1))
  for name,shape in [('w13_input_scale',(8,2)),('w2_input_scale',(8,)),('w13_weight_scale_2',(8,2)),('w2_weight_scale_2',(8,)),('w13_weight',(8,64,64)),('w2_weight',(8,64,32)),('w13_weight_scale',(8,64,4)),('w2_weight_scale',(8,64,2))]:
   dtype = torch.uint8 if name.endswith('_weight') else torch.float8_e4m3fn if name.endswith('_weight_scale') else torch.float32
   data=torch.full(shape,1,device='cuda',dtype=dtype)
   p=ModelWeightParameter(data=data,input_dim=1,output_dim=2,weight_loader=self.weight_loader) if len(shape)==3 else PerTensorScaleParameter(data=data,weight_loader=self.weight_loader)
   if name.endswith('weight_scale'):p.quant_method='block'
   self.register_parameter(name,p)
 def _map_global_expert_id_to_local_expert_id(self,i):return i
 def get_expert_mapping(self,include_fused=True):
  return [(f'experts.{"w2" if p=="down_proj" else "w13"}_',f'experts.{e}.{p}.',e,s) for e in range(8) for p,s in [('gate_proj','w1'),('up_proj','w3'),('down_proj','w2')]]
for name in ['weight_loader','_load_single_value','_load_per_tensor_weight_scale','_load_model_weight_or_group_weight_scale','_load_w13','_load_w2']:
 setattr(Fixture,name,extract(base,name))
Fixture._to_scalar=staticmethod(extract(base,'_to_scalar'))
Fixture._get_hidden_dim=staticmethod(extract(base,'_get_hidden_dim'))
Fixture._narrow_expert_data_for_padding=staticmethod(extract(base,'_narrow_expert_data_for_padding'))
old=extract(base,'load_weights');new=extract(optimized,'load_weights')
weights=[]
for e in range(8):
 for p in ['gate_proj','up_proj','down_proj']:
  for suffix in ['input_scale','weight_scale_2','weight','weight_scale']:
   shape=(1,)
   if suffix=='weight':shape=(64,32) if p=='down_proj' else (32,64)
   if suffix=='weight_scale':shape=(64,2) if p=='down_proj' else (32,4)
   dtype=torch.uint8 if suffix=='weight' else torch.float8_e4m3fn if suffix=='weight_scale' else torch.float32
   weights.append((f'{e}.{p}.{suffix}',torch.full(shape,1+len(weights)%8,dtype=dtype)))
# Shard boundaries: each call must retain previously loaded entries.
a=Fixture();b=Fixture();original_parameters=dict(b.named_parameters());os.environ['VLLM_NVFP4_STAGE_EXPERTS']='1'
for start in range(0,len(weights),12):
 part=weights[start:start+12];la=list(old(a,part));lb=list(new(b,part));torch.cuda.synchronize();assert la==lb
 for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
  assert pb is original_parameters[nb], (start, nb, 'Parameter replaced')
  assert na==nb and torch.equal(pa.view(torch.uint8),pb.view(torch.uint8)),(start,na,pa,pb)
print('EXPERT_STAGING_BYTE_EQUAL entries=96 parameter_identity_preserved=True',flush=True)
# Repeat full loading for a same-process copy-cost comparison; not model startup timing.
for label,fn,obj in [('original',old,a),('staged',new,b)]:
 t=time.perf_counter()
 for _ in range(20): list(fn(obj,weights))
 torch.cuda.synchronize();print(label,'seconds',time.perf_counter()-t,flush=True)
