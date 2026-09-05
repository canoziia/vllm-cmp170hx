import ast,types
p='/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py'
t=ast.parse(open(p).read());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_mamba_block_aligned_split');f.decorator_list=[]
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])),p,'exec'),ns)
self=types.SimpleNamespace(mamba_partial_cache_hit=True,use_v2_model_runner=True,scheduler_config=types.SimpleNamespace(async_scheduling=True),parallel_config=types.SimpleNamespace(pipeline_parallel_size=1),hash_block_size=32)
for start in range(128,224):
 for count in range(1,5):
  req=types.SimpleNamespace(num_computed_tokens=start,num_prompt_tokens=100,num_tokens=start+1,num_in_flight_tokens=0)
  got=ns['_mamba_block_aligned_split'](self,req,count)
  assert got==min(count,32-start%32)
  req.num_in_flight_tokens=4
  got=ns['_mamba_block_aligned_split'](self,req,count)
  assert got==(0 if start%32==0 or start%32+count>=32 else count)
print('DECODE_CHECKPOINT_SCHEDULER cases=768 PASS')
