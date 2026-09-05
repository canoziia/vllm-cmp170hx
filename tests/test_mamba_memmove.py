import ast,torch
from vllm.triton_utils import triton,tl
p='/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py';tree=ast.parse(open(p).read());wanted={'batch_memcpy_kernel','batch_memcpy'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
ns={'triton':triton,'tl':tl,'torch':torch,'__name__':'mamba_test'}
# Triton needs a real source file for inspect.getsourcelines.
import importlib.util,sys
spec=importlib.util.spec_from_file_location('patched_mamba',p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
for dtype in [torch.uint8,torch.bfloat16,torch.float32]:
 for length in [1025,4097,8193]:
  for shift in [1,7,127]:
   x=torch.arange(length+shift,device='cuda',dtype=torch.int32).to(dtype);expected=x.clone();expected[:length]=x[shift:shift+length].clone()
   src=torch.tensor([x.data_ptr()+shift*x.element_size()],device='cuda',dtype=torch.int64);dst=torch.tensor([x.data_ptr()],device='cuda',dtype=torch.int64);sizes=torch.tensor([length*x.element_size()],device='cuda',dtype=torch.int64)
   mod.batch_memcpy(src,dst,sizes);torch.cuda.synchronize();assert torch.equal(x,expected),(dtype,length,shift)
print('MAMBA_OVERLAP_BYTE_EQUAL cases=27')
