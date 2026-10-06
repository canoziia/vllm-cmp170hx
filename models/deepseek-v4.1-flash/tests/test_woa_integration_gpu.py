import torch, sys, importlib.util
spec=importlib.util.spec_from_file_location('dsv41_thin_woa','dsv41_thin_woa.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert torch.cuda.device_count()==1
for seed in (17,31,93):
 torch.manual_seed(seed)
 w=torch.randn(8,1024,4096,device='cuda',dtype=torch.bfloat16)
 for m in (1,6,48):
  for kind in ('normal','scaled','cancellation'):
   w=torch.randn(8,1024,4096,device='cuda',dtype=torch.bfloat16)
   x=torch.randn(m,8,4096,device='cuda',dtype=torch.bfloat16)
   if kind=='scaled':x*=0.03125
   if kind=='cancellation':
    x[:,:,1::2]=x[:,:,::2];w[:,:,1::2]=-w[:,:,::2]
   y=mod.thin_woa(x,w);old=torch.einsum('mgk,gnk->mgn',x,w)
   ref=torch.einsum('mgk,gnk->mgn',x.double(),w.double())
   e=(y.double()-ref).abs();eo=(old.double()-ref).abs()
   ok=e.mean()<=eo.mean() and e.max()<=eo.max()
   print('STRICT',seed,m,kind,'pass',bool(ok),'mean',e.mean().item(),eo.mean().item(),'max',e.max().item(),eo.max().item(),flush=True)
   assert ok
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):yg=mod.thin_woa(x,w)
   for _ in range(10):g.replay()
   torch.cuda.synchronize();assert torch.equal(yg,y)
# Fallback outside tuned shape: bitwise incumbent.
x=torch.randn(49,8,4096,device='cuda',dtype=torch.bfloat16)
assert torch.equal(mod.thin_woa(x,w),torch.einsum('mgk,gnk->mgn',x,w))
print('DONE',flush=True)
