"""Split-list correctness at real E/K/N, fp64 sampled GEMMs; graph + streams."""
import sys
sys.argv=['bench.py'];import bench as b
import torch,json,importlib.util,pathlib
import vllm.model_executor.layers.fused_moe as pkg
pkg.__path__.insert(0,'/t/candidate/vllm/model_executor/layers/fused_moe')
from vllm.model_executor.layers.fused_moe import moe_prefill as pp
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.scalar_type import scalar_types
from vllm import _custom_ops as ops
E,K,N,TK=384,5120,2304,6;dev='cuda';bf=torch.bfloat16
def weight(n,k):
 triples=[rand_marlin_weight_mxfp4_like(torch.empty(n,k,device=dev,dtype=bf),32) for _ in range(4)]
 refs,qs,ss=zip(*triples)
 return torch.stack(refs),torch.stack(qs).repeat(E//4,1,1),torch.stack(ss).repeat(E//4,1,1)
r1,w1,s1=weight(2*N,K);r2,w2,s2=weight(K,N)
ws=marlin_make_workspace_new(torch.device(dev),4)
for M in [512,1024,4096]:
 b.M=M
 for skew in [False,True]:
  x=torch.randn(M,K,device=dev,dtype=bf);ids=torch.rand(M,E if not skew else 24,device=dev).topk(TK,dim=1).indices.int();tw=torch.rand(M,TK,device=dev);tw/=tw.sum(-1,keepdim=True)
  c1=torch.empty(M*TK,2*N,device=dev,dtype=bf);c2=torch.randn(M*TK,N,device=dev,dtype=bf);c3=torch.empty(M*TK,K,device=dev,dtype=bf)
  assert pp.eligible(x,w1,w2,s1,s2,ids,scalar_types.float4_e2m1f,None,None,None,None,None,None,None,None,None,None,False)
  for bs in [8,16,32,48,64]:
   if M*TK/E/bs<.9:break
  ls=moe_align_block_size(ids,bs,E,None,ignore_invalid_experts=True);spl=pp.lists(ids)
  flat=torch.cat([s[:int(nt)] for _,s,e,nt in spl]);valid=flat[flat<ids.numel()]
  assert torch.equal(valid.sort().values,torch.arange(ids.numel(),device=dev,dtype=torch.int32))
  for block,sorted_ids,experts,nt in spl:
   count=int(nt);sl=sorted_ids[:count];ev=experts[:count//block].repeat_interleave(block);mask=sl<ids.numel()
   assert torch.equal(ids.flatten()[sl[mask]],ev[mask])
  for which in [1,2]:
   inp,out,w,s,n,k=(x,c1,w1,s1,2*N,K) if which==1 else (c2,c3,w2,s2,K,N)
   args=(inp,out,w,None,s,None,None,None,ws,*ls,tw)
   kw=dict(moe_block_size=bs,top_k=TK if which==1 else 1,mul_topk_weights=which==2,b_q_type=scalar_types.float4_e2m1f,size_m=M if which==1 else M*TK,size_n=n,size_k=k,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
   base=lambda:ops.moe_wna16_marlin_gemm(*args,**kw)
   new=lambda:pp.split_gemm(spl,*args,**kw)
   base();old=out[:64].clone();new();got=out[:64].clone()
   ri=torch.arange(64,device=dev)//TK if which==1 else torch.arange(64,device=dev)
   ref=torch.empty(64,128,device=dev,dtype=torch.float64)
   for ex in range(4):
    mask=(ids.flatten()[:64]%4)==ex
    ref[mask]=inp[ri[mask]].double()@(r1 if which==1 else r2)[ex,:,:128].double()
   if which==2:ref*=tw.flatten()[:64,None].double()
   e0=(old[:,:128].double()-ref).abs();e1=(got[:,:128].double()-ref).abs()
   row=dict(M=M,skew=skew,gemm=which,base_mean=e0.mean().item(),new_mean=e1.mean().item(),base_max=e0.max().item(),new_max=e1.max().item())
   print('ACCURACY',json.dumps(row),flush=True)
   assert e1.mean()<=1.01*e0.mean()+1e-8 and e1.max()<=1.01*e0.max()+1e-8
   new();assert torch.equal(got,out[:64]);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):new()
   g.replay();torch.cuda.synchronize();assert torch.equal(got,out[:64])
   if not skew:
    b.bench(f'w{which}_base',base,n=2);b.bench(f'w{which}_split',new,n=2)
 # Different streams get distinct persistent alignment buffers.
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):alt=pp.lists(ids)
 torch.cuda.current_stream().wait_stream(st)
 assert alt[0][1].data_ptr()!=spl[0][1].data_ptr()
print('PASS',flush=True)
open('result_moe_correctness.json','w').write(json.dumps(b.results,indent=2))
