#!/usr/bin/env python3
"""GPU sm80: large dispatch vs the same arithmetic at <=2312 query rows.
KDA split at a multiple of 64; carry recurrent state, preserve global order.
No MoE routing is split by this patch (Marlin never had a 2312 ceiling).
"""
import gc
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from test_pp_kda_prefill_gpu import make_inputs, call
from vllm.models.glm5next.nvidia.ops import kda_prefill_pp as kda
from vllm.models.glm5next.nvidia.ops import sparse_prefill_mla_pp as mla

assert torch.cuda.get_device_capability() == (8, 0)
for T in (2312, 5120, 8192, 10240):
    torch.manual_seed(42)
    q=torch.randn(T,64,512,device='cuda',dtype=torch.bfloat16)*0.2
    kv=torch.randn(8192,1,512,device='cuda',dtype=torch.bfloat16)*0.2
    idx=torch.randint(0,8192,(T,1,2176),device='cuda',dtype=torch.int32)
    idx[:, :, 2048:]=-1
    assert not mla.closed(q,kv,idx,512,0,None)
    out=mla.sparse_mla_prefill(q,kv,idx,512**-0.5)[0]
    ref=torch.cat([mla.sparse_mla_prefill(q[s:s+2304],kv,idx[s:s+2304],512**-0.5)[0] for s in range(0,T,2304)])
    diff=(out.float()-ref.float()).abs().max().item()
    print(json.dumps(dict(kernel='sparse_mla',rows=T,max_abs=diff,bitwise=torch.equal(out,ref))),flush=True)
    assert torch.equal(out,ref)
    del q,kv,idx,out,ref;gc.collect();torch.cuda.empty_cache()
    args=make_inputs([T],[True],42)
    whole,state=call(kda.chunk_kda_with_fused_gate,*args)
    buf,g,beta,a,dt,h0,cu=args
    pieces=[];h=h0
    for start in range(0,T,2304):
        end=min(start+2304,T)
        o,h=call(kda.chunk_kda_with_fused_gate,buf[start:end],g[:,start:end],beta[:,start:end],a,dt,h,torch.tensor([0,end-start],device='cuda',dtype=torch.int32))
        pieces.append(o)
    ref=torch.cat(pieces,dim=1)
    od=(whole.float()-ref.float()).abs().max().item();sd=(state-h).abs().max().item()
    print(json.dumps(dict(kernel='kda',rows=T,output_max_abs=od,state_max_abs=sd,output_bitwise=torch.equal(whole,ref),state_bitwise=torch.equal(state,h))),flush=True)
    torch.testing.assert_close(whole,ref,atol=0.002,rtol=0.02)
    torch.testing.assert_close(state,h,atol=0.002,rtol=0.02)
    del args,whole,state,buf,g,beta,a,dt,h0,cu,pieces,h,o,ref;gc.collect();torch.cuda.empty_cache()
print('LARGE_PREFILL_GPU_PASS',flush=True)
