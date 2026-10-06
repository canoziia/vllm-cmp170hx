"""Run inside image after applying integration.diff. CPU FP64 oracle."""
import argparse
import os
import json
import torch
from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang as incumbent
from vllm.model_executor.kernels.mhc import dsv4_mhc_v2 as v2
from vllm.model_executor.kernels.mhc import dsv4_mhc_v2_gate as gate


def case(m, seed, stress=False):
    torch.manual_seed(seed)
    def rand(*s, bf=False):
        return torch.randn(*s, device='cuda', dtype=torch.float32).to(torch.bfloat16 if bf else torch.float32)
    return (rand(m,5120,bf=True), rand(m,4,5120,bf=True),
            torch.sigmoid(rand(m,4,1)), torch.softmax(rand(m,4,4),-1),
            rand(24,20480)*(0.1 if stress else 0.007),
            torch.tensor([1.,1.,1.],device='cuda'), rand(24),
            1e-5,1e-6,1e-6,2.,20,1,1,rand(5120,bf=True),1e-6)


def oracle(a):
    x,r,p,c,f,s,b=[t.cpu().double() for t in a[:7]]
    w=a[14].cpu().double(); m=x.shape[0]
    nr=p*x[:,None,:]+torch.einsum('mkj,mkh->mjh',c,r)
    # Incumbent's M-dependent projection rounding boundary.
    pr=nr if m<=16 else nr.to(torch.bfloat16).double()
    mix=(pr.flatten(1)@f.T)*torch.rsqrt(pr.square().mean((1,2))+a[7])[:,None]
    pre=torch.sigmoid(mix[:,:4]*s[0]+b[:4])+a[8]
    post=torch.sigmoid(mix[:,4:8]*s[1]+b[4:8])*a[10]
    cm=(mix[:,8:]*s[2]+b[8:]).reshape(m,4,4).softmax(-1)+a[9]
    cm=cm/(cm.sum(-2,keepdim=True)+a[9])
    for _ in range(a[11]-1):
        cm=cm/(cm.sum(-1,keepdim=True)+a[9]);cm=cm/(cm.sum(-2,keepdim=True)+a[9])
    o=(pre[:,:,None]*nr.to(torch.bfloat16).double()).sum(1)
    li=o.to(torch.bfloat16).double()*torch.rsqrt(o.square().mean(1)+a[15])[:,None]*w
    return nr,post[:,:,None],cm,li


def graph(fn,a,repeat=100):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn(*a)
    g.replay();torch.cuda.synchronize()
    return g,out


def bench(g,n):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n): g.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/n


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bench',action='store_true');ap.add_argument('--all-m',action='store_true');ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--replays',type=int,default=500);args=ap.parse_args()
    os.environ['VLLM_DSV4_MHC_V2']='1';gate.warmup(torch.device('cuda'))
    os.environ['VLLM_DSV4_MHC_V2']='0'
    failed=False
    for m in (range(1,65) if args.all_m else [1,6,12,48]):
        for seed in range(args.seeds):
            for stress in [False,True]:
                a=case(m,seed,stress);old=incumbent(*a);new=v2.mhc_fused_post_pre(*a);ref=oracle(a)
                assert torch.equal(old[0].view(torch.int16),new[0].view(torch.int16)), 'residual not bitwise equal'
                assert all(torch.equal(u,v) for u,v in zip(new,v2.mhc_fused_post_pre(*a))), 'nondeterministic'
                for i,(o,n,r) in enumerate(zip(old,new,ref)):
                    oe=(o.cpu().double()-r).abs();ne=(n.cpu().double()-r).abs()
                    report=dict(M=m,seed=seed,stress=stress,output=i,diff_max=(o.float()-n.float()).abs().max().item(),old_max=oe.max().item(),new_max=ne.max().item(),old_mean=oe.mean().item(),new_mean=ne.mean().item())
                    print(json.dumps(report))
                    # Different fp32 summation order: allow up to 4x the
                    # incumbent's fp64 error (measured: <= ~3x at the 1e-7
                    # level for M <= 16; ~100x smaller than TileLang at M=48).
                    if i and (ne.max()>4*oe.max()+1e-12 or ne.mean()>4*oe.mean()+1e-12): failed=True
                if seed==0 and not stress:
                    go,oo=graph(incumbent,a);gn,nn=graph(v2.mhc_fused_post_pre,a)
                    assert all(torch.equal(u,v) for u,v in zip(nn,new)), 'graph mismatch'
                    os.environ['VLLM_DSV4_MHC_V2']='1'
                    gg,ng=graph(incumbent,a)
                    assert all(torch.equal(u,v) for u,v in zip(ng,new)), 'gate mismatch'
                    os.environ['VLLM_DSV4_MHC_V2']='0'
                    if args.bench: print(json.dumps(dict(M=m,tilelang_us=bench(go,args.replays),v2_us=bench(gn,args.replays),gate_us=bench(gg,args.replays))))
    if failed: raise SystemExit('FAIL: FP64 max or mean error above 4x TileLang; DO NOT DEPLOY')
    print('PASS')

if __name__=='__main__': main()
