#!/usr/bin/env python3
"""Run in image: python3 test_decode_metadata_gpu.py --bench."""
import argparse, importlib.util, random
import torch
p=argparse.ArgumentParser(); p.add_argument('--module'); p.add_argument('--bench',action='store_true'); a=p.parse_args()
if a.module:
    spec=importlib.util.spec_from_file_location('ks_decode',a.module); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
else:
    from vllm.v1.attention.backends.mla import decode_metadata as mod
fn=mod.prepare_flattened_decode
random.seed(42)
for n in (1, 2, 4, 8, 16, 32):
    for _ in range(30):
        lens = [random.randrange(7) for _ in range(n)]
        if sum(lens) == 0:
            lens[0] = 1
        starts = torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device='cuda',dtype=torch.int32)
        rows = sum(lens) + 3
        out = torch.empty(rows,device='cuda',dtype=torch.int32)
        result = mod.prepare_token_owners(starts, out, rows)
        ref = torch.zeros_like(out)
        ref[:sum(lens)] = torch.repeat_interleave(torch.arange(n,device='cuda',dtype=torch.int32),torch.tensor(lens,device='cuda'))
        assert torch.equal(result,ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mod.prepare_token_owners(starts, out, rows)
        lens.reverse()
        starts.copy_(torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device='cuda',dtype=torch.int32))
        graph.replay()
        ref[:sum(lens)] = torch.repeat_interleave(torch.arange(n,device='cuda',dtype=torch.int32),torch.tensor(lens,device='cuda'))
        assert torch.equal(result,ref)
print('PASS: shared token owners random zero-length requests/padding/changed graph replay')

def case(lens, rows, cols, capture=False):
    n=len(lens); dev='cuda'
    starts=torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device=dev,dtype=torch.int32)
    seq=torch.tensor([200+i+x if x else 0 for i,x in enumerate(lens)],device=dev,dtype=torch.int32)
    # Non-contiguous source strides, as in real block tables.
    bt=torch.randint(0,10000,(n,cols+13),device=dev,dtype=torch.int32)[:,:cols]
    out_seq=torch.full((rows+17,),-19,device=dev,dtype=torch.int32)
    out_bt=torch.full((rows,cols),-19,device=dev,dtype=torch.int32)
    out_lens=torch.zeros(max(n,rows),device=dev,dtype=torch.int32)
    def run(): fn(seq,bt,starts,out_seq,out_bt,out_lens,rows)
    run()
    if capture:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): run()
        # Change ownership on replay, retaining rows but not the CPU split.
        lens=list(reversed(lens)); starts.copy_(torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device=dev,dtype=torch.int32))
        seq.copy_(torch.tensor([300+i+x if x else 0 for i,x in enumerate(lens)],device=dev,dtype=torch.int32))
        g.replay()
    owners=torch.repeat_interleave(torch.arange(n,device=dev),torch.tensor(lens,device=dev))
    actual=sum(lens)
    ref_seq=torch.zeros_like(out_seq)
    ref_seq[:actual]=seq[owners]-(starts[owners+1]-starts[owners])+torch.arange(actual,device=dev)-starts[owners]+1
    assert torch.equal(out_seq,ref_seq), (lens,rows,cols)
    assert torch.equal(out_bt[:actual],bt[owners])
    assert torch.count_nonzero(out_bt[actual:])==0
    assert torch.equal(out_lens[:rows],torch.ones_like(out_lens[:rows]))
    if a.bench and cols==8192:
        for _ in range(10): run()
        beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        beg.record()
        for _ in range(100): run()
        end.record(); end.synchronize()
        print('bench',n,rows,cols,round(beg.elapsed_time(end)*10,2),'us')

for n in (1,2,4,8,16,32):
    for width in range(1,7):
        lens=[width]*n
        for cols in (1,17,8192):
            case(lens,sum(lens),cols)
        mixed=[random.randint(0,6) for _ in range(n)]
        if sum(mixed)==0: mixed[0]=1
        case(mixed,sum(mixed)+3,129,True)
        case(lens+[0,0],sum(lens)+3,257,True)
print('PASS: exact integer metadata, strided block tables, zero/padded requests, CUDA graph replay with changed ragged ownership')
