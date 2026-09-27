#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual compression/cache kernels: speculative suffix then prefix rollback.

Compare valid cache/ring contents against a clean accepted-prefix-only stream.
The candidate writes rejected suffixes and later overwrites them; no test-side
cache cleanup. Graph replay changes request lengths, positions and raw values.
This tests the compressor's rollback contract, not model/scheduler integration.
"""
import torch
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    fused_save_compress_norm, rope_quant_insert,
)


class Stream:
    def __init__(self, ratio, norm, rope):
        self.ratio,self.norm,self.rope=ratio,norm,rope
        self.raw=torch.zeros(18,512*ratio,device='cuda')
        self.pos=torch.zeros(18,device='cuda',dtype=torch.int64)
        self.owners=torch.zeros(18,device='cuda',dtype=torch.int32)
        self.offsets=torch.zeros(4,device='cuda',dtype=torch.int32)
        self.slots=torch.full((18,),-1,device='cuda',dtype=torch.int64)
        self.cache_slots=torch.full_like(self.slots,-1)
        self.ring=torch.zeros(3,8,1024,device='cuda')
        self.latent=torch.empty(18,512,device='cuda',dtype=torch.bfloat16)
        self.cache=torch.full((3,128,584),165,device='cuda',dtype=torch.uint8)
        self.graph=None

    def install(self, starts, chunks):
        offsets=[0];positions=[];owners=[];slots=[];cache_slots=[]
        for i,(start,chunk) in enumerate(zip(starts,chunks)):
            for p in range(start,start+len(chunk)):
                positions.append(p);owners.append(i)
                slots.append(i*8+p%8 if self.ratio==2 else i*128+p)
                cache_slots.append(i*128+p//self.ratio)
            offsets.append(offsets[-1]+len(chunk))
        count=offsets[-1]
        self.raw.zero_();self.pos.zero_();self.owners.zero_()
        self.slots.fill_(-1);self.cache_slots.fill_(-1)
        if count:
            self.raw[:count].copy_(torch.cat(chunks))
            self.pos[:count].copy_(torch.tensor(positions,device='cuda'))
            self.owners[:count].copy_(torch.tensor(owners,device='cuda',dtype=torch.int32))
            self.slots[:count].copy_(torch.tensor(slots,device='cuda'))
            self.cache_slots[:count].copy_(torch.tensor(cache_slots,device='cuda'))
        self.offsets.copy_(torch.tensor(offsets,device='cuda',dtype=torch.int32))

    def run(self):
        fused_save_compress_norm(self.raw,self.pos,self.ring if self.ratio==2 else None,
            self.slots,self.offsets if self.ratio==2 else None,
            self.owners if self.ratio==2 else None,self.norm,1e-20,self.ratio,self.latent)
        rope_quant_insert(self.latent,self.pos,self.rope,self.cache,self.cache_slots,self.ratio)

    def execute(self):
        self.graph.replay() if self.graph is not None else self.run()


def check(candidate,reference,ends):
    for i,end in enumerate(ends):
        # Main-cache slabs: compare only completed/committed groups.
        rows=end//candidate.ratio
        a=candidate.cache[i].view(-1);b=reference.cache[i].view(-1)
        torch.testing.assert_close(a[:rows*576],b[:rows*576],rtol=0,atol=0)
        torch.testing.assert_close(a[128*576:128*576+rows*8],b[128*576:128*576+rows*8],rtol=0,atol=0)
        # Compression ratio2 only needs the retained immediate predecessor to
        # close the next group. Rejected future rows are intentionally ignored.
        if candidate.ratio==2 and end%2:
            torch.testing.assert_close(candidate.ring[i,(end-1)%8],reference.ring[i,(end-1)%8],rtol=0,atol=0)


@torch.inference_mode()
def main():
    assert torch.cuda.get_device_capability()==(8,0)
    torch.manual_seed(884)
    norm=torch.randn(512,device='cuda',dtype=torch.bfloat16)
    angles=torch.randn(256,32,device='cuda')
    rope=torch.cat((angles.cos(),angles.sin()),dim=-1)
    cases=0
    for ratio in (1,2):
        for graph in (False,True):
            candidate=Stream(ratio,norm,rope);reference=Stream(ratio,norm,rope)
            if graph:
                warm=[torch.randn(6,512*ratio,device='cuda') for _ in range(3)]
                candidate.install([0,0,0],warm)
                for _ in range(3):candidate.run()
                torch.cuda.synchronize()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):candidate.run()
                candidate.graph=g
                candidate.ring.zero_();candidate.cache.fill_(165)
            ends=[0,0,0]
            # First fill history, then repeated accepts/rejects across odd/even
            # compression boundaries and ring wraps. Every later proposal uses
            # new raw rows, so stale rejected data cannot accidentally match.
            schedules=[([6,6,6],[6,6,6]),([6,2,5],[1,1,2]),
                       ([2,6,4],[2,3,1]),([6,1,6],[1,1,1]),
                       ([1,6,2],[1,2,2]),([5,3,6],[2,1,4]),
                       ([6,6,6],[6,6,6]),([0,2,5],[0,1,1])]*3
            for proposed,accepted in schedules:
                chunks=[torch.randn(k,512*ratio,device='cuda') for k in proposed]
                candidate.install(ends,chunks);candidate.execute()
                reference.install(ends,[x[:a] for x,a in zip(chunks,accepted)])
                reference.execute()
                ends=[end+a for end,a in zip(ends,accepted)]
                check(candidate,reference,ends)
                cases+=1
    print(f'COMPRESSOR_REJECTED_SUFFIX {cases} steps eager/graph valid packed-cache bytes PASS')


if __name__=='__main__':main()
