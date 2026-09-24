#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual installed indexer preparation, no AST mocks or adaptive patch.

Checks device boundaries vs balanced CPU boundaries, persistent outputs and
real paged MQA logits + top-k eager/captured output against a Torch reference.
Metadata preparation occurs outside capture, as in the V2 runner.
"""
from types import SimpleNamespace as NS
import torch
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton


@torch.inference_mode()
def main():
    assert torch.cuda.get_device_capability() == (8, 0)
    n, rows, width = 4, 16, 129
    table = (torch.arange(n*(width+7), dtype=torch.int32, device='cuda') % 24).reshape(n,width+7)[:,:width]
    state = NS(vllm_config=NS(speculative_config=NS(enable_adaptive_verification=True)),
        supports_varlen=False,
        decode_seq_lens_buffer=torch.empty(rows,dtype=torch.int32,device='cuda'),
        expanded_block_table_buffer=torch.empty(rows,width,dtype=torch.int32,device='cuda'),
        decode_lens_buffer=torch.empty(rows,dtype=torch.int32,device='cuda'),
        arange_buffer=torch.arange(rows,dtype=torch.int32,device='cuda'))
    pointers=tuple(t.data_ptr() for t in (state.decode_seq_lens_buffer,state.expanded_block_table_buffer,state.decode_lens_buffer))
    qsl=torch.zeros(n+1,dtype=torch.int32,device='cuda')
    seq=torch.zeros(n,dtype=torch.int32,device='cuda')
    def install(lengths):
        offsets=[0]
        for length in lengths:offsets.append(offsets[-1]+length)
        qsl.copy_(torch.tensor(offsets,dtype=torch.int32,device='cuda'))
        seq.copy_(torch.tensor([129+i*128+k for i,k in enumerate(lengths)],dtype=torch.int32,device='cuda'))
        total=sum(lengths)
        cpu=torch.tensor([total//n+(i<total%n) for i in range(n)],dtype=torch.int32)
        return DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors(state,
            seq,table,torch.diff(qsl),cpu,qsl[:-1],n,rows,False,6,int(cpu.max()))
    install([1,6,2,4])
    torch.manual_seed(103)
    heads,dim,block,max_len=16,128,128,640
    q=(torch.randn(rows,1,heads,dim,device='cuda')*.25).to(torch.float8_e4m3fn)
    cache_keys=(torch.randn(24,block,dim,device='cuda')*.25).to(torch.float8_e4m3fn)
    scales=torch.ones(24,block,device='cuda',dtype=torch.float32)
    cache=torch.empty(24,block,1,dim+4,device='cuda',dtype=torch.uint8)
    flat=cache.view(24,-1)
    flat[:,:block*dim]=cache_keys.view(torch.uint8).reshape(24,-1)
    flat[:,block*dim:]=scales.view(torch.uint8).reshape(24,-1)
    weights=torch.rand(rows,heads,device='cuda')
    logits_out=torch.empty(rows,max_len,device='cuda')
    topk_out=torch.empty(rows,8,device='cuda',dtype=torch.int64)
    def consume():
        logits=fp8_paged_mqa_logits_triton(q,cache,weights,
            state.decode_seq_lens_buffer,state.expanded_block_table_buffer,max_len)
        logits_out.copy_(logits)
        topk_out.copy_(logits.topk(8,dim=-1).indices)
    for _ in range(3):consume()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):consume()
    for lengths in ([1,6,2,4],[6,1,4,2],[0,6,0,1],[1,1,1,1],[0,0,0,0]):
        result=install(lengths)
        expected_seq=[];expected_rows=[]
        for i,k in enumerate(lengths):
            expected_seq += [129+i*128+j+1 for j in range(k)]
            expected_rows += [i]*(k)
        total=sum(lengths)
        expected_seq += [0]*(rows-total)
        torch.testing.assert_close(result[0].cpu(),torch.tensor(expected_seq,dtype=torch.int32))
        if total:
            torch.testing.assert_close(result[1][:total],table[expected_rows])
        assert (result[1][total:,0]==0).all()
        assert (result[2]==1).all()
        assert result[3:]==(rows,False)
        assert pointers==tuple(t.data_ptr() for t in result[:3])
        reference=torch.full_like(logits_out,-float('inf'))
        for row,context in enumerate(expected_seq):
            if context:
                positions=torch.arange(context,device='cuda')
                physical=result[1][row,positions//block].long()
                keys=cache_keys.float()[physical,positions%block]
                score=(q[row,0].float()@keys.T).relu()
                reference[row,:context]=(score*weights[row,:,None]).sum(0)
        consume()
        torch.testing.assert_close(logits_out,reference,rtol=.02,atol=.03)
        expected_topk=topk_out.clone()
        logits_out.fill_(float('nan'));topk_out.fill_(-1);graph.replay()
        torch.testing.assert_close(logits_out,reference,rtol=.02,atol=.03)
        torch.testing.assert_close(topk_out,expected_topk,rtol=0,atol=0)
    print('EXISTING_SM80_INDEXER five layouts + real MQA/top-k eager/graph PASS; no capability flags changed')


if __name__=='__main__':main()
