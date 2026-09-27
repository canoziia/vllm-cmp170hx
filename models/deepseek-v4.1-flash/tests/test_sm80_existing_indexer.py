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
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.config import AttentionConfig
from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton


@torch.inference_mode()
def main(ratio, depth):
    assert torch.cuda.get_device_capability() == (8, 0)
    n, rows, width = 4, 16, 129
    table = (torch.arange(n*(width+7), dtype=torch.int32, device='cuda') % 24).reshape(n,width+7)[:,:width]
    config=NS(model_config=NS(max_model_len=640,architectures=['DeepseekV41ForCausalLM']),
        scheduler_config=NS(max_num_batched_tokens=rows,max_num_seqs=n),
        parallel_config=NS(decode_context_parallel_size=1,prefill_context_parallel_size=1,
                           cp_kv_cache_interleave_size=1),
        attention_config=AttentionConfig(),num_speculative_tokens=depth,
        speculative_config=NS(enable_adaptive_verification=True,num_speculative_tokens=depth))
    spec=MLAAttentionSpec(block_size=128,num_kv_heads=1,head_size=128,
        dtype=torch.uint8,tokens_per_state=ratio,cache_dtype_str='fp8',model_version='deepseek_v4')
    state=DeepseekV32IndexerMetadataBuilder(spec,['indexer'],config,torch.device('cuda',0),block_table_width=width)
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
        cpu_qsl=torch.cat((torch.zeros(1,dtype=torch.int32),cpu.cumsum(0).to(torch.int32)))
        slots=torch.full((rows,),-1,device='cuda',dtype=torch.int64)
        slots[:total]=torch.arange(total,device='cuda')
        cm=CommonAttentionMetadata(query_start_loc=qsl,query_start_loc_cpu=cpu_qsl,
            seq_lens=seq,seq_lens_cpu_upper_bound=torch.full((n,),640,dtype=torch.int32),
            num_reqs=n,num_actual_tokens=rows,max_query_len=depth+1,max_seq_len=640,
            block_table_tensor=table,slot_mapping=slots,causal=True)
        metadata=state.build_for_cudagraph_capture(cm)
        assert metadata.num_prefills==0 and metadata.num_decode_tokens==rows
        decode=metadata.decode
        return (decode.seq_lens.view(-1),decode.block_table,decode.decode_lens,rows,decode.requires_padding)
    layouts=([1,6,2,4],[6,1,4,2],[0,6,0,1],[1,1,1,1],[0,0,0,0]) if depth==5 else (
        [1,2,1,2],[2,1,2,1],[0,2,0,1],[1,1,1,1],[0,0,0,0])
    install(layouts[0])
    torch.manual_seed(103)
    heads,dim,block,max_len=16,128,int(spec.num_states),640
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
    for lengths in layouts:
        result=install(lengths)
        expected_seq=[];expected_rows=[]
        for i,k in enumerate(lengths):
            expected_seq += [129+i*128+j+1 for j in range(k)]
            expected_rows += [i]*(k)
        total=sum(lengths)
        expected_seq += [0]*(rows-total)
        expected_seq=[value//ratio for value in expected_seq]
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
    print(f'EXISTING_SM80_INDEXER ratio={ratio} depth={depth} five layouts + real MQA/top-k eager/graph PASS; no capability flags changed')


if __name__=='__main__':
    import os
    depths=(1,5) if os.environ.get('TEST_SM80_ADAPTIVE_BACKENDS','0')=='1' else (5,)
    for depth in depths:
        for ratio in (1,2):main(ratio,depth)
