#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe existing SM80 MLA/SWA builders, unchanged, across graph replays.

No adaptive patches, capability overrides, row-map replacement, model weights
or PP mocks. Synthetic packed KV + actual builder/attention output vs FP32
reference. This does not enable varlen FULL graphs in the model runner.
Run in the pinned DeepSeek image on one SM80 GPU.
"""
from types import SimpleNamespace as NS
import math
import os
import torch

from vllm.models.deepseek_v4_1.ampere.ampere_sparse import DeepseekV41AmpereMLASparseBackend
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def pack(keys, block):
    keys = keys.clone()
    keys[:, :448] = keys[:, :448].to(torch.float8_e4m3fn).to(torch.bfloat16)
    raw = torch.zeros(keys.shape[0], 584, dtype=torch.uint8, device='cuda')
    raw[:, :448] = keys[:, :448].to(torch.float8_e4m3fn).view(torch.uint8)
    raw[:, 448:576] = keys[:, 448:].contiguous().view(torch.uint8)
    raw[:, 576:583] = 127
    pages = raw.reshape(-1, block, 584)
    packed = torch.cat((pages[:, :, :576].reshape(-1, block*576),
                        pages[:, :, 576:].reshape(-1, block*8)), dim=1)
    return packed.reshape(-1, block, 584), keys.float()


@torch.inference_mode()
def main(padded_requests):
    assert torch.cuda.get_device_capability() == (8, 0)
    device = torch.device('cuda', 0)
    n, rows, block, window, heads = 3+padded_requests, 18, 128, 32, 16
    adaptive = os.environ.get('TEST_SM80_ADAPTIVE_BACKENDS', '0') == '1'
    swa_builder_cls = DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    if adaptive:
        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import DeepseekV41AmpereSWAMetadataBuilder
        swa_builder_cls = DeepseekV41AmpereSWAMetadataBuilder
    config = NS(
        model_config=NS(max_model_len=512, hf_config=NS(
            compress_ratios=[0,1,2], index_topk=8, sliding_window=window)),
        scheduler_config=NS(max_num_batched_tokens=rows, max_num_seqs=n),
        parallel_config=NS(decode_context_parallel_size=1),
        speculative_config=NS(num_speculative_tokens=5, parallel_drafting=False,
                              enable_adaptive_verification=adaptive, use_dspark=lambda:True),
    )
    swa_spec = SlidingWindowMLASpec(block_size=block,num_kv_heads=1,head_size=512,
        dtype=torch.uint8,sliding_window=window,cache_dtype_str='fp8_ds_mla',model_version='deepseek_v4')
    torch.manual_seed(37)
    table = torch.tensor([[4,1,8],[7,3,0],[2,6,5]]+[[0,0,0]]*padded_requests,dtype=torch.int32,device=device)
    q = torch.randn(rows,heads,512,dtype=torch.bfloat16,device=device)*.125
    sink = torch.randn(heads,device=device)*.2
    swa_cache, swa_keys = pack(torch.randn(9*block,512,dtype=torch.bfloat16,device=device)*.125,block)
    starts = torch.zeros(n+1,dtype=torch.int32,device=device)
    seq = torch.zeros(n,dtype=torch.int32,device=device)
    slot_groups = torch.full((2,rows),-1,dtype=torch.int64,device=device)
    slots = slot_groups[0]
    cpu_starts = torch.zeros(n+1,dtype=torch.int32)
    cpu_seq = torch.tensor([384]*n,dtype=torch.int32)
    topk = torch.arange(8,dtype=torch.int32,device=device).expand(rows,-1).contiguous()
    output = torch.empty_like(q)
    cases = 0
    for ratio in (1,2):
        mla_spec = MLAAttentionSpec(block_size=block,num_kv_heads=1,head_size=512,
            dtype=torch.bfloat16,tokens_per_state=ratio,cache_dtype_str='fp8_ds_mla',model_version='deepseek_v4')
        mla_builder = DeepseekV41AmpereMLASparseBackend.get_builder_cls()(mla_spec,['c'],config,device)
        swa_builder = swa_builder_cls(swa_spec,['c'],config,device)
        # Run the actual V2 metadata orchestration, not direct builder calls.
        model_state=DefaultModelState.__new__(DefaultModelState)
        model_state.max_model_len=384
        model_state.supports_mm_inputs=False
        groups=[
            [AttentionGroup(DeepseekV41AmpereMLASparseBackend,['mla'],mla_spec,0,[mla_builder])],
            [AttentionGroup(DeepseekV41AmpereMLASparseBackend,['swa'],swa_spec,1,[swa_builder])],
        ]
        kv_config=NS(kv_cache_groups=[NS(layer_names=['mla']),NS(layer_names=['swa'])])
        compressed_block = int(mla_spec.num_states)
        compressed, comp_keys = pack(torch.randn(9*compressed_block,512,dtype=torch.bfloat16,device=device)*.125,compressed_block)
        def install(lengths,contexts,for_capture=False):
            lengths=list(lengths)+[0]*padded_requests
            contexts=list(contexts)+[0]*padded_requests
            off=[0]
            for count in lengths:off.append(off[-1]+count)
            starts.copy_(torch.tensor(off,dtype=torch.int32,device=device))
            # Upstream contract: CPU boundaries may differ, but total must
            # equal the device total. Graph padding is num_actual_tokens,
            # NOT an extra nonempty request in query_start_loc.
            total=sum(lengths)
            balanced=[total//3+(i<total%3) for i in range(3)]+[0]*padded_requests
            cpu_starts[0]=0
            torch.cumsum(torch.tensor(balanced,dtype=torch.int32),0,out=cpu_starts[1:])
            seq.copy_(torch.tensor([c+k for c,k in zip(contexts,lengths)],dtype=torch.int32,device=device))
            slot_values=[]
            table_cpu=table.cpu().tolist()
            for i,(c,k) in enumerate(zip(contexts,lengths)):
                slot_values.extend(table_cpu[i][p//block]*block+p%block for p in range(c,c+k))
            slots.fill_(-1)
            slots[:len(slot_values)].copy_(torch.tensor(slot_values,dtype=torch.int64,device=device))
            slot_groups[1].copy_(slots)
            import numpy as np
            batch=NS(num_reqs=3,num_reqs_after_padding=n,num_tokens=sum(lengths),
                num_tokens_after_padding=rows,query_start_loc_np=cpu_starts.numpy(),
                query_start_loc=starts,max_query_len=6,seq_lens_cpu_upper_bound=cpu_seq,
                seq_lens=seq,dcp_local_seq_lens=None,positions=torch.zeros(rows,device=device,dtype=torch.int64),
                is_prefilling_np=np.zeros(n,dtype=bool),prompt_lens=None)
            metadata=model_state.prepare_attn(batch,CUDAGraphMode.FULL,(table,table),
                slot_groups,groups,kv_config,for_capture=for_capture)
            return metadata['mla'],metadata['swa']
        def pointers(m,s):
            return tuple(x.data_ptr() for x in (m.req_id_per_token,m.slot_mapping,
                s.token_to_req_indices,s.decode_swa_ragged_indices,s.decode_swa_ragged_indptr))
        def invoke(m,s):
            ri,rp,rl=compute_global_topk_ragged_indices_and_indptr(topk,
                s.token_to_req_indices,m.block_table,compressed_block,s.is_valid_token)
            rocm_sparse_attn_decode(q=q,kv_cache=compressed,swa_k_cache=swa_cache,swa_only=False,
                topk_indices=None,topk_lens=rl,swa_indices=s.decode_swa_indices,swa_lens=s.decode_swa_lens,
                swa_ragged_indices=s.decode_swa_ragged_indices,swa_ragged_indptr=s.decode_swa_ragged_indptr,
                topk_ragged_indices=ri,topk_ragged_indptr=rp,attn_sink=sink,scale=1/math.sqrt(512),
                head_dim=512,nope_head_dim=448,rope_head_dim=64,output=output)
        def check(m,s,lengths,contexts):
            owners=[i for i,k in enumerate(lengths) for _ in range(k)]
            active=sum(lengths)
            assert m.req_id_per_token[:active].tolist()==owners
            assert s.token_to_req_indices[:active].tolist()==owners
            assert s.is_valid_token.tolist()==[True]*active+[False]*(rows-active)
            assert s.decode_swa_lens[active:].count_nonzero().item()==0
            row=0;table_cpu=table.cpu().tolist();expected_slots=[]
            for i,(k,c) in enumerate(zip(lengths,contexts)):
                for j in range(k):
                    pos=c+j
                    physical=[table_cpu[i][p//block]*block+p%block for p in range(max(0,pos-window+1),pos+1)]
                    extra=[table_cpu[i][p//compressed_block]*compressed_block+p%compressed_block for p in range(8)]
                    keys=torch.cat((swa_keys[physical],comp_keys[extra]))
                    scores=q[row].float()@keys.T/math.sqrt(512)
                    probs=torch.softmax(torch.cat((scores,sink[:,None]),dim=1),dim=1)[:,:-1]
                    torch.testing.assert_close(output[row],(probs@keys).to(torch.bfloat16),rtol=.02,atol=.003)
                    if ratio==1:expected_slots.append(table_cpu[i][pos//block]*block+pos%block)
                    else:
                        compressed_pos=(pos+1)//ratio-1
                        expected_slots.append(table_cpu[i][compressed_pos//compressed_block]*compressed_block+compressed_pos%compressed_block if (pos+1)%ratio==0 else -1)
                    row+=1
            assert m.slot_mapping[:active].tolist()==expected_slots
            assert (m.slot_mapping[active:]==-1).all()
            torch.testing.assert_close(output[active:],torch.zeros_like(output[active:]),rtol=0,atol=0)
        m,s=install([6,6,6],[125,255,129],for_capture=True);stable=pointers(m,s)
        for _ in range(3):invoke(m,s)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):invoke(m,s)
        for lengths in ([6,6,6],[2,6,4],[6,2,4],[1,1,1],[0,6,0]):
            for contexts in ([125,255,129],[126,256,130]):
                m,s=install(lengths,contexts)
                assert pointers(m,s)==stable
                output.fill_(float('nan'));invoke(m,s);check(m,s,lengths,contexts)
                output.fill_(float('nan'));graph.replay();check(m,s,lengths,contexts)
                cases+=1
    print(f'EXISTING_SM80_BUILDERS padded_requests={padded_requests} {cases} eager+graph oracle cases PASS; adaptive_backend_opt_in={adaptive}')


if __name__=='__main__':
    for padding in (0,3):main(padding)
