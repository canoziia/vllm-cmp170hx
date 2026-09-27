#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patched actual V2 gather/prepare methods + existing manager/CUDA kernels.

Method extraction avoids full model load; token/position/layout kernels and
manager are real. No custom plan adapter. Not execute_model/PP6 validation.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runner',type=Path)
    source=parser.parse_args().runner
    from vllm.v1.worker.gpu import model_runner as installed
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
    names=('gather_batch_req_state','prepare_inputs')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(installed.__dict__);exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),ns)
    cases=0;device=torch.device('cuda',0)
    for mixed in (False,True):
        for lengths in ([5,2],[2,5],[0,0],[5,5]):
            ids=['a','b']+(['p'] if mixed else [])
            mapping=[3,1]+([5] if mixed else [])
            lengths=lengths+([0] if mixed else [])
            caps=np.array([5,5]+([0] if mixed else []),dtype=np.int32)
            counts=np.array([6,6]+([4] if mixed else []),dtype=np.int32)
            computed=np.zeros(8,dtype=np.int32);computed[3]=17;computed[1]=31
            prefill=np.zeros(8,dtype=np.int32);prefill[3]=4;prefill[1]=4;prefill[5]=20
            all_ids=torch.arange(8*128,device=device,dtype=torch.int32).reshape(8,128)+1000
            last=torch.arange(8,device=device,dtype=torch.int32)[:,None]+200
            drafts=torch.arange(40,device=device,dtype=torch.int32).reshape(8,5)+300
            states=NS(device=device,num_speculative_steps=5,max_num_reqs=8,
                req_id_to_index=dict(zip(ids,mapping)),max_num_batched_tokens=64,
                prefill_len=NS(np=prefill,gpu=torch.tensor(prefill,device=device)),
                num_computed_prefill_tokens=np.minimum(computed,prefill),
                num_computed_tokens_np=computed,num_computed_tokens=NS(gpu=torch.tensor(computed,device=device)),
                next_prefill_tokens=torch.zeros(8,device=device,dtype=torch.int32),
                all_token_ids=NS(gpu=all_ids),last_sampled_tokens=last,draft_tokens=drafts)
            buffers=InputBuffers(8,64,device)
            manager=AdaptiveVerificationManager(states,buffers.query_start_loc,1,48)
            state=NS(device=device,max_num_reqs=8,max_num_tokens=64,decode_query_len=6,
                pcp_manager=None,model_config=NS(rswa_window=None),
                model_state=NS(num_new_sampled_tokens_per_step=1),input_buffers=buffers,
                req_states=states,adaptive_verification=manager)
            for name in names:setattr(state,name,MethodType(ns[name],state))
            scheduler=NS(num_scheduled_tokens=dict(zip(ids,map(int,counts))),
                total_num_scheduled_tokens=int(counts.sum()),scheduled_spec_decode_tokens={'a':[0]*5,'b':[0]*5},
                has_structured_output_requests=False)
            batch,_=state.gather_batch_req_state(scheduler,False,authoritative_budget=sum(lengths))
            result=state.prepare_inputs(scheduler,batch,NS(num_tokens=32,num_reqs=None),
                authoritative_capacities=torch.tensor(lengths,device=device,dtype=torch.int32))
            expected_ids=[];expected_pos=[];expected_logits=[];offsets=[0]
            for i,(slot,k) in enumerate(zip(mapping,lengths)):
                count=(1+k) if i<2 else 4
                expected_ids += [int(last[slot,0])]+drafts[slot,:k].tolist() if i<2 else all_ids[slot,:4].tolist()
                expected_pos += list(range(int(computed[slot]),int(computed[slot])+count))
                expected_logits += list(range(offsets[-1],offsets[-1]+count)) if i<2 else [offsets[-1]+3]
                offsets.append(offsets[-1]+count)
            assert result.query_start_loc.tolist()==offsets
            assert result.input_ids[:offsets[-1]].tolist()==expected_ids
            assert result.positions[:offsets[-1]].tolist()==expected_pos
            assert result.logits_indices.tolist()==expected_logits
            assert result.num_scheduled_tokens.tolist()==counts.tolist()
            assert result.num_draft_tokens_per_req.tolist()==caps.tolist()
            assert result.num_draft_tokens==sum(lengths)
            assert result.query_start_loc.data_ptr()==buffers.query_start_loc.data_ptr()
            if not sum(lengths):assert result.cu_num_logits_np.tolist()==list(range(len(ids)+1))
            cases+=1
    print(f'ADAPTIVE_V2_INPUTS {cases} actual-manager CUDA cases PASS; original query buffer preserved')


if __name__=='__main__':main()
