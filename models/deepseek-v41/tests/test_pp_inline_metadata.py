#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM P2P: non-tensor control rides in existing object metadata.

Two GPUs, no model. Validate that budget metadata can be read before waiting
for GPU payload completion; then use actual AsyncIntermediateTensors wait and
check hidden/length tensors. This does NOT establish PP6 lifecycle or latency.
"""
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


KEY = '__test_verification_budget'


def worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,
        rank=rank,world_size=2,timeout=timedelta(seconds=45))
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
    group=GroupCoordinator([[0,1]],rank,'nccl',use_device_communicator=False,
                           group_name='inline-budget-audit')
    slots=[3,1] if rank==0 else [2,7]
    request_state=NS(device=device,num_speculative_steps=5,max_num_reqs=8,
        req_id_to_index=dict(zip(('a','b'),slots)),max_num_batched_tokens=32)
    manager=AdaptiveVerificationManager(request_state,
        torch.zeros(9,device=device,dtype=torch.int32),1,32)
    scheduled={'a':6,'b':6};drafts={'a':[0]*5,'b':[0]*5}
    mapping=torch.tensor(slots,device=device)
    observed=torch.empty(3,device=device,dtype=torch.int32)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):observed.copy_(manager.query_start_loc[:3])
    for step,lengths in enumerate(([5,2],[2,5],[0,0],[5,5])):
        # Metadata integers are host-known before allocation: not read from GPU.
        metadata=(1,step,len(lengths),sum(lengths))
        expected=torch.tensor(lengths,dtype=torch.int32,device=device)
        if rank==0:
            manager.get_num_tokens(scheduled,drafts,authoritative_budget=sum(lengths))
            manager.reallocate_drafts(['a','b'],mapping,authoritative_capacities=expected)
            graph.replay()
            torch.testing.assert_close(observed,torch.tensor([0,1+lengths[0],2+sum(lengths)],device=device,dtype=torch.int32))
            source=expected.clone()
            payload={KEY:metadata,'lengths':source.clone(),
                     'hidden_states':torch.full((16,8),step+.5,device=device)}
            handles=group.isend_tensor_dict(payload,dst=1)
            source.fill_(-999)
            # Existing metadata handle + two CUDA handles, no CPU tensor send.
            assert len(handles)==3, len(handles)
            for h in handles:h.wait()
            reply=group.recv_tensor_dict(src=1)
            assert reply[KEY]==metadata
            torch.testing.assert_close(reply['lengths'],expected)
            torch.testing.assert_close(reply['hidden_states'],torch.full((16,8),step+1.5,device=device))
        else:
            tensors,handles,callbacks=group.irecv_tensor_dict(src=0)
            # Metadata receive has completed; no CPU tensor or new receive
            # handle is needed for this small static-schema budget tuple.
            assert tensors.pop(KEY)==metadata
            assert len(handles)==2, len(handles)
            intermediate=AsyncIntermediateTensors(tensors,handles,callbacks)
            assert not intermediate._comm_waited
            # Local CPU dispatch based on metadata can happen right here.
            budget=metadata[3];real_rows=metadata[2]+budget
            assert real_rows==sum(lengths)+len(lengths)
            assert not intermediate._comm_waited
            # Later model input staging establishes CUDA receive dependency.
            # CPU budget known before touching any received GPU tensor.
            manager.get_num_tokens(scheduled,drafts,authoritative_budget=budget)
            compact,_=manager.compact_batch(np.array([5,5]),np.array([6,6]),np.array([0,6,12]))
            assert compact.sum()==real_rows
            completed=intermediate.tensors
            assert intermediate._comm_waited
            manager.reallocate_drafts(['a','b'],mapping,authoritative_capacities=completed['lengths'])
            graph.replay()
            torch.testing.assert_close(observed,torch.tensor([0,1+lengths[0],2+sum(lengths)],device=device,dtype=torch.int32))
            torch.testing.assert_close(completed['lengths'],expected)
            completed['hidden_states'].add_(1)
            for h in group.isend_tensor_dict(completed | {KEY:metadata},dst=0):h.wait()
        torch.cuda.synchronize();dist.barrier(device_ids=[rank])
    if rank==0:print('INLINE_PP_METADATA four roundtrips + actual manager + graph offsets PASS; no additional CPU tensor/handle',flush=True)
    group.destroy();dist.destroy_process_group()


if __name__=='__main__':
    assert torch.cuda.device_count()==2
    with tempfile.TemporaryDirectory(prefix='inline-budget-') as root:
        mp.spawn(worker,args=(str(Path(root)/'init'),),nprocs=2,join=True)
