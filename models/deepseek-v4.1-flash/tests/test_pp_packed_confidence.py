#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual PPHandler with a real GroupCoordinator/NCCL group, no model.

Uses distributed module _PP to install the test's two-rank group; no CUDA,
collective, payload or PPHandler methods are mocked. Padded prefill rows,
slot reuse/cancellation, source overwrite and OFF wire behavior are tested.
"""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,rendezvous):
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,
        world_size=2,timeout=timedelta(seconds=45))
    import vllm.distributed.parallel_state as ps
    from vllm.v1.worker.gpu.pp_utils import PPHandler
    group=ps.GroupCoordinator([[0,1]],rank,'nccl',use_device_communicator=False,
                              group_name='packed-confidence-test')
    ps._PP=group
    scenarios=0
    for enabled in (False,True):
        for invalidate in (False,True):
            handler=PPHandler(8,5,device,relay_draft_confidences=enabled)
            for cycle in range(3):
                mapping=[3,1,5] if rank else [2,7,0]
                batch=NS(num_reqs=3,idx_mapping=torch.tensor(mapping,device=device),
                    idx_mapping_np=np.array(mapping),num_computed_tokens_np=np.array([10,10,0]),
                    num_scheduled_tokens=np.array([6,6,3]),prefill_len_np=np.array([4,4,20]))
                tokens=torch.arange(40,device=device).reshape(8,5)+cycle*100
                scores=torch.tensor([[0.,.2,.5,.9,1.],[1.,.8,.5,.1,0.],[.3]*5],device=device)
                expected=scores.clone()
                token_state=torch.full((8,5),-1,device=device,dtype=torch.int64)
                confidence_state=torch.full((8,5),-1.,device=device)
                def consume(scores,indices):confidence_state[indices]=scores
                if rank:
                    handler.broadcast(torch.zeros(3,1,device=device,dtype=torch.int64),
                        torch.ones(3,device=device,dtype=torch.int32),
                        torch.zeros(3,device=device,dtype=torch.int32),batch)
                    handler.broadcast_drafts(tokens,batch,scores if enabled else None)
                    scores.fill_(float('nan'))
                else:
                    assert handler.get_prev_sampled_outputs(token_state,consume) is None
                    handler.receive(batch)
                    if invalidate:handler.on_req_idx_freed(7)
                    assert handler.get_prev_sampled_outputs(token_state,consume) is None
                    result=handler.get_prev_sampled_outputs(token_state,consume)
                    assert result['idx_mapping'].tolist()==[2,-1 if invalidate else 7,-1]
                    torch.testing.assert_close(token_state[2],tokens[3])
                    if enabled:torch.testing.assert_close(confidence_state[2],expected[0],rtol=0,atol=0)
                    else:assert (confidence_state==-1).all()
                    if invalidate:
                        assert (token_state[7]==-1).all() and (confidence_state[7]==-1).all()
                    else:
                        torch.testing.assert_close(token_state[7],tokens[1])
                        if enabled:torch.testing.assert_close(confidence_state[7],expected[1],rtol=0,atol=0)
                    assert (token_state[0]==-1).all() and (confidence_state[0]==-1).all()
                torch.cuda.synchronize();dist.barrier(device_ids=[rank]);scenarios+=1
            dist.destroy_process_group(handler.broadcast_group)
    if rank==0:print(f'PACKED_PP_CONFIDENCE {scenarios} real FIFO cycles PASS',flush=True)
    ps._PP=None;group.destroy();dist.destroy_process_group()


if __name__=='__main__':
    assert torch.cuda.device_count()==2
    with tempfile.TemporaryDirectory(prefix='packed-confidence-') as path:
        mp.spawn(worker,args=(str(Path(path)/'init'),),nprocs=2,join=True)
