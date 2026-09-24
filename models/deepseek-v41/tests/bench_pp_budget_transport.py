#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU P2P transport-only microbenchmark, not model speedup evidence.

Compare existing hidden-only, inline scalar budget, inline budget+GPU lengths,
and prototype JSON-header+GPU-length tensor envelopes with identical hidden
payload. CPU wall-clock roundtrip timing includes metadata and waits.
"""
import json
import statistics
import tempfile
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,rendezvous):
    torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,
        world_size=2,timeout=timedelta(seconds=90))
    from vllm.distributed.parallel_state import GroupCoordinator
    group=GroupCoordinator([[0,1]],rank,'nccl',use_device_communicator=False,
                           group_name='budget-transport-bench')
    for rows in (6,96):
        hidden=torch.ones(rows,4,7168,device=device,dtype=torch.bfloat16)
        requests=max(1,rows//6)
        lengths=torch.full((requests,),5,device=device,dtype=torch.int32)
        metadata=(1,requests,requests*5)
        header={'step':0,'request_keys':[(f'request-{i}',0) for i in range(requests)],
            'draft_caps':[5]*requests,'non_draft_counts':[1]*requests,
            'draft_budget':5*requests,'graph_rows':rows,'version':1}
        def payload(mode):
            value={'hidden_states':hidden}
            if mode.startswith('inline'):value['budget']=metadata
            if mode=='inline_partial':value['lengths']=lengths.clone()
            if mode=='prototype':
                value['header']=torch.tensor(list(json.dumps(header,separators=(',',':')).encode()),dtype=torch.uint8)
                value['lengths']=lengths.clone()
            return value
        def exchange(mode):
            if rank==0:
                value=payload(mode)
                for handle in group.isend_tensor_dict(value,dst=1):handle.wait()
                got=group.recv_tensor_dict(src=1)
            else:
                got=group.recv_tensor_dict(src=0)
                if mode=='prototype':json.loads(bytes(got['header'].tolist()))
                if mode.startswith('inline'):assert got['budget']==metadata
                for handle in group.isend_tensor_dict(got,dst=0):handle.wait()
            return got
        modes=('hidden_only','inline_full','inline_partial','prototype')
        for mode in modes:
            for _ in range(5):exchange(mode)
        observations={mode:[] for mode in modes}
        for repeat in range(4):
            order=modes if repeat%2==0 else tuple(reversed(modes))
            for mode in order:
                torch.cuda.synchronize();dist.barrier(device_ids=[rank])
                start=time.perf_counter()
                for _ in range(30):last=exchange(mode)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000/30
                assert last['hidden_states'].shape==hidden.shape
                observations[mode].append(elapsed)
        if rank==0:
            print(json.dumps({'rows':rows,'roundtrip_ms':{k:round(statistics.median(v),4) for k,v in observations.items()},'repeats_ms':observations}),flush=True)
    group.destroy();dist.destroy_process_group()


if __name__=='__main__':
    assert torch.cuda.device_count()==2
    with tempfile.TemporaryDirectory(prefix='pp-budget-bench-') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'init'),),nprocs=2,join=True)
