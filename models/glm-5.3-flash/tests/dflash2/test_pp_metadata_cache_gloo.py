"""Patch 0030 (dev 0025): two local CPU processes, real Gloo, finite timeout; no GPU/node2."""
import datetime
import importlib.util
import os
import tempfile
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class Retained:
    def __init__(self, works, retained): self.works=works; self.retained=retained
    def wait(self):
        for work in self.works: work.wait()
        self.retained=()

def worker(rank, path):
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    spec=importlib.util.spec_from_file_location('cache', os.path.join(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages"), 'vllm', 'distributed', 'pp_metadata_cache.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cache=module.PPMetadataCache(SimpleNamespace(ranks=[0,1], cpu_group=dist.group.WORLD),Retained)
    pending=[]
    for step in range(160):
        rows=(8,8,64,8,0)[step%5]
        metadata=[('hidden_states',('cuda', 'bf16', (rows,4,4096))),
                  ('aux_fc_partial',('cuda','fp32',(rows,4096))),
                  ('__verification_budget', step//4%3)]
        # CPU payload mirrors ordered tensor p2p after metadata. Header size
        # cache may change independently of tensor values and row count.
        if rank==0:
            pending.append(cache.send(metadata,1))
            payload=torch.full((max(rows,1),),step,dtype=torch.int64)
            pending.append(Retained([dist.isend(payload,dst=1)],(payload,)))
            if len(pending)>16:
                pending.pop(0).wait()
        else:
            assert cache.recv(0)==metadata
            payload=torch.empty(max(rows,1),dtype=torch.int64)
            dist.recv(payload,src=0)
            assert torch.equal(payload,torch.full_like(payload,step))
    for handle in pending: handle.wait()
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='pp0025-gloo-') as folder:
        mp.spawn(worker,args=(folder+'/init',),nprocs=2,join=True)
    print('PASS: real Gloo, 160 ordered steps, bounded retained handles')
