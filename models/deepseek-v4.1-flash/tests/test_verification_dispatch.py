#!/usr/bin/env python3
"""CPU-only exact candidate test, run with applied pinned SOURCE_TREE argument."""
import ast,os,sys
from dataclasses import dataclass
from collections import defaultdict
from itertools import groupby,product
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
class CUDAGraphMode(Enum):
    NONE=0; PIECEWISE=1; FULL=2
class Mode:
    def decode_mode(self): return CUDAGraphMode.FULL
    def mixed_mode(self): return CUDAGraphMode.PIECEWISE
    def separate_routine(self): return True
@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: object
    num_tokens: int
    num_reqs: object
    uniform_token_count: object=None
    max_query_len: object=None
    num_active_loras: int=0
    num_ubatches: int=1

def load(path,flag,adaptive):
    os.environ['VLLM_DSV41_VERIFICATION']=str(flag)
    tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CudaGraphManager')
    f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_init_candidates')
    ns=dict(globals(),round_up=lambda x,y:(x+y-1)//y*y)
    exec(compile(ast.Module([f],type_ignores=[]),str(path),'exec'),ns)
    s=NS(compilation_config=NS(cudagraph_capture_sizes=sorted(set([1,2,4,6,8,12,16]+list(range(24,257,8))+list(range(272,385,16)))),max_cudagraph_capture_size=384),cudagraph_mode=Mode(),max_num_reqs=32,decode_query_len=6,varlen_decode=adaptive,lora_capture_cases=[0],vllm_config=NS(speculative_config=None,model_config=NS(architectures=['DeepseekV41ForCausalLM']),cache_config=NS(use_kda_recoverssm=False)),_capture_descs={},_candidates={})
    ns['_init_candidates'](s);return s
path=Path(sys.argv[1])/'vllm/v1/worker/gpu/cudagraph_utils.py'
for adaptive in (False,True):
    s=load(path,1,adaptive)
    for reqs in range(1,9):
        for width in range(1,7):
            rows=reqs*width
            ds=[d for d in s._candidates[rows,0] if d.cg_mode==CUDAGraphMode.FULL and d.num_tokens>=rows and (d.uniform_token_count is None or d.uniform_token_count==width) and d.num_reqs>=reqs]
            assert ds[0].num_tokens==rows,(adaptive,reqs,width,ds)
            assert ds[0].num_reqs==reqs,(adaptive,reqs,width,ds)
    for ds in s._capture_descs.values():assert len(ds)==len(set(ds))
    if adaptive:
        for rows in range(1,49):
            assert any(d.cg_mode==CUDAGraphMode.FULL and d.num_tokens==rows and d.uniform_token_count is None for d in s._candidates[rows,0])
print('PASS uniform widths1..6 cohorts1..8 exact tokens/requests, ragged rows1..48, no duplicate captures')
