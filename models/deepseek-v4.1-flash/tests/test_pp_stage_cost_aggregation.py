#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract candidate startup aggregation block; check sum and disagreement.
CPU-only fake collective isolates algebra, not model profiling correctness.
"""
import ast
import argparse
from pathlib import Path
from types import SimpleNamespace as NS
from vllm.v1.worker.gpu.async_utils import StepTimingSample


def main():
    p=argparse.ArgumentParser();p.add_argument('runner',type=Path);path=p.parse_args().runner
    tree=ast.parse(path.read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
        ast.unparse(n.test)=='self.use_pp' and 'rank_timings' in ast.unparse(n))
    source=ast.Module(body=block.body,type_ignores=[])
    peers=[[StepTimingSample(float(rank+1),3. if rank==5 else 0.,6,1,True),
            StepTimingSample(float((rank+1)*2),6. if rank==5 else 0.,12,2,True)] for rank in range(6)]
    def run(data):
        def gather(out,local,group):out[:]=data
        ns={'self':NS(parallel_config=NS(pipeline_parallel_size=6)),
            'timings':data[0], 'torch':NS(distributed=NS(all_gather_object=gather)),
            'get_pp_group':lambda:NS(cpu_group=None)}
        exec(compile(source,str(path),'exec'),ns)
        return ns['timings']
    result=run(peers)
    assert result==[StepTimingSample(21.,3.,6,1,True),StepTimingSample(42.,6.,12,2,True)]
    peers[-1][1]=StepTimingSample(12.,6.,13,2,True)
    try:run(peers)
    except RuntimeError:pass
    else:raise AssertionError('shape disagreement accepted')
    print('PP_STAGE_COST_AGGREGATION sum/shape rejection PASS (CPU algebra only)')


if __name__=='__main__':main()
