#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute actual GPUWorker control method with fake transport/model boundary.

No CUDA in this test: checks parsing order, stripping and guarded kwargs. Real
transport and actual manager/CUDA input behavior have separate tests.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('worker',type=Path)
    path=parser.parse_args().worker
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Worker')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
    method.decorator_list=[]
    class Output:pass
    class Intermediate:
        def __init__(self,tensors,comm_handles=None,comm_postprocess=None):self.tensors=tensors
    for enabled in (False,True):
        for budget in (None,0,5,10):
            events=[]
            class Handle:
                def wait(self):events.append('wait')
            payload={'hidden_states':torch.zeros(12,4)}
            if budget is not None:payload['__verification_budget']=budget
            if budget==5:payload['__verification_lengths']=torch.tensor([2,3],dtype=torch.int32)
            def recv(**kwargs):events.append('receive');return payload,[Handle()],[]
            group=NS(is_first_rank=False,is_last_rank=True,irecv_tensor_dict=recv)
            def execute(scheduler,intermediates,**kwargs):
                events.append('runner')
                assert set(intermediates.tensors)=={'hidden_states'}
                if budget is None:assert kwargs=={}
                else:
                    assert kwargs['pp_verification_budget']==budget
                    assert (kwargs['pp_verification_lengths'] is not None)==(budget==5)
                return Output()
            from contextlib import nullcontext
            state=NS(_pp_send_work=[],vllm_config=NS(
                compilation_config=NS(pass_config=NS(enable_sp=False)),
                parallel_config=NS(pipeline_parallel_size=6)),use_v2_model_runner=True,
                model_runner=NS(adaptive_verification=object() if enabled else None,
                    execute_model=execute,is_pooling_model=False),annotate_profile=lambda x:nullcontext())
            ns=dict(torch=torch,get_pp_group=lambda:group,get_tp_group=lambda:None,
                AsyncIntermediateTensors=Intermediate,IntermediateTensors=Intermediate,
                ModelRunnerOutput=Output,AsyncModelRunnerOutput=Output,NoneType=type(None),
                SchedulerOutput=Any)
            exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
            try:ns['execute_model'](state,NS(total_num_scheduled_tokens=12))
            except RuntimeError:
                assert not enabled and budget is not None
            else:
                assert enabled or budget is None
                assert events==(['receive','wait','runner'] if budget==5 else ['receive','runner'])
    print('WORKER_BUDGET_CONTROL 8 schema/ordering scenarios PASS (CPU boundary fakes)')


if __name__=='__main__':main()
