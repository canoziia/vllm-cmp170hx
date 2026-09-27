#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute actual V2 output relay block with a reused FULL graph result."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('runner',type=Path)
    path=p.parse_args().runner
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GPUModelRunner')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='execute_model')
    statement=next(x for x in method.body if isinstance(x,ast.If) and ast.unparse(x.test)=='not self.is_last_pp_rank' and 'relay_aux_hidden_states' in ast.unparse(x))
    class Intermediate:
        def __init__(self,tensors):self.tensors=tensors
    output=Intermediate({'hidden_states':torch.zeros(12,4)})
    state=NS(is_last_pp_rank=False,pp_handler=NS(relay_aux_hidden_states=lambda recv,out:out),
        adaptive_verification=NS(partial_capacities=lambda *args:torch.tensor([2,3])))
    ns=dict(IntermediateTensors=Intermediate,self=state,output_intermediate_tensors=output,
            model_inputs={'intermediate_tensors':None},input_batch=NS(num_reqs=2),scheduled_drafts=10)
    fn=ast.FunctionDef(name='relay',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='needs_pp_budget'),ast.arg(arg='pp_verification_budget')],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[statement],decorator_list=[])
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    first=ns['relay'](True,5)
    assert first is not output and set(first.tensors)=={'hidden_states','__verification_budget','__verification_lengths'}
    second=ns['relay'](False,None)
    assert second is output and set(second.tensors)=={'hidden_states'}
    assert set(output.tensors)=={'hidden_states'}
    third=ns['relay'](True,0)
    assert third.tensors['__verification_budget']==0
    assert set(output.tensors)=={'hidden_states'}
    print('PP_PERSISTENT_OUTPUT budget/no-budget/reuse actual runner relay PASS')


if __name__=='__main__':main()
