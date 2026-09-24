#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract actual warmup_kernels control flow and test paired PP relay state.
No model load; asserts both success/failure restore flags on all ranks.
"""
import argparse,ast
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    path=parser.parse_args().source
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='warmup_kernels')
    fn.decorator_list=[]
    for has_pp in (False,True):
        for enabled in (False,True):
            for failure in (False,True):
                handler=NS(relay_draft_confidences=enabled) if has_pp else None
                manager=object() if enabled else None
                runner=NS(vllm_config=NS(is_mm_encoder_only=False),
                          adaptive_verification=manager,pp_handler=handler)
                seen=[]
                def warmup(current,execute,sample):
                    seen.append((current.adaptive_verification,
                                 handler.relay_draft_confidences if handler else None))
                    if failure:raise ArithmeticError('injected warmup failure')
                ns={'_warmup_kernels':warmup}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
                try:ns['warmup_kernels'](runner,lambda x:x,lambda x:x)
                except ArithmeticError:assert failure
                else:assert not failure
                assert seen==[(None,False if has_pp else None)]
                assert runner.adaptive_verification is manager
                if has_pp:assert handler.relay_draft_confidences is enabled
    print('ADAPTIVE_WARMUP_RELAY 8 success/failure/off/PP lifecycle cases PASS')


if __name__=='__main__':main()
