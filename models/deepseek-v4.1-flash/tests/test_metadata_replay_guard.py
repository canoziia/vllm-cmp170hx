#!/usr/bin/env python3
"""CPU test: metadata replay can only consume exact uniform live batches."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace as NS
path=Path(sys.argv[1])/'vllm/v1/worker/gpu/model_runner.py'
tree=ast.parse(path.read_text())
node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'metadata_graph is not None' in ast.unparse(n.test))
expression=compile(ast.Expression(node.test),str(path),'eval')
for dummy in (False,True):
 for adaptive in (None,object()):
  for prefill in (False,True):
   for reqs in (1,2):
    for rows in (5,6):
     for width in (None,5,6):
      env=dict(metadata_graph=object(),dummy_run=dummy,self=NS(adaptive_verification=adaptive),input_batch=NS(has_prefill=prefill,num_reqs=reqs,num_tokens=rows),batch_desc=NS(num_reqs=1,num_tokens=6,uniform_token_count=6),uniform_tok_count=width)
      actual=eval(expression,env)
      assert actual == (not dummy and adaptive is None and not prefill and reqs==1 and rows==6 and width==6),env
print('PASS: prefill/ragged/padded/dummy/adaptive batches cannot replay static metadata')
