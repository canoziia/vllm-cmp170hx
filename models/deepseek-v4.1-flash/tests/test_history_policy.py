#!/usr/bin/env python3
"""Small stdlib acceptance/cohort policy checks against applied source."""
import importlib.util,sys,logging,ast
from pathlib import Path
from types import SimpleNamespace as NS
p=Path(sys.argv[1])/'vllm/v1/core/sched/dsv41_history_policy.py'
s=importlib.util.spec_from_file_location('history_policy_test',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
h=m.History();h.observe(2,2);assert h.risk==[1,1,0,0,0] and h.success==[1,1,0,0,0]
h=m.History();h.observe(5,2);assert h.risk==[1,1,1,0,0] and h.success==[1,1,0,0,0]
assert h.estimate()[0][0]==1
pol=m.HistoryPolicy(logging.getLogger('test'))
req=[NS(request_id=str(i)) for i in range(2)]
for i in range(150):
 for r in req:pol.observe(r.request_id,5,5)
assert all(pol.choose(req,8)[r.request_id]==5 for r in req)
req[0].verification_draft_limit=0;req[1].verification_draft_limit=2
assert pol.choose(req,8)=={}
assert pol.last_plan=={'0':0,'1':2}
pol.forget('0');assert '0' not in pol.history and '0' not in pol.last_plan
assert pol.cost(8,[1,5])>pol.cost(8,[3,3]) # equal rows, different graph path
assert pol.cost(32,[2,5,2,5,2,5])>0
# Feedback recorded before _update_request_with_output/EOS slicing, gates
# stale and grammar-invalid feedback. Auto limit is not stored as manual.
src=(p.parent/'scheduler.py').read_text();ast.parse(src)
assert src.index('self.dsv41_history_policy.observe(')<src.index('new_token_ids = generated_token_ids')
assert 'history_limits.get(request.request_id)' in src
assert 'spec.enable_adaptive_verification' in src
print('PASS at-risk censoring; high acceptance k5; manual k0/2; cleanup; ragged cost; integration gates')
