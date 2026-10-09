#!/usr/bin/env python3
"""Profile defaults do not silently enable the new execution paths."""
import importlib.util,os,sys
from pathlib import Path
p=Path(sys.argv[1])/'vllm/dsv41_opt_profile.py'
s=importlib.util.spec_from_file_location('profile_under_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
keys=('VLLM_DSV41_VERIFICATION','VLLM_DSV41_BALANCED_COHORTS','VLLM_DSV41_FAST_METADATA','VLLM_DSV41_METADATA_GRAPHS')
for k in keys:assert k not in m.PROFILES['default']
original=dict(os.environ)
try:
 for profile in ('','0','off','default','verification'):
  os.environ.clear();os.environ['VLLM_DSV41_OPT_PROFILE']=profile;m.apply()
  for k in keys:assert os.environ.get(k)==('1' if profile=='verification' else None),(profile,k)
  assert 'VLLM_DSV41_HISTORY_POLICY' not in os.environ
  assert all('enable_adaptive_verification' not in k.lower() for k in os.environ)
 os.environ.clear();os.environ['VLLM_DSV41_OPT_PROFILE']='verification';os.environ[keys[0]]='0';m.apply();assert os.environ[keys[0]]=='0'
finally:os.environ.clear();os.environ.update(original)
print('PASS default/off inert; verification opt-in; explicit component override retained')
