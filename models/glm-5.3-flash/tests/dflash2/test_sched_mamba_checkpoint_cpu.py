"""CPU regression: execute actual scheduler method extracted from pinned source."""
import ast,sys,unittest
from types import SimpleNamespace as NS
class MambaSpec:
    def __init__(self,block_size):self.block_size=block_size
p=sys.argv.pop(1);tree=ast.parse(open(p).read());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_mamba_block_aligned_split')
fn.decorator_list=[]
for a in fn.args.args:a.annotation=None
fn.returns=None
ns=dict(MambaSpec=MambaSpec,get_mamba_prefill_checkpoint_position=lambda *a,**k:0,is_mamba_prefill_checkpoint_valid=lambda **k:False)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),p,'exec'),ns)
def fixture(mamba_bs=5120):
 return NS(cache_config=NS(block_size=1024),kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=MambaSpec(mamba_bs))]),use_eagle_block_drop=False,hash_block_size=1024,mamba_has_prefill_checkpoint_blocks=False,mamba_prefill_checkpoint_alignment=64,max_num_scheduled_tokens=5184,scheduler_config=NS(long_prefill_token_threshold=0),mamba_partial_cache_hit=False,mamba_fine_grained_prefix_cache=False)
def run(s,start,end,budget):
 req=NS(num_computed_tokens=start,num_prompt_tokens=end,num_tokens=end,shared_prefix_boundary=0)
 return ns['_mamba_block_aligned_split'](s,req,budget)
class Tests(unittest.TestCase):
 def test_chunks(self):
  s=fixture()
  for suffix in [16,504]:
   self.assertEqual(run(s,97280,102400+suffix,5176),5120)
   self.assertEqual(run(s,102400,102400+suffix,suffix),suffix)
 def test_8192(self):
  s=fixture(8192);s.max_num_scheduled_tokens=8256
  self.assertEqual(run(s,0,107000,8248),8192)
 def test_budget_must_include_draft_slots(self):
  self.assertEqual(run(fixture(),0,107000,5112),0)
 def test_without_mamba_unchanged(self):
  s=fixture();s.kv_cache_config.kv_cache_groups=[]
  self.assertEqual(run(s,0,107000,5112),4096)
 def test_midblock(self):
  self.assertEqual(run(fixture(),101376,102416,1040),1024)
if __name__=='__main__':unittest.main()
