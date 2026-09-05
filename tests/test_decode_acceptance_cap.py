"""The checkpoint cap keeps a prefix of already verified tokens, conserving
sampled+rejected counts and preserving every prefill row/sentinel."""
import torch
from vllm.v1.worker.gpu.model_runner import _cap_decode_acceptance
positions=[]; counts=[]; prompts=[]
for p in range(120,140):
    for n in range(1,5):
        positions.append(p); counts.append(n); prompts.append(100)
positions += [127,128]; counts += [4,4]; prompts += [200,200]
size=len(positions)
idx=torch.arange(size,device='cuda',dtype=torch.int32)
computed=torch.tensor(positions,device='cuda',dtype=torch.int32)
prompt=torch.tensor(prompts,device='cuda',dtype=torch.int32)
sampled=torch.tensor(counts,device='cuda',dtype=torch.int32)
rejected=4-sampled.clone()
_cap_decode_acceptance[(size,)](idx,computed,None,prompt,sampled,rejected,UNIT=128)
torch.cuda.synchronize()
for i,(p,n,pl) in enumerate(zip(positions,counts,prompts)):
    expected=min(n,128-p%128) if p>=pl else n
    assert sampled[i].item()==expected
    assert (sampled[i]+rejected[i]).item()==4
print('DECODE_ACCEPTANCE_CAP PASS cases=',size)
