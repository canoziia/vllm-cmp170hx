import torch
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
for page in (64,1032,16384):
 a=torch.arange(16*page,device='cuda',dtype=torch.int64).view(16,page)
 b=(a+100000).clone();aa=a.clone();bb=b.clone()
 copies=[KVCacheBlockCopy(1,10,0),KVCacheBlockCopy(2,11,0),KVCacheBlockCopy(3,12,1)]
 copy_kv_cache_blocks_inplace([a,b],16,copies,[[a,a],[b]])
 aa[10]=aa[1];aa[11]=aa[2];bb[12]=bb[3]
 torch.cuda.synchronize();assert torch.equal(a,aa);assert torch.equal(b,bb)
print('GROUP_COW_BYTE_EQUAL_AND_UNRELATED_PAGES_UNCHANGED PASS')
