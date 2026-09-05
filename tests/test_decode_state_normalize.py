import torch,importlib.util,sys
import os
p=os.environ.get('MAMBA_UTILS_PATH','/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py')
s=importlib.util.spec_from_file_location('normtest',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
dev='cuda';cases=0
for dtype in (torch.bfloat16,torch.float32):
 for ds in (False,True):
  for accepted in (1,2,3,4):
   width,dim=7,128
   conv=torch.arange(8*width*dim,device=dev,dtype=torch.float32).reshape(8,dim,width) if ds else torch.arange(8*width*dim,device=dev,dtype=torch.float32).reshape(8,width,dim)
   conv=conv.to(dtype);temp=torch.arange(8*128*128,device=dev,dtype=torch.float32).reshape(8,128,128).to(dtype)
   original_conv=conv.clone();original_temp=temp.clone()
   bt=torch.arange(8,device=dev,dtype=torch.int32).reshape(1,8)
   i64=lambda v:torch.tensor(v,device=dev,dtype=torch.int64)
   i32=lambda v:torch.tensor(v,device=dev,dtype=torch.int32)
   acc=i32([accepted]);out=acc.clone();idx=i32([2]);computed=i32([2080]);mapping=i32([0])
   m.postprocess_mamba_fused_kernel[(1,2,1)](acc,idx,None,computed,None,i64([bt.data_ptr()]),8,i64([conv.data_ptr(),temp.data_ptr()]),i64([conv.stride(0)*conv.element_size(),temp.stride(0)*temp.element_size()]),i32([conv.element_size(),temp.element_size()]),i64([dim,128*128]),i32([width,0]),i32([0,0]),i32([dim if ds else 0,0]),i64([width*conv.element_size() if ds else 0,0]),out,mapping,1,block_size=832,COPY_BLOCK_SIZE=1024,CONV_STATE_DIM_FIRST=ds,HAS_IDX_MAPPING=True,PRECOMPUTED_NEW_COMPUTED=True,DECODE_CHECKPOINT_UNIT=32,TEMPORAL_TILES=1)
   torch.cuda.synchronize();bias=accepted-1
   assert torch.equal(temp[2],original_temp[2+bias])
   if ds:assert torch.equal(conv[2,:,:width-bias],original_conv[2,:,bias:])
   else:assert torch.equal(conv[2,:width-bias],original_conv[2,bias:])
   assert out.item()==1
   cases+=1
print('DECODE_NORMALIZE_BYTE_EQUAL',cases)
