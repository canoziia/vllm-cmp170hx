#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Existing sampler's adaptive logprob contract with heterogeneous prefixes.
Actual flatten/top-k/logsoftmax kernels. Does not test the rejection decision.
"""
from types import SimpleNamespace as NS
import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


@torch.inference_mode()
def main():
    torch.manual_seed(63)
    cases=0
    for lengths in ([1,6,3],[6,1,3],[2,4,1],[1,1,1]):
        offsets=[0]
        for n in lengths:offsets.append(offsets[-1]+n)
        logits=torch.randn(sum(lengths),128,device='cuda')
        chosen=logits.argmax(-1)
        sampled=torch.zeros(3,6,device='cuda',dtype=torch.int64)
        for i,n in enumerate(lengths):sampled[i,:n]=chosen[offsets[i]:offsets[i+1]]
        count=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        cu=torch.tensor(offsets,device='cuda',dtype=torch.int32)
        sampler=RejectionSampler(NS(logprobs_mode='raw_logprobs'),
            NS(num_speculative_tokens=5,enable_adaptive_verification=True,
               rejection_sample_method='standard'),torch.device('cuda',0))
        result=sampler._get_logprobs_tensors(sampled,count,logits,cu,
            np.array([0,6,12,18],dtype=np.int32),3)
        if sum(lengths)>3:
            assert result.cu_num_generated_tokens is None
            assert result.cu_num_generated_tokens_tensor.tolist()==offsets
        else:
            assert result.cu_num_generated_tokens_tensor is None
        assert result.logprob_token_ids[:,0].tolist()==chosen.tolist()
        torch.testing.assert_close(result.logprobs[:,0],logits.log_softmax(-1).gather(1,chosen[:,None]).flatten())
        cpu=result.to_cpu_nonblocking();torch.cuda.synchronize()
        if sum(lengths)>3:assert cpu.cu_num_generated_tokens_tensor.tolist()==offsets
        cases+=1
    print(f'ADAPTIVE_LOGPROB_BOUNDARIES {cases} heterogeneous CUDA cases PASS')


if __name__=='__main__':main()
