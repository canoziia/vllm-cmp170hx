#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real AdaptiveVerificationManager producer/consumer; no runner/PP mock.

Consumer accepts CPU total budget and allocator-produced GPU lengths, using
normal manager buffers. No model or communication; transport is tested elsewhere.
"""
import numpy as np
import torch
from types import SimpleNamespace as NS
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager


@torch.inference_mode()
def main():
    device=torch.device('cuda',0)
    def manager(mapping):
        states=NS(num_speculative_steps=5,device=device,max_num_reqs=8,
            max_num_batched_tokens=64,req_id_to_index=mapping,
            num_computed_tokens_np=np.full(8,10,dtype=np.int32),
            prefill_len=NS(np=np.full(8,4,dtype=np.int32)))
        return AdaptiveVerificationManager(states,torch.zeros(9,device=device,dtype=torch.int32),1,48)
    producer=manager({'a':3,'b':1,'p':5})
    consumer=manager({'a':2,'b':7,'p':0})
    ids=['a','b','p'];drafts={'a':[0]*5,'b':[0]*2}
    scheduled={'a':6,'b':3,'p':9}
    caps=np.array([5,2,0],dtype=np.int32);counts=np.array([6,3,9],dtype=np.int32)
    cpu_logits=np.array([0,6,9,10],dtype=np.int32)
    producer._confidence_probs.fill_(.5)
    producer._confidence_probs[3]=torch.tensor([.95,.8,.7,.6,.5],device=device)
    producer._confidence_probs[1]=torch.tensor([.9,.3,.2,.1,.1],device=device)
    pidx=torch.tensor([3,1,5],device=device)
    cidx=torch.tensor([2,7,0],device=device)
    ptrs=(consumer.query_start_loc.data_ptr(),consumer._cu_num_logits.data_ptr())
    for budget in range(8):
        for m in (producer,consumer):
            rows=m.get_num_tokens(scheduled,drafts,authoritative_budget=budget)
            assert rows==11+budget
            cpu_counts,cl=m.compact_batch(caps,counts,cpu_logits)
            assert cpu_counts.sum()==rows
            if budget==7:assert np.array_equal(cpu_counts,counts)
            if budget==0:assert cl.tolist()==[0,1,2,3]
        pl,pq,_=producer.reallocate_drafts(ids,pidx)
        snapshot=producer.partial_capacities(3,7,budget)
        assert (snapshot is None)==(budget in (0,7))
        wire=producer._batch_draft_capacity[:3].clone()
        cl,cq,_=consumer.reallocate_drafts(ids,cidx,authoritative_capacities=wire)
        torch.testing.assert_close(pl,cl,rtol=0,atol=0)
        torch.testing.assert_close(pq,cq,rtol=0,atol=0)
        actual=wire.tolist()
        assert sum(actual)==budget and all(0<=x<=cap for x,cap in zip(actual,caps))
        assert ptrs==(consumer.query_start_loc.data_ptr(),consumer._cu_num_logits.data_ptr())
        assert (torch.diff(cq[:4])==wire+torch.tensor([1,1,9],device=device)).all()
    for budget in (-1,8,True):
        try:consumer.get_num_tokens(scheduled,drafts,authoritative_budget=budget)
        except ValueError:pass
        else:raise AssertionError('invalid budget accepted')
    consumer.get_num_tokens(scheduled,drafts,authoritative_budget=3)
    try:consumer.reallocate_drafts(ids,cidx,authoritative_capacities=torch.zeros(2,device=device,dtype=torch.int32))
    except ValueError:pass
    else:raise AssertionError('bad wire shape accepted')
    # Original policy is still available: consumers need no cost table but the
    # producer's automatic mode must use it and respect the same logit limit.
    # Publish through the same manager interface that consumes PP feedback.
    producer.record_confidence_rows(torch.full((3,5),.75,device=device),pidx)
    producer.record_confidence_rows(torch.full((3,5),.25,device=device),pidx)
    torch.testing.assert_close(producer._confidence_probs[pidx],torch.full((3,5),.25,device=device))
    producer.cost_tables=(np.ones(9),np.ones(65))
    assert producer.get_num_tokens(scheduled,drafts)==18
    print('ADAPTIVE_MANAGER_AUTHORITY 8 budgets + original policy + schema gates PASS')


if __name__=='__main__':main()
