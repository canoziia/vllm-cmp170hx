#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scoped opt-in policy on the actual SM80 platform, no hardware mocks."""
from types import SimpleNamespace as NS
import torch
from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
    DeepseekV41AmpereMLAMetadataBuilder, DeepseekV41AmpereSWAMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer


def main():
    assert torch.cuda.get_device_capability()==(8,0)
    assert indexer.DeepseekV41IndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert not indexer.DeepseekV32IndexerBackend.supports_device_cpu_query_lens_mismatch()
    for depth in (1,5):
        for enabled in (False,True):
            config=NS(num_speculative_tokens=depth,
                model_config=NS(architectures=['DeepseekV41ForCausalLM']),
                speculative_config=NS(enable_adaptive_verification=enabled))
            for builder in (DeepseekV41AmpereMLAMetadataBuilder,DeepseekV41AmpereSWAMetadataBuilder):
                expected=AttentionCGSupport.ALWAYS if enabled else AttentionCGSupport.UNIFORM_BATCH
                assert builder.get_cudagraph_support(config,None)==expected
            assert indexer._use_flattening(config)==(enabled or depth==5)
            config.model_config.architectures=['OtherModel']
            assert indexer._use_flattening(config)==(depth==5)
    print('SM80_BACKEND_POLICY scoped V4.1 opt-in/default-path checks PASS')


if __name__=='__main__':main()
