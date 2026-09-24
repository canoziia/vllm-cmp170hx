#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual configuration validator with minimal attribute fixture, on SM80."""
from types import SimpleNamespace as NS
from vllm.config import VllmConfig,CUDAGraphMode


def config():
    return NS(lora_config=None,compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
        model_config=NS(architectures=['DeepseekV41ForCausalLM']),
        speculative_config=NS(enable_adaptive_verification=True,method='dspark',draft_sample_method='greedy'),
        parallel_config=NS(pipeline_parallel_size=6,tensor_parallel_size=1,data_parallel_size=1,
            prefill_context_parallel_size=1,decode_context_parallel_size=1,use_ubatching=False))


def main():
    VllmConfig._validate_adaptive_verification(config())
    for attr,value in [('tensor_parallel_size',2),('data_parallel_size',2),
        ('prefill_context_parallel_size',2),('decode_context_parallel_size',2),('use_ubatching',True)]:
        c=config();setattr(c.parallel_config,attr,value)
        try:VllmConfig._validate_adaptive_verification(c)
        except ValueError:pass
        else:raise AssertionError('unsupported config accepted: '+attr)
    c=config();c.speculative_config.enable_adaptive_verification=False
    c.parallel_config.tensor_parallel_size=2
    VllmConfig._validate_adaptive_verification(c)
    print('ADAPTIVE_PP_CONFIG tested scope and feature-OFF checks PASS')


if __name__=='__main__':main()
