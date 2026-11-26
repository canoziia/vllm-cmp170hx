# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import typing
from collections.abc import Callable, Iterable
from functools import partial
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS, NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
    TrtLlmMxfp4Mxfp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    drop_checkpoint_cache,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MixtureOfExperts,
    MegaGateRoutingMetadata,
    _use_mega_gate,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    DeepseekV4MegaAttnAttention,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    NgramHashState,
    can_share_engram_tables,
    gather_engram_hashes,
)
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask
from .ops.mhc import (
    MHC_OVERLAP_MAX_TOKENS,
    ampere_all_reduce_ready,
    ampere_mhc_max_tokens,
    ampere_projection_splits,
    boundary_all_reduce,
    hc_expand,
    init_ampere_all_reduce,
    init_mhc_all_reduce,
    mhc_all_reduce_max_tokens,
    mhc_boundary_weight,
    mhc_pre_delayed_ampere,
    mhc_pre_delayed_overlap,
    mhc_shifted_post_pre,
    prepare_ampere_mhc,
    supports_ampere_all_reduce,
    supports_ampere_mhc_overlap,
    supports_mhc_all_reduce,
    supports_mhc_overlap,
    use_ampere_mhc,
)
from .pipeline import PipelineBoundary, PipelineRelay, get_pipeline_boundaries
from .quantization import DeferringMarlinExperts
from .specialized_decode import records_decode_metadata, specialized_decode_config
from .specialized_decode.ampere import is_ampere
from .specialized_decode.hopper import is_hopper

if typing.TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    from .specialized_decode.runtime.decode import DecodeStep, SpecializedDecode

logger = init_logger(__name__)

_PREFILL_AR_STATE: dict[
    int,
    tuple[
        torch.cuda.Stream,
        list[torch.cuda.Event],
        torch.Tensor,
    ],
] = {}


def _get_prefill_ar_state(
    dev: torch.device,
    min_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple[
    torch.cuda.Stream,
    list[torch.cuda.Event],
    torch.Tensor,
]:
    idx = dev.index or 0
    st = _PREFILL_AR_STATE.get(idx)
    alloc_m = max(min_tokens, 2048)
    if st is None:
        with torch.cuda.device(idx):
            comm_stream = torch.cuda.Stream(device=dev)
            evs = [torch.cuda.Event() for _ in range(8)]
            comm_buf = torch.empty((alloc_m, hidden_size), dtype=dtype, device=dev)
            st = (comm_stream, evs, comm_buf)
        _PREFILL_AR_STATE[idx] = st
    else:
        comm_stream, evs, comm_buf = st
        if (
            comm_buf.shape[0] < min_tokens
            or comm_buf.shape[1] != hidden_size
            or comm_buf.dtype != dtype
        ):
            with torch.cuda.device(idx):
                comm_stream.synchronize()
                alloc_m = max(min_tokens, comm_buf.shape[0])
                comm_buf = torch.empty((alloc_m, hidden_size), dtype=dtype, device=dev)
                st = (comm_stream, evs, comm_buf)
            _PREFILL_AR_STATE[idx] = st
    return st


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        super().__init__(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=use_sequence_parallel,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            num_hash_layers=0,
            reduce_results=reduce_results,
            image_sentinel_lo=IMAGE_SENTINEL_BASE_ID,
        )

    def defer_finalize(self) -> None:
        """Leave the routed top-k reduction to the next fused all-reduce + mHC.

        Only experts that can stop after GEMM2 defer: the modular TRTLLM MXFP4
        experts (SM100) and the SM80 and SM90 Marlin experts. The fused kernel
        takes as many tokens as its boundaries do.
        """
        moe_config = self.experts.moe_config
        experts_cls = getattr(
            self.experts.routed_experts.quant_method, "experts_cls", None
        )
        if (
            experts_cls
            in (
                TrtLlmMxfp4ExpertsModular,
                TrtLlmMxfp4Mxfp8ExpertsModular,
                DeferringMarlinExperts,
            )
            and self.experts.routed_scaling_factor == 1.0
        ):
            max_num_tokens = mhc_all_reduce_max_tokens()
            moe_config.defer_moe_finalize(max_num_tokens)
            if moe_config.use_deferred_moe_finalize:
                logger.info_once(
                    "DSV4.1 mHC: MoE top-k finalize fused into the all-reduce for "
                    "up to %d tokens.",
                    max_num_tokens,
                )

    def defers_finalize(self, num_tokens: int) -> bool:
        return (
            not self.use_native_mega_moe
            and self.experts.moe_config.should_defer_moe_finalize(num_tokens)
        )

    def forward_unfinalized(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> MoEOutput:
        """`forward` with the routed top-k reduction and all-reduce left open."""
        # The runner's custom op returns tensors only, so run its body directly.
        shared_output, routed = _unpack(
            self.experts._forward_impl(
                hidden_states, hidden_states, hidden_states, input_ids
            )
        )
        assert shared_output is not None
        assert isinstance(routed, UnfinalizedMoEOutput)
        return MoEOutput(routed=routed, shared_output=shared_output)


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    The generic CUDA backend selector does not instantiate DSv4 layers directly,
    so map generic sparse-MLA choices to the DSv4-specialized attention class.
    Without an explicit backend: SM12 takes FlashInfer, SM100 takes mega
    attention where the topology allows it, SM90 takes the FlashMLA layer
    with CuTe DSL decode attention, and everything else keeps the FlashMLA
    path.
    """
    backend = vllm_config.attention_config.backend
    if is_ampere():
        # No FlashMLA or DeepGEMM build runs on SM80; its own kernels do.
        if backend is not None or (
            vllm_config.parallel_config.prefill_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 on SM80 runs its own sparse-MLA kernels: it takes "
                "no attention backend choice and no prefill context parallelism."
            )
        from .specialized_decode.ampere.attention import DeepseekV41AmpereAttention

        return DeepseekV41AmpereAttention
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        if backend not in (
            None,
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP requires FlashMLA sparse attention."
            )
        return DeepseekV4FlashMLAAttention
    if backend is None and is_hopper():
        # Decode without FlashMLA's 64-head padding; an explicit FlashMLA
        # backend keeps the stock decode.
        from .specialized_decode.hopper.attention import DeepseekV41HopperAttention

        return DeepseekV41HopperAttention
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV41 for DeepSeek V4.1 FlashInfer "
            "sparse MLA."
        )
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
    ):
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV4FlashInferSM120Attention
        return DeepseekV4FlashInferMLAAttention
    if backend is AttentionBackendEnum.FLASHMLA_MEGA_ATTN_DSV41:
        # FlashMLA mega attention reads the grouped ``wo_a`` weight directly
        # from its fused quantized output. LoRA wraps that layer after model
        # construction, so use the split-KV path whose grouped projection
        # explicitly applies the adapter delta.
        if vllm_config.lora_config is not None:
            return DeepseekV4FlashMLAAttention
        return DeepseekV4MegaAttnAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
    ):
        return DeepseekV4FlashMLAAttention

    if device_capability is not None and device_capability.major == 12:
        return DeepseekV4FlashInferSM120Attention
    # Mega attention is the SM100 default: it fuses Q RoPE, sparse attention,
    # the output's inverse RoPE and its FP8 cast into one launch, and brings
    # the 288 B NVFP4 compressed record -- the format the reference
    # implementation itself stores. It declines topologies it cannot serve
    # (non-SM100, TP that leaves fewer than WV_GROUP_SIZE heads per wo_a
    # group, a build without the kernel), which then fall through to FlashMLA.
    if vllm_config.lora_config is None and DeepseekV4MegaAttnAttention.is_available_for(
        vllm_config
    ):
        return DeepseekV4MegaAttnAttention
    return DeepseekV4FlashMLAAttention


def _yoco_prefill_only_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_YOCO_PREFILL_ONLY", "0") == "1"


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    moe_needs_token_sharded_input = (
        vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    )
    return (
        parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (moe_needs_token_sharded_input or parallel_config.data_parallel_size > 1)
    )


def _shard_fused_wqa_wkv(vllm_config: VllmConfig) -> bool:
    """Whether large batches shard fused_wqa_wkv by token and all-gather the output."""
    ca_comm = getattr(get_tp_group().device_communicator, "ca_comm", None)
    return (
        current_platform.is_device_capability(90)
        and not _use_sequence_parallel(vllm_config)
        and vllm_config.lora_config is None
        and not vllm_config.parallel_config.use_ubatching
        and not envs.VLLM_BATCH_INVARIANT
        and ca_comm is not None
        and not ca_comm.disabled
        and ca_comm.fully_connected
    )


def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
    """Set up the fused `wo_b` GEMM + reduce-scatter when opted in.

    Sequence parallel is the only topology where `wo_b` ends in a
    reduce-scatter, so the kernel is bound to RS mode. The workspace is a
    process-wide singleton shared with any other model code in this worker
    (the DSpark drafter reuses it), and its NVLink multicast rendezvous is
    collective, so every TP rank must take the same decision here.
    """
    if not (use_sequence_parallel and envs.VLLM_ENABLE_GEMM_RS):
        return False
    if vllm_config.lora_config is not None:
        # The fused kernel reads wo_b's base weight and has no LoRA epilogue.
        logger.warning_once(
            "DeepSeek-V4.1 GEMM-RS is disabled because LoRA is enabled."
        )
        return False

    # The kernel module pulls in cute_dsl, so import it only once opted in.
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
        maybe_init_gemm_rs_ar,
    )

    hidden_size = vllm_config.model_config.hf_config.hidden_size
    if not maybe_init_gemm_rs_ar(vllm_config, N=hidden_size, all_reduce=False):
        return False
    logger.info_once("To disable DeepSeek-V4.1 GEMM-RS, set VLLM_ENABLE_GEMM_RS=0.")
    return True


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
        engram_prefetch_stream: torch.cuda.Stream | None = None,
        run_gemm_rs: bool = False,
        mhc_stream: torch.cuda.Stream | None = None,
        fuse_mhc_all_reduce: bool = False,
        kv_write_only: bool = False,
        replicated_attn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.mhc_stream = mhc_stream
        self.fuse_mhc_all_reduce = fuse_mhc_all_reduce
        self.kv_write_only = kv_write_only
        self.replicated_attn = replicated_attn
        # SM80 runs every mHC boundary through its CuTe DSL kernels, SM90 the
        # small ones, with the coefficients on mhc_stream in FULL graphs of up
        # to this many tokens.
        self.ampere_mhc = use_ampere_mhc()
        self.mhc_overlap_max_tokens = (
            ampere_mhc_max_tokens() if self.ampere_mhc else MHC_OVERLAP_MAX_TOKENS
        )
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                    prefetch_stream=engram_prefetch_stream,
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
            replicated_attn=replicated_attn,
        )
        if replicated_attn:
            self.attn.wo_b.reduce_results = False
            self.attn.shard_fused_wqa_wkv = False
        else:
            if self.use_sequence_parallel or fuse_mhc_all_reduce:
                self.attn.wo_b.reduce_results = False
                if run_gemm_rs:
                    # Binds only when wo_b's kernel and shape qualify; otherwise
                    # forward keeps the separate reduce-scatter below.
                    self.attn.bind_gemm_rs()
            self.attn.shard_fused_wqa_wkv = _shard_fused_wqa_wkv(vllm_config)
        if kv_write_only:
            self.ffn = PPMissingLayer()
            self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
            self.ffn_norm = PPMissingLayer()
        else:
            self.ffn = DeepseekV4MoE(
                vllm_config,
                prefix=f"{prefix}.ffn",
                use_sequence_parallel=self.use_sequence_parallel,
                reduce_results=not fuse_mhc_all_reduce,
            )

            self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
            self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        if vllm_config.kernel_config.enable_jit_warmup and current_platform.is_cuda():
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_fused_post_pre_splits,
            )
            from vllm.model_executor.kernels.mhc.warmup import (
                HC_PRENORM_GEMM_KERNEL,
                MHC_PRE_NORM_KERNEL,
                compute_mhc_pre_num_splits,
            )
            from vllm.utils.deep_gemm import is_deep_gemm_supported

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if self.use_sequence_parallel:
                tp_size = vllm_config.parallel_config.tensor_parallel_size
                max_tokens = (max_tokens + tp_size - 1) // tp_size
            if is_deep_gemm_supported():
                hc_dim = self.hc_mult * self.hidden_size
                is_broadcast_layer = (
                    get_pp_group().is_first_rank and extract_layer_index(prefix) == 0
                )
                for k in (hc_dim, self.hidden_size)[: 1 + is_broadcast_layer]:
                    HC_PRENORM_GEMM_KERNEL.register_warmup(
                        max_tokens=max_tokens,
                        n=self.hc_mult * (2 + self.hc_mult),
                        k=k,
                        num_splits=compute_mhc_pre_num_splits,
                    )
            # The epilogue compiles per projection width and per pre-mix mode.
            # The first layer projects the broadcast embedding, so it reads one
            # hidden_size-wide row and selects stream zero; every later sublayer
            # projects the full hc stream, and collapses it with the pre-mix the
            # previous sublayer carried in. Those later shapes also arrive
            # through the fused post + pre-norm GEMM, which picks split-k
            # factors the token sweep below never produces, and on SM80
            # through the split-k prefill projection.
            broadcast_embedding = {
                "rms_numel": self.hidden_size,
                "use_pre_mix_in": False,
                "extra_splits": ampere_projection_splits(),
            }
            hc_stream = {
                "rms_numel": self.hc_mult * self.hidden_size,
                "extra_splits": (
                    *mhc_fused_post_pre_splits(self.hidden_size, self.hc_mult),
                    *ampere_projection_splits(),
                ),
            }
            variants = [
                broadcast_embedding,
                {**hc_stream, "use_pre_mix_in": False},
                {**hc_stream, "use_pre_mix_in": True},
            ]
            if vllm_config.speculative_config is not None:
                # Draft setups read aux hidden states out of the same collapse,
                # which is its own specialization of the kernel.
                variants.append(
                    {**hc_stream, "use_pre_mix_in": True, "write_aux": True}
                )
            # Shifted mHC splits the epilogue in two: the input collapse on the
            # caller stream and coefficient generation on a side stream.
            split_modes = (
                ("fused", "stats", "input")
                if mhc_stream is not None and not self.ampere_mhc
                else ("fused",)
            )
            for variant in variants:
                for split_mode in split_modes:
                    MHC_PRE_NORM_KERNEL.register_warmup(
                        split_mode=split_mode,
                        max_tokens=(
                            max_tokens
                            if split_mode == "fused"
                            else min(max_tokens, MHC_OVERLAP_MAX_TOKENS)
                        ),
                        hidden_size=self.hidden_size,
                        rms_eps=self.rms_norm_eps,
                        hc_pre_eps=self.hc_eps,
                        hc_sinkhorn_eps=self.hc_eps,
                        hc_post_mult_value=self.hc_post_alpha,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        norm_eps=self.rms_norm_eps,
                        hc_mult=self.hc_mult,
                        **variant,
                    )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if not kv_write_only:
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(
                    (mix_hc, hc_dim),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.hc_ffn_base = nn.Parameter(
                torch.empty(
                    mix_hc,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.hc_ffn_scale = nn.Parameter(
                torch.empty(
                    3,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        # The mixer weights laid out for the CuTe DSL mHC boundary, filled after
        # loading; None where that kernel cannot run.
        self.register_buffer(
            "hc_attn_boundary_weight",
            mhc_boundary_weight(
                self.hidden_size, self.hc_mult, self.hc_attn_fn.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "hc_ffn_boundary_weight",
            None
            if kv_write_only
            else mhc_boundary_weight(
                self.hidden_size, self.hc_mult, self.hc_attn_fn.device
            ),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        """Lay out the mHC mixer weights for the CuTe DSL boundary kernel (see
        ops/mhc_boundary.py), in place so captured graphs see reloads."""
        if self.hc_attn_boundary_weight is None:
            return
        from .ops.mhc_boundary import prepare_boundary_weight

        self.hc_attn_boundary_weight.copy_(prepare_boundary_weight(self.hc_attn_fn))
        if not self.kv_write_only and self.hc_ffn_boundary_weight is not None:
            self.hc_ffn_boundary_weight.copy_(prepare_boundary_weight(self.hc_ffn_fn))

    def forward(
        self,
        x: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
    ) -> tuple[
        torch.Tensor | MoEOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        previous_aux: torch.Tensor | None = None
        mhc_stream = self.mhc_stream
        if mhc_stream is not None and (
            in_piecewise_cudagraph()
            or not 0 < positions.shape[0] <= self.mhc_overlap_max_tokens
            or not torch.cuda.is_current_stream_capturing()
        ):
            # Use overlap only in FULL graphs; eager warmup initializes Mega mHC.
            mhc_stream = None
        mhc_pre: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
        if self.ampere_mhc:
            mhc_pre = partial(mhc_pre_delayed_ampere, stream=mhc_stream)
        else:
            mhc_pre = (
                partial(mhc_pre_delayed_overlap, stream=mhc_stream)
                if mhc_stream is not None
                else mhc_pre_delayed_tilelang
            )
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        if residual is None:
            assert isinstance(x, torch.Tensor)
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                residual = hc_expand(x, self.hc_mult)
                post_mix, res_mix, x, attn_pre = mhc_pre(
                    residual,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    x=x,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
            else:
                residual = x
                if self.engram is not None and engram_hashes is not None:
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        engram_mask,
                    )
                post_mix, res_mix, x, attn_pre = mhc_pre(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injection happens between the previous sublayer's post
            # and this block's pre, on the full hc stream, so the mix
            # coefficients see the injected stream. The injection also keeps
            # the post out of the pre-norm GEMM's fused prologue. The layer
            # before an engram layer finalizes its own MoE.
            assert isinstance(x, torch.Tensor)
            if self.fuse_mhc_all_reduce:
                x = boundary_all_reduce(x)
            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            residual = self.engram(
                previous_post,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
            del previous_post
            post_mix, res_mix, x, attn_pre = mhc_pre(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            # The collapse already reads the post-mapped streams, so the mean
            # aux consumers want comes out of the same kernel.
            residual, post_mix, res_mix, x, attn_pre, aux = mhc_shifted_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
                capture_aux=capture_previous_aux,
                stream=mhc_stream,
                reduce_results=self.fuse_mhc_all_reduce,
                boundary_weight=self.hc_attn_boundary_weight,
                defer_coeff=(
                    mhc_stream is not None
                    and self.ampere_mhc
                    and positions.shape[0] <= 192
                    and self.fuse_mhc_all_reduce
                    and not self.use_sequence_parallel
                ),
            )
            if capture_previous_aux:
                previous_aux = aux

        if self.use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.attn(positions, x, None)
        if post_mix is None:
            from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import (
                finish_pending_mhc_coeff,
            )

            post_mix, res_mix, attn_pre = finish_pending_mhc_coeff(residual.device)
        # With GEMM-RS bound, the attention output is already this rank's
        # sequence-parallel shard (see DeepseekV4Attention._wo_b_proj).
        if self.use_sequence_parallel and self.attn.gemm_rs is None:
            x = sp_reduce_scatter(x)

        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
            stream=mhc_stream,
            reduce_results=self.fuse_mhc_all_reduce and not self.replicated_attn,
            boundary_weight=self.hc_ffn_boundary_weight,
            defer_coeff=(mhc_stream is not None and self.ampere_mhc and x.shape[0] <= 192 and self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0])),
        )
        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):
            # The next layer's first boundary finalizes it with the all-reduce.
            x = self.ffn.forward_unfinalized(x, input_ids)
        else:
            x = self.ffn(x, input_ids, mega_gate_metadata)
        if post_mix is None:
            from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import (
                finish_pending_mhc_coeff,
            )

            post_mix, res_mix, ffn_pre = finish_pending_mhc_coeff(residual.device)
        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    def forward_attn_only(
        self,
        x: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the attention half of the decoder layer up to FFN entry (for DSpark tail split)."""
        assert self.engram is None and not self.use_sequence_parallel
        mhc_stream = self.mhc_stream
        if mhc_stream is not None and (
            in_piecewise_cudagraph()
            or not 0 < positions.shape[0] <= self.mhc_overlap_max_tokens
            or not torch.cuda.is_current_stream_capturing()
        ):
            mhc_stream = None
        if self.ampere_mhc:
            mhc_pre = partial(mhc_pre_delayed_ampere, stream=mhc_stream)
        else:
            mhc_pre = (
                partial(mhc_pre_delayed_overlap, stream=mhc_stream)
                if mhc_stream is not None
                else mhc_pre_delayed_tilelang
            )
        if residual is None:
            assert isinstance(x, torch.Tensor)
            residual = x
            post_mix, res_mix, x, attn_pre = mhc_pre(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            residual, post_mix, res_mix, x, attn_pre, _ = mhc_shifted_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
                capture_aux=False,
                stream=mhc_stream,
                reduce_results=self.fuse_mhc_all_reduce,
                boundary_weight=self.hc_attn_boundary_weight,
                defer_coeff=(
                    mhc_stream is not None
                    and self.ampere_mhc
                    and positions.shape[0] <= 192
                    and self.fuse_mhc_all_reduce
                    and not self.use_sequence_parallel
                ),
            )
        x = self.attn(positions, x, None)
        if post_mix is None:
            from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import (
                finish_pending_mhc_coeff,
            )

            post_mix, res_mix, attn_pre = finish_pending_mhc_coeff(residual.device)
        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        return x, residual, post_mix, res_mix, attn_pre

    def forward_ffn_only(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        attn_pre: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | MoEOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the token-wise FFN half of the decoder layer (for DSpark tail split)."""
        mhc_stream = self.mhc_stream
        if mhc_stream is not None and (
            in_piecewise_cudagraph()
            or not 0 < x.shape[0] <= self.mhc_overlap_max_tokens
            or not torch.cuda.is_current_stream_capturing()
        ):
            mhc_stream = None
        residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
            stream=mhc_stream,
            reduce_results=self.fuse_mhc_all_reduce and not self.replicated_attn,
            boundary_weight=self.hc_ffn_boundary_weight,
            defer_coeff=(
                mhc_stream is not None
                and self.ampere_mhc
                and x.shape[0] <= 192
                and self.fuse_mhc_all_reduce
                and self.ffn.defers_finalize(x.shape[0])
            ),
        )
        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):
            x = self.ffn.forward_unfinalized(x, input_ids)
        else:
            x = self.ffn(x, input_ids, None)
        if post_mix is None:
            from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import (
                finish_pending_mhc_coeff,
            )

            post_mix, res_mix, ffn_pre = finish_pending_mhc_coeff(residual.device)
        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        return x, residual, post_mix, res_mix, ffn_pre

    def write_kv(
        self,
        x: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Write the KV `forward` would for the replay layers' inputs, skipping
        the rest of the layer; the replay batch then reruns its rows."""
        # The replay batch rules out Engram and sequence parallel here.
        assert self.engram is None and not self.use_sequence_parallel
        *_, x, _, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
            reduce_results=self.fuse_mhc_all_reduce,
            boundary_weight=self.hc_attn_boundary_weight,
        )
        self.attn.forward_kv(positions, x)


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        pp_size = self.parallel_config.pipeline_parallel_size
        boundaries = (
            get_pipeline_boundaries(
                config,
                [
                    get_pp_indices(config.num_hidden_layers, rank, pp_size)
                    for rank in range(pp_size)
                ],
            )
            if pp_size > 1
            else ()
        )
        needs_relay = any(boundary.needs_relay for boundary in boundaries)
        if needs_relay and (
            self.parallel_config.prefill_context_parallel_size != 1
            or self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.use_ubatching
            or vllm_config.speculative_config is not None
            or vllm_config.kv_transfer_config is not None
            or _select_dsv4_attn_cls(vllm_config)
            not in (DeepseekV4FlashMLAAttention, DeepseekV4MegaAttnAttention)
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 cross-stage cache relay requires packed FlashMLA "
                "caches and PCP=DCP=1, without microbatching, "
                "speculative decoding or KV transfer."
            )
        pp_rank = get_pp_group().rank_in_group if needs_relay else 0
        self.pipeline_relay = (
            PipelineRelay(
                vllm_config,
                prefix,
                boundaries[pp_rank - 1] if pp_rank > 0 else PipelineBoundary(),
                boundaries[pp_rank] if pp_rank < pp_size - 1 else PipelineBoundary(),
            )
            if needs_relay
            else None
        )
        if self.parallel_config.prefill_context_parallel_size > 1 and (
            self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.use_ubatching
            or (
                vllm_config.speculative_config is not None
                and (
                    self.parallel_config.pipeline_parallel_size > 1
                    or not vllm_config.speculative_config.use_dspark()
                    or vllm_config.speculative_config.enable_adaptive_verification
                )
            )
            or vllm_config.kv_transfer_config is not None
            or (
                vllm_config.cache_config.enable_prefix_caching
                and not vllm_config.cache_config.swa_bounded_replay
            )
            or vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP currently requires DCP=1, with no "
                "microbatching, KV transfer, or MegaMoE. "
                "Speculative decoding requires fixed-length DSpark with PP=1. "
                "Prefix caching requires SWA bounded replay."
            )
        self.use_native_mega_moe = (
            vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        )
        self.use_mega_gate = _use_mega_gate(vllm_config)
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if (
            self.use_native_mega_moe
            and not vllm_config.parallel_config.enable_expert_parallel
        ):
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        # Keep mHC independent of the streams used inside attention.
        mhc_overlap = supports_mhc_overlap(vllm_config)
        ampere_mhc_overlap = supports_ampere_mhc_overlap(vllm_config)
        mhc_stream = torch.cuda.Stream(priority=-1) if mhc_overlap or ampere_mhc_overlap else None
        # SM100's overlap path and SM90's SM80-kernel boundaries (which
        # supports_mhc_all_reduce tells apart) fuse the all-reduce; SM80's
        # boundaries take it with their own two-shot kernel.
        fused_all_reduce = supports_mhc_all_reduce(vllm_config)
        ampere_all_reduce = (
            ampere_mhc_overlap
            and not self.use_sequence_parallel
            and supports_ampere_all_reduce(vllm_config)
        )
        self.fuse_mhc_all_reduce = (
            (mhc_overlap or ampere_mhc_overlap) and fused_all_reduce
        ) or ampere_all_reduce
        if use_ampere_mhc():
            prepare_ampere_mhc(config.hidden_size)
        if self.fuse_mhc_all_reduce:
            if fused_all_reduce:
                init_mhc_all_reduce(vllm_config)
            else:
                from .specialized_decode.ampere.indexer import init_ampere_all_gather

                init_ampere_all_reduce(vllm_config)
                # The row-sharded indexers' gathers need the same peer access.
                init_ampere_all_gather(vllm_config)

        self._pcie_dma_ring = None
        if (
            get_tensor_model_parallel_world_size() > 1
            and not self.use_sequence_parallel
            and current_platform.is_cuda()
            and current_platform.get_device_capability().to_int() == 80
        ):
            from .specialized_decode.ampere.pcie_dma_ring import (
                get_or_init_pcie_dma_ring,
            )

            self._pcie_dma_ring = get_or_init_pcie_dma_ring(
                get_tp_group(),
                config.hidden_size,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            # On SM80 embed_input_ids sums the vocabulary shards with the
            # two-shot kernel too (and so does the DSpark draft model, which
            # shares this module).
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                reduce_results=not ampere_all_reduce,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)
        engram_config = vllm_config.engram_config
        self.vllm_config = vllm_config
        engram_prefetch_stream = (
            torch.cuda.Stream()
            if self.engram_layout is not None
            and engram_config is not None
            and engram_config.cpu_offload
            else None
        )
        if (
            self.engram_layout is not None
            and engram_config is not None
            and engram_config.dp_shared_memory
        ):
            engram_config.dp_shared_memory = can_share_engram_tables(self.engram_layout)

        if self.engram_layout is not None and engram_config and engram_config.use_thp:
            # Release old checkpoint cache before allocating the Engram host tables.
            model_config = vllm_config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=vllm_config.load_config.download_dir,
            )

        # GEMM-RS uses NCCL symmetric-memory multicast, which requires all TP
        # ranks to belong to one NVLink domain. Collective: run before layers.
        self.run_gemm_rs = maybe_init_gemm_rs(vllm_config, self.use_sequence_parallel)

        self.yoco_prefill_only = _yoco_prefill_only_enabled()
        if self.yoco_prefill_only:
            if pp_size != 1:
                raise ValueError(
                    "VLLM_DSV41_YOCO_PREFILL_ONLY=1 requires pipeline_parallel_size == 1"
                )
            if vllm_config.speculative_config is not None:
                raise ValueError(
                    "VLLM_DSV41_YOCO_PREFILL_ONLY=1 is incompatible with speculative decoding"
                )
            if self.use_sequence_parallel:
                raise ValueError(
                    "VLLM_DSV41_YOCO_PREFILL_ONLY=1 is incompatible with sequence parallelism"
                )
            kv_source_ids = getattr(config, "kv_source_layer_ids", None)
            if not kv_source_ids:
                raise ValueError(
                    "VLLM_DSV41_YOCO_PREFILL_ONLY=1 requires config.kv_source_layer_ids"
                )
            cut = max(kv_source_ids)
            self.start_layer = 0
            self.end_layer = cut + 1
            from .specialized_decode.ampere.pcie_dma_ring import (
                prefill_row_shard_attn_enabled,
            )

            use_row_shard_attn = (
                prefill_row_shard_attn_enabled()
                and get_tensor_model_parallel_world_size() > 1
                and is_ampere()
                and not self.use_sequence_parallel
            )
            self.layers = nn.ModuleList(
                [
                    DeepseekV4DecoderLayer(
                        vllm_config,
                        prefix=f"{prefix}.layers.{i}",
                        topk_indices_buffer=self.topk_indices_buffer,
                        aux_stream_list=aux_stream_list,
                        candidate_block_buffer=self.candidate_block_buffer,
                        engram_layout=self.engram_layout,
                        engram_prefetch_stream=engram_prefetch_stream,
                        run_gemm_rs=self.run_gemm_rs,
                        mhc_stream=mhc_stream,
                        fuse_mhc_all_reduce=self.fuse_mhc_all_reduce,
                        kv_write_only=(i == cut),
                        replicated_attn=(use_row_shard_attn and i < cut),
                    )
                    if i <= cut
                    else PPMissingLayer()
                    for i in range(config.num_hidden_layers)
                ]
            )
            logger.info_once(
                "YOCO prefill-only mode: loaded encoder layers 0-%d "
                "(layer %d KV-write-only, row_shard_attn=%s, "
                "MoE/layers %d-%d/norm/lm_head skipped).",
                cut,
                cut,
                use_row_shard_attn,
                cut + 1,
                config.num_hidden_layers - 1,
            )
        else:
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.num_hidden_layers,
                lambda prefix: DeepseekV4DecoderLayer(
                    vllm_config,
                    prefix=prefix,
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                    candidate_block_buffer=self.candidate_block_buffer,
                    engram_layout=self.engram_layout,
                    engram_prefetch_stream=engram_prefetch_stream,
                    run_gemm_rs=self.run_gemm_rs,
                    mhc_stream=mhc_stream,
                    fuse_mhc_all_reduce=self.fuse_mhc_all_reduce,
                ),
                prefix=f"{prefix}.layers",
            )
        if self.pipeline_relay is not None:
            self.pipeline_relay.configure(
                self.layers[self.start_layer].attn, vllm_config
            )
            replica = self.pipeline_relay.incoming.kv_source
            if replica is not None:
                # The replica is updated each step, so its first reader always gathers.
                replica_prefix = f"{prefix}.layers.{replica}.attn"
                first_reader = next(
                    (
                        layer.attn
                        for layer in islice(
                            self.layers, self.start_layer, self.end_layer
                        )
                        if layer.attn.compressed_cache_prefix == replica_prefix
                    ),
                    None,
                )
                if first_reader is not None:
                    first_reader.always_gathers_compressed_kv = True
        self.engram_batch = EngramBatch.create(
            [
                layer.engram
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if getattr(layer, "engram", None) is not None
            ],
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        # SM80 and SM90 gather the Engram columns, and the logits (in the
        # causal LM), with the Lamport kernel of columns.py.
        self.column_gathers = False
        if (is_ampere() or is_hopper()) and not self.use_sequence_parallel:
            from .specialized_decode.ampere.columns import supports_column_gathers

            self.column_gathers = supports_column_gathers(vllm_config)
        if self.column_gathers:
            self._init_engram_column_gathers(vllm_config)
        # Decoder-side SWA bounded replay: in eager prefill steps the layers past
        # the last KV source run on each request's trailing window only
        # (decoder_replay_layers.py).
        self.decoder_replay_layers: DecoderReplayLayers | None = None
        self.decoder_replay_start = self.end_layer
        cut = max(config.kv_source_layer_ids)
        if self.yoco_prefill_only:
            self.decoder_replay_start = cut
        elif (
            cut < self.end_layer - 1
            and self._decoder_replay_supported(vllm_config, cut)
            and self.layers[cut].attn.swa_cache_layer.bounded_replay
            and vllm_config.lora_config is None
        ):
            assert self.layers[cut].attn.compress_ratio <= 1
            # The last KV source writes every row's KV, then replays.
            self.decoder_replay_start = cut
            # The attention metadata keys the replay layers read.
            metadata_prefixes = set()
            for layer in islice(self.layers, cut, self.end_layer):
                attn = typing.cast(DeepseekV4DecoderLayer, layer).attn
                metadata_prefixes.add(attn.swa_cache_layer.prefix)
                if attn.compressed_cache_prefix is not None:
                    metadata_prefixes.add(attn.compressed_cache_prefix)
                if attn.indexer is not None:
                    metadata_prefixes.add(attn.indexer.k_cache.prefix)
            self.decoder_replay_layers = DecoderReplayLayers(
                config.sliding_window,
                self._run_replay_layers,
                self.layers[cut].write_kv,
                metadata_prefixes,
                self.layers[cut].attn.swa_cache_layer.prefix,
                lambda: any(i < cut for i in self.aux_hidden_state_layers),
            )
            logger.info_once(
                "Decoder SWA bounded replay: in eager prefill steps, layer %d "
                "writes its KV for every token, and the rest of it and layers "
                "%d-%d run on each request's last %d tokens only.",
                cut,
                cut + 1,
                self.end_layer - 1,
                config.sliding_window,
            )

        if (
            self.fuse_mhc_all_reduce
            and not self.parallel_config.enable_expert_parallel
            and vllm_config.lora_config is None
        ):
            # A MoE's top-k finalize folds into the next layer's first mHC
            # boundary, which the last local layer lacks, an engram layer
            # replaces with its own all-reduce, and `decoder_replay_layers`
            # requires a finalized Tensor before trimming rows (when
            # `mhc_all_reduce_max_tokens()` exceeds `sliding_window`, e.g.
            # 192 > 128, or when `clip_prefix` trims a shorter chunk).
            # Experts with LoRA always finalize themselves.
            local_layers = list(islice(self.layers, self.start_layer, self.end_layer))
            for layer_idx, (layer, successor) in enumerate(
                zip(local_layers, local_layers[1:]), start=self.start_layer
            ):
                assert isinstance(layer, DeepseekV4DecoderLayer)
                assert isinstance(successor, DeepseekV4DecoderLayer)
                if (
                    (self.decoder_replay_layers is not None or self.yoco_prefill_only)
                    and layer_idx == self.decoder_replay_start - 1
                ):
                    continue
                if successor.engram is None:
                    layer.ffn.defer_finalize()

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_dp_shared_memory = bool(
            vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory
        )
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            engram = next(
                (
                    layer.engram
                    for layer in islice(self.layers, self.start_layer, self.end_layer)
                    if isinstance(layer, DeepseekV4DecoderLayer)
                    and layer.engram is not None
                ),
                None,
            )
            if engram is not None:
                # Shared host tables and GPU peer reads look up only local tokens.
                self.engram_dp_shared_memory = engram.embed_tokens.dp_size == 1
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank and not self.yoco_prefill_only:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        self._mtp_eager_hidden: torch.Tensor | None = None
        if get_pp_group().is_last_rank and needs_mtp_hidden_buffer(spec_config):
            max_decode_tokens = max(
                vllm_config.scheduler_config.max_num_seqs
                * (1 + (spec_config.num_speculative_tokens if spec_config else 0)),
                vllm_config.compilation_config.max_cudagraph_capture_size or 0,
                1,
            )
            self._mtp_hidden_buffer = torch.empty(
                min(vllm_config.scheduler_config.max_num_batched_tokens, max_decode_tokens),
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if not self.embed_tokens.reduce_results:
            x = boundary_all_reduce(x)
        return x

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
        }
        if self.pipeline_relay is not None:
            tensors.update(
                self.pipeline_relay.empty_tensors(
                    self.pipeline_relay.incoming, batch_size, device
                )
            )
        return IntermediateTensors(tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        decode_step: "DecodeStep | None" = None,
    ) -> torch.Tensor | IntermediateTensors:
        # A specialized decode step embeds from the executor's copy of the
        # table, which writes the first layer's hc streams too.
        embedding = (
            decode_step.executor.embedding
            if decode_step is not None and not self.use_sequence_parallel
            else None
        )
        streams: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            elif embedding is not None and embedding.accepts(input_ids):
                hidden_states, streams = embedding(input_ids)
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if self.pipeline_relay is not None:
                self.pipeline_relay.transfer(
                    positions,
                    self.topk_indices_buffer,
                    self.candidate_block_buffer,
                    intermediate_tensors,
                )

        if self.use_native_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Profile runs
        # (KV cache unbound) look up dead rows instead.
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    raise NotImplementedError(
                        "engram needs `lookback_token_ids` from the model runner"
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                )
            else:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them; the memory profile also has to
                # see the Engram activations.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                if self.engram_batch is not None:
                    self.engram_batch.prepare_embeddings(engram_hashes)
                else:
                    gathered_hashes = gather_engram_hashes(
                        engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                    )
                    for layer in islice(self.layers, self.start_layer, self.end_layer):
                        engram = getattr(layer, "engram", None)
                        if engram is not None:
                            engram.prepare_embeddings(
                                gathered_hashes[:, engram.layer_hash_index]
                            )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        local_input_ids = input_ids
        if (
            input_ids is not None
            and self.parallel_config.prefill_context_parallel_size > 1
            and not self.parallel_config.enable_expert_parallel
        ):
            input_ids = get_pcp_group().all_gather(input_ids, dim=0)

        if decode_step is not None:
            return decode_step.run(
                hidden_states,
                positions,
                input_ids,
                engram_hashes,
                engram_mask,
                streams=streams,
            )

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
            if self.use_sequence_parallel:
                pre_mix = sp_shard(pre_mix)
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
            engram_hashes,
            engram_mask,
        )
        if self.yoco_prefill_only:
            cut_layer = typing.cast(
                DeepseekV4DecoderLayer, self.layers[self.decoder_replay_start]
            )
            cut_layer.write_kv(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
            return torch.zeros(
                (full_num_tokens, self.config.hidden_size),
                dtype=self.vllm_config.model_config.dtype,
                device=positions.device,
            )
        late_aux: list[torch.Tensor] = []
        if self.decoder_replay_layers is not None:
            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
                hidden_states,
                positions,
                local_input_ids
                if self.decoder_replay_layers.replay_batch is not None
                else input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        else:
            hidden_states = self._collapse(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                aux_hidden_by_layer,
                full_num_tokens,
            )

        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        ] + late_aux

        if not get_pp_group().is_last_rank:
            # PP transport expects tensors replicated across TP ranks.
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "pre_mix": pre_mix,
                    **(
                        self.pipeline_relay.transfer(
                            positions,
                            self.topk_indices_buffer,
                            self.candidate_block_buffer,
                        )
                        if self.pipeline_relay is not None
                        else {}
                    ),
                }
            )

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            num_tokens = hidden_states.shape[0]
            if num_tokens <= self._mtp_hidden_buffer.shape[0]:
                self._mtp_eager_hidden = None
                self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
            else:
                self._mtp_eager_hidden = hidden_states.flatten(1)

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _mega_gate_metadata(
        self, input_ids: torch.Tensor | None
    ) -> MegaGateRoutingMetadata | None:
        if not self.use_mega_gate:
            return None
        assert input_ids is not None
        return prepare_mega_gate_routing_metadata(
            input_ids,
            has_hash_routing=False,
            image_sentinel_base_id=IMAGE_SENTINEL_BASE_ID
            if getattr(self.config, "vision_n_layers", 0) > 0
            else None,
        )

    def _run_layers(
        self,
        layer_ids: range,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Every layer's post runs inside the next layer's fused pre, so aux
        # hidden states are read back from there instead of recomputed.
        full_num_tokens = positions.shape[0]
        mega_gate_metadata = self._mega_gate_metadata(input_ids)
        if (
            full_num_tokens >= 1024
            and len(layer_ids) > 0
            and isinstance(hidden_states, torch.Tensor)
            and not self.use_sequence_parallel
            and not self.use_native_mega_moe
            and not torch.cuda.is_current_stream_capturing()
            and get_tensor_model_parallel_world_size() > 1
        ):
            from .specialized_decode.ampere.pcie_dma_ring import (
                prefill_row_shard_enabled,
            )

            if prefill_row_shard_enabled() and all(
                getattr(self.layers[i].attn, "gemm_rs", None) is None
                for i in layer_ids
            ):
                return self._run_layers_prefill_row_shard(
                    layer_ids=layer_ids,
                    hidden_states=hidden_states,
                    positions=positions,
                    input_ids=input_ids,
                    pre_mix=pre_mix,
                    post_mix=post_mix,
                    res_mix=res_mix,
                    residual=residual,
                    aux_hidden_by_layer=aux_hidden_by_layer,
                    engram_hashes=engram_hashes,
                    engram_mask=engram_mask,
                    mega_gate_metadata=mega_gate_metadata,
                )
            if os.environ.get("VLLM_DSV41_PREFILL_AR_OVERLAP", "0") == "1":
                dev_comm = getattr(get_tp_group(), "device_communicator", None)
                cand = getattr(dev_comm, "pynccl_comm", None)
                if cand is not None and not getattr(cand, "disabled", True):
                    m0 = ((full_num_tokens // 3) // 64) * 64
                    if m0 >= 256 and (full_num_tokens - m0) >= 256:
                        return self._run_layers_prefill_ar_overlap(
                            layer_ids=layer_ids,
                            hidden_states=hidden_states,
                            positions=positions,
                            input_ids=input_ids,
                            pre_mix=pre_mix,
                            post_mix=post_mix,
                            res_mix=res_mix,
                            residual=residual,
                            aux_hidden_by_layer=aux_hidden_by_layer,
                            engram_hashes=engram_hashes,
                            engram_mask=engram_mask,
                            mega_gate_metadata=mega_gate_metadata,
                            m0=m0,
                            pynccl=cand,
                        )
        can_defer_w2 = (
            os.environ.get("VLLM_DSV41_FUSED_W2_AR", "0") == "1"
            and not self.use_sequence_parallel
            and 3 <= full_num_tokens <= 32
            and ampere_all_reduce_ready()
        )
        _set_defer = None
        if can_defer_w2:
            from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import (
                set_defer_w2_sum as _set_defer,
            )
        dev = positions.device
        for idx in layer_ids:
            defer_this = (
                _set_defer is not None
                and (idx + 1) < layer_ids.stop
                and getattr(self.layers[idx + 1], "engram", True) is None
                and getattr(self.layers[idx + 1], "fuse_mhc_all_reduce", False)
            )
            if defer_this:
                _set_defer(dev, True)
            try:
                hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = (
                    self.layers[idx](
                        hidden_states,
                        positions,
                        input_ids,
                        pre_mix,
                        post_mix,
                        res_mix,
                        residual,
                        engram_hashes,
                        engram_mask,
                        capture_previous_aux=idx in self.aux_hidden_state_layers,
                        mega_gate_metadata=mega_gate_metadata,
                    )
                )
            finally:
                if defer_this:
                    _set_defer(dev, False)
            if previous_aux is not None:
                # idx is the one-based id of the layer whose post this is.
                if self.use_sequence_parallel:
                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]
                aux_hidden_by_layer[idx] = previous_aux
        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        return hidden_states, residual, post_mix, res_mix, pre_mix

    def _run_layers_prefill_row_shard(
        self,
        *,
        layer_ids: range,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
        mega_gate_metadata: MegaGateRoutingMetadata | None,
    ) -> tuple[
        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Lossless BF16 Row-Sharded Prefill with Copy-Engine Ring DMA.

        Keeps mHC residual streams `[S, 4, H]` (`S = ceil(M / W)`), mixes,
        and per-row mHC/Engram post-updates sharded across TP ranks (`1/W`
        memory & compute), using Copy-Engine Ring `reduce_scatter` / `all_gather`
        with zero `torch.cat` scratch spikes. Automatically enables 2-microbatch
        MoE splitting only when `M >= 4096` or `EP >= 6` (whole MoE on 4-card
        `M <= 3072` to avoid extra Marlin weight reads).
        """
        import torch.distributed as dist

        from .specialized_decode.ampere.pcie_dma_ring import (
            prefill_2mb_moe_enabled,
            prefill_2mb_oproj_enabled,
            prefill_ag_int8_enabled,
            prefill_async_engram_enabled,
        )

        assert isinstance(hidden_states, torch.Tensor)
        M = positions.shape[0]
        H = self.config.hidden_size
        dev = hidden_states.device
        dtype = hidden_states.dtype

        tp_group = get_tp_group()
        W = tp_group.world_size
        r = tp_group.rank_in_group
        tp_dev_group = tp_group.device_group
        dma = self._pcie_dma_ring

        M_pad = ((M + W - 1) // W) * W
        S = M_pad // W
        r_lo = r * S
        r_hi = r_lo + S
        valid = max(0, min(r_hi, M) - r_lo)

        def_stream = torch.cuda.current_stream(dev)
        comm_stream, evs, comm_buf = _get_prefill_ar_state(dev, M_pad, H, dtype)
        ev_h0_ready, ev_h0_done, ev_h1_ready, ev_h1_done = evs[:4]
        full_buf = comm_buf[:M_pad]
        shard_buf = torch.empty((S, H), dtype=dtype, device=dev)

        ep_size = (
            W if self.parallel_config.enable_expert_parallel else 1
        )
        use_2mb_moe = prefill_2mb_moe_enabled(M, ep_size, world_size=W)
        m0_2mb = ((M // 2) // (W * 64)) * (W * 64) if use_2mb_moe else 0
        if m0_2mb < 256 or (M - m0_2mb) < 256:
            use_2mb_moe = False

        def _slice_pad_local(t: torch.Tensor) -> torch.Tensor:
            if valid == S:
                return t[r_lo:r_hi].contiguous()
            out = t.new_zeros((S, *t.shape[1:]))
            if valid > 0:
                out[:valid].copy_(t[r_lo : r_lo + valid])
            return out

        def _pad_full_bf16(t: torch.Tensor) -> torch.Tensor:
            if M == M_pad and t.is_contiguous():
                return t
            full_buf[:M].copy_(t)
            if M_pad > M:
                full_buf[M:].zero_()
            return full_buf

        def _rs_bf16(full_t: torch.Tensor, out_shard: torch.Tensor) -> None:
            inp = _pad_full_bf16(full_t)
            if dma is not None and dma.supports_full(inp):
                dma.reduce_scatter(inp, out_shard, stream=def_stream)
            else:
                dist.reduce_scatter_single(out_shard, inp, group=tp_dev_group)

        use_ag_int8 = (
            dma is not None
            and prefill_ag_int8_enabled()
            and hasattr(dma, "supports_shard_int8")
            and (S * H) % 32 == 0
        )
        use_2mb_oproj = (
            dma is not None
            and prefill_2mb_oproj_enabled()
            and M >= 1024
            and M == M_pad
            and (S % 16) == 0
        )
        use_async_engram = (
            dma is not None
            and prefill_async_engram_enabled()
            and not use_2mb_moe
        )
        q_scratch = (
            torch.empty(S * H + (S * H // 32) * 2, dtype=torch.uint8, device=dev)
            if use_ag_int8
            else None
        )

        def _ag_bf16(shard: torch.Tensor, out_full: torch.Tensor) -> None:
            sc = shard.contiguous()
            if (
                q_scratch is not None
                and dma is not None
                and dma.supports_shard_int8(sc)
            ):
                dma.all_gather_int8(sc, out_full[:M_pad], q_scratch, stream=def_stream)
            elif dma is not None and dma.supports_shard(sc):
                dma.all_gather(sc, out_full[:M_pad], stream=def_stream)
            else:
                dist.all_gather_single(out_full[:M_pad], sc, group=tp_dev_group)

        def _ag_generic(shard: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (M_pad, *shard.shape[1:]), dtype=shard.dtype, device=shard.device
            )
            sc = shard.contiguous()
            if (
                sc.dtype == torch.bfloat16
                and sc.ndim == 2
                and dma is not None
                and dma.supports_shard(sc)
            ):
                dma.all_gather(sc, out, stream=def_stream)
            else:
                dist.all_gather_single(out, sc, group=tp_dev_group)
            return out[:M]

        def _engram_prep_kv(
            engram_mod: typing.Any,
            layer_hashes: torch.Tensor,
            run_stream: torch.cuda.Stream,
            ev_dma_done: torch.cuda.Event | None = None,
        ) -> torch.Tensor:
            with torch.cuda.stream(run_stream):
                if engram_mod._batch is not None:
                    if ev_dma_done is not None:
                        ev_dma_done.record(run_stream)
                    kv = engram_mod.wkv(
                        engram_mod._batch.wkv_input(engram_mod, layer_hashes.shape[0])
                    )
                else:
                    if (
                        engram_mod._gather_columns is not None
                        and dma is not None
                        and engram_mod.embed_tokens.tp_size == W
                    ):
                        emb_rows = engram_mod._ready_rows(layer_hashes.shape[0])
                        emb_flat = emb_rows.flatten(1).contiguous()
                        if dma.supports_shard(emb_flat):
                            emb_g = torch.empty(
                                (W, emb_flat.shape[0], emb_flat.shape[1]),
                                dtype=emb_flat.dtype,
                                device=dev,
                            )
                            dma.all_gather(
                                emb_flat,
                                emb_g.view(W * emb_flat.shape[0], emb_flat.shape[1]),
                                stream=run_stream,
                            )
                            if ev_dma_done is not None:
                                ev_dma_done.record(run_stream)
                            emb_full = (
                                emb_g.permute(1, 0, 2)
                                .reshape(emb_flat.shape[0], -1, emb_rows.shape[-1])[
                                    :, : engram_mod.embed_tokens.n_hash_cols
                                ]
                                .flatten(-2)
                                .contiguous()
                            )
                            del emb_g
                            kv = engram_mod.wkv(emb_full)
                            del emb_full
                        else:
                            if ev_dma_done is not None:
                                ev_dma_done.record(run_stream)
                            kv = engram_mod.wkv(engram_mod.embed(layer_hashes).flatten(-2))
                    else:
                        if ev_dma_done is not None:
                            ev_dma_done.record(run_stream)
                        kv = engram_mod.wkv(engram_mod.embed(layer_hashes).flatten(-2))
                if engram_mod._gather_columns is not None:
                    c_part = kv.shape[1]
                    if M == M_pad and kv.is_contiguous():
                        send = kv.view(W, S, c_part)
                    else:
                        send = kv.new_zeros((M_pad, c_part))
                        send[:M].copy_(kv)
                        send = send.view(W, S, c_part)
                    del kv
                    recv = torch.empty_like(send)
                    dist.all_to_all_single(recv, send, group=tp_dev_group)
                    del send
                    kv_local = recv.permute(1, 0, 2).reshape(S, W * c_part).contiguous()
                    del recv
                else:
                    kv_local = _slice_pad_local(kv)
                    del kv
                return kv_local

        def _engram_apply_kv(
            engram_mod: typing.Any,
            prev_post_local: torch.Tensor,
            kv_local: torch.Tensor,
            tok_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            mask_local: torch.Tensor | None = None
            if tok_mask is not None:
                if valid == S:
                    mask_local = tok_mask[r_lo:r_hi].contiguous()
                else:
                    mask_local = torch.zeros(S, dtype=tok_mask.dtype, device=dev)
                    if valid > 0:
                        mask_local[:valid].copy_(tok_mask[r_lo : r_lo + valid])
            return engram_mod._post_wkv(prev_post_local, kv_local, mask_local, valid)

        def _engram_local(
            engram_mod: typing.Any,
            prev_post_local: torch.Tensor,
            layer_hashes: torch.Tensor,
            tok_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            kv_local = _engram_prep_kv(engram_mod, layer_hashes, def_stream)
            return _engram_apply_kv(engram_mod, prev_post_local, kv_local, tok_mask)

        def _mhc_pre_fn(layer: DeepseekV4DecoderLayer):
            return (
                partial(mhc_pre_delayed_ampere, stream=None)
                if layer.ampere_mhc
                else mhc_pre_delayed_tilelang
            )

        # ---- Entry: prepare local [S, ...] mHC state & first layer's x_in ----
        first_layer = self.layers[layer_ids.start]
        assert isinstance(first_layer, DeepseekV4DecoderLayer)
        if residual is None:
            if hidden_states.dim() == 2:
                assert first_layer.hc_attn_fn_broadcast is not None
                h_loc = _slice_pad_local(hidden_states)
                del hidden_states
                res_loc = hc_expand(h_loc, first_layer.hc_mult)
                pm_loc, rm_loc, x_loc, attn_pre_loc = _mhc_pre_fn(first_layer)(
                    res_loc,
                    first_layer.hc_attn_fn_broadcast,
                    first_layer.hc_attn_scale,
                    first_layer.hc_attn_base,
                    first_layer.rms_norm_eps,
                    first_layer.hc_eps,
                    first_layer.hc_eps,
                    first_layer.hc_post_alpha,
                    first_layer.hc_sinkhorn_iters,
                    x=h_loc,
                    norm_weight=first_layer.attn_norm.weight,
                    norm_eps=first_layer.attn_norm.variance_epsilon,
                )
                del h_loc
            else:
                res_loc = _slice_pad_local(hidden_states)
                del hidden_states
                pre_loc_in = _slice_pad_local(pre_mix) if pre_mix is not None else None
                del pre_mix
                if first_layer.engram is not None and engram_hashes is not None:
                    res_loc = _engram_local(
                        first_layer.engram,
                        res_loc,
                        engram_hashes[:, first_layer.engram.layer_hash_index],
                        engram_mask,
                    )
                pm_loc, rm_loc, x_loc, attn_pre_loc = _mhc_pre_fn(first_layer)(
                    res_loc,
                    first_layer.hc_attn_fn,
                    first_layer.hc_attn_scale,
                    first_layer.hc_attn_base,
                    first_layer.rms_norm_eps,
                    first_layer.hc_eps,
                    first_layer.hc_eps,
                    first_layer.hc_post_alpha,
                    first_layer.hc_sinkhorn_iters,
                    pre_mix=pre_loc_in,
                    norm_weight=first_layer.attn_norm.weight,
                    norm_eps=first_layer.attn_norm.variance_epsilon,
                )
                del pre_loc_in
        else:
            assert (
                post_mix is not None
                and res_mix is not None
                and pre_mix is not None
            )
            res_loc = _slice_pad_local(residual)
            pm_loc = _slice_pad_local(post_mix)
            rm_loc = _slice_pad_local(res_mix)
            pre_loc_in = _slice_pad_local(pre_mix)
            del residual, post_mix, res_mix, pre_mix
            if first_layer.fuse_mhc_all_reduce:
                _rs_bf16(hidden_states, shard_buf)
                h_loc = shard_buf
            else:
                h_loc = _slice_pad_local(hidden_states)
            del hidden_states
            capture_first_aux = layer_ids.start in self.aux_hidden_state_layers
            if first_layer.engram is not None and engram_hashes is not None:
                prev_post_loc = mhc_post_tilelang(h_loc, res_loc, pm_loc, rm_loc)
                if capture_first_aux:
                    aux_hidden_by_layer[layer_ids.start] = _ag_generic(
                        prev_post_loc.mean(dim=1)
                    )
                res_loc = _engram_local(
                    first_layer.engram,
                    prev_post_loc,
                    engram_hashes[:, first_layer.engram.layer_hash_index],
                    engram_mask,
                )
                del prev_post_loc
                pm_loc, rm_loc, x_loc, attn_pre_loc = _mhc_pre_fn(first_layer)(
                    res_loc,
                    first_layer.hc_attn_fn,
                    first_layer.hc_attn_scale,
                    first_layer.hc_attn_base,
                    first_layer.rms_norm_eps,
                    first_layer.hc_eps,
                    first_layer.hc_eps,
                    first_layer.hc_post_alpha,
                    first_layer.hc_sinkhorn_iters,
                    pre_mix=pre_loc_in,
                    norm_weight=first_layer.attn_norm.weight,
                    norm_eps=first_layer.attn_norm.variance_epsilon,
                )
            else:
                res_loc, pm_loc, rm_loc, x_loc, attn_pre_loc, aux_loc = (
                    mhc_shifted_post_pre(
                        h_loc,
                        res_loc,
                        pm_loc,
                        rm_loc,
                        first_layer.hc_attn_fn,
                        first_layer.hc_attn_scale,
                        first_layer.hc_attn_base,
                        first_layer.rms_norm_eps,
                        first_layer.hc_eps,
                        first_layer.hc_eps,
                        first_layer.hc_post_alpha,
                        first_layer.hc_sinkhorn_iters,
                        pre_mix=pre_loc_in,
                        norm_weight=first_layer.attn_norm.weight,
                        norm_eps=first_layer.attn_norm.variance_epsilon,
                        capture_aux=capture_first_aux,
                        stream=None,
                        reduce_results=False,
                        boundary_weight=first_layer.hc_attn_boundary_weight,
                    )
                )
                if capture_first_aux and aux_loc is not None:
                    aux_hidden_by_layer[layer_ids.start] = _ag_generic(aux_loc)
            del pre_loc_in

        attn_metadata = get_forward_context().attn_metadata
        use_row_shard_attn = (
            isinstance(attn_metadata, dict)
            and all(
                getattr(self.layers[i].attn, "replicated_attn", False)
                for i in layer_ids
            )
            and hasattr(first_layer.attn, "_prepare_and_attn_row_sharded")
        )
        if use_row_shard_attn:
            swa_meta0 = attn_metadata.get(first_layer.attn.swa_cache_layer.prefix)
            if swa_meta0 is None or getattr(swa_meta0, "num_decode_tokens", 1) != 0:
                use_row_shard_attn = False

        if use_row_shard_attn:
            x_in = None
        else:
            _ag_bf16(x_loc, full_buf)
            del x_loc
            x_in = full_buf[:M]

        final_hidden: torch.Tensor | MoEOutput | None = None
        pre_loc: torch.Tensor | None = None

        # ---- Main Row-Sharded Layer Loop ----
        for idx in layer_ids:
            layer = self.layers[idx]
            assert isinstance(layer, DeepseekV4DecoderLayer)
            is_last = (idx + 1) >= layer_ids.stop

            if use_row_shard_attn:
                # 1. Row-sharded Attention: ZERO H=5120 AllGather/ReduceScatter!
                #    Each rank runs projections & 64-head MLA on ONLY its S local rows,
                #    all-gathering only the tiny 512-dim KV (+ 1024-dim compressor on 3 layers).
                qr_kv_loc, kv_score_loc, idx_w_loc = (
                    layer.attn._run_parallel_input_projections(x_loc)
                )
                qr_loc, _, kv_loc = layer.attn._split_qkv_and_norm(qr_kv_loc)
                del x_loc, qr_kv_loc
                assert kv_loc is not None
                kv_full = _ag_generic(kv_loc)
                kv_score_full = (
                    _ag_generic(kv_score_loc) if kv_score_loc is not None else None
                )
                del kv_loc, kv_score_loc

                qr_v = qr_loc[:valid].contiguous() if valid < S else qr_loc
                idx_w_v = (
                    (idx_w_loc[:valid].contiguous() if valid < S else idx_w_loc)
                    if idx_w_loc is not None
                    else None
                )
                del qr_loc, idx_w_loc
                attn_out_loc = layer.attn._alloc_attn_out(valid, qr_v)
                layer.attn._prepare_and_attn_row_sharded(
                    qr_v,
                    kv_full,
                    kv_score_full,
                    idx_w_v,
                    positions,
                    r_lo,
                    r_lo + valid,
                    attn_out_loc,
                )
                del qr_v, kv_full, kv_score_full, idx_w_v
                if valid > 0:
                    pos_loc = positions[r_lo : r_lo + valid].contiguous()
                    layer.attn._o_proj(attn_out_loc, pos_loc, out=shard_buf[:valid])
                    del pos_loc
                if valid < S:
                    shard_buf[valid:].zero_()
                del attn_out_loc
            else:
                assert x_in is not None
                # 1. Attention on full M rows
                qr_kv, kv_score, indexer_weights = (
                    layer.attn._run_parallel_input_projections(x_in)
                )
                qr, qr_scale, kv = layer.attn._split_qkv_and_norm(qr_kv)
                del qr_kv
                attn_out = layer.attn._alloc_attn_out(M, x_in)
                assert kv is not None
                layer.attn._prepare_and_attn(
                    x_in,
                    qr,
                    kv,
                    qr_scale,
                    kv_score,
                    indexer_weights,
                    positions,
                    attn_out,
                )
                del x_in, qr, qr_scale, kv, kv_score, indexer_weights

                orig_wob_reduce = layer.attn.wo_b.reduce_results
                if getattr(layer.attn, "replicated_attn", False):
                    y_full = layer.attn._o_proj(attn_out, positions)
                    del attn_out
                    shard_buf.copy_(_slice_pad_local(y_full))
                    del y_full
                elif (
                    use_2mb_oproj
                    and isinstance(attn_out, torch.Tensor)
                    and attn_out.dim() >= 2
                    and dma is not None
                ):
                    s_half = S // 2
                    m_half = W * s_half
                    a_view = attn_out.view(W, 2, s_half, *attn_out.shape[1:])
                    y0_buf = full_buf[:m_half]
                    y1_buf = full_buf[m_half:M]
                    try:
                        layer.attn.wo_b.reduce_results = False
                        a0 = a_view[:, 0].reshape(m_half, *attn_out.shape[1:]).contiguous()
                        layer.attn._o_proj(a0, positions[:m_half], out=y0_buf)
                        del a0
                        ev_h0_ready.record(def_stream)
                        comm_stream.wait_event(ev_h0_ready)
                        if dma.supports_full(y0_buf):
                            dma.reduce_scatter(y0_buf, shard_buf[:s_half], stream=comm_stream)
                        else:
                            with torch.cuda.stream(comm_stream):
                                dist.reduce_scatter_single(
                                    shard_buf[:s_half], y0_buf, group=tp_dev_group
                                )
                        ev_h0_done.record(comm_stream)

                        a1 = a_view[:, 1].reshape(m_half, *attn_out.shape[1:]).contiguous()
                        del attn_out, a_view
                        layer.attn._o_proj(a1, positions[:m_half], out=y1_buf)
                        del a1
                        ev_h1_ready.record(def_stream)
                        comm_stream.wait_event(ev_h1_ready)
                        if dma.supports_full(y1_buf):
                            dma.reduce_scatter(y1_buf, shard_buf[s_half:], stream=comm_stream)
                        else:
                            with torch.cuda.stream(comm_stream):
                                dist.reduce_scatter_single(
                                    shard_buf[s_half:], y1_buf, group=tp_dev_group
                                )
                        ev_h1_done.record(comm_stream)
                    finally:
                        layer.attn.wo_b.reduce_results = orig_wob_reduce
                    def_stream.wait_event(ev_h1_done)
                else:
                    try:
                        layer.attn.wo_b.reduce_results = False
                        y_partial = layer.attn._o_proj(attn_out, positions)
                    finally:
                        layer.attn.wo_b.reduce_results = orig_wob_reduce
                    del attn_out

                    # Reduce-Scatter attention partial output into local S rows
                    _rs_bf16(y_partial, shard_buf)
                    del y_partial

            # 2. FFN mHC + RMSNorm on ONLY S local rows (1/W memory & compute!)
            res_loc, pm_loc, rm_loc, x_ffn_loc, pre_loc, _ = mhc_shifted_post_pre(
                shard_buf,
                res_loc,
                pm_loc,
                rm_loc,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                pre_mix=attn_pre_loc,
                norm_weight=layer.ffn_norm.weight,
                norm_eps=layer.ffn_norm.variance_epsilon,
                stream=None,
                reduce_results=False,
                boundary_weight=layer.hc_ffn_boundary_weight,
            )
            del attn_pre_loc

            _ag_bf16(x_ffn_loc, full_buf)
            del x_ffn_loc
            x_ffn_in = full_buf[:M]

            # 3. FFN / MoE
            moe_cfg = getattr(layer.ffn.experts, "moe_config", None)
            if not is_last:
                if (
                    use_2mb_moe
                    and moe_cfg is not None
                    and mega_gate_metadata is None
                    and dma is not None
                ):
                    # 6/8-card large-chunk path: 2-microbatch MoE overlapped
                    # with Copy-Engine Ring AllReduce on comm_stream!
                    orig_skip = moe_cfg.skip_final_all_reduce
                    try:
                        moe_cfg.skip_final_all_reduce = True
                        ids0 = input_ids[:m0_2mb] if input_ids is not None else None
                        ids1 = input_ids[m0_2mb:] if input_ids is not None else None
                        h0 = layer.ffn(x_ffn_in[:m0_2mb], ids0, None)
                        ev_h0_ready.record(def_stream)
                        comm_stream.wait_event(ev_h0_ready)
                        if dma.supports_full(h0):
                            dma.all_reduce_inplace(h0, stream=comm_stream)
                        else:
                            with torch.cuda.stream(comm_stream):
                                dist.all_reduce(h0, group=tp_dev_group)
                        ev_h0_done.record(comm_stream)

                        h1 = layer.ffn(x_ffn_in[m0_2mb:], ids1, None)
                        del x_ffn_in
                        m1 = M - m0_2mb
                        m1_pad = ((m1 + W - 1) // W) * W
                        h1_buf = full_buf[:m1_pad]
                        h1_buf[:m1].copy_(h1)
                        if m1_pad > m1:
                            h1_buf[m1:].zero_()
                        del h1
                        ev_h1_ready.record(def_stream)
                        comm_stream.wait_event(ev_h1_ready)
                        if dma.supports_full(h1_buf):
                            dma.all_reduce_inplace(h1_buf, stream=comm_stream)
                        else:
                            with torch.cuda.stream(comm_stream):
                                dist.all_reduce(h1_buf, group=tp_dev_group)
                        ev_h1_done.record(comm_stream)
                    finally:
                        moe_cfg.skip_final_all_reduce = orig_skip

                    def_stream.wait_event(ev_h0_done)
                    def_stream.wait_event(ev_h1_done)
                    # Copy rank r's canonical [r_lo : r_lo + valid] into shard_buf
                    if valid < S:
                        shard_buf[valid:].zero_()
                    if valid > 0:
                        lo0 = min(r_lo, m0_2mb)
                        hi0 = min(r_lo + valid, m0_2mb)
                        n0 = max(0, hi0 - lo0)
                        if n0 > 0:
                            shard_buf[:n0].copy_(h0[lo0:hi0])
                        lo1 = max(r_lo, m0_2mb) - m0_2mb
                        hi1 = max(r_lo + valid, m0_2mb) - m0_2mb
                        n1 = max(0, hi1 - lo1)
                        if n1 > 0:
                            shard_buf[n0 : n0 + n1].copy_(h1_buf[lo1:hi1])
                    del h0
                    async_eng_g = None
                elif moe_cfg is not None:
                    # 4-card M<=3072 optimal path: whole MoE (0 extra weight reads)
                    # followed by Copy-Engine Ring Reduce-Scatter (1.31x vs NCCL RS).
                    # If the next layer has Engram, overlap its DMA AllGather + wkv +
                    # AllToAll on comm_stream behind this layer's MoE FFN!
                    nxt_peek = self.layers[idx + 1]
                    async_eng_g = None
                    async_eng_rows_dim = 0
                    async_eng_flat = None
                    if (
                        use_async_engram
                        and isinstance(nxt_peek, DeepseekV4DecoderLayer)
                        and nxt_peek.engram is not None
                        and engram_hashes is not None
                        and nxt_peek.engram._batch is None
                        and nxt_peek.engram._gather_columns is not None
                        and nxt_peek.engram.embed_tokens.tp_size == W
                    ):
                        emb_rows_p = nxt_peek.engram._ready_rows(M)
                        emb_flat_p = emb_rows_p.flatten(1).contiguous()
                        if dma.supports_shard(emb_flat_p):
                            async_eng_g = torch.empty(
                                (W, emb_flat_p.shape[0], emb_flat_p.shape[1]),
                                dtype=emb_flat_p.dtype,
                                device=dev,
                            )
                            async_eng_rows_dim = emb_rows_p.shape[-1]
                            async_eng_flat = emb_flat_p
                            ev_h0_ready.record(def_stream)
                            comm_stream.wait_event(ev_h0_ready)
                            dma.all_gather(
                                emb_flat_p,
                                async_eng_g.view(
                                    W * emb_flat_p.shape[0], emb_flat_p.shape[1]
                                ),
                                stream=comm_stream,
                            )
                            ev_h0_done.record(comm_stream)

                    orig_skip = moe_cfg.skip_final_all_reduce
                    try:
                        moe_cfg.skip_final_all_reduce = True
                        h_raw = layer.ffn(x_ffn_in, input_ids, mega_gate_metadata)
                    finally:
                        moe_cfg.skip_final_all_reduce = orig_skip
                    del x_ffn_in
                    if async_eng_g is not None:
                        def_stream.wait_event(ev_h0_done)
                        del async_eng_flat
                    _rs_bf16(h_raw, shard_buf)
                    del h_raw
                else:
                    async_eng_g = None
                    h_full = layer.ffn(x_ffn_in, input_ids, mega_gate_metadata)
                    del x_ffn_in
                    shard_buf.copy_(_slice_pad_local(h_full))
                    del h_full

                # 4. Next layer's Engram + Attn mHC on ONLY S local rows
                nxt_idx = idx + 1
                nxt = self.layers[nxt_idx]
                assert isinstance(nxt, DeepseekV4DecoderLayer)
                capture_nxt_aux = nxt_idx in self.aux_hidden_state_layers
                if nxt.engram is not None and engram_hashes is not None:
                    prev_post_loc = mhc_post_tilelang(
                        shard_buf, res_loc, pm_loc, rm_loc
                    )
                    if capture_nxt_aux:
                        aux_hidden_by_layer[nxt_idx] = _ag_generic(
                            prev_post_loc.mean(dim=1)
                        )
                    if async_eng_g is not None:
                        emb_full = (
                            async_eng_g.permute(1, 0, 2)
                            .reshape(M, -1, async_eng_rows_dim)[
                                :, : nxt.engram.embed_tokens.n_hash_cols
                            ]
                            .flatten(-2)
                            .contiguous()
                        )
                        del async_eng_g
                        kv_e = nxt.engram.wkv(emb_full)
                        del emb_full
                        c_part = kv_e.shape[1]
                        if M == M_pad and kv_e.is_contiguous():
                            send = kv_e.view(W, S, c_part)
                        else:
                            send = kv_e.new_zeros((M_pad, c_part))
                            send[:M].copy_(kv_e)
                            send = send.view(W, S, c_part)
                        del kv_e
                        recv = torch.empty_like(send)
                        dist.all_to_all_single(recv, send, group=tp_dev_group)
                        del send
                        kv_loc_e = recv.permute(1, 0, 2).reshape(S, W * c_part).contiguous()
                        del recv
                        res_loc = _engram_apply_kv(
                            nxt.engram,
                            prev_post_loc,
                            kv_loc_e,
                            engram_mask,
                        )
                        del kv_loc_e
                    else:
                        res_loc = _engram_local(
                            nxt.engram,
                            prev_post_loc,
                            engram_hashes[:, nxt.engram.layer_hash_index],
                            engram_mask,
                        )
                    del prev_post_loc
                    pm_loc, rm_loc, x_nxt_loc, attn_pre_loc = _mhc_pre_fn(nxt)(
                        res_loc,
                        nxt.hc_attn_fn,
                        nxt.hc_attn_scale,
                        nxt.hc_attn_base,
                        nxt.rms_norm_eps,
                        nxt.hc_eps,
                        nxt.hc_eps,
                        nxt.hc_post_alpha,
                        nxt.hc_sinkhorn_iters,
                        pre_mix=pre_loc,
                        norm_weight=nxt.attn_norm.weight,
                        norm_eps=nxt.attn_norm.variance_epsilon,
                    )
                else:
                    res_loc, pm_loc, rm_loc, x_nxt_loc, attn_pre_loc, aux_loc = (
                        mhc_shifted_post_pre(
                            shard_buf,
                            res_loc,
                            pm_loc,
                            rm_loc,
                            nxt.hc_attn_fn,
                            nxt.hc_attn_scale,
                            nxt.hc_attn_base,
                            nxt.rms_norm_eps,
                            nxt.hc_eps,
                            nxt.hc_eps,
                            nxt.hc_post_alpha,
                            nxt.hc_sinkhorn_iters,
                            pre_mix=pre_loc,
                            norm_weight=nxt.attn_norm.weight,
                            norm_eps=nxt.attn_norm.variance_epsilon,
                            capture_aux=capture_nxt_aux,
                            stream=None,
                            reduce_results=False,
                            boundary_weight=nxt.hc_attn_boundary_weight,
                        )
                    )
                    if capture_nxt_aux and aux_loc is not None:
                        aux_hidden_by_layer[nxt_idx] = _ag_generic(aux_loc)
                del pre_loc
                if use_row_shard_attn:
                    x_loc = x_nxt_loc
                else:
                    _ag_bf16(x_nxt_loc, full_buf)
                    del x_nxt_loc
                    x_in = full_buf[:M]
            else:
                # Last layer: preserve native reduce_results contract for _collapse
                # / decoder_replay_layers.
                if layer.fuse_mhc_all_reduce and layer.ffn.defers_finalize(
                    x_ffn_in.shape[0]
                ):
                    final_hidden = layer.ffn.forward_unfinalized(x_ffn_in, input_ids)
                else:
                    final_hidden = layer.ffn(x_ffn_in, input_ids, mega_gate_metadata)
                del x_ffn_in

        # ---- Exit: gather local mHC state once to match _run_layers contract ----
        assert final_hidden is not None and pre_loc is not None
        hc_mult = res_loc.shape[1]
        res_out = torch.empty((M, hc_mult, H), dtype=dtype, device=dev)
        for c_idx in range(hc_mult):
            _ag_bf16(res_loc[:, c_idx, :], full_buf)
            res_out[:, c_idx, :].copy_(full_buf[:M])
        del res_loc

        post_out = _ag_generic(pm_loc)
        res_mix_out = _ag_generic(rm_loc)
        pre_out = _ag_generic(pre_loc)
        del pm_loc, rm_loc, pre_loc, shard_buf
        return final_hidden, res_out, post_out, res_mix_out, pre_out

    def _run_layers_prefill_ar_overlap(
        self,
        *,
        layer_ids: range,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
        mega_gate_metadata: MegaGateRoutingMetadata | None,
        m0: int,
        pynccl: typing.Any,
    ) -> tuple[
        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        assert isinstance(hidden_states, torch.Tensor)
        full_num_tokens = positions.shape[0]
        dev = hidden_states.device
        def_stream = torch.cuda.current_stream(dev)
        dma = self._pcie_dma_ring
        split_moe_in_ar = (
            os.environ.get("VLLM_DSV41_PREFILL_AR_SPLIT_MOE", "1") == "1"
        )

        def _ar_inplace(buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
            if dma is not None and dma.supports_full(buf):
                dma.all_reduce_inplace(buf, stream=stream)
            else:
                pynccl.all_reduce(buf, buf, stream=stream)

        comm_stream, evs, comm_buf = _get_prefill_ar_state(
            dev,
            full_num_tokens,
            self.config.hidden_size,
            hidden_states.dtype,
        )
        (
            ev_sync0,
            ev_sync1,
            ev_ar0,
            ev_ar1,
            ev_fsync0,
            ev_fsync1,
            ef0,
            ef1,
        ) = evs
        pending_ffn_ar: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        split_mhc: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None

        try:
            for idx in layer_ids:
                layer = self.layers[idx]
                assert isinstance(layer, DeepseekV4DecoderLayer)
                capture_aux = idx in self.aux_hidden_state_layers
                did_split_attn_prep = False
                split_attn_mhc: (
                    tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                    ]
                    | None
                ) = None
                qr: typing.Any = None
                qr_scale: torch.Tensor | None = None
                kv: torch.Tensor | None = None
                kv_score: torch.Tensor | None = None
                indexer_weights: torch.Tensor | None = None

                if pending_ffn_ar is not None and (
                    layer.engram is None or engram_hashes is None
                ):
                    w_ef0, w_ef1 = pending_ffn_ar
                    pending_ffn_ar = None
                    assert isinstance(hidden_states, torch.Tensor)
                    if split_mhc is not None:
                        (
                            fres0,
                            fres1,
                            fpm0,
                            fpm1,
                            frm0,
                            frm1,
                            fpre0,
                            fpre1,
                        ) = split_mhc
                        split_mhc = None
                    else:
                        assert (
                            residual is not None
                            and post_mix is not None
                            and res_mix is not None
                            and pre_mix is not None
                        )
                        fres0, fres1 = residual[:m0], residual[m0:]
                        fpm0, fpm1 = post_mix[:m0], post_mix[m0:]
                        frm0, frm1 = res_mix[:m0], res_mix[m0:]
                        fpre0, fpre1 = pre_mix[:m0], pre_mix[m0:]
                        residual = post_mix = res_mix = pre_mix = None
                    def_stream.wait_event(w_ef0)
                    res0, pm0, rm0, x_in0, apre0, aux0 = mhc_shifted_post_pre(
                        hidden_states[:m0],
                        fres0,
                        fpm0,
                        frm0,
                        layer.hc_attn_fn,
                        layer.hc_attn_scale,
                        layer.hc_attn_base,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters,
                        pre_mix=fpre0,
                        norm_weight=layer.attn_norm.weight,
                        norm_eps=layer.attn_norm.variance_epsilon,
                        capture_aux=capture_aux,
                        stream=None,
                        reduce_results=False,
                        boundary_weight=layer.hc_attn_boundary_weight,
                    )
                    del fres0, fpm0, frm0, fpre0
                    qr_kv0, kvs0, idxw0 = layer.attn._run_parallel_input_projections(
                        x_in0
                    )

                    def_stream.wait_event(w_ef1)
                    res1, pm1, rm1, x_in1, apre1, aux1 = mhc_shifted_post_pre(
                        hidden_states[m0:],
                        fres1,
                        fpm1,
                        frm1,
                        layer.hc_attn_fn,
                        layer.hc_attn_scale,
                        layer.hc_attn_base,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters,
                        pre_mix=fpre1,
                        norm_weight=layer.attn_norm.weight,
                        norm_eps=layer.attn_norm.variance_epsilon,
                        capture_aux=capture_aux,
                        stream=None,
                        reduce_results=False,
                        boundary_weight=layer.hc_attn_boundary_weight,
                    )
                    del fres1, fpm1, frm1, fpre1, hidden_states
                    if capture_aux and aux0 is not None and aux1 is not None:
                        aux_hidden_by_layer[idx] = torch.cat([aux0, aux1], dim=0)
                    del aux0, aux1
                    if layer.attn.indexer is not None:
                        x_in = torch.cat([x_in0, x_in1], dim=0)
                    else:
                        x_in = torch.empty(
                            (0, x_in0.shape[1]),
                            dtype=x_in0.dtype,
                            device=x_in0.device,
                        )
                    del x_in0

                    qr_kv1, kvs1, idxw1 = layer.attn._run_parallel_input_projections(
                        x_in1
                    )
                    del x_in1
                    qr_kv = torch.cat([qr_kv0, qr_kv1], dim=0)
                    del qr_kv0, qr_kv1
                    qr, qr_scale, kv = layer.attn._split_qkv_and_norm(qr_kv)
                    del qr_kv
                    kv_score = (
                        torch.cat([kvs0, kvs1], dim=0)
                        if kvs0 is not None and kvs1 is not None
                        else None
                    )
                    del kvs0, kvs1
                    indexer_weights = (
                        torch.cat([idxw0, idxw1], dim=0)
                        if idxw0 is not None and idxw1 is not None
                        else None
                    )
                    del idxw0, idxw1
                    split_attn_mhc = (
                        res0,
                        res1,
                        pm0,
                        pm1,
                        rm0,
                        rm1,
                        apre0,
                        apre1,
                    )
                    del res0, res1, pm0, pm1, rm0, rm1, apre0, apre1
                    did_split_attn_prep = True
                else:
                    if split_mhc is not None:
                        (
                            fres0,
                            fres1,
                            fpm0,
                            fpm1,
                            frm0,
                            frm1,
                            fpre0,
                            fpre1,
                        ) = split_mhc
                        split_mhc = None
                        residual = torch.cat([fres0, fres1], dim=0)
                        del fres0, fres1
                        post_mix = torch.cat([fpm0, fpm1], dim=0)
                        del fpm0, fpm1
                        res_mix = torch.cat([frm0, frm1], dim=0)
                        del frm0, frm1
                        pre_mix = torch.cat([fpre0, fpre1], dim=0)
                        del fpre0, fpre1
                    if pending_ffn_ar is not None:
                        def_stream.wait_event(pending_ffn_ar[0])
                        def_stream.wait_event(pending_ffn_ar[1])
                        pending_ffn_ar = None
                        was_pre_reduced = True
                    else:
                        was_pre_reduced = False
                    mhc_pre_fn = (
                        partial(mhc_pre_delayed_ampere, stream=None)
                        if layer.ampere_mhc
                        else mhc_pre_delayed_tilelang
                    )
                    if residual is None:
                        assert isinstance(hidden_states, torch.Tensor)
                        if hidden_states.dim() == 2:
                            assert layer.hc_attn_fn_broadcast is not None
                            residual = hc_expand(hidden_states, layer.hc_mult)
                            post_mix, res_mix, x_in, attn_pre = mhc_pre_fn(
                                residual,
                                layer.hc_attn_fn_broadcast,
                                layer.hc_attn_scale,
                                layer.hc_attn_base,
                                layer.rms_norm_eps,
                                layer.hc_eps,
                                layer.hc_eps,
                                layer.hc_post_alpha,
                                layer.hc_sinkhorn_iters,
                                x=hidden_states,
                                norm_weight=layer.attn_norm.weight,
                                norm_eps=layer.attn_norm.variance_epsilon,
                            )
                        else:
                            residual = hidden_states
                            if layer.engram is not None and engram_hashes is not None:
                                residual = layer.engram(
                                    residual,
                                    engram_hashes[:, layer.engram.layer_hash_index],
                                    engram_mask,
                                )
                            post_mix, res_mix, x_in, attn_pre = mhc_pre_fn(
                                residual,
                                layer.hc_attn_fn,
                                layer.hc_attn_scale,
                                layer.hc_attn_base,
                                layer.rms_norm_eps,
                                layer.hc_eps,
                                layer.hc_eps,
                                layer.hc_post_alpha,
                                layer.hc_sinkhorn_iters,
                                pre_mix=pre_mix,
                                norm_weight=layer.attn_norm.weight,
                                norm_eps=layer.attn_norm.variance_epsilon,
                            )
                        del hidden_states
                        pre_mix = None
                    elif layer.engram is not None and engram_hashes is not None:
                        assert isinstance(hidden_states, torch.Tensor)
                        if layer.fuse_mhc_all_reduce and not was_pre_reduced:
                            hidden_states = boundary_all_reduce(hidden_states)
                        previous_post = mhc_post_tilelang(
                            hidden_states, residual, post_mix, res_mix
                        )
                        if capture_aux:
                            aux_hidden_by_layer[idx] = previous_post.mean(dim=1)
                        residual = layer.engram(
                            previous_post,
                            engram_hashes[:, layer.engram.layer_hash_index],
                            engram_mask,
                        )
                        del previous_post, hidden_states, post_mix, res_mix
                        post_mix, res_mix, x_in, attn_pre = mhc_pre_fn(
                            residual,
                            layer.hc_attn_fn,
                            layer.hc_attn_scale,
                            layer.hc_attn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            pre_mix=pre_mix,
                            norm_weight=layer.attn_norm.weight,
                            norm_eps=layer.attn_norm.variance_epsilon,
                        )
                        pre_mix = None
                    else:
                        residual, post_mix, res_mix, x_in, attn_pre, aux = (
                            mhc_shifted_post_pre(
                                hidden_states,
                                residual,
                                post_mix,
                                res_mix,
                                layer.hc_attn_fn,
                                layer.hc_attn_scale,
                                layer.hc_attn_base,
                                layer.rms_norm_eps,
                                layer.hc_eps,
                                layer.hc_eps,
                                layer.hc_post_alpha,
                                layer.hc_sinkhorn_iters,
                                pre_mix=pre_mix,
                                norm_weight=layer.attn_norm.weight,
                                norm_eps=layer.attn_norm.variance_epsilon,
                                capture_aux=capture_aux,
                                stream=None,
                                reduce_results=layer.fuse_mhc_all_reduce
                                and not was_pre_reduced,
                                boundary_weight=layer.hc_attn_boundary_weight,
                            )
                        )
                        if capture_aux and aux is not None:
                            aux_hidden_by_layer[idx] = aux
                        del hidden_states
                        pre_mix = None

                if not did_split_attn_prep:
                    qr_kv, kv_score, indexer_weights = (
                        layer.attn._run_parallel_input_projections(x_in)
                    )
                    qr, qr_scale, kv = layer.attn._split_qkv_and_norm(qr_kv)
                    del qr_kv

                attn_out = layer.attn._alloc_attn_out(full_num_tokens, x_in)
                assert kv is not None
                layer.attn._prepare_and_attn(
                    x_in,
                    qr,
                    kv,
                    qr_scale,
                    kv_score,
                    indexer_weights,
                    positions,
                    attn_out,
                )
                del x_in, qr, qr_scale, kv, kv_score, indexer_weights

                moe_cfg = getattr(layer.ffn.experts, "moe_config", None)
                next_layer = (
                    self.layers[idx + 1] if (idx + 1) < layer_ids.stop else None
                )
                can_defer_ffn_ar = (
                    moe_cfg is not None
                    and next_layer is not None
                    and (
                        getattr(next_layer, "engram", None) is None
                        or engram_hashes is None
                    )
                )
                did_split_ffn = False

                if isinstance(attn_out, torch.Tensor) and layer.attn.gemm_rs is None:
                    y_comm = comm_buf[:full_num_tokens]
                    orig_wob_reduce = layer.attn.wo_b.reduce_results
                    try:
                        layer.attn.wo_b.reduce_results = False
                        layer.attn._o_proj(
                            attn_out[:m0], positions[:m0], out=y_comm[:m0]
                        )
                        ev_sync0.record(def_stream)
                        comm_stream.wait_event(ev_sync0)
                        _ar_inplace(y_comm[:m0], comm_stream)
                        ev_ar0.record(comm_stream)

                        layer.attn._o_proj(
                            attn_out[m0:], positions[m0:], out=y_comm[m0:]
                        )
                        del attn_out
                        ev_sync1.record(def_stream)
                        comm_stream.wait_event(ev_sync1)
                        _ar_inplace(y_comm[m0:], comm_stream)
                        ev_ar1.record(comm_stream)
                    finally:
                        layer.attn.wo_b.reduce_results = orig_wob_reduce

                    if split_attn_mhc is not None:
                        (
                            ares0,
                            ares1,
                            apm0,
                            apm1,
                            arm0,
                            arm1,
                            aapre0,
                            aapre1,
                        ) = split_attn_mhc
                        split_attn_mhc = None
                    else:
                        assert (
                            residual is not None
                            and post_mix is not None
                            and res_mix is not None
                            and attn_pre is not None
                        )
                        ares0, ares1 = residual[:m0], residual[m0:]
                        apm0, apm1 = post_mix[:m0], post_mix[m0:]
                        arm0, arm1 = res_mix[:m0], res_mix[m0:]
                        aapre0, aapre1 = attn_pre[:m0], attn_pre[m0:]
                        del attn_pre
                        residual = post_mix = res_mix = None

                    def_stream.wait_event(ev_ar0)
                    fres0, fpm0, frm0, x_ffn0, fpre0, _ = mhc_shifted_post_pre(
                        y_comm[:m0],
                        ares0,
                        apm0,
                        arm0,
                        layer.hc_ffn_fn,
                        layer.hc_ffn_scale,
                        layer.hc_ffn_base,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters,
                        pre_mix=aapre0,
                        norm_weight=layer.ffn_norm.weight,
                        norm_eps=layer.ffn_norm.variance_epsilon,
                        stream=None,
                        reduce_results=False,
                        boundary_weight=layer.hc_ffn_boundary_weight,
                    )
                    del ares0, apm0, arm0, aapre0

                    if can_defer_ffn_ar and split_moe_in_ar and mega_gate_metadata is None:
                        assert moe_cfg is not None
                        ids_ok = (
                            input_ids is not None
                            and input_ids.shape[0] == full_num_tokens
                        )
                        ids0 = input_ids[:m0] if ids_ok else None
                        ids1 = input_ids[m0:] if ids_ok else None
                        orig_skip = moe_cfg.skip_final_all_reduce
                        try:
                            moe_cfg.skip_final_all_reduce = True
                            # Overlap Half-0 MoE FFN with Half-1 Attention PCIe AR!
                            h0 = layer.ffn(x_ffn0, ids0, None)
                            del x_ffn0
                            y_comm[:m0].copy_(h0)
                            del h0
                            ev_fsync0.record(def_stream)
                            comm_stream.wait_event(ev_fsync0)
                            _ar_inplace(y_comm[:m0], comm_stream)
                            ef0.record(comm_stream)

                            def_stream.wait_event(ev_ar1)
                            fres1, fpm1, frm1, x_ffn1, fpre1, _ = mhc_shifted_post_pre(
                                y_comm[m0:],
                                ares1,
                                apm1,
                                arm1,
                                layer.hc_ffn_fn,
                                layer.hc_ffn_scale,
                                layer.hc_ffn_base,
                                layer.rms_norm_eps,
                                layer.hc_eps,
                                layer.hc_eps,
                                layer.hc_post_alpha,
                                layer.hc_sinkhorn_iters,
                                pre_mix=aapre1,
                                norm_weight=layer.ffn_norm.weight,
                                norm_eps=layer.ffn_norm.variance_epsilon,
                                stream=None,
                                reduce_results=False,
                                boundary_weight=layer.hc_ffn_boundary_weight,
                            )
                            del ares1, apm1, arm1, aapre1

                            # Overlap Half-1 MoE FFN with Half-0 FFN PCIe AR!
                            h1 = layer.ffn(x_ffn1, ids1, None)
                            del x_ffn1
                            y_comm[m0:].copy_(h1)
                            del h1
                            ev_fsync1.record(def_stream)
                            comm_stream.wait_event(ev_fsync1)
                            _ar_inplace(y_comm[m0:], comm_stream)
                            ef1.record(comm_stream)
                        finally:
                            moe_cfg.skip_final_all_reduce = orig_skip

                        hidden_states = y_comm
                        del y_comm
                        pending_ffn_ar = (ef0, ef1)
                        split_mhc = (
                            fres0,
                            fres1,
                            fpm0,
                            fpm1,
                            frm0,
                            frm1,
                            fpre0,
                            fpre1,
                        )
                        del fres0, fres1, fpm0, fpm1, frm0, frm1, fpre0, fpre1
                        did_split_ffn = True
                    else:
                        def_stream.wait_event(ev_ar1)
                        fres1, fpm1, frm1, x_ffn1, fpre1, _ = mhc_shifted_post_pre(
                            y_comm[m0:],
                            ares1,
                            apm1,
                            arm1,
                            layer.hc_ffn_fn,
                            layer.hc_ffn_scale,
                            layer.hc_ffn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            pre_mix=aapre1,
                            norm_weight=layer.ffn_norm.weight,
                            norm_eps=layer.ffn_norm.variance_epsilon,
                            stream=None,
                            reduce_results=False,
                            boundary_weight=layer.hc_ffn_boundary_weight,
                        )
                        del ares1, apm1, arm1, aapre1, y_comm
                        x_ffn_in = torch.cat([x_ffn0, x_ffn1], dim=0)
                        del x_ffn0, x_ffn1
                        if can_defer_ffn_ar:
                            split_mhc = (
                                fres0,
                                fres1,
                                fpm0,
                                fpm1,
                                frm0,
                                frm1,
                                fpre0,
                                fpre1,
                            )
                            del fres0, fres1, fpm0, fpm1, frm0, frm1, fpre0, fpre1
                        else:
                            residual = torch.cat([fres0, fres1], dim=0)
                            del fres0, fres1
                            post_mix = torch.cat([fpm0, fpm1], dim=0)
                            del fpm0, fpm1
                            res_mix = torch.cat([frm0, frm1], dim=0)
                            del frm0, frm1
                            pre_mix = torch.cat([fpre0, fpre1], dim=0)
                            del fpre0, fpre1
                else:
                    if split_attn_mhc is not None:
                        (
                            ares0,
                            ares1,
                            apm0,
                            apm1,
                            arm0,
                            arm1,
                            aapre0,
                            aapre1,
                        ) = split_attn_mhc
                        split_attn_mhc = None
                        residual = torch.cat([ares0, ares1], dim=0)
                        del ares0, ares1
                        post_mix = torch.cat([apm0, apm1], dim=0)
                        del apm0, apm1
                        res_mix = torch.cat([arm0, arm1], dim=0)
                        del arm0, arm1
                        attn_pre = torch.cat([aapre0, aapre1], dim=0)
                        del aapre0, aapre1
                    y_full = layer.attn._o_proj(attn_out, positions)
                    del attn_out
                    residual, post_mix, res_mix, x_ffn_in, pre_mix, _ = (
                        mhc_shifted_post_pre(
                            y_full,
                            residual,
                            post_mix,
                            res_mix,
                            layer.hc_ffn_fn,
                            layer.hc_ffn_scale,
                            layer.hc_ffn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            pre_mix=attn_pre,
                            norm_weight=layer.ffn_norm.weight,
                            norm_eps=layer.ffn_norm.variance_epsilon,
                            stream=None,
                            reduce_results=layer.fuse_mhc_all_reduce and not layer.replicated_attn,
                            boundary_weight=layer.hc_ffn_boundary_weight,
                        )
                    )
                    del y_full, attn_pre

                if not did_split_ffn:
                    if can_defer_ffn_ar:
                        h_comm = comm_buf[:full_num_tokens]
                        orig_skip = moe_cfg.skip_final_all_reduce
                        try:
                            moe_cfg.skip_final_all_reduce = True
                            h_raw = layer.ffn(
                                x_ffn_in, input_ids, mega_gate_metadata
                            )
                            del x_ffn_in
                            h_comm.copy_(h_raw)
                            del h_raw
                            ev_fsync0.record(def_stream)
                            comm_stream.wait_event(ev_fsync0)
                            _ar_inplace(h_comm[:m0], comm_stream)
                            ef0.record(comm_stream)
                            _ar_inplace(h_comm[m0:], comm_stream)
                            ef1.record(comm_stream)
                        finally:
                            moe_cfg.skip_final_all_reduce = orig_skip
                        hidden_states = h_comm
                        del h_comm
                        pending_ffn_ar = (ef0, ef1)
                    else:
                        if layer.fuse_mhc_all_reduce and layer.ffn.defers_finalize(
                            x_ffn_in.shape[0]
                        ):
                            hidden_states = layer.ffn.forward_unfinalized(
                                x_ffn_in, input_ids
                            )
                        else:
                            hidden_states = layer.ffn(
                                x_ffn_in, input_ids, mega_gate_metadata
                            )
                        del x_ffn_in
        finally:
            def_stream.wait_event(ev_ar0)
            def_stream.wait_event(ev_ar1)
            if pending_ffn_ar is not None:
                def_stream.wait_event(pending_ffn_ar[0])
                def_stream.wait_event(pending_ffn_ar[1])

        if split_mhc is not None:
            fres0, fres1, fpm0, fpm1, frm0, frm1, fpre0, fpre1 = split_mhc
            split_mhc = None
            residual = torch.cat([fres0, fres1], dim=0)
            post_mix = torch.cat([fpm0, fpm1], dim=0)
            res_mix = torch.cat([frm0, frm1], dim=0)
            pre_mix = torch.cat([fpre0, fpre1], dim=0)

        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        return hidden_states, residual, post_mix, res_mix, pre_mix

    def _collapse(
        self,
        hidden_states: torch.Tensor | MoEOutput,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        # Without a successor boundary, the last layer finalized its own MoE.
        assert isinstance(hidden_states, torch.Tensor)
        # The last layer has no successor to fold its post into.
        if self.fuse_mhc_all_reduce:
            hidden_states = boundary_all_reduce(hidden_states)
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            final_aux = hidden_states.mean(dim=1)
            if self.use_sequence_parallel:
                final_aux = sp_all_gather(final_aux)[:full_num_tokens]
            aux_hidden_by_layer[self.end_layer] = final_aux
        return hidden_states

    def _run_replay_layers(
        self,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Layers `decoder_replay_start`.. on the given rows; returns their output,
        the last FFN's pre-mix and the aux hidden states they capture."""
        assert self.decoder_replay_layers is not None
        if (
            input_ids is not None
            and self.parallel_config.prefill_context_parallel_size > 1
            and not self.parallel_config.enable_expert_parallel
            and self.decoder_replay_layers.replay_batch is not None
        ):
            input_ids = get_pcp_group().all_gather(input_ids, dim=0)
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
        )
        hidden_states = self._collapse(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            aux_hidden_by_layer,
            positions.shape[0],
        )
        return (
            hidden_states,
            pre_mix,
            *(
                aux_hidden_by_layer[layer_id]
                for layer_id in self.aux_hidden_state_layers
                if layer_id in aux_hidden_by_layer
            ),
        )

    def _init_engram_column_gathers(self, vllm_config: VllmConfig) -> None:
        """All-gather the TP-sharded Engram layers' heads and wkv output (plus
        16-byte TP-local greedy argmax stats) with the side-by-side Lamport
        kernel (`columns.py`). Collective over the TP group."""
        from .specialized_decode.ampere.columns import (
            all_gather_columns,
            init_column_gathers,
            supports_column_gathers,
        )

        row_sizes: list[int] = [16, 32] if supports_column_gathers(vllm_config) else []
        engrams = [
            layer.engram
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if getattr(layer, "engram", None) is not None
            and layer.engram.embed_tokens.tp_size > 1
            and layer.engram.embed_tokens.dp_size == 1
            and layer.engram._batch is None
        ]
        if engrams:
            dtype = vllm_config.model_config.dtype
            assert isinstance(dtype, torch.dtype)
            dtype_bytes = dtype.itemsize
            table = engrams[0].embed_tokens
            row_sizes.extend(
                [
                    # The staged BF16 rows of a rank's heads, and its wkv columns.
                    table.part_n_hash_cols * table.dim * 2,
                    engrams[0].wkv.output_size_per_partition * dtype_bytes,
                ]
            )
        if row_sizes:
            init_column_gathers(row_sizes)
        for engram in engrams:
            engram.set_column_gather(all_gather_columns)

    def _decoder_replay_supported(self, vllm_config: VllmConfig, cut: int) -> bool:
        """Whether this rank may replay from layer `cut` on; warns when not."""
        parallel_config = vllm_config.parallel_config
        spec_config = vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        draft_hf_config = getattr(draft_config, "hf_config", None)
        draft_window = getattr(draft_hf_config, "sliding_window", None)
        draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
        window = self.config.sliding_window
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = (
                "the pipeline stage holding the last KV source layer must also "
                "hold every layer after it"
            )
        elif self.use_sequence_parallel or parallel_config.use_ubatching:
            reason = (
                "the replay-layer batch shrinks per rank, which sequence and "
                "microbatching cannot follow"
            )
        elif any(i >= cut for i in getattr(self.config, "engram_layer_ids", ())):
            reason = "an Engram layer sits at or after the last KV source layer"
        elif draft_config is not None and (
            draft_window is None
            or draft_window > window
            or any(t != "sliding_attention" for t in draft_layer_types)
        ):
            reason = (
                f"the drafter (sliding window {draft_window}) reads hidden states "
                f"outside the target's {window}-token window"
            )
        else:
            return True
        logger.warning_once("Decoder SWA bounded replay is off: %s.", reason)
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Pre-compute expert mapping ONCE and index by weight_name for O(1)
        # lookup instead of scanning all 3*n_routed_experts entries per tensor.
        expert_mapping = self.get_expert_mapping()
        expert_mapping_by_wname: dict[
            str, list[tuple[str, str, int, str]]
        ] = {}
        for mapping in expert_mapping:
            expert_mapping_by_wname.setdefault(mapping[1], []).append(mapping)

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.use_sequence_parallel
        )

        import time

        last_log = time.monotonic()
        loaded_count = 0
        for name, loaded_weight in weights:
            loaded_count += 1
            now = time.monotonic()
            if now - last_log >= 15.0:
                logger.info(
                    "Loading weights: %d tensors processed so far (current: %s)",
                    loaded_count,
                    name,
                )
                last_log = now
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if ".shared_experts." in name:
                if is_pp_missing_parameter(name, self):
                    continue
                experts = self.layers[extract_layer_index(name)].ffn.experts
                if getattr(experts, "has_fused_shared_experts", False):
                    loaded_params.add(
                        experts.load_shared_expert_weight(name, loaded_weight)
                    )
                    continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    name_mapped = None
                    # O(1) lookup by "experts.<id>.<proj>." substring.
                    candidates = None
                    exp_pos = name.find("experts.")
                    if exp_pos >= 0:
                        d1 = name.find(".", exp_pos + 8)
                        d2 = name.find(".", d1 + 1) if d1 >= 0 else -1
                        if d2 >= 0:
                            candidates = expert_mapping_by_wname.get(
                                name[exp_pos : d2 + 1]
                            )
                    if candidates is None:
                        candidates = expert_mapping
                    for mapping in candidates:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    if name_mapped is not None:
                        loaded_params.add(name_mapped)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if (
                        self.yoco_prefill_only
                        and ".hc_ffn_" in name
                        and name not in params_dict
                    ):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_native_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=first_layer.ffn.n_redundant_experts,
        )

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers `weight_scale`;
    block-FP8 linear layers register `weight_scale_inv`.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        # decode_backend=latency_specialized: a decode executor over self.model, built
        # by the model state once the weights are loaded.
        self.specialized_decode_config = specialized_decode_config(vllm_config)
        self.specialized_decode: SpecializedDecode | None = None
        # SM80 and SM90 record the metadata builds of FULL decode steps.
        self.records_decode_metadata = records_decode_metadata(vllm_config)
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank and not self.model.yoco_prefill_only:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        if self.model.column_gathers and isinstance(self.lm_head, ParallelLMHead):
            # The vocabulary shards go through the Lamport gather; the DSpark
            # draft model's head (same shape) shares it.
            from .specialized_decode.ampere.columns import (
                AmpereLogitsProcessor,
                init_column_gathers,
            )

            head_dtype = vllm_config.model_config.head_dtype
            init_column_gathers(
                [self.lm_head.num_embeddings_per_partition * head_dtype.itemsize]
            )
            self.logits_processor = AmpereLogitsProcessor(config.vocab_size)
        else:
            self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.model.yoco_prefill_only:
            logits = torch.zeros(
                (hidden_states.shape[0], self.config.vocab_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            eos_id = getattr(self.config, "eos_token_id", 0)
            if isinstance(eos_id, list):
                eos_id = eos_id[0] if eos_id else 0
            logits[:, int(eos_id or 0)] = 1.0
            return logits
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.model.yoco_prefill_only:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_vocab = cdiv(self.config.vocab_size, tp_size)
            logits = torch.zeros(
                (hidden_states.shape[0], shard_vocab),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if tp_rank == 0:
                logits[:, 0] = 1.0
            return logits
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def build_specialized_decode(self) -> "SpecializedDecode":
        from .specialized_decode.logits.logits import SpecializedDecodeLogitsProcessor
        from .specialized_decode.runtime.decode import SpecializedDecode

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode.for_target(
            self.model, self.specialized_decode_config, self.lm_head
        )
        self.logits_processor = SpecializedDecodeLogitsProcessor(
            self.config.vocab_size, self.specialized_decode.all_gather
        )
        return self.specialized_decode

    def get_model_state_cls(self):
        if self.specialized_decode_config is not None:
            from .specialized_decode.runtime.model_state import (
                DeepseekV41SpecializedDecodeModelState,
            )

            return DeepseekV41SpecializedDecodeModelState
        if self.records_decode_metadata:
            from .specialized_decode.runtime.model_state import (
                DeepseekV41MetadataGraphModelState,
            )

            return DeepseekV41MetadataGraphModelState
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    @property
    def decoder_replay_layers(self) -> DecoderReplayLayers | None:
        return self.model.decoder_replay_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        decode = self.specialized_decode
        step = None if decode is None else decode.plan(positions, intermediate_tensors)
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            decode_step=step,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_decode_tokens,
        hc_mult * hidden_size) or zero-copy eager view for the MTP draft model."""
        eager = getattr(self.model, "_mtp_eager_hidden", None)
        if eager is not None:
            return eager
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        import gc
        import os

        self.model.finalize_mhc_broadcast_weights()
        config = self.model.vllm_config
        if config.engram_config and config.engram_config.use_thp:
            # Loading weights refills the file cache; release it before MADV_COLLAPSE.
            model_config = config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=config.load_config.download_dir,
            )
            for layer in islice(
                self.model.layers, self.model.start_layer, self.model.end_layer
            ):
                if layer.engram is not None:
                    layer.engram.embed_tokens.collapse_huge_pages()

        if (
            os.environ.get("VLLM_DSV41_OFFLOAD_VISION_EMBED", "0") == "1"
            and isinstance(self.model.embed_tokens, VocabParallelEmbedding)
            and self.model.embed_tokens.weight.is_cuda
        ):
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
            from ..common.engram import _exact_pinned_tensor

            w = self.model.embed_tokens.weight.data
            dev_idx = w.device.index
            pinned = _exact_pinned_tensor(tuple(w.shape), w.dtype, dev_idx)
            if pinned is None:
                pinned = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
            pinned.copy_(w, non_blocking=False)
            uva_view = get_accelerator_view_from_cpu_tensor(pinned)
            self.model._pinned_embed_host = pinned
            self.model.embed_tokens.weight = nn.Parameter(uva_view, requires_grad=False)
            del w

        if (
            os.environ.get("VLLM_DSV41_MARLIN_CLEAN_REPACK", "0") == "1"
            or os.environ.get("VLLM_DSV41_OFFLOAD_VISION_EMBED", "0") == "1"
        ):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
