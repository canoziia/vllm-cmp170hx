# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of `num_speculative_tokens` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly `N =
    num_speculative_tokens` query tokens (anchor + N-1 noise), NOT `1 + N`.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and `sample_pos = query_pos + 1`
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the `sample_from_anchor` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash `1 + N` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.

Under batch-sharded sampling, draft steps also shard the sequential sampling
across TP ranks: each rank runs the Markov loop only for the requests it owns
(the same owners as verification), see `_sample_sequential_sharded`.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.sample.batch_shard import (
    all_to_all_vocab_shards,
    batch_sharded_sampling_enabled,
    get_tp_pynccl_comm,
)
from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
    FusedMarkovSampler,
    build_fused_markov_sampler,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False

        # Set when the sequential sampling can be batch-sharded across TP.
        self._sharded_sequential_comm: PyNcclCommunicator | None = None

        # Whether the markov bias is fused into the Gumbel sample kernel.
        self._fuse_markov_bias: bool = False
        self._fused_markov: FusedMarkovSampler | None = None
        # Round 12: k-aware DSpark tail split (layers[-1].ffn + draft_head + markov
        # run only on num_reqs * active_k rows; 100% bitwise identical to full k=5).
        self._tail_split_ready: bool = False
        self._capturing_tail_split: bool = False
        self._tail_bufs: tuple[torch.Tensor, ...] | None = None
        self._tail_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._tail_k_offsets: dict[tuple[int, int], torch.Tensor] = {}
        # Round 13: context-prep bucket split (avoids running main_proj+ctx_wkv on 192 rows when M=64)
        self._ctx_split_ready: bool = False
        self._capturing_ctx_split: bool = False
        self._ctx_prep_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._ctx_buckets_by_nr: dict[int, tuple[int, ...]] = {}

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        tp_size = get_tp_group().world_size
        if tp_size > 1 and batch_sharded_sampling_enabled(
            self.vllm_config, target_model
        ):
            self._try_enable_sharded_sequential_sampling()

        self._fuse_markov_bias = (
            self.draft_logits is not None
            and self._d2t_scatter_index is None
            and self.draft_watermarker is None
            and self.acceptance_estimator is None
            and not hasattr(self.model, "apply_output_logit_transform")
        )

    def _try_enable_sharded_sequential_sampling(self) -> None:
        tp_size = get_tp_group().world_size
        lm_head = getattr(self.model, "lm_head", None)
        pynccl_comm = get_tp_pynccl_comm()
        reason = None
        if self.draft_logits is not None and self.draft_logits.row_mapping is None:
            reason = "the draft logits cache is replicated"
        elif self.use_acceptance_estimator:
            reason = (
                "the online acceptance estimator fits on every request's "
                "draft logits on every rank"
            )
        elif self._draft_topk is not None:
            reason = "top-k Markov sampling is not supported"
        elif hasattr(self.model, "sample_sequential"):
            reason = "the draft model fuses its own sequential sampling"
        elif not hasattr(self.model, "compute_draft_logits_local") or (
            getattr(lm_head, "tp_size", None) != tp_size
        ):
            reason = "the draft model cannot compute vocab-sharded draft logits"
        elif pynccl_comm is None:
            reason = "PyNCCL is unavailable"

        if reason is not None:
            logger.info(
                "DSpark sequential sampling stays replicated across TP ranks: %s.",
                reason,
            )
            return

        self._sharded_sequential_comm = pynccl_comm
        logger.info(
            "DSpark sequential sampling is batch-sharded across %d TP ranks.",
            tp_size,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if os.environ.get("VLLM_DSPARK_DRAFT_HEAD_FP8", "0") == "1":
            if not hasattr(model, "build_fp8_draft_head"):
                raise ValueError(
                    "VLLM_DSPARK_DRAFT_HEAD_FP8=1 is not supported by "
                    f"{type(model).__name__}"
                )
            model.build_fp8_draft_head()
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.use_draft_logits and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits_spec(self.vllm_config)[0],
                device=self.device,
            )
        self.use_confidence_head = (
            self.speculative_config.uses_gpu_adaptive_verification()
            and model.model.confidence_head is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        if (
            self.use_local_argmax_reduction
            and self.draft_logits is None
            and os.environ.get("VLLM_DSPARK_FUSED_MARKOV", "1") == "1"
        ):
            self._fused_markov = build_fused_markov_sampler(
                model, self.max_num_reqs, self.num_speculative_steps, self.device
            )
        return model

    def _validate_local_argmax_reduction(self) -> None:
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic'."
            )
        missing = [
            name
            for name in ("compute_draft_logits_shard", "select_draft_token_shard")
            if not hasattr(self.model, name)
        ]
        if missing:
            raise ValueError(
                "use_local_argmax_reduction is enabled but DSpark draft model "
                f"{self.model.__class__.__name__} does not implement "
                f"{', '.join(missing)}()."
            )
        logger.info(
            "DSpark: vocab-sharded Markov head with local argmax reduction "
            "(communication: O(2*tp_size) vs O(vocab_size) per draft position)."
        )

    def supports_batch_sharded_draft_logits(self) -> bool:
        # A model's own sample_sequential kernels write the full-size cache.
        return not hasattr(self.model, "sample_sequential")

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample draft ids from the logits. If `bias` is provided, the Gumbel
        kernel adds it."""
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = greedy_argmax(logits)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        if self.draft_watermarker is not None:
            return self.draft_watermarker.sample(
                logits,
                idx_map,
                self.temperature,
                self.seeds,
                sample_pos - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=self.draft_logits.values,
                logits_cache_col=self._step_cols[step],
                use_fp64=self.use_fp64_gumbel,
            )
        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits.values,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
            logits_cache_row_mapping=self.draft_logits.row_mapping,
            bias=bias,
        )

    def _markov_step(
        self,
        prev: torch.Tensor,
        draft_hidden: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Markov embedding of the previous token at one draft position.

        Recurrent heads also read the backbone hidden state and carry a state
        across the positions of one proposal, starting from None (zeros).
        """
        if getattr(self.model, "has_recurrent_markov_head", False):
            return self.model.markov_step(prev, draft_hidden, state)
        return self.model.markov_embed(prev), None

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return
        # A draft model can run the Markov steps below with its own kernels.
        # It returns False, having done nothing, for a step that the loop
        # below must run.
        sample_sequential = getattr(self.model, "sample_sequential", None)
        if sample_sequential is not None and sample_sequential(
            self, num_reqs, head_hidden
        ):
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        if self.use_local_argmax_reduction:
            base_logits = self.model.compute_draft_logits_shard(sample_hidden)
        else:
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        if self._fused_markov is not None:
            prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
            self._fused_markov.sample(num_reqs, base_logits, prev, self.draft_tokens)
            if self.use_confidence_head:
                previous_ids = torch.cat(
                    (prev[:, None], self.draft_tokens[:num_reqs, : n_spec - 1]),
                    dim=1,
                )
                markov_embeds = self.model.markov_embed(previous_ids.flatten()).view(
                    num_reqs, n_spec, -1
                )
                conf = self._confidence(
                    sample_hidden, markov_embeds, (num_reqs, n_spec)
                )
                if conf is not None:
                    self.draft_token_confidence_probs[:num_reqs] = conf
            return

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        confidence = self._sample_markov_chain(
            base_logits,
            sample_hidden,
            self.input_buffers.input_ids,
            self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec),
            self.sample_pos[:num_sample].view(num_reqs, n_spec),
            self.draft_tokens[:num_reqs],
            prev_indices=self._anchor_idx[:num_reqs],
        )
        if confidence is not None:
            self.draft_token_confidence_probs[:num_reqs] = confidence

    def _sample_markov_chain(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        prev: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        out_tokens: torch.Tensor,
        *,
        prev_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run the sequential Markov steps for a block of requests, writing the
        drafts to `out_tokens`.

        Returns:
            [num_rows, num_speculative_steps] draft confidences, or None
            without a confidence head.

        """
        fused_chain = getattr(self.model, "sample_markov_chain", None)
        if fused_chain is not None:
            # The per-step Markov embeddings, or None where it does not apply.
            fused_embeds = fused_chain(
                self,
                base_logits,
                prev,
                idx_map,
                sample_pos,
                out_tokens,
                prev_indices=prev_indices,
            )
            if fused_embeds is not None:
                return self._confidence(hidden, fused_embeds, out_tokens.shape)
        if prev_indices is not None:
            prev = prev[prev_indices]
        markov_embeds = []
        hidden_by_step = hidden.view(*out_tokens.shape, -1)
        state: torch.Tensor | None = None
        for i in range(self.num_speculative_steps):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed, state = self._markov_step(prev, hidden_by_step[:, i], state)
            if self.use_confidence_head:
                markov_embeds.append(markov_embed)
            if self.use_local_argmax_reduction:
                draft_sampled_i = self.model.map_draft_to_target(
                    self.model.select_draft_token_shard(markov_embed, base_logits[:, i])
                )
                out_tokens[:, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i]
            # The kernel rounds the sum to the logits dtype, as `logits + bias`
            # does only when the dtypes match.
            if not self._fuse_markov_bias or bias.dtype != logits_i.dtype:
                logits_i = logits_i + bias
                if hasattr(self.model, "apply_output_logit_transform"):
                    logits_i = self.model.apply_output_logit_transform(logits_i)
                bias = None
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i, bias
            )
            out_tokens[:, i] = draft_sampled_i
            prev = draft_sampled_i

        if not self.use_confidence_head:
            return None
        return self._confidence(
            hidden, torch.stack(markov_embeds, dim=1), out_tokens.shape
        )

    def _confidence(
        self,
        hidden: torch.Tensor,
        markov_embeds: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor | None:
        """Draft confidences from `[num_rows, num_steps, rank]` Markov
        embeddings, or None without a confidence head."""
        if not self.use_confidence_head:
            return None
        confidence = self.model.compute_confidence(hidden, markov_embeds.flatten(0, 1))
        return confidence.view(shape)

    def _sample_sequential_sharded(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        """_sample_sequential with each TP rank sampling only its own requests.

        Request i is owned by rank shard_owner(i), as in DraftLogitsCache, so no
        rank owns more than cdiv(num_reqs, tp_size) requests. Each rank computes
        vocab-shard base logits for every request, in owner-sorted order. An
        all-to-all then gives each rank the full-vocab rows of the requests it
        owns, the Markov loop runs on just those rows, and an all-gather returns
        the drafts in batch order. All shapes depend only on num_reqs, so the
        step can be captured.
        """
        assert self._sharded_sequential_comm is not None
        tp_group = get_tp_group()
        tp_size, tp_rank = tp_group.world_size, tp_group.rank_in_group
        n_spec = self.num_speculative_steps
        capacity = cdiv(num_reqs, tp_size)
        # Rank r owns requests r, r + tp_size, r + 2 * tp_size, ...
        send_reqs = torch.arange(capacity, device=self.device) * tp_size + torch.arange(
            tp_size, device=self.device
        ).unsqueeze(-1)
        # Mark padding slots with -1.
        send_reqs = torch.where(send_reqs < num_reqs, send_reqs, -1)
        req_idxs = torch.arange(num_reqs, device=self.device)
        # Inverse of send_reqs (ignoring padding), used for restoring the
        # all-gathered drafts/confidences into the original request order.
        gathered_src_idxs = (req_idxs % tp_size) * capacity + req_idxs // tp_size

        # Re-shard the draft logits from the vocab dimension to the batch dimension.
        # Afterwards, each rank holds the full-vocab logits of its own requests.
        # Padding slots read request 0's rows, and their results are discarded.
        send_rows = send_reqs.clamp(min=0).unsqueeze(-1) * n_spec + self._step_cols
        send_hidden = head_hidden[self.sample_indices[send_rows.flatten()]]
        local_logits = self.model.compute_draft_logits_local(send_hidden)
        num_local_rows = capacity * n_spec
        base_logits = all_to_all_vocab_shards(
            local_logits, [num_local_rows] * tp_size, num_local_rows
        ).view(capacity, n_spec, -1)[..., : self.model.logits_processor.org_vocab_size]

        # Sample draft tokens for the locally assigned requests.
        local_reqs = send_reqs[tp_rank]
        local_rows = send_rows[tp_rank]
        local_tokens = self.draft_tokens.new_empty(capacity, n_spec)
        local_hidden = send_hidden.view(tp_size, capacity * n_spec, -1)[tp_rank]
        confidence = self._sample_markov_chain(
            base_logits,
            local_hidden,
            self.input_buffers.input_ids[self._anchor_idx[local_reqs.clamp(min=0)]],
            # Empty slots are inert: no draft logits cache writes.
            torch.where(
                local_reqs.view(-1, 1) >= 0, self.sample_idx_mapping[local_rows], -1
            ),
            self.sample_pos[local_rows],
            local_tokens,
        )

        # Gather the draft tokens and confidences sampled by each rank.
        packed = local_tokens
        if confidence is not None:
            packed = torch.cat(
                [local_tokens, confidence.float().view(torch.int32).long()], dim=1
            )
        gathered = packed.new_empty(tp_size * capacity, packed.shape[1])
        self._sharded_sequential_comm.all_gather(gathered, packed.contiguous())

        # Write the output draft tokens and confidences.
        packed = gathered[gathered_src_idxs]
        self.draft_tokens[:num_reqs] = packed[:, :n_spec]
        if confidence is not None:
            self.draft_token_confidence_probs[:num_reqs] = (
                packed[:, n_spec:].int().view(torch.float32)
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to `-inf`. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        hidden_by_step = sample_hidden.view(num_reqs, n_spec, -1)
        state: torch.Tensor | None = None

        for i in range(n_spec):
            markov_embed, state = self._markov_step(prev, hidden_by_step[:, i], state)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _can_tail_split(self) -> bool:
        import os

        if os.environ.get("VLLM_DSV41_DSPARK_TAIL_SPLIT", "1") != "1":
            return False
        if (
            self.num_speculative_steps <= 1
            or self._draft_topk is not None
            or self._sharded_sequential_comm is not None
            or self._fused_markov is None
            or self.use_confidence_head
            or getattr(self.model, "sample_sequential", None) is not None
            or not self.use_local_argmax_reduction
        ):
            return False
        can_fn = getattr(self.model, "can_split_tail", None)
        return bool(can_fn is not None and can_fn())

    @torch.inference_mode()
    def _run_model_prefix(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor, ...]:
        from vllm.forward_context import BatchDescriptor, set_forward_context

        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=self.input_buffers.is_padding[:num_tokens],
        ):
            return self.model.forward_prefix(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )

    @torch.inference_mode()
    def _run_tail(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        active_k: int,
    ) -> None:
        from vllm.forward_context import BatchDescriptor, set_forward_context

        assert self._tail_bufs is not None and self._fused_markov is not None
        n_spec = self.num_speculative_steps
        if active_k >= n_spec:
            x, res, pm, rm, pre = (b[:num_tokens_padded] for b in self._tail_bufs)
            ids = self.input_buffers.input_ids[:num_tokens_padded]
            num_tail = num_tokens_padded
            is_pad = self.input_buffers.is_padding[:num_tokens_padded]
        else:
            keep_idx = self.sample_indices[self._tail_k_offsets[(num_reqs, active_k)]]
            x, res, pm, rm, pre = (b.index_select(0, keep_idx) for b in self._tail_bufs)
            ids = self.input_buffers.input_ids.index_select(0, keep_idx)
            num_tail = num_reqs * active_k
            is_pad = self.input_buffers.is_padding.index_select(0, keep_idx)

        batch_descriptor = BatchDescriptor(num_tokens=num_tail)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tail,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            num_tokens_across_dp=None,
            slot_mapping=None,
            batch_descriptor=batch_descriptor,
            is_padding=is_pad,
        ):
            head_hidden = self.model.forward_tail(x, res, pm, rm, pre, ids)

        if active_k >= n_spec:
            sample_hidden = head_hidden[self.sample_indices[: num_reqs * n_spec]]
            k_eff = n_spec
        else:
            sample_hidden = head_hidden
            k_eff = active_k

        base_logits = self.model.compute_draft_logits_shard(sample_hidden)
        base_logits = base_logits.view(num_reqs, k_eff, base_logits.shape[-1])
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        self._fused_markov.sample(
            num_reqs, base_logits, prev, self.draft_tokens, num_steps=k_eff
        )

    def capture(self) -> None:
        if not self._can_tail_split() or (
            self.query_cudagraph_manager is None
            or self.query_cudagraph_manager.cudagraph_mode == CUDAGraphMode.NONE
        ):
            super().capture()
            return

        import os
        use_ctx_split = os.environ.get("VLLM_DSV41_DSPARK_CTX_SPLIT", "1") == "1"
        self._capturing_tail_split = True
        self._capturing_ctx_split = use_ctx_split
        try:
            super().capture()
        finally:
            self._capturing_tail_split = False
            self._capturing_ctx_split = False

        if self._tail_bufs is None:
            return

        mgr = self.query_cudagraph_manager
        pool = mgr.pool
        cap_stream = mgr.capture_stream
        n_spec = self.num_speculative_steps
        # Collect unique FULL-captured (num_reqs, num_tokens) pairs.
        full_pairs = sorted(
            {
                (desc.num_reqs, desc.num_tokens)
                for desc in mgr.graphs
                if desc.cg_mode == CUDAGraphMode.FULL
                and desc.num_reqs is not None
                and desc.num_reqs >= 8
            },
            reverse=True,
        )
        # Ensure sample_indices holds valid in-range indices during tail capture.
        self.sample_indices.zero_()
        ctx_stream = (
            torch.cuda.stream(cap_stream) if cap_stream is not None else torch.cuda.stream(torch.cuda.current_stream())
        )
        with ctx_stream:
            for nr, nt in full_pairs:
                for k in range(n_spec, 0, -1):
                    if k < n_spec:
                        offs = (
                            torch.arange(nr, device=self.device, dtype=torch.int64)[:, None] * n_spec
                            + torch.arange(k, device=self.device, dtype=torch.int64)[None, :]
                        ).reshape(-1)
                        self._tail_k_offsets[(nr, k)] = offs
                    # Eager warmup outside graph capture for JIT/attribute init.
                    self._run_tail(nr, nt, k)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=cap_stream):
                        self._run_tail(nr, nt, k)
                    self._tail_graphs[(nr, k)] = g
        if use_ctx_split:
            with ctx_stream:
                for nr, _nt in full_pairs:
                    max_c = self._num_graph_context_tokens(nr)
                    buckets = tuple(sorted({
                        min(nr * 2, max_c),
                        min(nr * 4, max_c),
                        max_c,
                    }))
                    self._ctx_buckets_by_nr[nr] = buckets
                    for nc in reversed(buckets):
                        self._run_ctx_prep(nr, nc)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=pool, stream=cap_stream):
                            self._run_ctx_prep(nr, nc)
                        self._ctx_prep_graphs[(nr, nc)] = g
            self._ctx_split_ready = True
        torch.cuda.synchronize()
        self._tail_split_ready = True
        logger.info(
            "DSpark k-aware tail+ctx split enabled: captured %d tail graphs + %d ctx graphs (%d req sizes)",
            len(self._tail_graphs),
            len(self._ctx_prep_graphs),
            len(full_pairs),
        )

    @torch.inference_mode()
    def _run_ctx_prep(self, num_reqs: int, num_context: int) -> None:
        if self._aux_staging is not None:
            self.hidden_states[:num_context].copy_(
                self.model.combine_hidden_states(self._aux_staging[:num_context])
            )
        self.prepare_context_anchor(
            num_reqs, self.target_input_buffers.query_start_loc, self._num_rejected
        )
        self._precompute_context_kv(0, num_context)

    def _select_ctx_bucket(self, num_graph_reqs: int, num_target_tokens: int) -> int:
        if not self._ctx_split_ready:
            return self._num_graph_context_tokens(num_graph_reqs)
        buckets = self._ctx_buckets_by_nr.get(num_graph_reqs)
        if not buckets:
            return self._num_graph_context_tokens(num_graph_reqs)
        for b in buckets:
            if b >= num_target_tokens:
                return b
        return buckets[-1]

    def _before_fullgraph_replay(self, batch_desc: Any, num_context: int) -> None:
        if not self._ctx_split_ready:
            return
        g = self._ctx_prep_graphs.get((batch_desc.num_reqs, num_context))
        if g is not None:
            g.replay()

    def _after_fullgraph_replay(
        self, batch_desc: Any, num_speculative_tokens: int | None
    ) -> None:
        if not self._tail_split_ready:
            return
        nr = batch_desc.num_reqs
        k = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else max(1, min(int(num_speculative_tokens), self.num_speculative_steps))
        )
        g = self._tail_graphs.get((nr, k))
        if g is not None:
            g.replay()

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if self._capturing_tail_split and num_reqs >= 8:
            outs = self._run_model_prefix(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            if self._tail_bufs is None:
                assert not torch.cuda.is_current_stream_capturing()
                max_q = self.max_num_reqs * self.num_query_per_req
                self._tail_bufs = tuple(
                    torch.zeros((max_q, *t.shape[1:]), dtype=t.dtype, device=t.device)
                    for t in outs
                )
            for dst, src in zip(self._tail_bufs, outs):
                dst[:num_tokens_padded].copy_(src)
            return
        # Full draft step (eager fallback): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if self._sharded_sequential_comm is not None:
            self._sample_sequential_sharded(num_reqs, head_hidden)
        else:
            self._sample_sequential(num_reqs, head_hidden)
