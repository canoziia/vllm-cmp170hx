#include "torch_utils.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>
#include <optional>

void fused_deepseek_v4_kv_rope_insert(
    torch::stable::Tensor const& kv, torch::stable::Tensor& k_cache,
    torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, int64_t cache_block_size,
    std::optional<torch::stable::Tensor> fp8_scale, bool kv_mxfp8);

torch::stable::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::stable::Tensor const& q_in, torch::stable::Tensor const& kv,
    torch::stable::Tensor& k_cache, torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, int64_t q_head_padded,
    double eps, int64_t cache_block_size, bool apply_q_norm, bool kv_mxfp8,
    bool apply_q_rope, bool is_q_interleaved);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    torch::stable::Tensor& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor& k_cache, torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, double eps,
    int64_t cache_block_size, bool apply_q_norm);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    torch::stable::Tensor const& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor& q_fp8, torch::stable::Tensor& k_cache,
    torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache,
    torch::stable::Tensor const& fp8_scale,
    torch::stable::Tensor const& q_fp8_scale_inv, double eps,
    int64_t cache_block_size, bool apply_q_norm);

STABLE_TORCH_LIBRARY(_dsv41_swa_C, ops) {
  ops.def(
      "fused_deepseek_v4_kv_rope_insert("
      "Tensor kv, Tensor! k_cache, Tensor slot_mapping, Tensor position_ids, "
      "Tensor cos_sin_cache, int cache_block_size, Tensor? fp8_scale=None, "
      "bool kv_mxfp8=False) -> ()");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor q_in, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size, "
      "bool apply_q_norm=True, bool kv_mxfp8=False, bool apply_q_rope=True, "
      "bool is_q_interleaved=False) -> Tensor");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor position_ids, Tensor cos_sin_cache, float eps, "
      "int cache_block_size, bool apply_q_norm=True) -> ()");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert("
      "Tensor q, Tensor kv, Tensor! q_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "Tensor fp8_scale, Tensor q_fp8_scale_inv, float eps, "
      "int cache_block_size, bool apply_q_norm=True) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_dsv41_swa_C, CUDA, ops) {
  ops.impl("fused_deepseek_v4_kv_rope_insert",
           TORCH_BOX(&fused_deepseek_v4_kv_rope_insert));
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
           TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert));
}

REGISTER_EXTENSION(_dsv41_swa_C)
