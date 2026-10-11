/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Qwen3.8: copy of csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu
 * (vLLM fork d63af5a) on the ATen API with a Record mode for GDN RecoverSSM.
 */

#include <cstdint>
#include <string>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state,
                                                 const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element,
                 state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4
load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(
    __nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate,
          bool Record>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_post_conv_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides, float* __restrict__ rec,
    const int* __restrict__ rec_slots, int rec_t, int rec_f) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  // Record (Qwen3.8 GDN RecoverSSM): verify off the committed running state.
  const int source_slot =
      Record ? state_indices[request * state_indices_width]
             : (accepted > 0 && accepted <= state_indices_width
                    ? state_indices[request * state_indices_width + accepted - 1]
                    : 0);
  const int rec_slot = Record ? rec_slots[request] : -1;
  float* rec_base = nullptr;
  if (Record && rec_slot >= 0) {
    rec_base = rec + static_cast<int64_t>(rec_slot) * rec_t * rec_f;
  }
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0],
                                              source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
      if (Record && rec_base != nullptr &&
          value_head % ValueHeadsPerKeyHead == 0) {
        rec_base[static_cast<int64_t>(t) * rec_f + HV + key_head * kDimK + dim] =
            k_values[i] * k_scale;
      }
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
      if (Record && rec_base != nullptr) {
        rec_base[static_cast<int64_t>(t) * rec_f + value_head] = shared_decay[t];
      }
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
        if (Record && rec_base != nullptr && lane == 0) {
          rec_base[static_cast<int64_t>(t) * rec_f + HV + H * kDimK +
                   value_head * kDimV + value] = delta;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }

      const int destination_slot =
          Record ? 0 : state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_state4(destination_state + value * kDimK + k_base,
                       make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}


template <typename StateT, int VPK, bool SigmoidGate, bool Record>
void launch(const at::Tensor& mixed_qkv, const at::Tensor& a, const at::Tensor& b,
            const at::Tensor& a_log, const at::Tensor& dt_bias,
            const at::Tensor& state_indices, const at::Tensor& cu_seqlens,
            const at::Tensor& num_accepted_tokens, at::Tensor& state,
            const at::Tensor& output_gate, const at::Tensor& norm_weight,
            at::Tensor& out, int H, int HV, int dt_bias_type, double scale,
            double norm_eps, const GdnDecodeStrides& strides, float* rec,
            const int* rec_slots, int rec_t, int rec_f) {
  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<int>(state_indices.size(0)), HV);
  gdn_decode_post_conv_mtp_kernel<StateT, VPK, SigmoidGate, Record>
      <<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          a_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          num_accepted_tokens.data_ptr<int>(),
          static_cast<StateT*>(state.data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
          norm_weight.data_ptr(), static_cast<__nv_bfloat16*>(out.data_ptr()),
          H, HV, static_cast<int>(state_indices.size(1)), dt_bias_type,
          norm_weight.scalar_type() == at::kBFloat16, static_cast<float>(scale),
          static_cast<float>(norm_eps), strides, rec, rec_slots, rec_t, rec_f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Same contract as torch.ops._C.fused_gdn_decode_post_conv_mtp, plus an
// optional record buffer [slots, T, HV + H*128 + HV*128] (fp32) and per-request
// record slots (int32, -1 = do not record). With a record buffer the kernel
// reads column 0 of state_indices, writes no per-token states and records
// (decay, normalised key, delta) for every verified token.
void post_conv_mtp(const at::Tensor& mixed_qkv, const at::Tensor& a,
                   const at::Tensor& b, const at::Tensor& a_log,
                   const at::Tensor& dt_bias, const at::Tensor& state_indices,
                   const at::Tensor& cu_seqlens,
                   const at::Tensor& num_accepted_tokens, at::Tensor state,
                   const at::Tensor& output_gate, const at::Tensor& norm_weight,
                   at::Tensor out, double scale, double norm_eps,
                   const std::string& activation,
                   const std::optional<at::Tensor>& rec,
                   const std::optional<at::Tensor>& rec_slots) {
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kBFloat16 && mixed_qkv.dim() == 2 &&
              mixed_qkv.stride(1) == 1);
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK &&
              state.stride(1) == kDimV * kDimK && state.stride(2) == kDimK &&
              state.stride(3) == 1);
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.is_contiguous() &&
              state_indices.size(1) <= kMaxMtpTokens);
  TORCH_CHECK(cu_seqlens.is_contiguous() && num_accepted_tokens.is_contiguous());
  TORCH_CHECK(output_gate.dim() == 3 && output_gate.stride(2) == 1 &&
              output_gate.stride(1) == kDimV);
  TORCH_CHECK(out.is_contiguous() && a.stride(1) == 1 && b.stride(1) == 1);
  TORCH_CHECK(activation == "silu" || activation == "sigmoid");
  const int HV = static_cast<int>(state.size(1));
  const int64_t key_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * kDimV;
  const int H = static_cast<int>(key_width / (2 * kDimK));
  const int vpk = HV / H;
  const auto dt = dt_bias.scalar_type();
  const int dt_bias_type =
      dt == at::kFloat ? kDtBiasFloat32
                       : (dt == at::kBFloat16 ? kDtBiasBFloat16 : kDtBiasFloat16);
  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                 output_gate.stride(0), state.stride(0)};
  const bool record = rec.has_value();
  float* rec_ptr = nullptr;
  const int* slots_ptr = nullptr;
  int rec_t = 1, rec_f = 1;
  if (record) {
    TORCH_CHECK(rec_slots.has_value() && rec->scalar_type() == at::kFloat &&
                rec->dim() == 3 && rec->is_contiguous() &&
                rec->size(2) == HV + H * kDimK + HV * kDimV &&
                rec_slots->scalar_type() == at::kInt &&
                rec_slots->numel() == state_indices.size(0));
    rec_ptr = rec->data_ptr<float>();
    slots_ptr = rec_slots->data_ptr<int>();
    rec_t = static_cast<int>(rec->size(1));
    rec_f = static_cast<int>(rec->size(2));
  }
  const bool sigmoid = activation == "sigmoid";
  const bool fp32 = state.scalar_type() == at::kFloat;
  TORCH_CHECK(vpk == 3 && fp32, "Qwen3.8 native GDN MTP: HV/H == 3, fp32 state");
#define QWEN38_LAUNCH(SG, REC)                                                  \
  launch<float, 3, SG, REC>(mixed_qkv, a, b, a_log, dt_bias, state_indices,     \
                            cu_seqlens, num_accepted_tokens, state, output_gate, \
                            norm_weight, out, H, HV, dt_bias_type, scale,        \
                            norm_eps, strides, rec_ptr, slots_ptr, rec_t, rec_f)
  if (sigmoid) {
    if (record) QWEN38_LAUNCH(true, true); else QWEN38_LAUNCH(true, false);
  } else {
    if (record) QWEN38_LAUNCH(false, true); else QWEN38_LAUNCH(false, false);
  }
#undef QWEN38_LAUNCH
}

int64_t abi() { return 1; }

TORCH_LIBRARY(_qwen38_gdn_mtp_C, m) {
  m.def(
      "post_conv_mtp(Tensor mixed_qkv, Tensor a, Tensor b, Tensor a_log, "
      "Tensor dt_bias, Tensor state_indices, Tensor cu_seqlens, "
      "Tensor num_accepted_tokens, Tensor(s!) state, Tensor output_gate, "
      "Tensor norm_weight, Tensor(o!) out, float scale, float norm_eps, "
      "str activation, Tensor(r!)? rec, Tensor? rec_slots) -> ()");
  m.def("abi() -> int");
}

TORCH_LIBRARY_IMPL(_qwen38_gdn_mtp_C, CUDA, m) {
  m.impl("post_conv_mtp", &post_conv_mtp);
}

TORCH_LIBRARY_IMPL(_qwen38_gdn_mtp_C, CompositeExplicitAutograd, m) {
  m.impl("abi", &abi);
}
