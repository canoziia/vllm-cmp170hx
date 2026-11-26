// SPDX-License-Identifier: Apache-2.0
// SM80 Copy-Engine Ring DMA collective primitives (lossless BF16).
//
// Data plane runs on dedicated Copy-Engine streams via cudaMemcpyAsync
// (zero SM usage, full PCIe Gen4/Gen5 bidirectional bandwidth, true hardware
// overlap with concurrent GEMM/MoE on compute streams). Control plane uses
// 1-CTA system-scope 64-bit PTX acquire/release flags and <=64-CTA vectorized
// BF16 local reduction kernels.
//
// O(1) Triple-Buffered Ring (K = 3 fixed slots regardless of world_size):
// matches the exact 3-stage slot lifecycle (inbound DMA -> local add/drain ->
// outbound DMA forward) with lightweight bidirectional ACK flow-control so
// slab VRAM stays constant across 2/4/6/8 GPUs with zero pipeline bubbles.
//
// Canonical chunk ownership: rank r owns chunk r = [r * S : (r + 1) * S],
// matching PyTorch dist.reduce_scatter_single / dist.all_gather_single.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRingSlots = 3;

static __device__ __forceinline__ uint64_t ld_acq_sys_u64(const uint64_t* ptr) {
  uint64_t val;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(val)
               : "l"(ptr)
               : "memory");
  return val;
}

static __device__ __forceinline__ void st_rel_sys_u64(uint64_t* ptr,
                                                      uint64_t val) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
}

__global__ void dma_publish_flag_kernel(uint64_t* remote_flag,
                                        uint64_t expected) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    st_rel_sys_u64(remote_flag, expected);
  }
}

__global__ void dma_wait_flag_kernel(const uint64_t* local_flag,
                                     uint64_t expected) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (ld_acq_sys_u64(local_flag) < expected) {
    }
  }
}

__device__ __forceinline__ uint32_t add_bf16x2_f32(uint32_t x, uint32_t y) {
  __nv_bfloat162 bx = *reinterpret_cast<__nv_bfloat162*>(&x);
  __nv_bfloat162 by = *reinterpret_cast<__nv_bfloat162*>(&y);
  float2 fx = __bfloat1622float2(bx);
  float2 fy = __bfloat1622float2(by);
  float2 fz = make_float2(fx.x + fy.x, fx.y + fy.y);
  __nv_bfloat162 bz = __float22bfloat162_rn(fz);
  return *reinterpret_cast<uint32_t*>(&bz);
}

// Vectorized BF16 local add: out[i] = bf16(f32(a[i]) + f32(b[i])), 8 BF16/thread.
__global__ void dma_add_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                    const __nv_bfloat16* __restrict__ b,
                                    __nv_bfloat16* __restrict__ out,
                                    int64_t n_vec8) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4* a4 = reinterpret_cast<const uint4*>(a);
  const uint4* b4 = reinterpret_cast<const uint4*>(b);
  uint4* o4 = reinterpret_cast<uint4*>(out);
  for (int64_t i = idx; i < n_vec8; i += stride) {
    uint4 va = a4[i];
    uint4 vb = b4[i];
    uint4 vo;
    vo.x = add_bf16x2_f32(va.x, vb.x);
    vo.y = add_bf16x2_f32(va.y, vb.y);
    vo.z = add_bf16x2_f32(va.z, vb.z);
    vo.w = add_bf16x2_f32(va.w, vb.w);
    o4[i] = vo;
  }
}

// 128-byte cacheline stride per flag slot (16 * 8 bytes).
static inline uint64_t* data_flag_slot(uint64_t* base, int slot) {
  return base + slot * 16;
}

static inline uint64_t* ack_flag_slot(uint64_t* base, int slot) {
  return base + (kRingSlots + slot) * 16;
}

struct CeRingChannel {
  int rank = 0;
  int world = 0;
  int64_t max_chunk_bytes = 0;
  void* local_slab = nullptr;
  void* next_slab = nullptr;
  uint64_t* local_flags = nullptr;
  uint64_t* next_flags = nullptr;
  uint64_t* prev_flags = nullptr;
  cudaStream_t copy_stream = nullptr;
  cudaStream_t add_stream = nullptr;
  cudaEvent_t ev_entry = nullptr;
  cudaEvent_t ev_copy0_done = nullptr;
  cudaEvent_t ev_add_hop[16] = {nullptr};
  cudaEvent_t ev_ag_hop[16] = {nullptr};
  cudaEvent_t ev_exit_add = nullptr;
  cudaEvent_t ev_exit_copy = nullptr;
  uint64_t seq = 0;

  ~CeRingChannel() {
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (add_stream) cudaStreamDestroy(add_stream);
    if (ev_entry) cudaEventDestroy(ev_entry);
    if (ev_copy0_done) cudaEventDestroy(ev_copy0_done);
    if (ev_exit_add) cudaEventDestroy(ev_exit_add);
    if (ev_exit_copy) cudaEventDestroy(ev_exit_copy);
    for (int i = 0; i < 16; ++i) {
      if (ev_add_hop[i]) cudaEventDestroy(ev_add_hop[i]);
      if (ev_ag_hop[i]) cudaEventDestroy(ev_ag_hop[i]);
    }
  }
};

static std::unordered_map<int, std::unique_ptr<CeRingChannel>> g_channels;

CeRingChannel* get_chan(int chan_id) {
  auto it = g_channels.find(chan_id);
  TORCH_CHECK(it != g_channels.end() && it->second != nullptr,
              "dsv41_pcie_dma channel not initialized: ", chan_id);
  return it->second.get();
}

}  // namespace

int ring_num_slots() { return kRingSlots; }

int can_access_peer(int dev, int peer) {
  int can = 0;
  cudaError_t err = cudaDeviceCanAccessPeer(&can, dev, peer);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return can;
}

int64_t alloc_ipc_buffer(int64_t nbytes) {
  TORCH_CHECK(nbytes > 0, "alloc_ipc_buffer requires nbytes > 0");
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, static_cast<size_t>(nbytes));
  TORCH_CHECK(err == cudaSuccess,
              "cudaMalloc(", nbytes, ") failed: ", cudaGetErrorString(err));
  err = cudaMemset(ptr, 0, static_cast<size_t>(nbytes));
  TORCH_CHECK(err == cudaSuccess,
              "cudaMemset failed: ", cudaGetErrorString(err));
  return reinterpret_cast<int64_t>(ptr);
}

void free_ipc_buffer(int64_t ptr) {
  if (ptr != 0) {
    cudaFree(reinterpret_cast<void*>(ptr));
  }
}

std::vector<uint8_t> get_ipc_handle_bytes(int64_t ptr) {
  cudaIpcMemHandle_t h;
  cudaError_t err = cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr));
  TORCH_CHECK(err == cudaSuccess,
              "cudaIpcGetMemHandle failed: ", cudaGetErrorString(err));
  std::vector<uint8_t> out(sizeof(cudaIpcMemHandle_t));
  std::memcpy(out.data(), &h, sizeof(cudaIpcMemHandle_t));
  return out;
}

int64_t open_ipc_handle_bytes(const std::vector<uint8_t>& bytes) {
  cudaIpcMemHandle_t h;
  TORCH_CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t),
              "invalid cudaIpcMemHandle byte length");
  std::memcpy(&h, bytes.data(), sizeof(cudaIpcMemHandle_t));
  void* ptr = nullptr;
  cudaError_t err =
      cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
  TORCH_CHECK(err == cudaSuccess,
              "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(err));
  return reinterpret_cast<int64_t>(ptr);
}

void close_ipc_handle(int64_t ptr) {
  if (ptr != 0) {
    cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
  }
}

void init_channel(int chan_id, int rank, int world, int64_t max_chunk_bytes,
                  int64_t local_slab_ptr, int64_t next_slab_ptr,
                  int64_t local_flags_ptr, int64_t next_flags_ptr,
                  int64_t prev_flags_ptr) {
  TORCH_CHECK(world >= 2 && world <= 16,
              "dsv41_pcie_dma supports world_size in [2, 16], got ", world);
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank ", rank);
  TORCH_CHECK(max_chunk_bytes > 0 && (max_chunk_bytes % 16) == 0,
              "max_chunk_bytes must be positive and 16-byte aligned");
  TORCH_CHECK(prev_flags_ptr != 0, "prev_flags_ptr must be non-null");

  auto c = std::make_unique<CeRingChannel>();
  c->rank = rank;
  c->world = world;
  c->max_chunk_bytes = max_chunk_bytes;
  c->local_slab = reinterpret_cast<void*>(local_slab_ptr);
  c->next_slab = reinterpret_cast<void*>(next_slab_ptr);
  c->local_flags = reinterpret_cast<uint64_t*>(local_flags_ptr);
  c->next_flags = reinterpret_cast<uint64_t*>(next_flags_ptr);
  c->prev_flags = reinterpret_cast<uint64_t*>(prev_flags_ptr);
  c->seq = 0;

  TORCH_CHECK(cudaStreamCreateWithFlags(&c->copy_stream,
                                        cudaStreamNonBlocking) == cudaSuccess);
  TORCH_CHECK(cudaStreamCreateWithFlags(&c->add_stream,
                                        cudaStreamNonBlocking) == cudaSuccess);
  TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_entry, cudaEventDisableTiming) ==
              cudaSuccess);
  TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_copy0_done,
                                       cudaEventDisableTiming) == cudaSuccess);
  TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_exit_add,
                                       cudaEventDisableTiming) == cudaSuccess);
  TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_exit_copy,
                                       cudaEventDisableTiming) == cudaSuccess);
  for (int i = 0; i < 16; ++i) {
    TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_add_hop[i],
                                         cudaEventDisableTiming) ==
                cudaSuccess);
    TORCH_CHECK(cudaEventCreateWithFlags(&c->ev_ag_hop[i],
                                         cudaEventDisableTiming) ==
                cudaSuccess);
  }
  g_channels[chan_id] = std::move(c);
}


// Vectorized Block-32 symmetric INT8 + FP16-scale quantize:
//   inp: [chunk_elems] bf16 (chunk_elems % 32 == 0)
//   out_bytes layout: [chunk_elems] int8_t followed by [chunk_elems / 32] __half scales.
// 4 threads per 32-element block, each thread handles 8 BF16 elements (128-bit load, 64-bit store).
__global__ void dma_quant_bf16_to_int8_blk32_kernel(
    const __nv_bfloat16* __restrict__ inp,
    uint8_t* __restrict__ out_bytes,
    int64_t n_blocks32) {
  int8_t* q_ptr = reinterpret_cast<int8_t*>(out_bytes);
  __half* s_ptr = reinterpret_cast<__half*>(out_bytes + n_blocks32 * 32);
  const uint4* inp4 = reinterpret_cast<const uint4*>(inp);
  uint2* q2 = reinterpret_cast<uint2*>(q_ptr);

  const int64_t lane_in_blk = threadIdx.x & 3;
  const int64_t blk_start = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 2;
  const int64_t blk_stride = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 2;

  for (int64_t b = blk_start; b < n_blocks32; b += blk_stride) {
    const int64_t v_idx = b * 4 + lane_in_blk;
    uint4 v = inp4[v_idx];
    float f[8];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 bx = *reinterpret_cast<const __nv_bfloat162*>(&words[i]);
      float2 fx = __bfloat1622float2(bx);
      f[2 * i] = fx.x;
      f[2 * i + 1] = fx.y;
      amax = fmaxf(amax, fmaxf(fabsf(fx.x), fabsf(fx.y)));
    }
    // 4-thread warp-subgroup reduction (zero SMEM, zero __syncthreads)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));

    // Clamp tiny/zero amax safely; round scale to __half so quant & dequant match exactly.
    float raw_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    __half h_scale = __float2half_rn(raw_scale);
    float f_scale = __half2float(h_scale);
    if (amax > 0.0f && f_scale == 0.0f) {
      // Handle subnormal amax that underflowed fp16: bump to smallest positive fp16
      unsigned short min_pos = 1;
      h_scale = *reinterpret_cast<const __half*>(&min_pos);
      f_scale = __half2float(h_scale);
    }
    float inv_scale = (f_scale > 0.0f) ? (1.0f / f_scale) : 0.0f;

    uint32_t p0 = 0, p1 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int qv = __float2int_rn(f[i] * inv_scale);
      qv = max(-127, min(127, qv));
      p0 |= (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(qv))) << (i * 8));
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int qv = __float2int_rn(f[i + 4] * inv_scale);
      qv = max(-127, min(127, qv));
      p1 |= (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(qv))) << (i * 8));
    }
    q2[v_idx] = make_uint2(p0, p1);
    if (lane_in_blk == 0) {
      s_ptr[b] = h_scale;
    }
  }
}

// Vectorized Block-32 symmetric INT8 + FP16-scale dequantize -> BF16.
__global__ void dma_dequant_int8_blk32_to_bf16_kernel(
    const uint8_t* __restrict__ inp_bytes,
    __nv_bfloat16* __restrict__ out,
    int64_t n_blocks32) {
  const int8_t* q_ptr = reinterpret_cast<const int8_t*>(inp_bytes);
  const __half* s_ptr = reinterpret_cast<const __half*>(inp_bytes + n_blocks32 * 32);
  const uint2* q2 = reinterpret_cast<const uint2*>(q_ptr);
  uint4* out4 = reinterpret_cast<uint4*>(out);

  const int64_t lane_in_blk = threadIdx.x & 3;
  const int64_t blk_start = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 2;
  const int64_t blk_stride = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 2;

  for (int64_t b = blk_start; b < n_blocks32; b += blk_stride) {
    const int64_t v_idx = b * 4 + lane_in_blk;
    const uint2 pv = q2[v_idx];
    const float sc = __half2float(s_ptr[b]);
    const int8_t* qb = reinterpret_cast<const int8_t*>(&pv);

    uint4 vo;
    __nv_bfloat162 b0 = __float22bfloat162_rn(make_float2(qb[0] * sc, qb[1] * sc));
    __nv_bfloat162 b1 = __float22bfloat162_rn(make_float2(qb[2] * sc, qb[3] * sc));
    __nv_bfloat162 b2 = __float22bfloat162_rn(make_float2(qb[4] * sc, qb[5] * sc));
    __nv_bfloat162 b3 = __float22bfloat162_rn(make_float2(qb[6] * sc, qb[7] * sc));
    vo.x = *reinterpret_cast<uint32_t*>(&b0);
    vo.y = *reinterpret_cast<uint32_t*>(&b1);
    vo.z = *reinterpret_cast<uint32_t*>(&b2);
    vo.w = *reinterpret_cast<uint32_t*>(&b3);
    out4[v_idx] = vo;
  }
}

// Block-32 INT8 compressed Ring All-Gather with hop-overlapped dequantization:
//   shard: [S, H] bf16 -> out: [W*S, H] bf16 (bitwise identical across ALL W ranks!)
//   q_scratch: contiguous uint8 buffer with numel >= chunk_elems * 17 / 16
void ce_all_gather_int8(int chan_id, torch::Tensor shard, torch::Tensor out,
                        torch::Tensor q_scratch, int64_t caller_stream_int) {
  auto* c = get_chan(chan_id);
  const int W = c->world;
  const int r = c->rank;
  TORCH_CHECK(shard.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "ce_all_gather_int8 requires bfloat16 shard and out");
  TORCH_CHECK(q_scratch.scalar_type() == torch::kUInt8 && q_scratch.is_contiguous(),
              "ce_all_gather_int8 requires contiguous uint8 q_scratch");
  TORCH_CHECK(shard.is_contiguous() && out.is_contiguous(),
              "ce_all_gather_int8 requires contiguous shard and out");
  const int64_t chunk_elems = shard.numel();
  TORCH_CHECK(chunk_elems % 32 == 0, "shard numel must be divisible by 32");
  TORCH_CHECK(out.numel() == chunk_elems * W, "out numel mismatch");
  const int64_t n_blocks32 = chunk_elems / 32;
  const int64_t packed_bytes = chunk_elems + n_blocks32 * 2;
  TORCH_CHECK(q_scratch.numel() >= packed_bytes, "q_scratch too small");
  TORCH_CHECK(packed_bytes <= c->max_chunk_bytes,
              "packed_bytes exceeds channel max_chunk_bytes");

  cudaStream_t caller_s = reinterpret_cast<cudaStream_t>(caller_stream_int);
  const __nv_bfloat16* shard_ptr =
      reinterpret_cast<const __nv_bfloat16*>(shard.data_ptr());
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  uint8_t* q_ptr = reinterpret_cast<uint8_t*>(q_scratch.data_ptr());
  uint8_t* local_base = reinterpret_cast<uint8_t*>(c->local_slab);
  uint8_t* next_base = reinterpret_cast<uint8_t*>(c->next_slab);

  // 1. Quantize local shard into q_scratch on caller stream (~6.5 us for S=768, H=5120)
  const int q_threads = 256;
  const int q_grid = static_cast<int>(
      std::min<int64_t>(256, (n_blocks32 * 4 + q_threads - 1) / q_threads));
  dma_quant_bf16_to_int8_blk32_kernel<<<q_grid, q_threads, 0, caller_s>>>(
      shard_ptr, q_ptr, n_blocks32);

  cudaEventRecord(c->ev_entry, caller_s);
  cudaStreamWaitEvent(c->copy_stream, c->ev_entry, 0);
  cudaStreamWaitEvent(c->add_stream, c->ev_entry, 0);

  // 2. Dequantize local shard into out[r] on add_stream so ALL W ranks get
  //    bitwise identical outputs (hidden behind hop 0 PCIe DMA!).
  const int dq_grid = static_cast<int>(
      std::min<int64_t>(128, (n_blocks32 * 4 + q_threads - 1) / q_threads));
  dma_dequant_int8_blk32_to_bf16_kernel<<<dq_grid, q_threads, 0, c->add_stream>>>(
      q_ptr, out_ptr + r * chunk_elems, n_blocks32);

  // 3. Pipelined W-1 ring hops with hop-overlapped dequantization on add_stream
  for (int h = 0; h < W - 1; ++h) {
    const uint64_t s_seq = ++c->seq;
    const int slot = static_cast<int>((s_seq - 1) % kRingSlots);
    const uint64_t prev_seq = s_seq - 1;
    const int prev_slot =
        (h > 0) ? static_cast<int>((prev_seq - 1) % kRingSlots) : 0;

    const int recv_chunk = (r - 1 - h + 2 * W) % W;
    const uint8_t* src_base =
        (h == 0) ? q_ptr : (local_base + prev_slot * c->max_chunk_bytes);
    uint8_t* dst_remote = next_base + slot * c->max_chunk_bytes;
    const uint8_t* recv_local = local_base + slot * c->max_chunk_bytes;

    if (h > 0) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          data_flag_slot(c->local_flags, prev_slot), prev_seq);
    }
    if (s_seq > static_cast<uint64_t>(kRingSlots)) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->local_flags, slot), s_seq - kRingSlots);
    }
    cudaMemcpyAsync(dst_remote, src_base, packed_bytes,
                    cudaMemcpyDeviceToDevice, c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        data_flag_slot(c->next_flags, slot), s_seq);
    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_ag_hop[h - 1], 0);
      dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->prev_flags, prev_slot), prev_seq);
    }

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        data_flag_slot(c->local_flags, slot), s_seq);
    dma_dequant_int8_blk32_to_bf16_kernel<<<dq_grid, q_threads, 0, c->add_stream>>>(
        recv_local, out_ptr + recv_chunk * chunk_elems, n_blocks32);
    if (h < W - 2) {
      cudaEventRecord(c->ev_ag_hop[h], c->add_stream);
    } else {
      dma_publish_flag_kernel<<<1, 1, 0, c->add_stream>>>(
          ack_flag_slot(c->prev_flags, slot), s_seq);
    }
  }

  cudaEventRecord(c->ev_exit_add, c->add_stream);
  cudaEventRecord(c->ev_exit_copy, c->copy_stream);
  cudaStreamWaitEvent(caller_s, c->ev_exit_add, 0);
  cudaStreamWaitEvent(caller_s, c->ev_exit_copy, 0);
}

// Canonical Ring Reduce-Scatter: inp [M, H] bf16 -> out [M/W, H] bf16
// Rank r receives the reduced sum of chunk r = [r * S : (r + 1) * S].
void ce_reduce_scatter(int chan_id, torch::Tensor inp, torch::Tensor out,
                       int64_t caller_stream_int) {
  auto* c = get_chan(chan_id);
  const int W = c->world;
  const int r = c->rank;
  TORCH_CHECK(inp.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "ce_reduce_scatter requires bfloat16 tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
              "ce_reduce_scatter requires contiguous tensors");
  const int64_t total_elems = inp.numel();
  TORCH_CHECK(total_elems % (W * 8) == 0,
              "numel must be divisible by world * 8");
  const int64_t chunk_elems = total_elems / W;
  TORCH_CHECK(out.numel() == chunk_elems, "out numel mismatch");
  const int64_t chunk_bytes = chunk_elems * 2;
  TORCH_CHECK(chunk_bytes <= c->max_chunk_bytes,
              "chunk_bytes exceeds channel max_chunk_bytes");

  cudaStream_t caller_s = reinterpret_cast<cudaStream_t>(caller_stream_int);
  const __nv_bfloat16* inp_ptr =
      reinterpret_cast<const __nv_bfloat16*>(inp.data_ptr());
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  uint8_t* local_base = reinterpret_cast<uint8_t*>(c->local_slab);
  uint8_t* next_base = reinterpret_cast<uint8_t*>(c->next_slab);

  cudaEventRecord(c->ev_entry, caller_s);
  cudaStreamWaitEvent(c->copy_stream, c->ev_entry, 0);
  cudaStreamWaitEvent(c->add_stream, c->ev_entry, 0);

  for (int h = 0; h < W - 1; ++h) {
    const uint64_t s_seq = ++c->seq;
    const int slot = static_cast<int>((s_seq - 1) % kRingSlots);
    const uint64_t prev_seq = s_seq - 1;
    const int prev_slot =
        (h > 0) ? static_cast<int>((prev_seq - 1) % kRingSlots) : 0;

    const int send_chunk = (r - 1 - h + 2 * W) % W;
    const int recv_chunk = (r - 2 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (inp_ptr + send_chunk * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + prev_slot * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot * c->max_chunk_bytes);
    const __nv_bfloat16* local_add_src = inp_ptr + recv_chunk * chunk_elems;
    __nv_bfloat16* local_add_dst = (h == W - 2) ? out_ptr : recv_local;

    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[h - 1], 0);
    }
    if (s_seq > static_cast<uint64_t>(kRingSlots)) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->local_flags, slot), s_seq - kRingSlots);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        data_flag_slot(c->next_flags, slot), s_seq);
    if (h > 0) {
      dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->prev_flags, prev_slot), prev_seq);
    }

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        data_flag_slot(c->local_flags, slot), s_seq);
    const int64_t n_vec8 = chunk_elems / 8;
    const int grid =
        static_cast<int>(std::min<int64_t>(64, (n_vec8 + 255) / 256));
    dma_add_bf16_kernel<<<grid, 256, 0, c->add_stream>>>(
        recv_local, local_add_src, local_add_dst, n_vec8);
    if (h < W - 2) {
      cudaEventRecord(c->ev_add_hop[h], c->add_stream);
    } else {
      dma_publish_flag_kernel<<<1, 1, 0, c->add_stream>>>(
          ack_flag_slot(c->prev_flags, slot), s_seq);
    }
  }

  cudaEventRecord(c->ev_exit_add, c->add_stream);
  cudaEventRecord(c->ev_exit_copy, c->copy_stream);
  cudaStreamWaitEvent(caller_s, c->ev_exit_add, 0);
  cudaStreamWaitEvent(caller_s, c->ev_exit_copy, 0);
}

// Canonical Ring All-Gather: shard [M/W, H] bf16 -> out [M, H] bf16
// Rank r supplies chunk r = [r * S : (r + 1) * S].
void ce_all_gather(int chan_id, torch::Tensor shard, torch::Tensor out,
                   int64_t caller_stream_int) {
  auto* c = get_chan(chan_id);
  const int W = c->world;
  const int r = c->rank;
  TORCH_CHECK(shard.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "ce_all_gather requires bfloat16 tensors");
  TORCH_CHECK(shard.is_contiguous() && out.is_contiguous(),
              "ce_all_gather requires contiguous tensors");
  const int64_t chunk_elems = shard.numel();
  TORCH_CHECK(chunk_elems % 8 == 0, "shard numel must be divisible by 8");
  TORCH_CHECK(out.numel() == chunk_elems * W, "out numel mismatch");
  const int64_t chunk_bytes = chunk_elems * 2;
  TORCH_CHECK(chunk_bytes <= c->max_chunk_bytes,
              "chunk_bytes exceeds channel max_chunk_bytes");

  cudaStream_t caller_s = reinterpret_cast<cudaStream_t>(caller_stream_int);
  const __nv_bfloat16* shard_ptr =
      reinterpret_cast<const __nv_bfloat16*>(shard.data_ptr());
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  uint8_t* local_base = reinterpret_cast<uint8_t*>(c->local_slab);
  uint8_t* next_base = reinterpret_cast<uint8_t*>(c->next_slab);

  cudaEventRecord(c->ev_entry, caller_s);
  cudaStreamWaitEvent(c->copy_stream, c->ev_entry, 0);
  cudaStreamWaitEvent(c->add_stream, c->ev_entry, 0);

  // Copy local shard into out[r] if not already aliased there
  if (out_ptr + r * chunk_elems != shard_ptr) {
    cudaMemcpyAsync(out_ptr + r * chunk_elems, shard_ptr, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
  }

  for (int h = 0; h < W - 1; ++h) {
    const uint64_t s_seq = ++c->seq;
    const int slot = static_cast<int>((s_seq - 1) % kRingSlots);
    const uint64_t prev_seq = s_seq - 1;
    const int prev_slot =
        (h > 0) ? static_cast<int>((prev_seq - 1) % kRingSlots) : 0;

    const int recv_chunk = (r - 1 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? shard_ptr
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + prev_slot * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot * c->max_chunk_bytes);

    if (h > 0) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          data_flag_slot(c->local_flags, prev_slot), prev_seq);
    }
    if (s_seq > static_cast<uint64_t>(kRingSlots)) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->local_flags, slot), s_seq - kRingSlots);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        data_flag_slot(c->next_flags, slot), s_seq);
    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_ag_hop[h - 1], 0);
      dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->prev_flags, prev_slot), prev_seq);
    }

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        data_flag_slot(c->local_flags, slot), s_seq);
    cudaMemcpyAsync(out_ptr + recv_chunk * chunk_elems, recv_local, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
    if (h < W - 2) {
      cudaEventRecord(c->ev_ag_hop[h], c->add_stream);
    } else {
      dma_publish_flag_kernel<<<1, 1, 0, c->add_stream>>>(
          ack_flag_slot(c->prev_flags, slot), s_seq);
    }
  }

  cudaEventRecord(c->ev_exit_add, c->add_stream);
  cudaEventRecord(c->ev_exit_copy, c->copy_stream);
  cudaStreamWaitEvent(caller_s, c->ev_exit_add, 0);
  cudaStreamWaitEvent(caller_s, c->ev_exit_copy, 0);
}

// Unified in-place Ring All-Reduce: zero caller-stream bounce between RS & AG.
void ce_all_reduce_inplace(int chan_id, torch::Tensor buf,
                           int64_t caller_stream_int) {
  auto* c = get_chan(chan_id);
  const int W = c->world;
  const int r = c->rank;
  TORCH_CHECK(buf.scalar_type() == torch::kBFloat16,
              "ce_all_reduce_inplace requires bfloat16 tensor");
  TORCH_CHECK(buf.is_contiguous(),
              "ce_all_reduce_inplace requires contiguous tensor");
  const int64_t total_elems = buf.numel();
  TORCH_CHECK(total_elems % (W * 8) == 0,
              "numel must be divisible by world * 8");
  const int64_t chunk_elems = total_elems / W;
  const int64_t chunk_bytes = chunk_elems * 2;
  TORCH_CHECK(chunk_bytes <= c->max_chunk_bytes,
              "chunk_bytes exceeds channel max_chunk_bytes");

  cudaStream_t caller_s = reinterpret_cast<cudaStream_t>(caller_stream_int);
  __nv_bfloat16* buf_ptr = reinterpret_cast<__nv_bfloat16*>(buf.data_ptr());
  uint8_t* local_base = reinterpret_cast<uint8_t*>(c->local_slab);
  uint8_t* next_base = reinterpret_cast<uint8_t*>(c->next_slab);

  cudaEventRecord(c->ev_entry, caller_s);
  cudaStreamWaitEvent(c->copy_stream, c->ev_entry, 0);
  cudaStreamWaitEvent(c->add_stream, c->ev_entry, 0);

  // ---- Phase 1: Reduce-Scatter into canonical chunk r ----
  for (int h = 0; h < W - 1; ++h) {
    const uint64_t s_seq = ++c->seq;
    const int slot = static_cast<int>((s_seq - 1) % kRingSlots);
    const uint64_t prev_seq = s_seq - 1;
    const int prev_slot =
        (h > 0) ? static_cast<int>((prev_seq - 1) % kRingSlots) : 0;

    const int send_chunk = (r - 1 - h + 2 * W) % W;
    const int recv_chunk = (r - 2 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (buf_ptr + send_chunk * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + prev_slot * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot * c->max_chunk_bytes);
    const __nv_bfloat16* local_add_src = buf_ptr + recv_chunk * chunk_elems;
    __nv_bfloat16* local_add_dst =
        (h == W - 2) ? (buf_ptr + r * chunk_elems) : recv_local;

    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[h - 1], 0);
    }
    if (s_seq > static_cast<uint64_t>(kRingSlots)) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->local_flags, slot), s_seq - kRingSlots);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    if (h == 0) {
      cudaEventRecord(c->ev_copy0_done, c->copy_stream);
    }
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        data_flag_slot(c->next_flags, slot), s_seq);
    if (h > 0) {
      dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->prev_flags, prev_slot), prev_seq);
    }

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        data_flag_slot(c->local_flags, slot), s_seq);
    const int64_t n_vec8 = chunk_elems / 8;
    const int grid =
        static_cast<int>(std::min<int64_t>(64, (n_vec8 + 255) / 256));
    dma_add_bf16_kernel<<<grid, 256, 0, c->add_stream>>>(
        recv_local, local_add_src, local_add_dst, n_vec8);
    cudaEventRecord(c->ev_add_hop[h], c->add_stream);
    if (h == W - 2) {
      dma_publish_flag_kernel<<<1, 1, 0, c->add_stream>>>(
          ack_flag_slot(c->prev_flags, slot), s_seq);
    }
  }

  // ---- Phase 2: All-Gather from canonical chunk r ----
  cudaStreamWaitEvent(c->add_stream, c->ev_copy0_done, 0);
  cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[W - 2], 0);

  for (int h = 0; h < W - 1; ++h) {
    const uint64_t s_seq = ++c->seq;
    const int slot = static_cast<int>((s_seq - 1) % kRingSlots);
    const uint64_t prev_seq = s_seq - 1;
    const int prev_slot =
        (h > 0) ? static_cast<int>((prev_seq - 1) % kRingSlots) : 0;

    const int recv_chunk = (r - 1 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (buf_ptr + r * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + prev_slot * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot * c->max_chunk_bytes);

    if (h > 0) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          data_flag_slot(c->local_flags, prev_slot), prev_seq);
    }
    if (s_seq > static_cast<uint64_t>(kRingSlots)) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->local_flags, slot), s_seq - kRingSlots);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        data_flag_slot(c->next_flags, slot), s_seq);
    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_ag_hop[h - 1], 0);
      dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          ack_flag_slot(c->prev_flags, prev_slot), prev_seq);
    }

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        data_flag_slot(c->local_flags, slot), s_seq);
    cudaMemcpyAsync(buf_ptr + recv_chunk * chunk_elems, recv_local, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
    if (h < W - 2) {
      cudaEventRecord(c->ev_ag_hop[h], c->add_stream);
    } else {
      dma_publish_flag_kernel<<<1, 1, 0, c->add_stream>>>(
          ack_flag_slot(c->prev_flags, slot), s_seq);
    }
  }

  cudaEventRecord(c->ev_exit_add, c->add_stream);
  cudaEventRecord(c->ev_exit_copy, c->copy_stream);
  cudaStreamWaitEvent(caller_s, c->ev_exit_add, 0);
  cudaStreamWaitEvent(caller_s, c->ev_exit_copy, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ring_num_slots", &ring_num_slots);
  m.def("can_access_peer", &can_access_peer);
  m.def("alloc_ipc_buffer", &alloc_ipc_buffer);
  m.def("free_ipc_buffer", &free_ipc_buffer);
  m.def("get_ipc_handle_bytes", &get_ipc_handle_bytes);
  m.def("open_ipc_handle_bytes", &open_ipc_handle_bytes);
  m.def("close_ipc_handle", &close_ipc_handle);
  m.def("init_channel", &init_channel);
  m.def("ce_reduce_scatter", &ce_reduce_scatter);
  m.def("ce_all_gather_int8", &ce_all_gather_int8, "Block-32 INT8 Ring All-Gather");
  m.def("ce_all_gather", &ce_all_gather);
  m.def("ce_all_reduce_inplace", &ce_all_reduce_inplace);
}
