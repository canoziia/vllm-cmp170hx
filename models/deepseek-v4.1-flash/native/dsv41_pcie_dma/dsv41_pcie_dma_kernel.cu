// SPDX-License-Identifier: Apache-2.0
// SM80 Copy-Engine Ring DMA collective primitives (lossless BF16).
//
// Data plane runs on dedicated Copy-Engine streams via cudaMemcpyAsync
// (zero SM usage, full PCIe Gen4/Gen5 bidirectional bandwidth, true hardware
// overlap with concurrent GEMM/MoE on compute streams). Control plane uses
// 1-CTA system-scope PTX acquire/release flags and <=64-CTA vectorized BF16
// local reduction kernels.
//
// Canonical chunk ownership: rank r owns chunk r = [r * S : (r + 1) * S],
// matching PyTorch dist.reduce_scatter_single / dist.all_gather_single.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

static __device__ __forceinline__ uint32_t ld_acq_sys_u32(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(val)
               : "l"(ptr)
               : "memory");
  return val;
}

static __device__ __forceinline__ void st_rel_sys_u32(uint32_t* ptr,
                                                      uint32_t val) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
}

__global__ void dma_publish_flag_kernel(uint32_t* remote_flag,
                                        uint32_t expected) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    st_rel_sys_u32(remote_flag, expected);
  }
}

__global__ void dma_wait_flag_kernel(const uint32_t* local_flag,
                                     uint32_t expected) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (ld_acq_sys_u32(local_flag) < expected) {
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

static inline uint32_t* flag_slot(uint32_t* base, int idx) {
  return base + idx * 32;  // 128-byte cacheline stride
}

struct CeRingChannel {
  int rank = 0;
  int world = 0;
  int64_t max_chunk_bytes = 0;
  void* local_slab = nullptr;
  void* next_slab = nullptr;
  uint32_t* local_flags = nullptr;
  uint32_t* next_flags = nullptr;
  cudaStream_t copy_stream = nullptr;
  cudaStream_t add_stream = nullptr;
  cudaEvent_t ev_entry = nullptr;
  cudaEvent_t ev_copy0_done = nullptr;
  cudaEvent_t ev_add_hop[16] = {nullptr};
  cudaEvent_t ev_exit_add = nullptr;
  cudaEvent_t ev_exit_copy = nullptr;
  uint32_t epoch = 0;

  ~CeRingChannel() {
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (add_stream) cudaStreamDestroy(add_stream);
    if (ev_entry) cudaEventDestroy(ev_entry);
    if (ev_copy0_done) cudaEventDestroy(ev_copy0_done);
    if (ev_exit_add) cudaEventDestroy(ev_exit_add);
    if (ev_exit_copy) cudaEventDestroy(ev_exit_copy);
    for (int i = 0; i < 16; ++i) {
      if (ev_add_hop[i]) cudaEventDestroy(ev_add_hop[i]);
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
                  int64_t local_flags_ptr, int64_t next_flags_ptr) {
  TORCH_CHECK(world >= 2 && world <= 16,
              "dsv41_pcie_dma supports world_size in [2, 16], got ", world);
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank ", rank);
  TORCH_CHECK(max_chunk_bytes > 0 && (max_chunk_bytes % 16) == 0,
              "max_chunk_bytes must be positive and 16-byte aligned");

  auto c = std::make_unique<CeRingChannel>();
  c->rank = rank;
  c->world = world;
  c->max_chunk_bytes = max_chunk_bytes;
  c->local_slab = reinterpret_cast<void*>(local_slab_ptr);
  c->next_slab = reinterpret_cast<void*>(next_slab_ptr);
  c->local_flags = reinterpret_cast<uint32_t*>(local_flags_ptr);
  c->next_flags = reinterpret_cast<uint32_t*>(next_flags_ptr);
  c->epoch = 0;

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
  }
  g_channels[chan_id] = std::move(c);
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

  c->epoch++;
  const uint32_t ep = c->epoch;
  const int gen_base = (ep & 1) * (W - 1);

  for (int h = 0; h < W - 1; ++h) {
    const int slot_idx = gen_base + h;
    const int send_chunk = (r - 1 - h + 2 * W) % W;
    const int recv_chunk = (r - 2 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (inp_ptr + send_chunk * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + (gen_base + h - 1) * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot_idx * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot_idx * c->max_chunk_bytes);
    const __nv_bfloat16* local_add_src = inp_ptr + recv_chunk * chunk_elems;
    __nv_bfloat16* local_add_dst = (h == W - 2) ? out_ptr : recv_local;

    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[h - 1], 0);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        flag_slot(c->next_flags, slot_idx), ep);

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        flag_slot(c->local_flags, slot_idx), ep);
    const int64_t n_vec8 = chunk_elems / 8;
    const int grid = static_cast<int>(std::min<int64_t>(64, (n_vec8 + 255) / 256));
    dma_add_bf16_kernel<<<grid, 256, 0, c->add_stream>>>(
        recv_local, local_add_src, local_add_dst, n_vec8);
    if (h < W - 2) {
      cudaEventRecord(c->ev_add_hop[h], c->add_stream);
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

  c->epoch++;
  const uint32_t ep = c->epoch;
  const int gen_base = (ep & 1) * (W - 1);

  // Copy local shard into out[r] if not already aliased there
  if (out_ptr + r * chunk_elems != shard_ptr) {
    cudaMemcpyAsync(out_ptr + r * chunk_elems, shard_ptr, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
  }

  for (int h = 0; h < W - 1; ++h) {
    const int slot_idx = gen_base + h;
    const int recv_chunk = (r - 1 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? shard_ptr
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + (gen_base + h - 1) * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot_idx * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot_idx * c->max_chunk_bytes);

    if (h > 0) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          flag_slot(c->local_flags, gen_base + h - 1), ep);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        flag_slot(c->next_flags, slot_idx), ep);

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        flag_slot(c->local_flags, slot_idx), ep);
    cudaMemcpyAsync(out_ptr + recv_chunk * chunk_elems, recv_local, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
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
  c->epoch++;
  const uint32_t ep_rs = c->epoch;
  const int gen_rs = (ep_rs & 1) * (W - 1);

  for (int h = 0; h < W - 1; ++h) {
    const int slot_idx = gen_rs + h;
    const int send_chunk = (r - 1 - h + 2 * W) % W;
    const int recv_chunk = (r - 2 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (buf_ptr + send_chunk * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + (gen_rs + h - 1) * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot_idx * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot_idx * c->max_chunk_bytes);
    const __nv_bfloat16* local_add_src = buf_ptr + recv_chunk * chunk_elems;
    __nv_bfloat16* local_add_dst =
        (h == W - 2) ? (buf_ptr + r * chunk_elems) : recv_local;

    if (h > 0) {
      cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[h - 1], 0);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    if (h == 0) {
      cudaEventRecord(c->ev_copy0_done, c->copy_stream);
    }
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        flag_slot(c->next_flags, slot_idx), ep_rs);

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        flag_slot(c->local_flags, slot_idx), ep_rs);
    const int64_t n_vec8 = chunk_elems / 8;
    const int grid = static_cast<int>(std::min<int64_t>(64, (n_vec8 + 255) / 256));
    dma_add_bf16_kernel<<<grid, 256, 0, c->add_stream>>>(
        recv_local, local_add_src, local_add_dst, n_vec8);
    cudaEventRecord(c->ev_add_hop[h], c->add_stream);
  }

  // ---- Phase 2: All-Gather from canonical chunk r (opposite slab gen) ----
  c->epoch++;
  const uint32_t ep_ag = c->epoch;
  const int gen_ag = (ep_ag & 1) * (W - 1);

  cudaStreamWaitEvent(c->add_stream, c->ev_copy0_done, 0);
  cudaStreamWaitEvent(c->copy_stream, c->ev_add_hop[W - 2], 0);

  for (int h = 0; h < W - 1; ++h) {
    const int slot_idx = gen_ag + h;
    const int recv_chunk = (r - 1 - h + 2 * W) % W;
    const __nv_bfloat16* src_base =
        (h == 0) ? (buf_ptr + r * chunk_elems)
                 : reinterpret_cast<const __nv_bfloat16*>(
                       local_base + (gen_ag + h - 1) * c->max_chunk_bytes);
    __nv_bfloat16* dst_remote = reinterpret_cast<__nv_bfloat16*>(
        next_base + slot_idx * c->max_chunk_bytes);
    __nv_bfloat16* recv_local = reinterpret_cast<__nv_bfloat16*>(
        local_base + slot_idx * c->max_chunk_bytes);

    if (h > 0) {
      dma_wait_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
          flag_slot(c->local_flags, gen_ag + h - 1), ep_ag);
    }
    cudaMemcpyAsync(dst_remote, src_base, chunk_bytes, cudaMemcpyDeviceToDevice,
                    c->copy_stream);
    dma_publish_flag_kernel<<<1, 1, 0, c->copy_stream>>>(
        flag_slot(c->next_flags, slot_idx), ep_ag);

    dma_wait_flag_kernel<<<1, 1, 0, c->add_stream>>>(
        flag_slot(c->local_flags, slot_idx), ep_ag);
    cudaMemcpyAsync(buf_ptr + recv_chunk * chunk_elems, recv_local, chunk_bytes,
                    cudaMemcpyDeviceToDevice, c->add_stream);
  }

  cudaEventRecord(c->ev_exit_add, c->add_stream);
  cudaEventRecord(c->ev_exit_copy, c->copy_stream);
  cudaStreamWaitEvent(caller_s, c->ev_exit_add, 0);
  cudaStreamWaitEvent(caller_s, c->ev_exit_copy, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("can_access_peer", &can_access_peer);
  m.def("alloc_ipc_buffer", &alloc_ipc_buffer);
  m.def("free_ipc_buffer", &free_ipc_buffer);
  m.def("get_ipc_handle_bytes", &get_ipc_handle_bytes);
  m.def("open_ipc_handle_bytes", &open_ipc_handle_bytes);
  m.def("close_ipc_handle", &close_ipc_handle);
  m.def("init_channel", &init_channel);
  m.def("ce_reduce_scatter", &ce_reduce_scatter);
  m.def("ce_all_gather", &ce_all_gather);
  m.def("ce_all_reduce_inplace", &ce_all_reduce_inplace);
}
