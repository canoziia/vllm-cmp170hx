// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace dsv41_r2 {

// ============================================================================
// 1. Single-CTA Shared-Memory Deterministic MoE Align (S <= 4096, E <= 1024)
//    Bitwise identical to _moe_align_block_size_deterministic, ~3 us at M=192!
// ============================================================================
template <typename IdT>
__global__ void __launch_bounds__(1024) det_align_v2_kernel(
    const IdT* __restrict__ ids,
    const int32_t* __restrict__ emap,
    int S,
    int global_e,
    int local_e,
    int BS,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  __shared__ int16_t sid[4096];
  __shared__ int32_t s_sorted[2048];
  __shared__ int cnt[1024];
  __shared__ int sblk[1024];
  __shared__ int wsum[32];

  const int tid = threadIdx.x;
  cnt[tid] = 0;
  __syncthreads();

  for (int i = tid; i < S; i += 1024) {
    const int64_t raw = static_cast<int64_t>(ids[i]);
    int e = -1;
    if (raw >= 0 && raw < global_e) {
      e = (emap != nullptr) ? emap[raw] : static_cast<int>(raw);
      if (e < 0 || e >= local_e) e = -1;
    }
    sid[i] = static_cast<int16_t>(e);
    if (e >= 0) atomicAdd(&cnt[e], 1);
  }
  // Pad tail up to multiple of 8 for vectorized int4 broadcast scan
  const int s_pad = (S + 7) & ~7;
  for (int i = S + tid; i < s_pad; i += 1024) {
    sid[i] = -1;
  }
  __syncthreads();

  const int my_cnt = (tid < local_e) ? cnt[tid] : 0;
  const int myblk = (my_cnt + BS - 1) / BS;
  const int lane = tid & 31, wid = tid >> 5;
  int inc = myblk;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += t;
  }
  if (lane == 31) wsum[wid] = inc;
  __syncthreads();
  if (wid == 0) {
    int v = wsum[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int t = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += t;
    }
    wsum[lane] = v;
  }
  __syncthreads();

  const int excl = inc - myblk + (wid > 0 ? wsum[wid - 1] : 0);
  sblk[tid] = excl;
  const int total_blk = wsum[31];
  const int total = total_blk * BS;

  for (int b = tid; b < nb; b += 1024) {
    if (b >= total_blk) eids[b] = -1;
  }
  if (tid < local_e) {
    for (int k = 0; k < myblk; ++k) {
      if (excl + k < nb) eids[excl + k] = tid;
    }
  }
  if (tid == 0) *ntpp = total;

  const bool use_smem_out = (total <= 2048);
  if (use_smem_out) {
    for (int p = tid; p < total; p += 1024) s_sorted[p] = S;
    // Fill tail [total, cap) in global memory directly
    const int tail_start4 = (total + 3) >> 2;
    const int cap4 = cap >> 2;
    const int4 fill4 = make_int4(S, S, S, S);
    int4* sorted4 = reinterpret_cast<int4*>(sorted);
    for (int p = total + tid; p < min(cap, tail_start4 << 2); p += 1024) sorted[p] = S;
    for (int p4 = tail_start4 + tid; p4 < cap4; p4 += 1024) sorted4[p4] = fill4;
    for (int p = (cap4 << 2) + tid; p < cap; p += 1024) sorted[p] = S;
  } else {
    const int4 fill4 = make_int4(S, S, S, S);
    int4* sorted4 = reinterpret_cast<int4*>(sorted);
    const int cap4 = cap >> 2;
    for (int p = tid; p < cap4; p += 1024) sorted4[p] = fill4;
    for (int p = (cap4 << 2) + tid; p < cap; p += 1024) sorted[p] = S;
  }
  __syncthreads();

  int32_t* dst_sorted = use_smem_out ? s_sorted : sorted;
  if (S <= 192) {
    for (int i = tid; i < S; i += 1024) {
      const int e = sid[i];
      if (e < 0) continue;
      int rank = 0;
      for (int j = 0; j < i; ++j) rank += (sid[j] == e);
      dst_sorted[sblk[e] * BS + rank] = i;
    }
  } else {
    if (tid < local_e && my_cnt > 0) {
      int out_pos = excl * BS;
      const int4* sid4 = reinterpret_cast<const int4*>(sid);
      const int n8 = s_pad >> 3;
      const int16_t target = static_cast<int16_t>(tid);
      for (int g = 0; g < n8; ++g) {
        const int4 v = sid4[g];
        const int16_t* s8 = reinterpret_cast<const int16_t*>(&v);
        const int base_j = g << 3;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          if (s8[u] == target) dst_sorted[out_pos++] = base_j + u;
        }
      }
    }
  }
  if (use_smem_out) {
    __syncthreads();
    const int total4 = total >> 2;
    int4* sorted4 = reinterpret_cast<int4*>(sorted);
    const int4* s_sorted4 = reinterpret_cast<const int4*>(s_sorted);
    for (int p4 = tid; p4 < total4; p4 += 1024) sorted4[p4] = s_sorted4[p4];
    for (int p = (total4 << 2) + tid; p < total; p += 1024) sorted[p] = s_sorted[p];
  }
}

void det_align_v2(
    const at::Tensor& ids,
    const c10::optional<at::Tensor>& emap_opt,
    int64_t global_e,
    int64_t local_e,
    int64_t BS,
    at::Tensor& sorted,
    at::Tensor& eids,
    at::Tensor& ntpp) {
  const int S = static_cast<int>(ids.numel());
  TORCH_CHECK(S > 0 && S <= 4096 && local_e > 0 && local_e <= 1024, "det_align_v2 bounds");
  const at::cuda::OptionalCUDAGuard guard(ids.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int32_t* emap_ptr =
      (emap_opt.has_value() && emap_opt->defined()) ? emap_opt->data_ptr<int32_t>() : nullptr;
  if (ids.scalar_type() == at::kInt) {
    det_align_v2_kernel<int32_t><<<1, 1024, 0, st>>>(
        ids.data_ptr<int32_t>(), emap_ptr, S, (int)global_e, (int)local_e, (int)BS,
        sorted.data_ptr<int32_t>(), (int)sorted.numel(),
        eids.data_ptr<int32_t>(), (int)eids.numel(), ntpp.data_ptr<int32_t>());
  } else {
    det_align_v2_kernel<int64_t><<<1, 1024, 0, st>>>(
        ids.data_ptr<int64_t>(), emap_ptr, S, (int)global_e, (int)local_e, (int)BS,
        sorted.data_ptr<int32_t>(), (int)sorted.numel(),
        eids.data_ptr<int32_t>(), (int)eids.numel(), ntpp.data_ptr<int32_t>());
  }
}

// ============================================================================
// 2. MXFP4 MoE Decode w2 Split-K (KS = 3, 4, 6) + L2-Fused EP moe_sum
//    Solves w2 SM/warp starvation (80..120 CTAs -> 320..720 CTAs across 108..126 SMs!)
// ============================================================================
__device__ __forceinline__ __nv_bfloat162 u2b(uint32_t v) {
  return *reinterpret_cast<__nv_bfloat162*>(&v);
}
__device__ __forceinline__ uint32_t b2u(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16_z(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                           uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ uint32_t pl(uint32_t t) {
  const uint32_t c = (t >> 6) & 0x01C001C0u;
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(t), "r"(0x80008000u), "r"(c));
  return r;
}
template <int DQ>
__device__ __forceinline__ void deq_raw(uint32_t q, uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                        uint32_t& a3) {
  if constexpr (DQ == 3) {
    a0 = ((q & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
    a2 = ((q & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
    const uint32_t qh = q >> 8;
    a1 = ((qh & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
    a3 = ((qh & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
    return;
  }
  if constexpr (DQ == 2) {
    a0 = ((q & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
    a2 = ((q & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
  } else {
    a0 = pl(q << 12);
    a2 = pl(q << 8);
  }
  a1 = pl(q << 4);
  a3 = pl(q);
}

constexpr int WARPS = 4;
constexpr uint32_t XPRE = 0x71807180u;
constexpr float EPI = 67108864.0f;

template <int CR, int TW>
struct Frags {
  uint32_t x[CR][2];
  uint2 s[CR / 2][TW];
};

__device__ __forceinline__ int atomic_add_acq_rel_gpu(int* addr, int val) {
  int prev;
  asm volatile("atom.global.acq_rel.gpu.add.s32 %0, [%1], %2;\n"
               : "=r"(prev)
               : "l"(addr), "r"(val)
               : "memory");
  return prev;
}

template <int DQ, int R, int CR, int TW, int K, int N>
__global__ void __launch_bounds__(WARPS * 32) mxfp4_w2_splitk_kernel(
    const __nv_bfloat16* __restrict__ A,
    int64_t a_stride,
    const int4* __restrict__ W,
    const uint8_t* __restrict__ S,
    const int32_t* __restrict__ sorted,
    const int32_t* __restrict__ eids,
    const int32_t* __restrict__ ntpp,
    int n_slots,
    int* __restrict__ ctr,
    int* __restrict__ ctr_other,
    float* __restrict__ part2) {
  static_assert(CR % 2 == 0 && R % CR == 0, "chunks");
  static_assert(K % (16 * R) == 0 && N % (64 * TW) == 0, "shape");
  constexpr int CH = R / CR;
  constexpr int TG = N / 64 / TW;
  constexpr int KS = K / 16 / R;
  constexpr int SLOT = CR * TW * 32;
  extern __shared__ int4 smem_raw[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c = lane & 3;
  int4* const ring0 = smem_raw + warp * (2 * SLOT) + lane;
  int4* const ring1 = ring0 + SLOT;
  if (blockIdx.x == 0 && threadIdx.x == 0) *ctr_other = 0;

  constexpr int krows = K >> 4;
  constexpr int wrow = N >> 1;
  const int nblk = ntpp[0] >> 3;
  const int items = nblk * TG * KS;

  auto split = [&](int item, int& b, int& ks, int& t0) {
    t0 = (item % TG) * TW;
    const int rest = item / TG;
    ks = rest % KS;
    b = rest / KS;
  };

  int l_item = 0, l_ch = 0;
  const int4* l_w = W;
  const uint8_t* l_s = S;
  const uint32_t* l_x = reinterpret_cast<const uint32_t*>(A);
  auto grab = [&]() {
    int v = 0;
    if (lane == 0) v = atomicAdd(ctr, 1);
    l_item = __shfl_sync(0xffffffffu, v, 0);
    l_ch = 0;
    if (l_item < items) {
      int b, ks, t0;
      split(l_item, b, ks, t0);
      const int e0 = eids[b];
      const int e = e0 < 0 ? 0 : e0;
      const int r0 = ks * R;
      l_w = W + ((int64_t)e * krows + r0) * wrow + t0 * 32 + lane;
      l_s = S + ((int64_t)e * (K / 32) + (r0 >> 1)) * N + t0 * 64 + 8 * g;
      const int sid = sorted[b * 8 + g];
      const int arow = (e0 >= 0 && sid < n_slots) ? sid : 0;
      l_x = reinterpret_cast<const uint32_t*>(A + (int64_t)arow * a_stride + r0 * 16 + 2 * c);
    }
  };

  auto load = [&](int4* dst, Frags<CR, TW>& f) {
    const int4* src = l_w + l_ch * (CR * wrow);
#pragma unroll
    for (int i = 0; i < CR; i++)
#pragma unroll
      for (int u = 0; u < TW; u++) cp_async16(dst + (i * TW + u) * 32, src + i * wrow + u * 32);
    cp_async_commit();
    const uint8_t* sp = l_s + (int64_t)(l_ch * (CR / 2)) * N;
#pragma unroll
    for (int gi = 0; gi < CR / 2; gi++)
#pragma unroll
      for (int u = 0; u < TW; u++)
        f.s[gi][u] = __ldg(reinterpret_cast<const uint2*>(sp + (int64_t)gi * N + u * 64));
    const uint32_t* ap = l_x + l_ch * CR * 8;
#pragma unroll
    for (int i = 0; i < CR; i++) {
      f.x[i][0] = __ldg(ap + i * 8);
      f.x[i][1] = __ldg(ap + i * 8 + 4);
    }
  };

  float acc[TW][4][4];
#pragma unroll
  for (int u = 0; u < TW; u++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;

  auto epilogue = [&](int item) {
    int b, ks, t0;
    split(item, b, ks, t0);
    const bool ok = eids[b] >= 0;
    const int s0 = sorted[b * 8 + 2 * c];
    const int s1 = sorted[b * 8 + 2 * c + 1];
    const bool v0 = ok && s0 < n_slots, v1 = ok && s1 < n_slots;
    if constexpr (DQ != 0) {
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int v = 0; v < 4; v++) acc[u][j][v] *= EPI;
    }
    float* p0 = part2 + ((int64_t)ks * n_slots + s0) * N + t0 * 64 + g;
    float* p1 = part2 + ((int64_t)ks * n_slots + s1) * N + t0 * 64 + g;
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (v0) p0[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h];
          if (v1) p1[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h + 1];
        }
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;
  };

  Frags<CR, TW> cur, nxt;
  grab();
  int c_item = l_item, c_ch = 0;
  if (l_item < items) load(ring0, cur);
  int par = 0;
  while (c_item < items) {
    cp_async_wait0();
    if (++l_ch == CH) grab();
    const int4* cs = par ? ring1 : ring0;
    if (l_item < items) load(par ? ring0 : ring1, nxt);
    if constexpr (DQ == 0) {
#pragma unroll
      for (int gi = 0; gi < CR / 2; gi++) {
#pragma unroll
        for (int u = 0; u < TW; u++) {
          const uint32_t w0 = cur.s[gi][u].x, w1 = cur.s[gi][u].y;
          __nv_bfloat162 sc[4];
          sc[0] = u2b(((w0 & 0x00FF00FFu) + 0x007E007Eu) << 7);
          sc[1] = u2b((((w0 >> 8) & 0x00FF00FFu) + 0x007E007Eu) << 7);
          sc[2] = u2b(((w1 & 0x00FF00FFu) + 0x007E007Eu) << 7);
          sc[3] = u2b((((w1 >> 8) & 0x00FF00FFu) + 0x007E007Eu) << 7);
#pragma unroll
          for (int ii = 0; ii < 2; ii++) {
            const int i = gi * 2 + ii;
            const int4 wv = cs[(i * TW + u) * 32];
            const uint32_t* qv = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const uint32_t qj = qv[j];
              const __nv_bfloat162 sl = __low2bfloat162(sc[j]);
              const __nv_bfloat162 sh = __high2bfloat162(sc[j]);
              const uint32_t a0 = b2u(__hmul2(u2b(pl(qj << 12)), sl));
              const uint32_t a1 = b2u(__hmul2(u2b(pl(qj << 4)), sh));
              const uint32_t a2 = b2u(__hmul2(u2b(pl(qj << 8)), sl));
              const uint32_t a3 = b2u(__hmul2(u2b(pl(qj)), sh));
              mma_bf16(acc[u][j], a0, a1, a2, a3, cur.x[i][0], cur.x[i][1]);
            }
          }
        }
      }
    } else {
      uint32_t xb[CR][2];
#pragma unroll
      for (int i = 0; i < CR; i++) {
        xb[i][0] = b2u(__hmul2(u2b(cur.x[i][0]), u2b(XPRE)));
        xb[i][1] = b2u(__hmul2(u2b(cur.x[i][1]), u2b(XPRE)));
      }
#pragma unroll
      for (int gi = 0; gi < CR / 2; gi++) {
#pragma unroll
        for (int u = 0; u < TW; u++) {
          const uint32_t w0 = cur.s[gi][u].x, w1 = cur.s[gi][u].y;
          float flo[4], fhi[4];
          flo[0] = __uint_as_float((w0 << 23) & 0x7F800000u);
          fhi[0] = __uint_as_float((w0 << 7) & 0x7F800000u);
          flo[1] = __uint_as_float((w0 << 15) & 0x7F800000u);
          fhi[1] = __uint_as_float((w0 >> 1) & 0x7F800000u);
          flo[2] = __uint_as_float((w1 << 23) & 0x7F800000u);
          fhi[2] = __uint_as_float((w1 << 7) & 0x7F800000u);
          flo[3] = __uint_as_float((w1 << 15) & 0x7F800000u);
          fhi[3] = __uint_as_float((w1 >> 1) & 0x7F800000u);
          const int i0 = gi * 2, i1 = gi * 2 + 1;
          const int4 wv0 = cs[(i0 * TW + u) * 32];
          const int4 wv1 = cs[(i1 * TW + u) * 32];
          const uint32_t* q0 = reinterpret_cast<const uint32_t*>(&wv0);
          const uint32_t* q1 = reinterpret_cast<const uint32_t*>(&wv1);
#pragma unroll
          for (int j = 0; j < 4; j++) {
            float t[4];
            uint32_t a0, a1, a2, a3;
            deq_raw<DQ>(q0[j], a0, a1, a2, a3);
            mma_bf16_z(t, a0, a1, a2, a3, xb[i0][0], xb[i0][1]);
            deq_raw<DQ>(q1[j], a0, a1, a2, a3);
            mma_bf16(t, a0, a1, a2, a3, xb[i1][0], xb[i1][1]);
            acc[u][j][0] = fmaf(t[0], flo[j], acc[u][j][0]);
            acc[u][j][1] = fmaf(t[1], flo[j], acc[u][j][1]);
            acc[u][j][2] = fmaf(t[2], fhi[j], acc[u][j][2]);
            acc[u][j][3] = fmaf(t[3], fhi[j], acc[u][j][3]);
          }
        }
      }
    }
    if (++c_ch == CH) {
      epilogue(c_item);
      c_ch = 0;
      c_item = l_item;
    }
    cur = nxt;
    par ^= 1;
  }
  cp_async_wait0();
}

template <int KS, typename IdT>
__global__ void __launch_bounds__(256) w2_ks_moe_sum_ep_kernel(
    const float* __restrict__ part2,
    const float* __restrict__ topk_w,
    const IdT* __restrict__ topk_ids,
    const int32_t* __restrict__ emap,
    int M,
    int topk,
    int N,
    int global_e,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y) {
  // Each CTA covers 1024 columns (256 threads * float4 = 1024 cols -> 5 CTAs per token row!)
  const int m = blockIdx.y;
  const int col4 = blockIdx.x * 256 + threadIdx.x;
  const int col = col4 * 4;
  if (m >= M || col >= N) return;

  const int n_slots = M * topk;
  const int64_t ks_stride = (int64_t)n_slots * N;

  float rx0 = 0.f, rx1 = 0.f, rx2 = 0.f, rx3 = 0.f;
  for (int k = 0; k < topk; ++k) {
    const int slot = m * topk + k;
    bool valid = true;
    if (topk_ids != nullptr) {
      const int64_t raw = static_cast<int64_t>(topk_ids[slot]);
      if (raw < 0 || raw >= global_e) {
        valid = false;
      } else if (emap != nullptr && emap[raw] < 0) {
        valid = false;
      }
    }
    if (!valid) continue;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const float* base_p = part2 + (int64_t)slot * N + col;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const float4 v = *reinterpret_cast<const float4*>(base_p + ks * ks_stride);
      s0 += v.x;
      s1 += v.y;
      s2 += v.z;
      s3 += v.w;
    }
    const float w = __bfloat162float(__float2bfloat16(topk_w[slot]));
    const float o0 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s0)) * w));
    const float o1 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s1)) * w));
    const float o2 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s2)) * w));
    const float o3 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s3)) * w));
    rx0 += o0;
    rx1 += o1;
    rx2 += o2;
    rx3 += o3;
  }

  __nv_bfloat162 out01 = __floats2bfloat162_rn(rx0, rx1);
  __nv_bfloat162 out23 = __floats2bfloat162_rn(rx2, rx3);
  if (bias != nullptr) {
    const uint2 b = *reinterpret_cast<const uint2*>(bias + (int64_t)m * N + col);
    out01 = __hadd2(u2b(b.x), out01);
    out23 = __hadd2(u2b(b.y), out23);
  }
  uint2 packed;
  packed.x = b2u(out01);
  packed.y = b2u(out23);
  *reinterpret_cast<uint2*>(y + (int64_t)m * N + col) = packed;
}

template <int KS, typename IdT>
__global__ void __launch_bounds__(128) w2_ks_moe_sum_ep_256_kernel(
    const float* __restrict__ part2,
    const float* __restrict__ topk_w,
    const IdT* __restrict__ topk_ids,
    const int32_t* __restrict__ emap,
    int M,
    int topk,
    int N,
    int global_e,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y) {
  // Small-M fast variant: 128 threads * float2 = 256 cols/CTA -> 20 CTAs per row (120 CTAs at M=6!)
  const int m = blockIdx.y;
  const int col = (blockIdx.x * 128 + threadIdx.x) * 2;
  if (m >= M || col >= N) return;

  const int n_slots = M * topk;
  const int64_t ks_stride = (int64_t)n_slots * N;

  float rx0 = 0.f, rx1 = 0.f;
  for (int k = 0; k < topk; ++k) {
    const int slot = m * topk + k;
    bool valid = true;
    if (topk_ids != nullptr) {
      const int64_t raw = static_cast<int64_t>(topk_ids[slot]);
      if (raw < 0 || raw >= global_e) {
        valid = false;
      } else if (emap != nullptr && emap[raw] < 0) {
        valid = false;
      }
    }
    if (!valid) continue;

    float s0 = 0.f, s1 = 0.f;
    const float* base_p = part2 + (int64_t)slot * N + col;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const float2 v = *reinterpret_cast<const float2*>(base_p + ks * ks_stride);
      s0 += v.x;
      s1 += v.y;
    }
    const float w = __bfloat162float(__float2bfloat16(topk_w[slot]));
    const float o0 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s0)) * w));
    const float o1 = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(s1)) * w));
    rx0 += o0;
    rx1 += o1;
  }

  __nv_bfloat162 out01 = __floats2bfloat162_rn(rx0, rx1);
  if (bias != nullptr) {
    const uint32_t b01 = *reinterpret_cast<const uint32_t*>(bias + (int64_t)m * N + col);
    out01 = __hadd2(u2b(b01), out01);
  }
  *reinterpret_cast<uint32_t*>(y + (int64_t)m * N + col) = b2u(out01);
}

template <int DQ, int R, int CR, int TW, int KK, int NN>
static void launch_w2_splitk(
    const at::Tensor& h,
    const at::Tensor& w,
    const at::Tensor& s,
    const at::Tensor& sorted,
    const at::Tensor& eids,
    const at::Tensor& ntpp,
    const at::Tensor& topk_w,
    const c10::optional<at::Tensor>& topk_ids_opt,
    const c10::optional<at::Tensor>& emap_opt,
    int64_t topk,
    int64_t global_e,
    at::Tensor& ctr,
    at::Tensor& part2,
    at::Tensor& y,
    const c10::optional<at::Tensor>& bias_opt) {
  constexpr int T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  constexpr int KS = KK / 16 / R;
  constexpr int TG = NN / 64 / TW;
  auto kern = mxfp4_w2_splitk_kernel<DQ, R, CR, TW, KK, NN>;
  static const int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM);
    return n > 0 ? n : 1;
  }();

  const int M = (int)(topk_w.numel() / topk);
  const int n_slots = M * (int)topk;
  const int64_t max_blk = std::min<int64_t>(sorted.numel() / 8, n_slots);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_items = max_blk * KS * TG;
  const int64_t grid = std::min<int64_t>((max_items + WARPS - 1) / WARPS, (int64_t)sms * occ);
  auto st = at::cuda::getCurrentCUDAStream();

  if (grid > 0) {
    kern<<<grid, T, SMEM, st>>>(
        (const __nv_bfloat16*)h.data_ptr(), h.stride(0),
        (const int4*)w.data_ptr(), (const uint8_t*)s.data_ptr(),
        sorted.data_ptr<int32_t>(), eids.data_ptr<int32_t>(), ntpp.data_ptr<int32_t>(),
        n_slots, ctr.data_ptr<int32_t>() + 1, ctr.data_ptr<int32_t>(),
        part2.data_ptr<float>());
  }

  // When y is empty (numel == 0), caller fuses the KS + topk + shared sum directly
  // into TwoShotAllReduceW2SplitKPostCollapse!
  if (y.numel() == 0) return;

  const int32_t* emap_ptr =
      (emap_opt.has_value() && emap_opt->defined()) ? emap_opt->data_ptr<int32_t>() : nullptr;
  const __nv_bfloat16* bias_ptr =
      (bias_opt.has_value() && bias_opt->defined())
          ? reinterpret_cast<const __nv_bfloat16*>(bias_opt->data_ptr())
          : nullptr;
  if (M <= 12) {
    dim3 sum_grid((NN + 255) / 256, M);
    if (topk_ids_opt.has_value() && topk_ids_opt->defined()) {
      if (topk_ids_opt->scalar_type() == at::kInt) {
        w2_ks_moe_sum_ep_256_kernel<KS, int32_t><<<sum_grid, 128, 0, st>>>(
            part2.data_ptr<float>(), topk_w.data_ptr<float>(),
            topk_ids_opt->data_ptr<int32_t>(), emap_ptr, M, (int)topk, NN, (int)global_e,
            bias_ptr, (__nv_bfloat16*)y.data_ptr());
      } else {
        w2_ks_moe_sum_ep_256_kernel<KS, int64_t><<<sum_grid, 128, 0, st>>>(
            part2.data_ptr<float>(), topk_w.data_ptr<float>(),
            topk_ids_opt->data_ptr<int64_t>(), emap_ptr, M, (int)topk, NN, (int)global_e,
            bias_ptr, (__nv_bfloat16*)y.data_ptr());
      }
    } else {
      w2_ks_moe_sum_ep_256_kernel<KS, int32_t><<<sum_grid, 128, 0, st>>>(
          part2.data_ptr<float>(), topk_w.data_ptr<float>(),
          nullptr, emap_ptr, M, (int)topk, NN, (int)global_e,
          bias_ptr, (__nv_bfloat16*)y.data_ptr());
    }
  } else {
    dim3 sum_grid((NN + 1023) / 1024, M);
    if (topk_ids_opt.has_value() && topk_ids_opt->defined()) {
      if (topk_ids_opt->scalar_type() == at::kInt) {
        w2_ks_moe_sum_ep_kernel<KS, int32_t><<<sum_grid, 256, 0, st>>>(
            part2.data_ptr<float>(), topk_w.data_ptr<float>(),
            topk_ids_opt->data_ptr<int32_t>(), emap_ptr, M, (int)topk, NN, (int)global_e,
            bias_ptr, (__nv_bfloat16*)y.data_ptr());
      } else {
        w2_ks_moe_sum_ep_kernel<KS, int64_t><<<sum_grid, 256, 0, st>>>(
            part2.data_ptr<float>(), topk_w.data_ptr<float>(),
            topk_ids_opt->data_ptr<int64_t>(), emap_ptr, M, (int)topk, NN, (int)global_e,
            bias_ptr, (__nv_bfloat16*)y.data_ptr());
      }
    } else {
      w2_ks_moe_sum_ep_kernel<KS, int32_t><<<sum_grid, 256, 0, st>>>(
          part2.data_ptr<float>(), topk_w.data_ptr<float>(),
          nullptr, emap_ptr, M, (int)topk, NN, (int)global_e,
          bias_ptr, (__nv_bfloat16*)y.data_ptr());
    }
  }
}

void w2_splitk_ep(
    const at::Tensor& h,
    const at::Tensor& w,
    const at::Tensor& s,
    const at::Tensor& sorted,
    const at::Tensor& eids,
    const at::Tensor& ntpp,
    const at::Tensor& topk_w,
    const c10::optional<at::Tensor>& topk_ids_opt,
    const c10::optional<at::Tensor>& emap_opt,
    int64_t topk,
    int64_t global_e,
    int64_t ks_mode,
    int64_t dq,
    at::Tensor& ctr,
    at::Tensor& part2,
    at::Tensor& y,
    const c10::optional<at::Tensor>& bias_opt) {
  const at::cuda::OptionalCUDAGuard guard(h.device());
  TORCH_CHECK(h.size(1) == 2304 && (y.numel() == 0 || y.size(1) == 5120), "w2_splitk_ep shape 2304x5120");
  if (y.numel() > 0 && bias_opt.has_value() && bias_opt->defined()) {
    TORCH_CHECK(
        bias_opt->is_contiguous() && bias_opt->scalar_type() == at::kBFloat16 &&
            bias_opt->sizes() == y.sizes(),
        "w2_splitk_ep bias");
  }
  if (ks_mode == 4) {
    if (dq == 0)
      launch_w2_splitk<0, 36, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else if (dq == 3)
      launch_w2_splitk<3, 36, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else
      launch_w2_splitk<2, 36, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
  } else if (ks_mode == 3) {
    if (dq == 0)
      launch_w2_splitk<0, 48, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else if (dq == 3)
      launch_w2_splitk<3, 48, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else
      launch_w2_splitk<2, 48, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
  } else if (ks_mode == 6) {
    if (dq == 0)
      launch_w2_splitk<0, 24, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else if (dq == 3)
      launch_w2_splitk<3, 24, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
    else
      launch_w2_splitk<2, 24, 12, 1, 2304, 5120>(h, w, s, sorted, eids, ntpp, topk_w, topk_ids_opt, emap_opt, topk, global_e, ctr, part2, y, bias_opt);
  } else {
    TORCH_CHECK(false, "unsupported ks_mode (use 3, 4, or 6)");
  }
}

// ============================================================================
// 3. 4-GPU PIX PCIe CUDA-IPC AllReduce (1-Stage & 2-Stage Full-Duplex)
//    Allocates raw cudaMalloc IPC buffers (immune to expandable_segments VMM!)
// ============================================================================
struct PcieArState {
  int rank;
  int world_size;
  size_t max_bytes;
  void* raw_alloc = nullptr;
  uint8_t* peer_in[4];
  uint8_t* peer_out[4];
  uint32_t* peer_flags[4];
  uint32_t* local_step = nullptr;
};

static std::vector<PcieArState*> g_ar_states;

__device__ __forceinline__ void sys_store_release(uint32_t* ptr, uint32_t val) {
  asm volatile("st.release.sys.global.u32 [%0], %1;\n" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ uint32_t sys_load_acquire(uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}

__device__ __forceinline__ uint4 load_nc_u4(const uint4* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ void p2p_barrier(
    uint32_t* const pflags0, uint32_t* const pflags1, uint32_t* const pflags2, uint32_t* const pflags3,
    int rank, int bid, uint32_t step) {
  __syncthreads();
  if (threadIdx.x < 4 && threadIdx.x != rank) {
    uint32_t* pf = (threadIdx.x == 0) ? pflags0 : (threadIdx.x == 1) ? pflags1 : (threadIdx.x == 2) ? pflags2 : pflags3;
    sys_store_release(pf + bid * 4 + rank, step);
    uint32_t* mf = (rank == 0) ? pflags0 : (rank == 1) ? pflags1 : (rank == 2) ? pflags2 : pflags3;
    while (sys_load_acquire(mf + bid * 4 + threadIdx.x) != step) {
    }
  }
  __syncthreads();
}

// 1-stage fused copy + all-reduce (double-buffered via in/out by step&1: 1 barrier/call, race-free!)
__global__ void __launch_bounds__(256) pcie_ar_1stage_kernel(
    const uint4* __restrict__ inp,
    uint4* __restrict__ out,
    int n_u4,
    int rank,
    uint4* __restrict__ in0, uint4* __restrict__ in1, uint4* __restrict__ in2, uint4* __restrict__ in3,
    uint4* __restrict__ out0, uint4* __restrict__ out1, uint4* __restrict__ out2, uint4* __restrict__ out3,
    uint32_t* __restrict__ fl0, uint32_t* __restrict__ fl1, uint32_t* __restrict__ fl2, uint32_t* __restrict__ fl3,
    uint32_t* __restrict__ local_step) {
  __shared__ uint32_t s_step;
  const int bid = blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t st = local_step[bid] + 1;
    local_step[bid] = st;
    s_step = st;
  }
  __syncthreads();
  const uint32_t step = s_step;
  const bool odd = (step & 1u) != 0;

  uint4* b0 = odd ? in0 : out0;
  uint4* b1 = odd ? in1 : out1;
  uint4* b2 = odd ? in2 : out2;
  uint4* b3 = odd ? in3 : out3;

  uint4* my_buf = (rank == 0) ? b0 : (rank == 1) ? b1 : (rank == 2) ? b2 : b3;
  const int idx0 = bid * 256 + threadIdx.x;
  const int stride = gridDim.x * 256;

  // Stage 0: copy local inp -> my_buf
  for (int i = idx0; i < n_u4; i += stride) {
    my_buf[i] = inp[i];
  }
  __threadfence_system();
  p2p_barrier(fl0, fl1, fl2, fl3, rank, bid, step);

  // Stage 1: read 3 peers + local inp, sum in fp32, write out
  const uint4* pA = (rank == 0) ? b1 : b0;
  const uint4* pB = (rank <= 1) ? b2 : b1;
  const uint4* pC = (rank <= 2) ? b3 : b2;

  for (int i = idx0; i < n_u4; i += stride) {
    const uint4 v0 = inp[i];
    const uint4 v1 = load_nc_u4(pA + i);
    const uint4 v2 = load_nc_u4(pB + i);
    const uint4 v3 = load_nc_u4(pC + i);

    const uint32_t a[4] = {v0.x, v0.y, v0.z, v0.w};
    const uint32_t b[4] = {v1.x, v1.y, v1.z, v1.w};
    const uint32_t c[4] = {v2.x, v2.y, v2.z, v2.w};
    const uint32_t d[4] = {v3.x, v3.y, v3.z, v3.w};
    uint32_t res[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float2 fa = __bfloat1622float2(u2b(a[w]));
      const float2 fb = __bfloat1622float2(u2b(b[w]));
      const float2 fc = __bfloat1622float2(u2b(c[w]));
      const float2 fd = __bfloat1622float2(u2b(d[w]));
      res[w] = b2u(__floats2bfloat162_rn(fa.x + fb.x + fc.x + fd.x,
                                         fa.y + fb.y + fc.y + fd.y));
    }
    out[i] = make_uint4(res[0], res[1], res[2], res[3]);
  }
}

// 2-stage Full-Duplex Reduce-Scatter (RX) + Direct Remote Push (TX) for medium/large M!
__global__ void __launch_bounds__(256) pcie_ar_2stage_kernel(
    const uint4* __restrict__ inp,
    uint4* __restrict__ out,
    int n_u4,
    int rank,
    uint4* __restrict__ in0, uint4* __restrict__ in1, uint4* __restrict__ in2, uint4* __restrict__ in3,
    uint4* __restrict__ out0, uint4* __restrict__ out1, uint4* __restrict__ out2, uint4* __restrict__ out3,
    uint32_t* __restrict__ fl0, uint32_t* __restrict__ fl1, uint32_t* __restrict__ fl2, uint32_t* __restrict__ fl3,
    uint32_t* __restrict__ local_step) {
  __shared__ uint32_t s_step;
  const int bid = blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t st = local_step[bid] + 2;
    local_step[bid] = st;
    s_step = st;
  }
  __syncthreads();
  const uint32_t step1 = s_step - 1;
  const uint32_t step2 = s_step;

  uint4* my_in = (rank == 0) ? in0 : (rank == 1) ? in1 : (rank == 2) ? in2 : in3;
  uint4* my_out = (rank == 0) ? out0 : (rank == 1) ? out1 : (rank == 2) ? out2 : out3;
  const int idx0 = bid * 256 + threadIdx.x;
  const int stride = gridDim.x * 256;

  // Stage 0: stage non-owned 3 quarters into my_in (local HBM @ 2000 GB/s: < 0.7 us!)
  const int q_u4 = n_u4 >> 2;
  const int my_q0 = rank * q_u4;
  const int my_q1 = my_q0 + q_u4;
  for (int i = idx0; i < n_u4; i += stride) {
    if (i < my_q0 || i >= my_q1) {
      my_in[i] = inp[i];
    }
  }
  __threadfence_system();
  p2p_barrier(fl0, fl1, fl2, fl3, rank, bid, step1);

  // Stage 1: Reduce own quarter [my_q0, my_q1) by reading 3 peers' my_in (PCIe RX)
  //          AND pushing reduced result directly to 3 peers' out buffers (PCIe TX simultaneously!)
  const uint4* p_inA = (rank == 0) ? in1 : in0;
  const uint4* p_inB = (rank <= 1) ? in2 : in1;
  const uint4* p_inC = (rank <= 2) ? in3 : in2;

  uint4* p_outA = (rank == 0) ? out1 : out0;
  uint4* p_outB = (rank <= 1) ? out2 : out1;
  uint4* p_outC = (rank <= 2) ? out3 : out2;

  for (int off = idx0; off < q_u4; off += stride) {
    const int i = my_q0 + off;
    const uint4 v0 = inp[i];
    const uint4 v1 = load_nc_u4(p_inA + i);
    const uint4 v2 = load_nc_u4(p_inB + i);
    const uint4 v3 = load_nc_u4(p_inC + i);

    const uint32_t a[4] = {v0.x, v0.y, v0.z, v0.w};
    const uint32_t b[4] = {v1.x, v1.y, v1.z, v1.w};
    const uint32_t c[4] = {v2.x, v2.y, v2.z, v2.w};
    const uint32_t d[4] = {v3.x, v3.y, v3.z, v3.w};
    uint32_t res[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float2 fa = __bfloat1622float2(u2b(a[w]));
      const float2 fb = __bfloat1622float2(u2b(b[w]));
      const float2 fc = __bfloat1622float2(u2b(c[w]));
      const float2 fd = __bfloat1622float2(u2b(d[w]));
      res[w] = b2u(__floats2bfloat162_rn(fa.x + fb.x + fc.x + fd.x,
                                         fa.y + fb.y + fc.y + fd.y));
    }
    const uint4 sum4 = make_uint4(res[0], res[1], res[2], res[3]);
    out[i] = sum4;
    p_outA[i] = sum4;
    p_outB[i] = sum4;
    p_outC[i] = sum4;
  }
  __threadfence_system();
  p2p_barrier(fl0, fl1, fl2, fl3, rank, bid, step2);

  // Stage 2: Copy the 3 received quarters from local my_out -> out (local HBM!)
  for (int i = idx0; i < n_u4; i += stride) {
    if (i < my_q0 || i >= my_q1) {
      out[i] = load_nc_u4(my_out + i);
    }
  }
}

at::Tensor pcie_ar_alloc_handle(int64_t rank, int64_t world_size, int64_t max_bytes) {
  TORCH_CHECK(world_size == 4 && rank >= 0 && rank < 4, "TP=4 required");
  int dev = -1;
  cudaGetDevice(&dev);
  // Enable P2P access to peers
  for (int p = 0; p < 4; ++p) {
    if (p != dev) {
      int can = 0;
      cudaDeviceCanAccessPeer(&can, dev, p);
      if (can) {
        cudaDeviceEnablePeerAccess(p, 0);
        cudaGetLastError();  // clear cudaErrorPeerAccessAlreadyEnabled
      }
    }
  }

  const size_t aligned_max = (static_cast<size_t>(max_bytes) + 4095) & ~size_t(4095);
  const size_t flags_bytes = 4096;  // up to 256 CTAs * 4 ints
  const size_t total_ipc = aligned_max * 2 + flags_bytes;

  auto* st = new PcieArState();
  st->rank = (int)rank;
  st->world_size = 4;
  st->max_bytes = aligned_max;
  TORCH_CHECK(cudaMalloc(&st->raw_alloc, total_ipc) == cudaSuccess, "cudaMalloc IPC failed");
  TORCH_CHECK(cudaMemset(st->raw_alloc, 0, total_ipc) == cudaSuccess, "cudaMemset IPC failed");
  TORCH_CHECK(cudaMalloc(&st->local_step, 4096) == cudaSuccess, "cudaMalloc step failed");
  TORCH_CHECK(cudaMemset(st->local_step, 0, 4096) == cudaSuccess, "cudaMemset step failed");
  cudaDeviceSynchronize();

  cudaIpcMemHandle_t handle;
  TORCH_CHECK(cudaIpcGetMemHandle(&handle, st->raw_alloc) == cudaSuccess, "cudaIpcGetMemHandle failed");

  int64_t handle_id = (int64_t)g_ar_states.size();
  g_ar_states.push_back(st);

  auto out = at::empty({(int64_t)sizeof(cudaIpcMemHandle_t) + 8}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  std::memcpy(out.data_ptr<uint8_t>(), &handle, sizeof(cudaIpcMemHandle_t));
  std::memcpy(out.data_ptr<uint8_t>() + sizeof(cudaIpcMemHandle_t), &handle_id, 8);
  return out;
}

int64_t pcie_ar_open_handles(int64_t handle_id, const std::vector<at::Tensor>& handles) {
  TORCH_CHECK(handle_id >= 0 && handle_id < (int64_t)g_ar_states.size() && handles.size() == 4, "open_handles args");
  PcieArState* st = g_ar_states[handle_id];
  for (int p = 0; p < 4; ++p) {
    uint8_t* base = nullptr;
    if (p == st->rank) {
      base = reinterpret_cast<uint8_t*>(st->raw_alloc);
    } else {
      cudaIpcMemHandle_t h;
      std::memcpy(&h, handles[p].data_ptr<uint8_t>(), sizeof(cudaIpcMemHandle_t));
      TORCH_CHECK(
          cudaIpcOpenMemHandle(reinterpret_cast<void**>(&base), h, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess,
          "cudaIpcOpenMemHandle failed");
    }
    st->peer_in[p] = base;
    st->peer_out[p] = base + st->max_bytes;
    st->peer_flags[p] = reinterpret_cast<uint32_t*>(base + st->max_bytes * 2);
  }
  cudaDeviceSynchronize();
  return handle_id;
}

void pcie_ar_run(int64_t handle_id, const at::Tensor& inp, at::Tensor& out, int64_t stage_mode, int64_t n_ctas) {
  PcieArState* st = g_ar_states[handle_id];
  const size_t nbytes = (size_t)inp.numel() * 2;
  TORCH_CHECK(nbytes <= st->max_bytes && nbytes % 64 == 0, "pcie_ar_run size");
  const int n_u4 = (int)(nbytes / 16);
  const at::cuda::OptionalCUDAGuard guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (stage_mode == 1) {
    int grid = (int)std::min<int64_t>(n_ctas, (n_u4 + 255) / 256);
    if (grid < 1) grid = 1;
    pcie_ar_1stage_kernel<<<grid, 256, 0, stream>>>(
        (const uint4*)inp.data_ptr(), (uint4*)out.data_ptr(), n_u4, st->rank,
        (uint4*)st->peer_in[0], (uint4*)st->peer_in[1], (uint4*)st->peer_in[2], (uint4*)st->peer_in[3],
        (uint4*)st->peer_out[0], (uint4*)st->peer_out[1], (uint4*)st->peer_out[2], (uint4*)st->peer_out[3],
        st->peer_flags[0], st->peer_flags[1], st->peer_flags[2], st->peer_flags[3],
        st->local_step);
  } else {
    int grid = (int)std::min<int64_t>(n_ctas, ((n_u4 >> 2) + 255) / 256);
    if (grid < 1) grid = 1;
    pcie_ar_2stage_kernel<<<grid, 256, 0, stream>>>(
        (const uint4*)inp.data_ptr(), (uint4*)out.data_ptr(), n_u4, st->rank,
        (uint4*)st->peer_in[0], (uint4*)st->peer_in[1], (uint4*)st->peer_in[2], (uint4*)st->peer_in[3],
        (uint4*)st->peer_out[0], (uint4*)st->peer_out[1], (uint4*)st->peer_out[2], (uint4*)st->peer_out[3],
        st->peer_flags[0], st->peer_flags[1], st->peer_flags[2], st->peer_flags[3],
        st->local_step);
  }
}

// ============================================================================
// 4. Single-Kernel C++ Tensor-Core wo_a MXFP8 Split-K + Fused Tile Reduction
//    Replaces 2-kernel Triton wo_mxfp8_splitk + reduce (16.7 us -> ~6 us!)
// ============================================================================
__device__ __forceinline__ uint32_t fp8_u16_to_bf16x2(uint16_t u) {
  const uint32_t p = (static_cast<uint32_t>(u) & 0xFFu) |
                     ((static_cast<uint32_t>(u) & 0xFF00u) << 8);
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n"
      : "=r"(r)
      : "r"(p << 8), "r"(0x80008000u), "r"((p << 4) & 0x07F007F0u));
  return r;
}

__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ void cp16z_smem(void* smem, const void* gmem, int nbytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(s), "l"(gmem), "r"(nbytes)
               : "memory");
}

template <int N_WAIT>
__device__ __forceinline__ void cp_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N_WAIT) : "memory");
}

__device__ __forceinline__ void ldsm4_smem(const void* smem, uint32_t* d) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(s)
               : "memory");
}

union WoaSmem {
  struct {
    uint4 w[3][4][16][5];
    uint4 x[3][16][9];
    uint16_t s[3][4][16];
  } pipe;
  struct {
    float tw[4][16][18];
    bool is_last;
  } epi;
};

template <int SPLIT, bool BF16_MUL>
__global__ void __launch_bounds__(128) woa_mxfp8_splitk_kernel(
    const __nv_bfloat16* __restrict__ X,
    int64_t stride_xm,
    int64_t stride_xg,
    const uint8_t* __restrict__ W_u8,
    const uint8_t* __restrict__ S_u8,
    float* __restrict__ part,
    int32_t* __restrict__ tile_cnt,
    __nv_bfloat16* __restrict__ Y,
    int M,
    int G) {
  constexpr int N = 1024;
  constexpr int K = 4096;
  constexpr int KS_ROW = K / 32;  // 128
  constexpr int KPER = K / SPLIT;
  constexpr int NSTEPS = KPER / 64;
  extern __shared__ __align__(128) uint8_t woa_smem_raw[];
  WoaSmem* const smem = reinterpret_cast<WoaSmem*>(woa_smem_raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g_lane = lane >> 2, c4 = lane & 3;
  const int pid0 = blockIdx.x;
  const int g = blockIdx.y;
  const int ch = blockIdx.z;
  const int ntg = pid0 / SPLIT;
  const int ks = pid0 % SPLIT;

  const int m0 = ch * 16;
  const int M_rem = min(16, M - m0);
  if (M_rem <= 0) return;

  const int gn0 = g * N + ntg * 64 + warp * 16;
  const int k_start = ks * KPER;

  const int xm = tid >> 3;  // 0..15
  const int xc = tid & 7;   // 0..7
  const int xb_n = (xm < M_rem) ? 16 : 0;
  const __nv_bfloat16* const x_row_base =
      X + (int64_t)(m0 + (xm < M_rem ? xm : 0)) * stride_xm + (int64_t)g * stride_xg + xc * 8;
  const uint8_t* const w_row0 = W_u8 + (int64_t)(gn0 + g_lane) * K + c4 * 16;
  const uint8_t* const w_row1 = W_u8 + (int64_t)(gn0 + g_lane + 8) * K + c4 * 16;

  auto issue = [&](int slot, int step) {
    const int k_base = k_start + step * 64;
    cp_async16(&smem->pipe.w[slot][warp][g_lane][c4], w_row0 + k_base);
    cp_async16(&smem->pipe.w[slot][warp][g_lane + 8][c4], w_row1 + k_base);
    cp16z_smem(&smem->pipe.x[slot][xm][xc], x_row_base + k_base, xb_n);
    if (lane < 16) {
      smem->pipe.s[slot][warp][lane] = *reinterpret_cast<const uint16_t*>(
          S_u8 + (int64_t)(gn0 + lane) * KS_ROW + (k_base >> 5));
    }
    cp_async_commit();
  };

  issue(0, 0);
  if constexpr (NSTEPS > 1) {
    issue(1, 1);
  } else {
    cp_async_commit();
  }

  float acc[2][4];
#pragma unroll
  for (int bb = 0; bb < 2; ++bb)
#pragma unroll
    for (int v = 0; v < 4; ++v) acc[bb][v] = 0.f;

  const int ldm_m = ((lane >> 4) << 3) + (lane & 7);

  auto deq_pair = [&](uint16_t u, uint32_t sb) -> uint32_t {
    const uint32_t raw = fp8_u16_to_bf16x2(u);
    if constexpr (BF16_MUL) {
      const uint32_t sbf = ((sb + 120u) << 7) * 0x00010001u;
      return hmul2_bf16(raw, sbf);
    } else {
      const float sf = __uint_as_float(sb << 23);
      const float2 vf = __bfloat1622float2(u2b(raw));
      return b2u(__floats2bfloat162_rn((vf.x * 1.3292279957849158e36f) * sf,
                                       (vf.y * 1.3292279957849158e36f) * sf));
    }
  };

  int step = 0;
  while (step < NSTEPS) {
#pragma unroll
    for (int slot = 0; slot < 3; ++slot) {
      if (step < NSTEPS) {
        cp_wait_group<1>();
        __syncthreads();
        if (step + 2 < NSTEPS) {
          issue((slot + 2) % 3, step + 2);
        } else {
          cp_async_commit();
        }

        const uint16_t sg0 = smem->pipe.s[slot][warp][g_lane];
        const uint16_t sg1 = smem->pipe.s[slot][warp][g_lane + 8];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint32_t sb0 = (kk < 2) ? (sg0 & 0xFFu) : (sg0 >> 8);
          const uint32_t sb1 = (kk < 2) ? (sg1 & 0xFFu) : (sg1 >> 8);
          const uint16_t* w0_u16 = reinterpret_cast<const uint16_t*>(&smem->pipe.w[slot][warp][g_lane][kk]);
          const uint16_t* w1_u16 = reinterpret_cast<const uint16_t*>(&smem->pipe.w[slot][warp][g_lane + 8][kk]);
          const uint32_t a0 = deq_pair(w0_u16[c4], sb0);
          const uint32_t a2 = deq_pair(w0_u16[4 + c4], sb0);
          const uint32_t a1 = deq_pair(w1_u16[c4], sb1);
          const uint32_t a3 = deq_pair(w1_u16[4 + c4], sb1);

          uint32_t xb[4];
          const int ldm_u4 = (kk << 1) + ((lane >> 3) & 1);
          ldsm4_smem(&smem->pipe.x[slot][ldm_m][ldm_u4], xb);
          mma_bf16(acc[0], a0, a1, a2, a3, xb[0], xb[1]);
          if (M_rem > 8) {
            mma_bf16(acc[1], a0, a1, a2, a3, xb[2], xb[3]);
          }
        }
        ++step;
      }
    }
  }
  cp_wait_group<0>();
  __syncthreads();

#pragma unroll
  for (int bb = 0; bb < 2; ++bb) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int n_loc = (v >> 1) * 8 + g_lane;
      const int m_loc = bb * 8 + 2 * c4 + (v & 1);
      smem->epi.tw[warp][m_loc][n_loc] = acc[bb][v];
    }
  }
  __syncthreads();

  const int col2 = lane * 2;
  const int w_src = col2 >> 4;
  const int n_in_w = col2 & 15;
  const int col_global = ntg * 64 + col2;

  if constexpr (SPLIT == 1) {
    for (int r = warp; r < M_rem; r += 4) {
      const float v0 = smem->epi.tw[w_src][r][n_in_w];
      const float v1 = smem->epi.tw[w_src][r][n_in_w + 1];
      *reinterpret_cast<__nv_bfloat162*>(Y + ((int64_t)(m0 + r) * G + g) * N + col_global) =
          __floats2bfloat162_rn(v0, v1);
    }
    return;
  }

  for (int r = warp; r < M_rem; r += 4) {
    const float2 v = make_float2(smem->epi.tw[w_src][r][n_in_w],
                                 smem->epi.tw[w_src][r][n_in_w + 1]);
    __stcg(reinterpret_cast<float2*>(
               part + (((int64_t)ks * M + m0 + r) * G + g) * N + col_global),
           v);
  }
  __syncthreads();

  if (tid == 0) {
    const int tile_id = (ch * G + g) * 16 + ntg;
    const int prev = atomic_add_acq_rel_gpu(tile_cnt + tile_id, 1);
    if (prev == SPLIT - 1) {
      __stcg(tile_cnt + tile_id, 0);
      smem->epi.is_last = true;
    } else {
      smem->epi.is_last = false;
    }
  }
  __syncthreads();
  if (!smem->epi.is_last) return;

  const int64_t slice_stride = (int64_t)M * G * N;
  for (int r = warp; r < M_rem; r += 4) {
    const float* base_p = part + ((int64_t)(m0 + r) * G + g) * N + col_global;
    float2 sum = __ldcg(reinterpret_cast<const float2*>(base_p));
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
      const float2 v = __ldcg(reinterpret_cast<const float2*>(base_p + s * slice_stride));
      sum.x += v.x;
      sum.y += v.y;
    }
    *reinterpret_cast<__nv_bfloat162*>(Y + ((int64_t)(m0 + r) * G + g) * N + col_global) =
        __floats2bfloat162_rn(sum.x, sum.y);
  }
}

void woa_mxfp8_splitk(
    const at::Tensor& x,
    const at::Tensor& w_u8,
    const at::Tensor& s_u8,
    bool bf16_mul,
    int64_t split,
    at::Tensor& part,
    at::Tensor& tile_cnt,
    at::Tensor& y) {
  const int M = (int)x.size(0);
  const int G = (int)x.size(1);
  if (M == 0) return;
  TORCH_CHECK(x.size(2) == 4096 && x.stride(2) == 1 && x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "woa_mxfp8_splitk: x alignment");
  TORCH_CHECK(w_u8.is_contiguous() && s_u8.is_contiguous() && y.is_contiguous(), "woa contiguous");
  const int NCH = (M + 15) >> 4;
  TORCH_CHECK(tile_cnt.numel() >= NCH * G * 16, "woa tile_cnt too small");
  if (split > 1) {
    TORCH_CHECK(part.numel() >= split * M * G * 1024, "woa part too small");
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  constexpr int SMEM = sizeof(WoaSmem);
  dim3 grid((unsigned)(split * 16), (unsigned)G, (unsigned)NCH);

#define LAUNCH_WOA(SP, BF)                                                                    \
  do {                                                                                        \
    auto kern = woa_mxfp8_splitk_kernel<SP, BF>;                                              \
    static bool init = [&] {                                                                  \
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);          \
      return true;                                                                            \
    }();                                                                                      \
    (void)init;                                                                               \
    kern<<<grid, 128, SMEM, st>>>(                                                            \
        (const __nv_bfloat16*)x.data_ptr(), x.stride(0), x.stride(1),                         \
        w_u8.data_ptr<uint8_t>(), s_u8.data_ptr<uint8_t>(),                                   \
        part.data_ptr<float>(), tile_cnt.data_ptr<int32_t>(),                                 \
        (__nv_bfloat16*)y.data_ptr(), M, G);                                                  \
  } while (0)

  if (split == 16) {
    if (bf16_mul) LAUNCH_WOA(16, true);
    else LAUNCH_WOA(16, false);
  } else if (split == 8) {
    if (bf16_mul) LAUNCH_WOA(8, true);
    else LAUNCH_WOA(8, false);
  } else if (split == 4) {
    if (bf16_mul) LAUNCH_WOA(4, true);
    else LAUNCH_WOA(4, false);
  } else if (split == 2) {
    if (bf16_mul) LAUNCH_WOA(2, true);
    else LAUNCH_WOA(2, false);
  } else if (split == 1) {
    if (bf16_mul) LAUNCH_WOA(1, true);
    else LAUNCH_WOA(1, false);
  } else {
    TORCH_CHECK(false, "woa_mxfp8_splitk: split must be 1, 2, 4, 8, or 16");
  }
#undef LAUNCH_WOA
}

// ============================================================================
// 4b. WO_A MARLIN-8BIT NO-BARRIER K-SPLIT (K=4096, N=1024, M=1..192)
//     Zero extra HBM (in-place Marlin-8bit repack at load time).
//     M<=8 (TB=1, W=4, ST=4, S=3): 9.72 us cold-L2 on 108-SM GA100!
//     M=9..192 (TB=2, W=4, ST=3, S=3/1): 1 fused launch, 0 copies!
// ============================================================================
namespace woa_marlin {

__device__ __forceinline__ uint32_t lop3_ea(uint32_t a, uint32_t m, uint32_t b) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t fp8_lo(uint32_t w) {
  return lop3_ea(w << 8, 0x80008000u, (w << 4) & 0x07F007F0u);
}
__device__ __forceinline__ uint32_t fp8_hi(uint32_t w) {
  return lop3_ea(w, 0x80008000u, (w >> 4) & 0x07F007F0u);
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ void cp16_u32(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16z_u32(uint32_t dst, const void* src, int n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n) : "memory");
}
template <int N>
__device__ __forceinline__ void cp_wait_g() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldsm4_u32(uint32_t addr, uint32_t* d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr)
               : "memory");
}

template <int TB, int W, int ST, int S>
__global__ void __launch_bounds__(W * 32) woa_marlin_ksplit_kernel(
    const __nv_bfloat16* __restrict__ X,
    int64_t sxm,
    int64_t sxg,
    int M,
    int G,
    const uint4* __restrict__ Pw,
    const uint8_t* __restrict__ Ps,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ part) {
  constexpr int K = 4096, N = 1024;
  constexpr int TOK = TB * 8;
  constexpr int SW = 132 + 32 * TB;
  constexpr int OFF_S = 128, OFF_X = 132;
  extern __shared__ __align__(128) uint4 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g_lane = lane >> 2, c4 = lane & 3;
  const int cs = blockIdx.x;       // 0..S-1
  const int tile_g = blockIdx.y;   // 0..G*16-1
  const int m_blk = blockIdx.z;    // 0..ceil(M/TOK)-1
  const int grp = tile_g >> 4;
  const int tile = tile_g & 15;
  const int m_base = m_blk * TOK;
  const int m_rem = min(TOK, M - m_base);
  if (m_rem <= 0) return;

  constexpr int GU = K >> 5;  // 128
  const int tot_k = S * W;
  const int my_k = cs * W + warp;
  const int u0 = (my_k * GU) / tot_k;
  const int u1 = ((my_k + 1) * GU) / tot_k;
  int l_left = u1 - u0;

  uint4* const wring = smem + warp * ST * SW;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(wring);

  const int q0 = lane, q1 = 32 + lane;
  const uint32_t d_w0 = (q0 ^ ((q0 >> 3) & 1)) * 16, d_w1 = (q1 ^ ((q1 >> 3) & 1)) * 16;
  const uint32_t d_s = (OFF_S + lane) * 16;
  const uint32_t d_x = (OFF_X + g_lane * 4 + (c4 ^ ((g_lane >> 1) & 3))) * 16;
  const int sw = (lane >> 2) & 1;
  const int r_w0 = (2 * lane) ^ sw, r_w1 = (2 * lane + 1) ^ sw;
  const uint32_t r_x = (OFF_X + (lane & 7) * 4 + ((lane >> 3) ^ (((lane & 7) >> 1) & 3))) * 16;

  constexpr int64_t rowN = N;
  const uint4* Pw_g = Pw + (int64_t)grp * (K / 16) * rowN;
  const uint8_t* Ps_g = Ps + (int64_t)grp * (K / 32) * N;
  const uint4* l_pw = Pw_g + (int64_t)u0 * 2 * rowN + tile * 64 + lane;
  const uint8_t* l_ps = Ps_g + (int64_t)u0 * N + tile * 64 + lane * 16;
  int xb_n[TB];
  const __nv_bfloat16* l_px[TB];
#pragma unroll
  for (int bb = 0; bb < TB; bb++) {
    const int r = bb * 8 + g_lane;
    xb_n[bb] = (r < m_rem) ? 16 : 0;
    l_px[bb] = X + (int64_t)(m_base + (r < m_rem ? r : 0)) * sxm + (int64_t)grp * sxg + u0 * 32 + c4 * 8;
  }

  auto issue = [&](const int slot) {
    if (l_left > 0) {
      const uint32_t dst = ring + slot * SW * 16;
#pragma unroll
      for (int r = 0; r < 2; r++) {
        cp16_u32(dst + r * 1024 + d_w0, l_pw + r * rowN);
        cp16_u32(dst + r * 1024 + d_w1, l_pw + r * rowN + 32);
      }
      if (lane < 4) cp16_u32(dst + d_s, l_ps);
#pragma unroll
      for (int bb = 0; bb < TB; bb++) cp16z_u32(dst + d_x + bb * 512, l_px[bb], xb_n[bb]);
      l_left--;
      l_pw += 2 * rowN;
      l_ps += N;
#pragma unroll
      for (int bb = 0; bb < TB; bb++) l_px[bb] += 32;
    }
    cp_async_commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) issue(s);

  float acc[TB][4][4];
#pragma unroll
  for (int bb = 0; bb < TB; bb++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[bb][j][v] = 0.f;

  int n = u1 - u0;
  while (n > 0) {
#pragma unroll
    for (int s = 0; s < ST; s++) {
      if (n > 0) {
        cp_wait_g<ST - 2>();
        __syncwarp();
        issue((s + ST - 1) % ST);
        const uint4* stp = wring + s * SW;
        const uint32_t st = ring + s * SW * 16;
        const uint2 sv = reinterpret_cast<const uint2*>(stp + OFF_S)[g_lane];
        uint32_t sd[8];
        const uint32_t e[2] = {sv.x, sv.y};
        constexpr uint32_t sel[4] = {0x4040u, 0x4242u, 0x4141u, 0x4343u};
#pragma unroll
        for (int j = 0; j < 8; j++)
          sd[j] = __byte_perm(e[j >> 2], 0, sel[j & 3]) * 128u + 0x3c003c00u;
        uint32_t xb[TB][4];
#pragma unroll
        for (int bb = 0; bb < TB; bb++) ldsm4_u32(st + r_x + bb * 512, xb[bb]);
        uint4 wv[2][2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
          wv[i][0] = stp[i * 64 + r_w0];
          wv[i][1] = stp[i * 64 + r_w1];
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const uint32_t wd[8] = {wv[i][0].x, wv[i][0].y, wv[i][0].z, wv[i][0].w,
                                  wv[i][1].x, wv[i][1].y, wv[i][1].z, wv[i][1].w};
#pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            const uint32_t qa = wd[2 * jj], qb = wd[2 * jj + 1];
            const uint32_t a0 = hmul2u(fp8_lo(qa), sd[2 * jj]);
            const uint32_t a2 = hmul2u(fp8_hi(qa), sd[2 * jj]);
            const uint32_t a1 = hmul2u(fp8_lo(qb), sd[2 * jj + 1]);
            const uint32_t a3 = hmul2u(fp8_hi(qb), sd[2 * jj + 1]);
#pragma unroll
            for (int bb = 0; bb < TB; bb++)
              mma_bf16(acc[bb][jj], a0, a1, a2, a3, xb[bb][2 * i], xb[bb][2 * i + 1]);
          }
        }
        n--;
      }
    }
  }
  cp_wait_g<0>();
  __syncthreads();

  float* tw = reinterpret_cast<float*>(smem) + warp * TOK * 68;
#pragma unroll
  for (int bb = 0; bb < TB; bb++)
#pragma unroll
    for (int jj = 0; jj < 4; jj++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int col = 16 * jj + 8 * (v >> 1) + g_lane;
        const int tok = bb * 8 + 2 * c4 + (v & 1);
        tw[tok * 68 + col] = acc[bb][jj][v];
      }
  __syncthreads();

  float* tile_base = reinterpret_cast<float*>(smem);
  if constexpr (S == 1) {
    for (int row = warp; row < m_rem; row += W) {
      float2 sum = *reinterpret_cast<const float2*>(tile_base + row * 68 + 2 * lane);
#pragma unroll
      for (int w = 1; w < W; w++) {
        const float2 v = *reinterpret_cast<const float2*>(tile_base + (w * TOK + row) * 68 + 2 * lane);
        sum.x += v.x;
        sum.y += v.y;
      }
      *reinterpret_cast<__nv_bfloat162*>(
          out + ((int64_t)(m_base + row) * G + grp) * N + tile * 64 + 2 * lane) =
          __floats2bfloat162_rn(sum.x, sum.y);
    }
  } else {
    if (cs > 0) {
      float* p_dst = part + (int64_t)(tile_g * (S - 1) + (cs - 1)) * (TOK * 64);
      for (int row = warp; row < m_rem; row += W) {
        float2 sum = *reinterpret_cast<const float2*>(tile_base + row * 68 + 2 * lane);
#pragma unroll
        for (int w = 1; w < W; w++) {
          const float2 v = *reinterpret_cast<const float2*>(tile_base + (w * TOK + row) * 68 + 2 * lane);
          sum.x += v.x;
          sum.y += v.y;
        }
        uint2* dst2 = reinterpret_cast<uint2*>(p_dst + row * 64) + lane;
        const uint32_t sx = __float_as_uint(sum.x), sy = __float_as_uint(sum.y);
        asm volatile("st.relaxed.gpu.global.v2.u32 [%0], {%1,%2};\n"
                     ::"l"(dst2), "r"(sx), "r"(sy) : "memory");
      }
    } else {
      for (int row = warp; row < m_rem; row += W) {
        float2 sum = *reinterpret_cast<const float2*>(tile_base + row * 68 + 2 * lane);
#pragma unroll
        for (int w = 1; w < W; w++) {
          const float2 v = *reinterpret_cast<const float2*>(tile_base + (w * TOK + row) * 68 + 2 * lane);
          sum.x += v.x;
          sum.y += v.y;
        }
        uint2 u[S - 1];
        bool pending;
        do {
          pending = false;
#pragma unroll
          for (int k = 0; k < S - 1; k++) {
            const uint2* src = reinterpret_cast<const uint2*>(
                part + (int64_t)(tile_g * (S - 1) + k) * (TOK * 64) + row * 64) + lane;
            asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1}, [%2];\n"
                         : "=r"(u[k].x), "=r"(u[k].y) : "l"(src) : "memory");
            pending |= (u[k].x == 0xFFFFFFFFu) | (u[k].y == 0xFFFFFFFFu);
          }
        } while (pending);
#pragma unroll
        for (int k = 0; k < S - 1; k++) {
          sum.x += __uint_as_float(u[k].x);
          sum.y += __uint_as_float(u[k].y);
          uint2* dst = reinterpret_cast<uint2*>(
              part + (int64_t)(tile_g * (S - 1) + k) * (TOK * 64) + row * 64) + lane;
          constexpr uint32_t SENT = 0xFFFFFFFFu;
          asm volatile("st.relaxed.gpu.global.v2.u32 [%0], {%1,%2};\n"
                       ::"l"(dst), "r"(SENT), "r"(SENT) : "memory");
        }
        *reinterpret_cast<__nv_bfloat162*>(
            out + ((int64_t)(m_base + row) * G + grp) * N + tile * 64 + 2 * lane) =
            __floats2bfloat162_rn(sum.x, sum.y);
      }
    }
  }
}

template <int TB, int W, int ST, int S>
static void launch_woa_marlin(
    const __nv_bfloat16* x, int64_t sxm, int64_t sxg, int M, int G,
    const uint4* w, const uint8_t* s, __nv_bfloat16* out, float* part, cudaStream_t st) {
  constexpr int SW = 132 + 32 * TB;
  constexpr int SM1 = W * ST * SW * 16;
  constexpr int SM2 = W * (TB * 8) * 68 * 4;
  constexpr int SMEM = SM1 > SM2 ? SM1 : SM2;
  auto kern = woa_marlin_ksplit_kernel<TB, W, ST, S>;
  static bool init = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return true;
  }();
  (void)init;
  dim3 grid(S, G * 16, (M + TB * 8 - 1) / (TB * 8));
  kern<<<grid, W * 32, SMEM, st>>>(x, sxm, sxg, M, G, w, s, out, part);
}

}  // namespace woa_marlin

void woa_marlin_ksplit(
    const at::Tensor& x,
    const at::Tensor& w_marlin,
    const at::Tensor& s_marlin,
    at::Tensor& part,
    at::Tensor& out) {
  const int M = (int)x.size(0), G = (int)x.size(1), K = (int)x.size(2);
  TORCH_CHECK(M >= 1 && M <= 192 && K == 4096 && G >= 1, "woa_marlin_ksplit shape");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(2) == 1 &&
                  x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "woa_marlin_ksplit x align");
  TORCH_CHECK(w_marlin.is_contiguous() && w_marlin.numel() * w_marlin.element_size() == (int64_t)G * 1024 * 4096,
              "woa_marlin_ksplit w_marlin");
  TORCH_CHECK(s_marlin.is_contiguous() && s_marlin.numel() * s_marlin.element_size() == (int64_t)G * 1024 * 128,
              "woa_marlin_ksplit s_marlin");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                  out.size(0) == M && out.size(1) == G && out.size(2) == 1024,
              "woa_marlin_ksplit out");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const uint4*>(w_marlin.data_ptr());
  const auto* sp = reinterpret_cast<const uint8_t*>(s_marlin.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* pp = part.data_ptr<float>();
  if (M <= 8) {
    if (G <= 2 && part.numel() >= (int64_t)G * 16 * 2 * 8 * 64) {
      woa_marlin::launch_woa_marlin<1, 4, 4, 3>(xp, x.stride(0), x.stride(1), M, G, wp, sp, op, pp, st);
    } else {
      woa_marlin::launch_woa_marlin<1, 4, 4, 1>(xp, x.stride(0), x.stride(1), M, G, wp, sp, op, pp, st);
    }
  } else if (M <= 16 && G <= 2 && part.numel() >= (int64_t)G * 16 * 2 * 16 * 64) {
    woa_marlin::launch_woa_marlin<2, 4, 3, 3>(xp, x.stride(0), x.stride(1), M, G, wp, sp, op, pp, st);
  } else {
    woa_marlin::launch_woa_marlin<2, 4, 3, 1>(xp, x.stride(0), x.stride(1), M, G, wp, sp, op, pp, st);
  }
}

// ============================================================================
// 5. TP-Local Unconstrained Greedy Argmax & Speculative Verification
//    Eliminates [M, 129280] BF16 AllGather (49.6 MB @ M=192 -> 3 KB!)
// ============================================================================
template <typename T>
__global__ void __launch_bounds__(512) tp_greedy_local_argmax_kernel(
    const T* __restrict__ logits,
    int64_t stride_m,
    int V_local,
    int valid_cols,
    int vocab_offset,
    int32_t* __restrict__ out_stats) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T* row_ptr = logits + (int64_t)row * stride_m;

  float best_val = -INFINITY;
  int best_col = 0;

  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const bool aligned16 =
        (reinterpret_cast<uintptr_t>(row_ptr) % 16 == 0) && ((valid_cols & 7) == 0);
    if (aligned16) {
      const int n_vec = valid_cols >> 3;
      const uint4* vec_ptr = reinterpret_cast<const uint4*>(row_ptr);
      for (int vi = tid; vi < n_vec; vi += 512) {
        const uint4 u = vec_ptr[vi];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        const int base_c = vi << 3;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __bfloat1622float2(u2b(w[k]));
          const int c0 = base_c + (k << 1);
          if (f.x == f.x && (f.x > best_val || (f.x == best_val && c0 < best_col))) {
            best_val = f.x;
            best_col = c0;
          }
          const int c1 = c0 + 1;
          if (f.y == f.y && (f.y > best_val || (f.y == best_val && c1 < best_col))) {
            best_val = f.y;
            best_col = c1;
          }
        }
      }
    } else {
      for (int c = tid; c < valid_cols; c += 512) {
        const float v = __bfloat162float(row_ptr[c]);
        if (v == v && (v > best_val || (v == best_val && c < best_col))) {
          best_val = v;
          best_col = c;
        }
      }
    }
  } else {
    for (int c = tid; c < valid_cols; c += 512) {
      const float v = static_cast<float>(row_ptr[c]);
      if (v == v && (v > best_val || (v == best_val && c < best_col))) {
        best_val = v;
        best_col = c;
      }
    }
  }

  // Warp reduction (ties go to smaller col)
  const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_val, off);
    const int oc = __shfl_xor_sync(0xffffffffu, best_col, off);
    if (ov > best_val || (ov == best_val && oc < best_col)) {
      best_val = ov;
      best_col = oc;
    }
  }

  __shared__ float s_val[16];
  __shared__ int s_col[16];
  if (lane == 0) {
    s_val[warp] = best_val;
    s_col[warp] = best_col;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < 16) ? s_val[lane] : -INFINITY;
    int c = (lane < 16) ? s_col[lane] : INT32_MAX;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oc = __shfl_xor_sync(0xffffffffu, c, off);
      if (ov > v || (ov == v && oc < c)) {
        v = ov;
        c = oc;
      }
    }
    if (lane == 0) {
      // Sanitize -0.0f (0x80000000) to +0.0f so AllGatherRows sentinel never alters bits.
      const int32_t v_bits = (v == 0.0f) ? 0 : __float_as_int(v);
      const int32_t g_idx = (valid_cols > 0 && c < valid_cols) ? (vocab_offset + c) : vocab_offset;
      *reinterpret_cast<int4*>(out_stats + row * 4) = make_int4(v_bits, g_idx, 0, 0);
    }
  }
}

void tp_greedy_local_argmax(
    const at::Tensor& local_logits,
    int64_t valid_cols,
    int64_t vocab_offset,
    at::Tensor& out_stats) {
  const int M = (int)local_logits.size(0);
  if (M == 0) return;
  const int V_local = (int)local_logits.size(1);
  TORCH_CHECK(local_logits.stride(1) == 1 && out_stats.is_contiguous() && out_stats.numel() >= M * 4,
              "tp_greedy_local_argmax shape");
  const at::cuda::OptionalCUDAGuard guard(local_logits.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int vc = std::max(0, std::min(V_local, (int)valid_cols));
  if (local_logits.scalar_type() == at::kBFloat16) {
    tp_greedy_local_argmax_kernel<__nv_bfloat16><<<M, 512, 0, st>>>(
        (const __nv_bfloat16*)local_logits.data_ptr(), local_logits.stride(0),
        V_local, vc, (int)vocab_offset, out_stats.data_ptr<int32_t>());
  } else {
    TORCH_CHECK(local_logits.scalar_type() == at::kFloat, "local_logits dtype");
    tp_greedy_local_argmax_kernel<float><<<M, 512, 0, st>>>(
        local_logits.data_ptr<float>(), local_logits.stride(0),
        V_local, vc, (int)vocab_offset, out_stats.data_ptr<int32_t>());
  }
}

__device__ __forceinline__ int64_t pick_global_argmax(
    const int32_t* __restrict__ gathered_row,
    int tp_size) {
  float best_val = -INFINITY;
  int64_t best_idx = 0;
  for (int r = 0; r < tp_size; ++r) {
    const float v = __int_as_float(gathered_row[r * 4]);
    const int64_t idx = static_cast<int64_t>(gathered_row[r * 4 + 1]);
    // Ranks are ordered 0..tp_size-1 with strictly increasing vocab offsets,
    // so strict '>' breaks ties to the lowest global token ID identically to stock!
    if (v == v && v > best_val) {
      best_val = v;
      best_idx = idx;
    }
  }
  return best_idx;
}

__global__ void __launch_bounds__(128) tp_greedy_nonspec_kernel(
    const int32_t* __restrict__ gathered_stats,
    int num_logits,
    int tp_size,
    int64_t* __restrict__ sampled_out) {
  const int row = blockIdx.x * 128 + threadIdx.x;
  if (row >= num_logits) return;
  sampled_out[row] = pick_global_argmax(gathered_stats + (int64_t)row * tp_size * 4, tp_size);
}

template <typename DraftT>
__global__ void __launch_bounds__(64) tp_greedy_spec_verify_kernel(
    const int32_t* __restrict__ gathered_stats,
    const DraftT* __restrict__ verify_draft_sampled,
    const int32_t* __restrict__ cu_num_logits,
    int num_reqs,
    int tp_size,
    int64_t sampled_stride,
    int64_t* __restrict__ sampled_out,
    int32_t* __restrict__ num_sampled_out) {
  const int req_idx = blockIdx.x * 64 + threadIdx.x;
  if (req_idx >= num_reqs) return;

  const int64_t start_idx = cu_num_logits[req_idx];
  const int64_t end_idx = cu_num_logits[req_idx + 1];
  if (end_idx <= start_idx) {
    num_sampled_out[req_idx] = 0;
    return;
  }
  const int64_t num_draft = end_idx - start_idx - 1;
  int64_t accepted_len = 0;
  bool verifying = true;
  int64_t* req_sampled = sampled_out + (int64_t)req_idx * sampled_stride;

  for (int64_t i = 0; i < num_draft; ++i) {
    if (!verifying) break;
    const int64_t logit_idx = start_idx + i;
    const int64_t raw_draft = static_cast<int64_t>(verify_draft_sampled[logit_idx + 1]);
    const bool is_valid = raw_draft >= 0;
    const int64_t draft_tok = is_valid ? raw_draft : 0;
    const int64_t target_argmax = pick_global_argmax(
        gathered_stats + logit_idx * tp_size * 4, tp_size);
    const bool accepted = is_valid && (target_argmax == draft_tok);
    req_sampled[i] = accepted ? draft_tok : target_argmax;
    if (accepted) {
      accepted_len += 1;
    } else {
      verifying = false;
    }
  }

  if (accepted_len == num_draft) {
    // Bonus token (or single token when num_draft == 0)
    req_sampled[accepted_len] = pick_global_argmax(
        gathered_stats + (end_idx - 1) * tp_size * 4, tp_size);
  }
  num_sampled_out[req_idx] = static_cast<int32_t>(accepted_len + 1);
}

void tp_greedy_nonspec(
    const at::Tensor& gathered_stats,
    int64_t tp_size,
    at::Tensor& sampled_out) {
  const int M = (int)gathered_stats.size(0);
  if (M == 0) return;
  TORCH_CHECK(gathered_stats.is_contiguous() && sampled_out.is_contiguous(), "contiguous");
  const at::cuda::OptionalCUDAGuard guard(gathered_stats.device());
  auto st = at::cuda::getCurrentCUDAStream();
  tp_greedy_nonspec_kernel<<<(M + 127) / 128, 128, 0, st>>>(
      gathered_stats.data_ptr<int32_t>(), M, (int)tp_size, sampled_out.data_ptr<int64_t>());
}

void tp_greedy_spec_verify(
    const at::Tensor& gathered_stats,
    const at::Tensor& verify_draft_sampled,
    const at::Tensor& cu_num_logits,
    int64_t tp_size,
    at::Tensor& sampled_out,
    at::Tensor& num_sampled_out) {
  const int num_reqs = (int)cu_num_logits.size(0) - 1;
  if (num_reqs <= 0) return;
  TORCH_CHECK(gathered_stats.is_contiguous() && verify_draft_sampled.is_contiguous() &&
                  cu_num_logits.is_contiguous(),
              "contiguous");
  const at::cuda::OptionalCUDAGuard guard(gathered_stats.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int grid = (num_reqs + 63) / 64;
  if (verify_draft_sampled.scalar_type() == at::kInt) {
    tp_greedy_spec_verify_kernel<int32_t><<<grid, 64, 0, st>>>(
        gathered_stats.data_ptr<int32_t>(), verify_draft_sampled.data_ptr<int32_t>(),
        cu_num_logits.data_ptr<int32_t>(), num_reqs, (int)tp_size,
        sampled_out.stride(0), sampled_out.data_ptr<int64_t>(), num_sampled_out.data_ptr<int32_t>());
  } else {
    TORCH_CHECK(verify_draft_sampled.scalar_type() == at::kLong, "draft dtype");
    tp_greedy_spec_verify_kernel<int64_t><<<grid, 64, 0, st>>>(
        gathered_stats.data_ptr<int32_t>(), verify_draft_sampled.data_ptr<int64_t>(),
        cu_num_logits.data_ptr<int32_t>(), num_reqs, (int)tp_size,
        sampled_out.stride(0), sampled_out.data_ptr<int64_t>(), num_sampled_out.data_ptr<int32_t>());
  }
}

// ============================================================================
// 6. Fast Marlin-Layout MXFP8 -> Row-Major BF16 Dequantization (~25 us)
//    Enables 260-TFLOPS cuBLAS torch.mm for large-M prefill (0 MiB permanent HBM!)
// ============================================================================
__device__ __forceinline__ float e4m3_byte_to_f32(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t em = b & 0x7Fu;
  if (em == 0x7Fu) return __uint_as_float(0x7FC00000u);
  if (em >= 8u) {
    return __uint_as_float(sign | ((em + (120u << 3)) << 20));
  }
  return __uint_as_float(sign | __float_as_uint(static_cast<float>(em) * 0.001953125f));
}

__global__ void __launch_bounds__(64) dequant_marlin_mxfp8_kernel(
    const uint4* __restrict__ w_u4,
    const uint8_t* __restrict__ s_u8,
    __nv_bfloat16* __restrict__ out,
    int padded_n,
    int padded_k,
    int size_n,
    int size_k) {
  __shared__ __align__(16) __nv_bfloat16 smem[64][40];
  const int t = blockIdx.x;   // 64-col tile [0 .. padded_n/64)
  const int kg = blockIdx.y;  // 32-k group  [0 .. padded_k/32)
  const int tid = threadIdx.x;
  const int warp = tid >> 5;  // 0 or 1 (selects k16 row 2*kg + warp)
  const int lane = tid & 31;
  const int g = lane >> 2;    // 0..7
  const int c = lane & 3;     // 0..3

  const uint8_t* sb = s_u8 + (int64_t)kg * padded_n + t * 64 + 8 * g;
  const uint2 sv = *reinterpret_cast<const uint2*>(sb);
  const uint8_t sbytes[8] = {
      static_cast<uint8_t>(sv.x & 0xFFu),
      static_cast<uint8_t>((sv.x >> 16) & 0xFFu),
      static_cast<uint8_t>((sv.x >> 8) & 0xFFu),
      static_cast<uint8_t>((sv.x >> 24) & 0xFFu),
      static_cast<uint8_t>(sv.y & 0xFFu),
      static_cast<uint8_t>((sv.y >> 16) & 0xFFu),
      static_cast<uint8_t>((sv.y >> 8) & 0xFFu),
      static_cast<uint8_t>((sv.y >> 24) & 0xFFu),
  };

  const int kr = 2 * kg + warp;
  const uint4* w_tile = w_u4 + (int64_t)kr * padded_n + t * 64;
  const uint4 u0 = w_tile[2 * lane];
  const uint4 u1 = w_tile[2 * lane + 1];
  const uint32_t wd[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};

  const int k_lo = warp * 16 + 2 * c;
  const int k_hi = k_lo + 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint8_t e8 = sbytes[j];
    const float sc = (e8 == 0) ? __uint_as_float(0x00400000u)
                               : __uint_as_float(static_cast<uint32_t>(e8) << 23);
    const uint32_t w = wd[j];
    const float v0 = e4m3_byte_to_f32(w & 0xFFu) * sc;
    const float v1 = e4m3_byte_to_f32((w >> 8) & 0xFFu) * sc;
    const float v2 = e4m3_byte_to_f32((w >> 16) & 0xFFu) * sc;
    const float v3 = e4m3_byte_to_f32((w >> 24) & 0xFFu) * sc;
    const int col = g + 8 * j;
    *reinterpret_cast<__nv_bfloat162*>(&smem[col][k_lo]) = __floats2bfloat162_rn(v0, v2);
    *reinterpret_cast<__nv_bfloat162*>(&smem[col][k_hi]) = __floats2bfloat162_rn(v1, v3);
  }
  __syncthreads();

  const int r_sub = tid >> 2;  // 0..15
  const int k_vec = tid & 3;   // 0..3 (8 bf16s = 1 uint4 each)
  const int gk = kg * 32 + k_vec * 8;
  if (gk >= size_k) return;
  const bool vec_ok = ((size_k & 7) == 0) && (gk + 8 <= size_k);
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int row = step * 16 + r_sub;
    const int gn = t * 64 + row;
    if (gn < size_n) {
      if (vec_ok) {
        *reinterpret_cast<uint4*>(out + (int64_t)gn * size_k + gk) =
            *reinterpret_cast<const uint4*>(&smem[row][k_vec * 8]);
      } else {
        for (int i = 0; i < 8 && gk + i < size_k; ++i) {
          out[(int64_t)gn * size_k + gk + i] = smem[row][k_vec * 8 + i];
        }
      }
    }
  }
}

void dequant_marlin_mxfp8(
    const at::Tensor& w_marlin,
    const at::Tensor& s_marlin,
    at::Tensor& out_bf16,
    int64_t size_n,
    int64_t size_k) {
  TORCH_CHECK(w_marlin.is_cuda() && s_marlin.is_cuda() && out_bf16.is_cuda(), "cuda");
  TORCH_CHECK(w_marlin.is_contiguous() && s_marlin.is_contiguous() && out_bf16.is_contiguous(),
              "contiguous");
  TORCH_CHECK(out_bf16.scalar_type() == at::kBFloat16, "out_bf16 dtype");
  const int padded_k = static_cast<int>(w_marlin.size(0)) * 16;
  const int padded_n = static_cast<int>(w_marlin.size(1)) / 4;
  TORCH_CHECK((padded_n % 64 == 0) && (padded_k % 32 == 0), "marlin tile shape");
  TORCH_CHECK(s_marlin.size(0) == padded_k / 32 && s_marlin.size(1) == padded_n, "scale shape");
  TORCH_CHECK(out_bf16.size(0) == size_n && out_bf16.size(1) == size_k, "out shape");
  const at::cuda::OptionalCUDAGuard guard(w_marlin.device());
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(padded_n / 64, padded_k / 32);
  dequant_marlin_mxfp8_kernel<<<grid, 64, 0, st>>>(
      reinterpret_cast<const uint4*>(w_marlin.data_ptr()),
      reinterpret_cast<const uint8_t*>(s_marlin.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_bf16.data_ptr()),
      padded_n, padded_k, static_cast<int>(size_n), static_cast<int>(size_k));
}

__global__ void __launch_bounds__(256, 4) dequant_row_mxfp8_kernel(
    const uint4* __restrict__ w_vec,
    const uint8_t* __restrict__ s_u8,
    uint4* __restrict__ out_vec,
    int total_vec16) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= total_vec16) return;
  const uint32_t sb = static_cast<uint32_t>(__ldg(s_u8 + (idx >> 1)));
  const float sf = (sb == 0u) ? __uint_as_float(0x00400000u)
                  : (sb == 255u) ? __uint_as_float(0x7FC00000u)
                                 : __uint_as_float(sb << 23);
  const uint4 w16 = __ldcs(w_vec + idx);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w16);
  uint32_t out_u32[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float f0 = e4m3_byte_to_f32(bytes[2 * i]) * sf;
    const float f1 = e4m3_byte_to_f32(bytes[2 * i + 1]) * sf;
    out_u32[i] = b2u(__floats2bfloat162_rn(f0, f1));
  }
  out_vec[2 * idx] = make_uint4(out_u32[0], out_u32[1], out_u32[2], out_u32[3]);
  out_vec[2 * idx + 1] = make_uint4(out_u32[4], out_u32[5], out_u32[6], out_u32[7]);
}

void dequant_row_mxfp8(
    const at::Tensor& w_u8,
    const at::Tensor& s_u8,
    at::Tensor& out_bf16) {
  TORCH_CHECK(w_u8.is_cuda() && s_u8.is_cuda() && out_bf16.is_cuda(), "cuda");
  TORCH_CHECK(w_u8.is_contiguous() && s_u8.is_contiguous() && out_bf16.is_contiguous(), "contiguous");
  TORCH_CHECK(out_bf16.scalar_type() == at::kBFloat16, "out_bf16 dtype");
  const int64_t numel = w_u8.numel();
  TORCH_CHECK(numel == out_bf16.numel() && (numel % 32 == 0) && s_u8.numel() * 32 == numel, "shape");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(w_u8.data_ptr()) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(out_bf16.data_ptr()) % 16 == 0),
              "align16");
  if (numel == 0) return;
  const int total_vec16 = static_cast<int>(numel >> 4);
  const at::cuda::OptionalCUDAGuard guard(w_u8.device());
  auto st = at::cuda::getCurrentCUDAStream();
  dequant_row_mxfp8_kernel<<<(total_vec16 + 255) / 256, 256, 0, st>>>(
      reinterpret_cast<const uint4*>(w_u8.data_ptr()),
      reinterpret_cast<const uint8_t*>(s_u8.data_ptr()),
      reinterpret_cast<uint4*>(out_bf16.data_ptr()),
      total_vec16);
}

// ============================================================================
// 7. Bitwise-Identical MoE Router + TopK + Optional Block-8 EP Align
//    Covers BOTH Backbone (E=384, TOPK=6) AND DSpark (E=128, TOPK=3) + bias_vl!
// ============================================================================
namespace r2_router {

__device__ __forceinline__ void bf16x8_to_f32(uint4 u, float* f) {
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 v = __bfloat1622float2(b2[i]);
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

__device__ __forceinline__ void bf16x4_to_f32(uint2 u, float* f) {
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float2 v = __bfloat1622float2(b2[i]);
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

__device__ __forceinline__ float r_warp_sum(float v) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, mask);
  return v;
}

__device__ __forceinline__ float r_fmax_ptx(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float r_triton_softplus_sqrt(float x) {
  float t;
  asm("{\n\t"
      ".reg .f32 %%e, %%y, %%y_scaled, %%y_norm, %%bias, %%m_f, %%f, %%p, %%t1, %%t2, %%ln, %%sel;\n\t"
      ".reg .s32 %%yi, %%tmp, %%exp_i, %%mant_i;\n\t"
      ".reg .pred %%p_sub, %%p_fin, %%p_zero, %%p_gt20;\n\t"
      "mul.f32 %%e, %1, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 %%e, %%e;\n\t"
      "add.f32 %%y, %%e, 0f3F800000;\n\t"
      "setp.lt.f32 %%p_sub, %%y, 0f00800000;\n\t"
      "mul.f32 %%y_scaled, %%y, 0f4B000000;\n\t"
      "selp.f32 %%y_norm, %%y_scaled, %%y, %%p_sub;\n\t"
      "selp.f32 %%bias, 0fC1B80000, 0f00000000, %%p_sub;\n\t"
      "mov.b32 %%yi, %%y_norm;\n\t"
      "add.s32 %%tmp, %%yi, -1059760811;\n\t"
      "and.b32 %%exp_i, %%tmp, -8388608;\n\t"
      "sub.s32 %%mant_i, %%yi, %%exp_i;\n\t"
      "cvt.rn.f32.s32 %%m_f, %%exp_i;\n\t"
      "fma.rn.ftz.f32 %%t1, %%m_f, 0f34000000, %%bias;\n\t"
      "mov.b32 %%f, %%mant_i;\n\t"
      "add.f32 %%f, %%f, 0fBF800000;\n\t"
      "fma.rn.ftz.f32 %%p, 0fBE055027, %%f, 0f3E1039F6;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0fBDF8CDCC;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0f3E0F2955;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0fBE2AD8B9;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0f3E4CED0B;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0fBE7FFF22;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0f3EAAAA78;\n\t"
      "fma.rn.ftz.f32 %%p, %%p, %%f, 0fBF000000;\n\t"
      "mul.f32 %%t2, %%f, %%p;\n\t"
      "fma.rn.ftz.f32 %%t2, %%t2, %%f, %%f;\n\t"
      "fma.rn.ftz.f32 %%ln, %%t1, 0f3F317218, %%t2;\n\t"
      "setp.lt.u32 %%p_fin, %%yi, 2139095040;\n\t"
      "@!%%p_fin fma.rn.ftz.f32 %%ln, %%y_norm, 0f7F800000, 0f7F800000;\n\t"
      "setp.eq.f32 %%p_zero, %%y_norm, 0f00000000;\n\t"
      "selp.f32 %%ln, 0fFF800000, %%ln, %%p_zero;\n\t"
      "setp.gt.f32 %%p_gt20, %1, 0f41A00000;\n\t"
      "selp.f32 %%sel, %1, %%ln, %%p_gt20;\n\t"
      "sqrt.approx.ftz.f32 %0, %%sel;\n\t"
      "}"
      : "=f"(t)
      : "f"(x));
  return t;
}

__device__ __forceinline__ bool r_signal_last_cta(int* done_cnt, int expected_prev) {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  uint32_t prev;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;\n"
               : "=r"(prev)
               : "l"(done_cnt)
               : "memory");
  return static_cast<int>(prev) == expected_prev;
}

template <int E, int TOPK, typename IdxT, bool CG_LOAD, bool PRECOMPUTED_W = false>
__device__ __forceinline__ void r_topk_warp_row(
    const float* __restrict__ row_logits,
    const float* __restrict__ corr_bias,
    float* __restrict__ row_weights,
    IdxT* __restrict__ row_ids,
    int32_t* __restrict__ smem_ids_row,
    float routed_scaling_factor,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr,
    int row,
    int lane) {
  if (is_padding_ptr != nullptr && is_padding_ptr[row]) {
    if (lane < TOPK) {
      row_weights[lane] = 0.0f;
      row_ids[lane] = static_cast<IdxT>(-1);
      if (smem_ids_row != nullptr) {
        smem_ids_row[lane] = -1;
      }
    }
    return;
  }
  constexpr int NE = E / 32;  // 4 for 128, 8 for 256, 12 for 384
  bool use_vl = false;
  if (bias_vl != nullptr && input_ids_ptr != nullptr && image_sentinel_lo > 0) {
    const int64_t tok = ids_is_i32
        ? static_cast<int64_t>(static_cast<const int32_t*>(input_ids_ptr)[row])
        : static_cast<const int64_t*>(input_ids_ptr)[row];
    use_vl = (tok >= image_sentinel_lo) && (tok < image_sentinel_lo + 5);
  }
  const float* const bptr = use_vl ? bias_vl : corr_bias;

  float w[NE];
  float cur[NE];
#pragma unroll
  for (int k = 0; k < NE; ++k) {
    const int idx = k * 32 + lane;
    const float lf = CG_LOAD ? __ldcg(row_logits + idx) : row_logits[idx];
    const float bf = bptr[idx];
    const float wi = PRECOMPUTED_W ? lf : r_triton_softplus_sqrt(lf);
    float ci = wi + bf;
    asm("{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.num.f32 %%p, %0, %0;\n\t"
        "selp.f32 %0, %0, 0fF149F2CA, %%p;\n\t"
        "}"
        : "+f"(ci));
    w[k] = wi;
    cur[k] = ci;
  }

  float sel_w[TOPK];
  uint32_t sel_id[TOPK];
#pragma unroll
  for (int slot = 0; slot < TOPK; ++slot) {
    float bv = cur[0];
    uint32_t bi = static_cast<uint32_t>(lane);
    int bj = 0;
#pragma unroll
    for (int k = 1; k < NE; ++k) {
      if (cur[k] > bv) {
        bv = cur[k];
        bi = static_cast<uint32_t>(k * 32 + lane);
        bj = k;
      }
    }
    float wv = bv;
    uint32_t wi = bi;
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, wv, mask);
      const uint32_t oi = __shfl_xor_sync(0xffffffffu, wi, mask);
      if (ov > wv || (ov == wv && oi < wi)) {
        wv = ov;
        wi = oi;
      }
    }
    float my_w = 0.0f;
    if (bi == wi) {
#pragma unroll
      for (int k = 0; k < NE; ++k) {
        if (k == bj) {
          my_w = w[k] + 0.0f;
          cur[k] = __uint_as_float(0xFF800000u);
        }
      }
    }
    sel_w[slot] = __shfl_sync(0xffffffffu, my_w, static_cast<int>(wi & 31u));
    sel_id[slot] = wi;
  }

  float wsum;
  if constexpr (TOPK == 6) {
    // Exact Triton tree-sum order over 8 padded slots [w0,w1,w2,w3,w4,w5,0,0]
    wsum = ((sel_w[0] + sel_w[4]) + sel_w[2]) + ((sel_w[1] + sel_w[5]) + sel_w[3]);
  } else {
    // Exact Triton tree-sum order over 4 padded slots [w0,w1,w2,0]
    wsum = (sel_w[0] + sel_w[2]) + (sel_w[1] + 0.0f);
  }
  const float denom = (wsum > 0.0f) ? wsum : 1.0f;
  float scale;
  asm("div.full.f32 %0, %1, %2;" : "=f"(scale) : "f"(routed_scaling_factor), "f"(denom));

  if (lane < TOPK) {
    float sw = 0.0f;
    uint32_t sid = 0;
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
      if (lane == s) {
        sw = sel_w[s];
        sid = sel_id[s];
      }
    }
    row_weights[lane] = scale * sw;
    row_ids[lane] = static_cast<IdxT>(sid);
    if (smem_ids_row != nullptr) {
      smem_ids_row[lane] = static_cast<int32_t>(sid);
    }
  }
}

template <int E, int TOPK, typename IdxT>
__global__ void __launch_bounds__(32) dsv4_topk_v2_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ corr_bias,
    float* __restrict__ topk_weights,
    IdxT* __restrict__ topk_ids,
    float routed_scaling_factor,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  r_topk_warp_row<E, TOPK, IdxT, false>(
      logits + (int64_t)row * E, corr_bias,
      topk_weights + (int64_t)row * TOPK, topk_ids + (int64_t)row * TOPK,
      nullptr, routed_scaling_factor, input_ids_ptr, ids_is_i32,
      bias_vl, image_sentinel_lo, is_padding_ptr, row, lane);
}

struct AlignBs8Smem {
  int32_t topk_ids[64];
  int16_t sid[64];
  int32_t cnt[128];
  int32_t sblk[128];
  int32_t wsum[8];
  int32_t s_sorted[512];
};

template <int NT>
__device__ __forceinline__ void epilogue_align_bs8(
    int tid,
    int S,
    int global_e,
    int local_e,
    AlignBs8Smem* as,
    const int32_t* __restrict__ emap,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  static_assert(NT >= 128, "NT must be >= 128");
  if (tid < 128) as->cnt[tid] = 0;
  __syncthreads();

  if (tid < S) {
    const int gid = as->topk_ids[tid];
    int e = -1;
    if (gid >= 0 && gid < global_e) {
      e = (emap != nullptr) ? emap[gid] : gid;
      if (e < 0 || e >= local_e) e = -1;
    }
    as->sid[tid] = static_cast<int16_t>(e);
    if (e >= 0) atomicAdd(&as->cnt[e], 1);
  }
  __syncthreads();

  int my_cnt = 0, myblk = 0, excl = 0;
  const int lane = tid & 31, wid = tid >> 5;
  if (tid < 128) {
    my_cnt = (tid < local_e) ? as->cnt[tid] : 0;
    myblk = (my_cnt + 7) >> 3;
    int inc = myblk;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int t = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += t;
    }
    if (lane == 31) as->wsum[wid] = inc;
    excl = inc - myblk;
  }
  __syncthreads();

  if (wid == 0 && lane < 4) {
    int v = as->wsum[lane];
    const int t1 = __shfl_up_sync(0x0fu, v, 1);
    if (lane >= 1) v += t1;
    const int t2 = __shfl_up_sync(0x0fu, v, 2);
    if (lane >= 2) v += t2;
    as->wsum[lane] = v;
  }
  __syncthreads();

  if (tid < 128) {
    if (wid > 0) excl += as->wsum[wid - 1];
    as->sblk[tid] = excl;
  }
  const int total_blk = as->wsum[3];
  const int total = total_blk * 8;

  for (int b = tid; b < nb; b += NT) {
    if (b >= total_blk) eids[b] = -1;
  }
  if (tid < local_e) {
    for (int k = 0; k < myblk; ++k) {
      if (excl + k < nb) eids[excl + k] = tid;
    }
  }
  if (tid == 0) *ntpp = total;

  for (int p = tid; p < total; p += NT) as->s_sorted[p] = S;
  for (int p = total + tid; p < cap; p += NT) sorted[p] = S;
  __syncthreads();

  if (tid < S) {
    const int e = as->sid[tid];
    if (e >= 0) {
      int rank = 0;
      for (int j = 0; j < tid; ++j) rank += (as->sid[j] == e);
      as->s_sorted[as->sblk[e] * 8 + rank] = tid;
    }
  }
  __syncthreads();

  const int n_vec = total >> 2;
  int4* sorted4 = reinterpret_cast<int4*>(sorted);
  const int4* src4 = reinterpret_cast<const int4*>(as->s_sorted);
  for (int q = tid; q < n_vec; q += NT) sorted4[q] = src4[q];
}

template <int E, int TOPK, typename IdxT>
__global__ void __launch_bounds__(256, 4) fgta_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ logits_out,
    int K,
    int64_t stride_wn,
    const float* __restrict__ corr_bias,
    float* __restrict__ topk_weights,
    IdxT* __restrict__ topk_ids,
    float routed_scaling_factor,
    int* __restrict__ done_cnt,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr,
    bool do_align,
    int local_e,
    const int32_t* __restrict__ emap,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  __shared__ union {
    struct {
      float sm[8];
      bool is_last;
    } g;
    AlignBs8Smem align;
  } u;

  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int c = tid * 8;
  const __nv_bfloat16* w_row = w + (int64_t)pid * stride_wn;

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k0 = 0; k0 < K; k0 += 2048) {
    if (k0 + c < K) {
      const uint4 wu = __ldcs(reinterpret_cast<const uint4*>(w_row + k0 + c));
      const uint4 xu = *reinterpret_cast<const uint4*>(x + k0 + c);
      float wf[8], xf[8];
      bf16x8_to_f32(wu, wf);
      bf16x8_to_f32(xu, xf);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = fmaf(wf[j], xf[j], acc[j]);
    }
  }
  float s = acc[0] + acc[1];
#pragma unroll
  for (int j = 2; j < 8; ++j) s = acc[j] + s;
  s = r_warp_sum(s);

  if (lane == 0) u.g.sm[warp] = s;
  __syncthreads();
  if (tid < 8) {
    float v = u.g.sm[tid];
    v += __shfl_xor_sync(0xffu, v, 4);
    v += __shfl_xor_sync(0xffu, v, 2);
    v += __shfl_xor_sync(0xffu, v, 1);
    if (tid == 0) __stcg(logits_out + pid, v);
  }
  __syncthreads();
  if (tid == 0) u.g.is_last = r_signal_last_cta(done_cnt, E - 1);
  __syncthreads();
  if (!u.g.is_last) return;
  __syncthreads();
  if (tid == 0) *done_cnt = 0;

  if (warp == 0) {
    r_topk_warp_row<E, TOPK, IdxT, true>(
        logits_out, corr_bias, topk_weights, topk_ids,
        do_align ? u.align.topk_ids : nullptr,
        routed_scaling_factor, input_ids_ptr, ids_is_i32,
        bias_vl, image_sentinel_lo, is_padding_ptr, 0, lane);
  }
  if (do_align) {
    __syncthreads();
    epilogue_align_bs8<256>(tid, TOPK, E, local_e, &u.align, emap, sorted, cap, eids, nb, ntpp);
  }
}

template <int E, int TOPK, typename IdxT>
__global__ void __launch_bounds__(128, 8) fgta_m2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ logits_out,
    int K,
    int64_t stride_xm,
    int64_t stride_wn,
    const float* __restrict__ corr_bias,
    float* __restrict__ topk_weights,
    IdxT* __restrict__ topk_ids,
    float routed_scaling_factor,
    int* __restrict__ done_cnt,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr,
    bool do_align,
    int local_e,
    const int32_t* __restrict__ emap,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  __shared__ union {
    struct {
      float sm[2][4];
      bool is_last;
    } g;
    AlignBs8Smem align;
  } u;

  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int c = tid * 8;
  const __nv_bfloat16* w_row = w + (int64_t)pid * stride_wn;
  const __nv_bfloat16* x0 = x;
  const __nv_bfloat16* x1 = x + stride_xm;

  float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k0 = 0; k0 < K; k0 += 1024) {
    if (k0 + c < K) {
      const uint4 wu = __ldcs(reinterpret_cast<const uint4*>(w_row + k0 + c));
      const uint4 xu0 = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + c));
      const uint4 xu1 = __ldg(reinterpret_cast<const uint4*>(x1 + k0 + c));
      float wf[8], xf0[8], xf1[8];
      bf16x8_to_f32(wu, wf);
      bf16x8_to_f32(xu0, xf0);
      bf16x8_to_f32(xu1, xf1);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc0[j] = fmaf(wf[j], xf0[j], acc0[j]);
        acc1[j] = fmaf(wf[j], xf1[j], acc1[j]);
      }
    }
  }

  float s0 = acc0[0] + acc0[1];
  float s1 = acc1[0] + acc1[1];
#pragma unroll
  for (int j = 2; j < 8; ++j) {
    s0 = acc0[j] + s0;
    s1 = acc1[j] + s1;
  }
  s0 = r_warp_sum(s0);
  s1 = r_warp_sum(s1);

  if (lane == 0) {
    u.g.sm[0][warp] = s0;
    u.g.sm[1][warp] = s1;
  }
  __syncthreads();
  if (tid < 2) {
    const float v = (u.g.sm[tid][0] + u.g.sm[tid][2]) + (u.g.sm[tid][1] + u.g.sm[tid][3]);
    __stcg(logits_out + (int64_t)tid * E + pid, v);
  }
  __syncthreads();
  if (tid == 0) u.g.is_last = r_signal_last_cta(done_cnt, E - 1);
  __syncthreads();
  if (!u.g.is_last) return;
  __syncthreads();
  if (tid == 0) *done_cnt = 0;

  if (warp < 2) {
    r_topk_warp_row<E, TOPK, IdxT, true>(
        logits_out + (int64_t)warp * E, corr_bias,
        topk_weights + (int64_t)warp * TOPK, topk_ids + (int64_t)warp * TOPK,
        do_align ? (u.align.topk_ids + warp * TOPK) : nullptr,
        routed_scaling_factor, input_ids_ptr, ids_is_i32,
        bias_vl, image_sentinel_lo, is_padding_ptr, warp, lane);
  }
  if (do_align) {
    __syncthreads();
    epilogue_align_bs8<128>(tid, 2 * TOPK, E, local_e, &u.align, emap, sorted, cap, eids, nb, ntpp);
  }
}

template <int M_MAX, int MIN_BLK, int E, int TOPK, typename IdxT>
__global__ void __launch_bounds__(128, MIN_BLK) fgta_m3_8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ logits_out,
    int M,
    int K,
    int64_t stride_xm,
    int64_t stride_wn,
    const float* __restrict__ corr_bias,
    float* __restrict__ topk_weights,
    IdxT* __restrict__ topk_ids,
    float routed_scaling_factor,
    int* __restrict__ done_cnt,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr,
    bool do_align,
    int local_e,
    const int32_t* __restrict__ emap,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  __shared__ union {
    struct {
      float sm[M_MAX][4];
      bool is_last;
    } g;
    AlignBs8Smem align;
  } u;

  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int c = tid * 8;
  const __nv_bfloat16* w_row = w + (int64_t)pid * stride_wn;

  float acc[M_MAX][8];
#pragma unroll
  for (int m = 0; m < M_MAX; ++m)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[m][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += 1024) {
    if (k0 + c < K) {
      const uint4 wu = __ldcs(reinterpret_cast<const uint4*>(w_row + k0 + c));
      float wf[8];
      bf16x8_to_f32(wu, wf);
#pragma unroll
      for (int m = 0; m < M_MAX; ++m) {
        if (m < M) {
          const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * stride_xm + k0 + c));
          float xf[8];
          bf16x8_to_f32(xu, xf);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[m][j] = fmaf(wf[j], xf[j], acc[m][j]);
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M_MAX; ++m) {
    if (m < M) {
      float s = acc[m][0] + acc[m][1];
#pragma unroll
      for (int j = 2; j < 8; ++j) s = acc[m][j] + s;
      s = r_warp_sum(s);
      if (lane == 0) u.g.sm[m][warp] = s;
    }
  }
  __syncthreads();
  if (tid < M) {
    const float v = (u.g.sm[tid][0] + u.g.sm[tid][2]) + (u.g.sm[tid][1] + u.g.sm[tid][3]);
    __stcg(logits_out + (int64_t)tid * E + pid, v);
  }
  __syncthreads();
  if (tid == 0) u.g.is_last = r_signal_last_cta(done_cnt, E - 1);
  __syncthreads();
  if (!u.g.is_last) return;
  __syncthreads();
  if (tid == 0) *done_cnt = 0;

  for (int row = warp; row < M; row += 4) {
    r_topk_warp_row<E, TOPK, IdxT, true>(
        logits_out + (int64_t)row * E, corr_bias,
        topk_weights + (int64_t)row * TOPK, topk_ids + (int64_t)row * TOPK,
        do_align ? (u.align.topk_ids + row * TOPK) : nullptr,
        routed_scaling_factor, input_ids_ptr, ids_is_i32,
        bias_vl, image_sentinel_lo, is_padding_ptr, row, lane);
  }
  if (do_align) {
    __syncthreads();
    epilogue_align_bs8<128>(tid, M * TOPK, E, local_e, &u.align, emap, sorted, cap, eids, nb, ntpp);
  }
}

template <int E, int TOPK, int SPLIT_K, typename IdxT, int ST = 3, bool USE_SENTINEL = false>
__global__ void __launch_bounds__(256, 4) fgta_tc_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ logits_part,
    int M,
    int K,
    int64_t stride_xm,
    int64_t stride_wn,
    const float* __restrict__ corr_bias,
    float* __restrict__ topk_weights,
    IdxT* __restrict__ topk_ids,
    float routed_scaling_factor,
    int* __restrict__ done_cnt,
    const void* __restrict__ input_ids_ptr,
    bool ids_is_i32,
    const float* __restrict__ bias_vl,
    int64_t image_sentinel_lo,
    const bool* __restrict__ is_padding_ptr,
    bool do_align,
    int local_e,
    const int32_t* __restrict__ emap,
    int32_t* __restrict__ sorted,
    int cap,
    int32_t* __restrict__ eids,
    int nb,
    int32_t* __restrict__ ntpp) {
  struct EpiSmem {
    float row_logits[8][E];
    AlignBs8Smem align;
  };
  extern __shared__ __align__(128) uint4 smem[];
  constexpr int WS_STR = 64;
  constexpr int XS_STR = 32;
  constexpr int ST_U4 = WS_STR + XS_STR;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int c4 = lane & 3;
  const int et = blockIdx.x / SPLIT_K;
  const int ks = blockIdx.x % SPLIT_K;
  const int k_slice = K / SPLIT_K;
  const int k_warp = k_slice >> 3;
  const int k_base = ks * k_slice + warp * k_warp;
  const int e_base = et * 16;

  uint4* const wring = smem + warp * ST * ST_U4;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(wring);

  const int w_r = lane >> 1;
  const int w_c0 = lane & 1;
  const int w_c1 = w_c0 + 2;
  const uint32_t dw0 = (w_r * 4 + (w_c0 ^ (w_r & 3))) * 16;
  const uint32_t dw1 = (w_r * 4 + (w_c1 ^ (w_r & 3))) * 16;
  const __nv_bfloat16* wp = w + (int64_t)(e_base + w_r) * stride_wn + k_base + w_c0 * 8;

  const int x_r = lane >> 2;
  const int x_c = lane & 3;
  const uint32_t dx = (WS_STR + x_r * 4 + (x_c ^ (x_r & 3))) * 16;
  const int x_bytes = (x_r < M) ? 16 : 0;
  const __nv_bfloat16* xp = x + (int64_t)(x_r < M ? x_r : 0) * stride_xm + k_base + x_c * 8;

  const int a_row = lane & 15;
  const int a_c_sub0 = (lane >> 4) & 1;
  const int a_c_sub1 = a_c_sub0 + 2;
  const uint32_t rw_sub0 = (a_row * 4 + (a_c_sub0 ^ (a_row & 3))) * 16;
  const uint32_t rw_sub1 = (a_row * 4 + (a_c_sub1 ^ (a_row & 3))) * 16;

  const int b_row = lane & 7;
  const int b_c_sub0 = (lane >> 3) & 1;
  const int b_c_sub1 = b_c_sub0 + 2;
  const uint32_t rx_sub0 = (WS_STR + b_row * 4 + (b_c_sub0 ^ (b_row & 3))) * 16;
  const uint32_t rx_sub1 = (WS_STR + b_row * 4 + (b_c_sub1 ^ (b_row & 3))) * 16;

  const int n_iters = k_warp >> 5;
  auto issue = [&](int slot, int it) {
    if (it < n_iters) {
      const uint32_t st = ring + slot * ST_U4 * 16;
      const int ko = it * 32;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(st + dw0), "l"(wp + ko) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(st + dw1), "l"(wp + ko + 16) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   ::"r"(st + dx), "l"(xp + ko), "r"(x_bytes) : "memory");
    }
    cp_async_commit();
  };

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) issue(s, s);

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int it = 0; it < n_iters; ++it) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2) : "memory");
    __syncwarp();
    const int cur = it % ST;
    issue((cur + ST - 1) % ST, it + ST - 1);
    const uint32_t st = ring + cur * ST_U4 * 16;
    uint32_t a0, a1, a2, a3, b0, b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(st + rw_sub0) : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(st + rx_sub0) : "memory");
    mma_bf16(acc, a0, a1, a2, a3, b0, b1);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(st + rw_sub1) : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(st + rx_sub1) : "memory");
    mma_bf16(acc, a0, a1, a2, a3, b0, b1);
  }
  cp_async_wait0();
  __syncthreads();

  float* red = reinterpret_cast<float*>(smem);
  red[(warp * 8 + 2 * c4) * 16 + g] = acc[0];
  red[(warp * 8 + 2 * c4 + 1) * 16 + g] = acc[1];
  red[(warp * 8 + 2 * c4) * 16 + g + 8] = acc[2];
  red[(warp * 8 + 2 * c4 + 1) * 16 + g + 8] = acc[3];
  __syncthreads();

  const int row_stride = USE_SENTINEL ? 8 : M;
  if (tid < 32) {
    const int tr = tid >> 2;
    const int er4 = (tid & 3) * 4;
    if (tr < M) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w_i = 0; w_i < 8; ++w_i) {
        const float4 v = *reinterpret_cast<const float4*>(red + (w_i * 8 + tr) * 16 + er4);
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      uint4* dst = reinterpret_cast<uint4*>(
          logits_part + ((int64_t)ks * row_stride + tr) * E + e_base + er4);
      const uint32_t ux = __float_as_uint(s.x), uy = __float_as_uint(s.y);
      const uint32_t uz = __float_as_uint(s.z), uw = __float_as_uint(s.w);
      if constexpr (USE_SENTINEL) {
        asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     ::"l"(dst), "r"(ux), "r"(uy), "r"(uz), "r"(uw) : "memory");
      } else {
        __stcg(reinterpret_cast<float4*>(dst), s);
      }
    }
  }
  __syncthreads();

  EpiSmem* epi = reinterpret_cast<EpiSmem*>(smem);
  constexpr int E4 = E >> 2;
  if constexpr (USE_SENTINEL) {
    if (blockIdx.x != 0) return;
    constexpr uint32_t SENT = 0xFFFFFFFFu;
    if (warp < M) {
      for (int e4 = lane; e4 < E4; e4 += 32) {
        float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int sk = 0; sk < SPLIT_K; ++sk) {
          uint4* p = reinterpret_cast<uint4*>(logits_part + ((int64_t)sk * 8 + warp) * E + e4 * 4);
          uint32_t vx, vy, vz, vw;
          do {
            asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(vx), "=r"(vy), "=r"(vz), "=r"(vw) : "l"(p) : "memory");
          } while ((vx == SENT) | (vy == SENT) | (vz == SENT) | (vw == SENT));
          asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                       ::"l"(p), "r"(SENT), "r"(SENT), "r"(SENT), "r"(SENT) : "memory");
          acc4.x += __uint_as_float(vx);
          acc4.y += __uint_as_float(vy);
          acc4.z += __uint_as_float(vz);
          acc4.w += __uint_as_float(vw);
        }
        *reinterpret_cast<float4*>(&epi->row_logits[warp][e4 * 4]) = acc4;
      }
      __syncwarp();
      r_topk_warp_row<E, TOPK, IdxT, false>(
          epi->row_logits[warp], corr_bias,
          topk_weights + (int64_t)warp * TOPK, topk_ids + (int64_t)warp * TOPK,
          do_align ? (epi->align.topk_ids + warp * TOPK) : nullptr,
          routed_scaling_factor, input_ids_ptr, ids_is_i32,
          bias_vl, image_sentinel_lo, is_padding_ptr, warp, lane);
    }
  } else {
    __shared__ bool is_last;
    if (tid == 0) is_last = r_signal_last_cta(done_cnt, (E / 16) * SPLIT_K - 1);
    __syncthreads();
    if (!is_last) return;
    __syncthreads();
    if (tid == 0) *done_cnt = 0;
    if (warp < M) {
      for (int e4 = lane; e4 < E4; e4 += 32) {
        float4 acc4 = __ldcg(reinterpret_cast<const float4*>(logits_part + (int64_t)warp * E + e4 * 4));
#pragma unroll
        for (int sk = 1; sk < SPLIT_K; ++sk) {
          float4 v = __ldcg(reinterpret_cast<const float4*>(
              logits_part + ((int64_t)sk * M + warp) * E + e4 * 4));
          acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
        }
        *reinterpret_cast<float4*>(&epi->row_logits[warp][e4 * 4]) = acc4;
      }
      __syncwarp();
      r_topk_warp_row<E, TOPK, IdxT, false>(
          epi->row_logits[warp], corr_bias,
          topk_weights + (int64_t)warp * TOPK, topk_ids + (int64_t)warp * TOPK,
          do_align ? (epi->align.topk_ids + warp * TOPK) : nullptr,
          routed_scaling_factor, input_ids_ptr, ids_is_i32,
          bias_vl, image_sentinel_lo, is_padding_ptr, warp, lane);
    }
  }
  if (do_align) {
    __syncthreads();
    epilogue_align_bs8<256>(tid, M * TOPK, E, local_e, &epi->align, emap, sorted, cap, eids, nb, ntpp);
  }
}


}  // namespace r2_router

void dsv4_topk_v2(
    const at::Tensor& logits,
    const at::Tensor& corr_bias,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    double routed_scaling_factor,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& bias_vl_opt,
    int64_t image_sentinel_lo,
    const c10::optional<at::Tensor>& is_padding_opt) {
  const int M = static_cast<int>(logits.size(0));
  const int E = static_cast<int>(logits.size(1));
  const int topk = static_cast<int>(topk_ids.size(1));
  if (M == 0) return;
  TORCH_CHECK(logits.is_contiguous() && logits.scalar_type() == at::kFloat, "logits");
  TORCH_CHECK(corr_bias.is_contiguous() && corr_bias.scalar_type() == at::kFloat && corr_bias.numel() == E,
              "corr_bias");
  TORCH_CHECK(topk_weights.is_contiguous() && topk_weights.scalar_type() == at::kFloat && topk_ids.is_contiguous(),
              "outputs");
  const bool* pad_ptr = nullptr;
  if (is_padding_opt.has_value() && is_padding_opt->defined()) {
    TORCH_CHECK(
        is_padding_opt->is_contiguous() && is_padding_opt->scalar_type() == at::kBool &&
            is_padding_opt->numel() >= M &&
            (topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong),
        "is_padding");
    pad_ptr = is_padding_opt->data_ptr<bool>();
  }
  const void* ids_ptr = nullptr;
  bool ids_is_i32 = true;
  const float* bvl_ptr = nullptr;
  if (bias_vl_opt.has_value() && bias_vl_opt->defined() && image_sentinel_lo > 0) {
    TORCH_CHECK(input_ids_opt.has_value() && input_ids_opt->defined() && input_ids_opt->is_contiguous(), "input_ids");
    TORCH_CHECK(bias_vl_opt->is_contiguous() && bias_vl_opt->scalar_type() == at::kFloat, "bias_vl");
    ids_ptr = input_ids_opt->data_ptr();
    ids_is_i32 = (input_ids_opt->scalar_type() == at::kInt);
    TORCH_CHECK(ids_is_i32 || input_ids_opt->scalar_type() == at::kLong, "input_ids dtype");
    bvl_ptr = bias_vl_opt->data_ptr<float>();
  }
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const float rsf = static_cast<float>(routed_scaling_factor);

#define LAUNCH_TV2(E_VAL, TOPK_VAL, IDX_T)                                                        \
  r2_router::dsv4_topk_v2_kernel<E_VAL, TOPK_VAL, IDX_T><<<M, 32, 0, st>>>(                       \
      logits.data_ptr<float>(), corr_bias.data_ptr<float>(),                                      \
      topk_weights.data_ptr<float>(), reinterpret_cast<IDX_T*>(topk_ids.data_ptr()),              \
      rsf, ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr)

  if (E == 384 && topk == 6) {
    if (topk_ids.scalar_type() == at::kInt) LAUNCH_TV2(384, 6, int32_t);
    else if (topk_ids.scalar_type() == at::kUInt32) LAUNCH_TV2(384, 6, uint32_t);
    else { TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "ids dtype"); LAUNCH_TV2(384, 6, int64_t); }
  } else if (E == 256 && topk == 6) {
    if (topk_ids.scalar_type() == at::kInt) LAUNCH_TV2(256, 6, int32_t);
    else if (topk_ids.scalar_type() == at::kUInt32) LAUNCH_TV2(256, 6, uint32_t);
    else { TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "ids dtype"); LAUNCH_TV2(256, 6, int64_t); }
  } else if (E == 128 && topk == 3) {
    if (topk_ids.scalar_type() == at::kInt) LAUNCH_TV2(128, 3, int32_t);
    else if (topk_ids.scalar_type() == at::kUInt32) LAUNCH_TV2(128, 3, uint32_t);
    else { TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "ids dtype"); LAUNCH_TV2(128, 3, int64_t); }
  } else {
    TORCH_CHECK(false, "unsupported (E, topk) in dsv4_topk_v2");
  }
#undef LAUNCH_TV2
}

void fused_gate_topk_align(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& corr_bias,
    at::Tensor& logits_scratch,
    at::Tensor& done_cnt,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    double routed_scaling_factor,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& bias_vl_opt,
    int64_t image_sentinel_lo,
    bool do_align,
    int64_t local_e,
    const c10::optional<at::Tensor>& emap_opt,
    const c10::optional<at::Tensor>& sorted_opt,
    const c10::optional<at::Tensor>& eids_opt,
    const c10::optional<at::Tensor>& ntpp_opt,
    const c10::optional<at::Tensor>& is_padding_opt,
    bool use_sentinel) {
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int E = static_cast<int>(w.size(0));
  const int topk = static_cast<int>(topk_ids.size(1));
  TORCH_CHECK(M >= 1 && M <= 8 && (K % 512 == 0), "shape M=1..8, K%512==0");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && (x.stride(0) % 8 == 0) && (w.stride(0) % 8 == 0) &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "align");
  TORCH_CHECK(logits_scratch.numel() >= M * E && done_cnt.numel() >= 1, "scratch");

  const bool* pad_ptr = nullptr;
  if (is_padding_opt.has_value() && is_padding_opt->defined()) {
    TORCH_CHECK(
        is_padding_opt->is_contiguous() && is_padding_opt->scalar_type() == at::kBool &&
            is_padding_opt->numel() >= M,
        "is_padding");
    pad_ptr = is_padding_opt->data_ptr<bool>();
  }

  const void* ids_ptr = nullptr;
  bool ids_is_i32 = true;
  const float* bvl_ptr = nullptr;
  if (bias_vl_opt.has_value() && bias_vl_opt->defined() && image_sentinel_lo > 0) {
    TORCH_CHECK(input_ids_opt.has_value() && input_ids_opt->defined() && input_ids_opt->is_contiguous(), "input_ids");
    TORCH_CHECK(bias_vl_opt->is_contiguous() && bias_vl_opt->scalar_type() == at::kFloat, "bias_vl");
    ids_ptr = input_ids_opt->data_ptr();
    ids_is_i32 = (input_ids_opt->scalar_type() == at::kInt);
    TORCH_CHECK(ids_is_i32 || input_ids_opt->scalar_type() == at::kLong, "input_ids dtype");
    bvl_ptr = bias_vl_opt->data_ptr<float>();
  }

  const int32_t* emap_ptr = nullptr;
  int32_t* sorted_ptr = nullptr;
  int32_t* eids_ptr = nullptr;
  int32_t* ntpp_ptr = nullptr;
  int cap = 0, nb = 0;
  if (do_align) {
    TORCH_CHECK(local_e > 0 && local_e <= 128, "local_e <= 128");
    TORCH_CHECK(sorted_opt.has_value() && eids_opt.has_value() && ntpp_opt.has_value(), "align tensors");
    if (emap_opt.has_value() && emap_opt->defined()) {
      emap_ptr = emap_opt->data_ptr<int32_t>();
    }
    sorted_ptr = sorted_opt->data_ptr<int32_t>();
    cap = static_cast<int>(sorted_opt->numel());
    eids_ptr = eids_opt->data_ptr<int32_t>();
    nb = static_cast<int>(eids_opt->numel());
    ntpp_ptr = ntpp_opt->data_ptr<int32_t>();
  }

  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  float* lp = logits_scratch.data_ptr<float>();
  const float* bp = corr_bias.data_ptr<float>();
  float* twp = topk_weights.data_ptr<float>();
  int* cntp = done_cnt.data_ptr<int>();
  const float rsf = static_cast<float>(routed_scaling_factor);

  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  constexpr int FGTA_SMEM = 8 * 3 * 96 * 16;  // 36864 B (ST=3; > sizeof(EpiSmem) for E<=384)

#define LAUNCH_FGTA(E_VAL, TOPK_VAL, SK_VAL, IDX_T)                                                 \
  do {                                                                                              \
    IDX_T* tip = reinterpret_cast<IDX_T*>(topk_ids.data_ptr());                                     \
    if (M == 1) {                                                                                   \
      r2_router::fgta_m1_kernel<E_VAL, TOPK_VAL, IDX_T><<<E_VAL, 256, 0, st>>>(                     \
          xp, wp, lp, K, w.stride(0), bp, twp, tip, rsf, cntp,                                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else if (M == 2) {                                                                            \
      r2_router::fgta_m2_kernel<E_VAL, TOPK_VAL, IDX_T><<<E_VAL, 128, 0, st>>>(                     \
          xp, wp, lp, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                         \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else if (K == 5120 && use_sentinel && sms * 3 >= (E_VAL / 16) * SK_VAL &&                     \
               logits_scratch.numel() >= SK_VAL * 8 * E_VAL) {                                      \
      auto kern = r2_router::fgta_tc_kernel<E_VAL, TOPK_VAL, SK_VAL, IDX_T, 3, true>;               \
      static bool init = [&] {                                                                      \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, FGTA_SMEM);         \
        return true;                                                                                \
      }();                                                                                          \
      (void)init;                                                                                   \
      kern<<<(E_VAL / 16) * SK_VAL, 256, FGTA_SMEM, st>>>(                                          \
          xp, wp, lp, M, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else if (K == 5120 && logits_scratch.numel() >= SK_VAL * M * E_VAL) {                         \
      auto kern = r2_router::fgta_tc_kernel<E_VAL, TOPK_VAL, SK_VAL, IDX_T, 3, false>;              \
      static bool init = [&] {                                                                      \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, FGTA_SMEM);         \
        return true;                                                                                \
      }();                                                                                          \
      (void)init;                                                                                   \
      kern<<<(E_VAL / 16) * SK_VAL, 256, FGTA_SMEM, st>>>(                                          \
          xp, wp, lp, M, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else if (M <= 4) {                                                                            \
      r2_router::fgta_m3_8_kernel<4, 8, E_VAL, TOPK_VAL, IDX_T><<<E_VAL, 128, 0, st>>>(             \
          xp, wp, lp, M, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else if (M <= 6) {                                                                            \
      r2_router::fgta_m3_8_kernel<6, 6, E_VAL, TOPK_VAL, IDX_T><<<E_VAL, 128, 0, st>>>(             \
          xp, wp, lp, M, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    } else {                                                                                        \
      r2_router::fgta_m3_8_kernel<8, 4, E_VAL, TOPK_VAL, IDX_T><<<E_VAL, 128, 0, st>>>(             \
          xp, wp, lp, M, K, x.stride(0), w.stride(0), bp, twp, tip, rsf, cntp,                      \
          ids_ptr, ids_is_i32, bvl_ptr, image_sentinel_lo, pad_ptr,                                 \
          do_align, (int)local_e, emap_ptr, sorted_ptr, cap, eids_ptr, nb, ntpp_ptr);               \
    }                                                                                               \
  } while (0)

  if (E == 384 && topk == 6) {
    if (topk_ids.scalar_type() == at::kInt) LAUNCH_FGTA(384, 6, 4, int32_t);
    else { TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "ids dtype"); LAUNCH_FGTA(384, 6, 4, int64_t); }
  } else if (E == 128 && topk == 3) {
    if (topk_ids.scalar_type() == at::kInt) LAUNCH_FGTA(128, 3, 10, int32_t);
    else { TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "ids dtype"); LAUNCH_FGTA(128, 3, 10, int64_t); }
  } else {
    TORCH_CHECK(false, "unsupported (E, topk) in fused_gate_topk_align");
  }
#undef LAUNCH_FGTA
}

}  // namespace dsv41_r2

TORCH_LIBRARY(_dsv41_r2_C, m) {
  m.def("det_align_v2(Tensor ids, Tensor? emap, int global_e, int local_e, int BS, Tensor(a!) sorted, Tensor(b!) eids, Tensor(c!) ntpp) -> ()");
  m.def("w2_splitk_ep(Tensor h, Tensor w, Tensor s, Tensor sorted, Tensor eids, Tensor ntpp, Tensor topk_w, Tensor? topk_ids, Tensor? emap, int topk, int global_e, int ks_mode, int dq, Tensor(a!) ctr, Tensor(b!) part2, Tensor(c!) y, Tensor? bias=None) -> ()");
  m.def("pcie_ar_alloc_handle", &dsv41_r2::pcie_ar_alloc_handle);
  m.def("pcie_ar_open_handles", &dsv41_r2::pcie_ar_open_handles);
  m.def("pcie_ar_run(int handle_id, Tensor inp, Tensor(a!) out, int stage_mode, int n_ctas) -> ()");
  m.def("woa_mxfp8_splitk(Tensor x, Tensor w_u8, Tensor s_u8, bool bf16_mul, int split, Tensor(a!) part, Tensor(b!) tile_cnt, Tensor(c!) y) -> ()");
  m.def("woa_marlin_ksplit(Tensor x, Tensor w_marlin, Tensor s_marlin, Tensor(a!) part, Tensor(b!) out) -> ()");
  m.def("tp_greedy_local_argmax(Tensor local_logits, int valid_cols, int vocab_offset, Tensor(a!) out_stats) -> ()");
  m.def("tp_greedy_nonspec(Tensor gathered_stats, int tp_size, Tensor(a!) sampled_out) -> ()");
  m.def("tp_greedy_spec_verify(Tensor gathered_stats, Tensor verify_draft_sampled, Tensor cu_num_logits, int tp_size, Tensor(a!) sampled_out, Tensor(b!) num_sampled_out) -> ()");
  m.def("dequant_marlin_mxfp8(Tensor w_marlin, Tensor s_marlin, Tensor(a!) out_bf16, int size_n, int size_k) -> ()");
  m.def("dequant_row_mxfp8(Tensor w_u8, Tensor s_u8, Tensor(a!) out_bf16) -> ()");
  m.def("dsv4_topk_v2(Tensor logits, Tensor corr_bias, Tensor(a!) topk_weights, Tensor(b!) topk_ids, float routed_scaling_factor, Tensor? input_ids, Tensor? bias_vl, int image_sentinel_lo, Tensor? is_padding=None) -> ()");
  m.def("fused_gate_topk_align(Tensor x, Tensor w, Tensor corr_bias, Tensor(a!) logits_scratch, Tensor(b!) done_cnt, Tensor(c!) topk_weights, Tensor(d!) topk_ids, float routed_scaling_factor, Tensor? input_ids, Tensor? bias_vl, int image_sentinel_lo, bool do_align, int local_e, Tensor? emap, Tensor(e!)? sorted, Tensor(f!)? eids, Tensor(g!)? ntpp, Tensor? is_padding=None, bool use_sentinel=False) -> ()");
}

TORCH_LIBRARY_IMPL(_dsv41_r2_C, CUDA, m) {
  m.impl("det_align_v2", &dsv41_r2::det_align_v2);
  m.impl("w2_splitk_ep", &dsv41_r2::w2_splitk_ep);
  m.impl("pcie_ar_run", &dsv41_r2::pcie_ar_run);
  m.impl("woa_mxfp8_splitk", &dsv41_r2::woa_mxfp8_splitk);
  m.impl("woa_marlin_ksplit", &dsv41_r2::woa_marlin_ksplit);
  m.impl("tp_greedy_local_argmax", &dsv41_r2::tp_greedy_local_argmax);
  m.impl("tp_greedy_nonspec", &dsv41_r2::tp_greedy_nonspec);
  m.impl("tp_greedy_spec_verify", &dsv41_r2::tp_greedy_spec_verify);
  m.impl("dequant_marlin_mxfp8", &dsv41_r2::dequant_marlin_mxfp8);
  m.impl("dequant_row_mxfp8", &dsv41_r2::dequant_row_mxfp8);
  m.impl("dsv4_topk_v2", &dsv41_r2::dsv4_topk_v2);
  m.impl("fused_gate_topk_align", &dsv41_r2::fused_gate_topk_align);
}
