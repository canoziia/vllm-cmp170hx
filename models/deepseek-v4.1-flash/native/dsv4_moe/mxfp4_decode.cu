// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * Weight-streaming MXFP4 MoE decode kernels for DeepSeek V4.1 Flash on sm_80.
 *
 * Adapted from models/glm-5.3-flash/native/ampere_marlin/decode_orig.cu (itself
 * ported from Morrowmake vllm-cmp170hx). Differences:
 *  - weights are Marlin-packed MXFP4 (float4_e2m1f, gptq_marlin_repack 4-bit
 *    layout - the same nibble positions as uint4b8);
 *  - scales are e8m0 bytes, one per 32 k (2 k16 rows), [E][K/32][N], permuted
 *    by marlin_permute_scales and byte-reordered [0,2,1,3] by
 *    mxfp4_marlin_process_scales;
 *  - K / N need not be powers of two (DeepSeek: 5120 / 4608 / 2304).
 *
 * Dequantisation is exact: each e2m1 nibble is placed in bf16 as v * 2^-126
 * (sign to bit 15, e1e0m to bits 8..6) and multiplied by the folded scale
 * 2^(S-1) = 2^(S-127) * 2^126 (bf16 exponent field S + 126; valid for every
 * e8m0 scale S <= 128, which the Python side checks per layer). The weights fed
 * to the tensor cores are therefore the values Marlin uses; only the fp32
 * accumulation order differs (split-K partials summed in a fixed order:
 * deterministic, no atomics on data).
 *
 * Measured on CMP 170HX (pure read 1.61 TB/s): FP32/bf16 FMA-class ALU is
 * limited to ~42 thread-instr/SM/clk, tensor cores are not; the dequant is
 * kept to shifts + one lop3 + one bf16 multiply per pair of weights.
 *
 *   w13: fp32 split-K partials part[ks][slot][N]
 *   act: fixed-order sum -> bf16 -> clamp-silu-mul -> h (bf16)
 *   w2 : one k-slice; bf16(acc) * bf16(router w) -> bf16 -> c3[slot][K]
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace dsv4_mxfp4_decode {

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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

constexpr int WARPS = 4;

template <int CR, int TW>
struct Frags {
  uint32_t x[CR][2];     // activation B fragments
  uint2 s[CR / 2][TW];   // per scale group, per tile: 8 e8m0 bytes of lane group g
};

// R: k16 rows per warp item; CR: k16 rows per chunk (even, one or more whole
// scale groups); TW: adjacent 64-col tiles per warp.  K / N are compile-time.
template <int R, int CR, int TW, int K, int N, bool W13>
__global__ void __launch_bounds__(WARPS * 32)
    mxfp4_dec_gemm(const __nv_bfloat16* __restrict__ A, int64_t a_stride,
                   const int4* __restrict__ W, const uint8_t* __restrict__ S,
                   const int32_t* __restrict__ sorted, const int32_t* __restrict__ eids,
                   const int32_t* __restrict__ ntpp, const float* __restrict__ topk_w,
                   int topk, int n_slots, int* __restrict__ ctr, int* __restrict__ ctr_other,
                   float* __restrict__ part, __nv_bfloat16* __restrict__ out) {
  static_assert(CR % 2 == 0 && R % CR == 0, "chunks hold whole 32-k scale groups");
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
  constexpr int wrow = N >> 1;  // int4 per k16 row
  const int items = (ntpp[0] >> 3) * TG * KS;

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
      const int arow = e0 >= 0 && sid < n_slots ? (W13 ? sid / topk : sid) : 0;
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
    if constexpr (W13) {
      float* p0 = part + ((int64_t)ks * n_slots + s0) * N + t0 * 64 + g;
      float* p1 = part + ((int64_t)ks * n_slots + s1) * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0) p0[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h];
            if (v1) p1[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h + 1];
          }
    } else {
      const float w0 = v0 ? __bfloat162float(__float2bfloat16(topk_w[s0])) : 0.f;
      const float w1 = v1 ? __bfloat162float(__float2bfloat16(topk_w[s1])) : 0.f;
      __nv_bfloat16* o0 = out + (int64_t)s0 * N + t0 * 64 + g;
      __nv_bfloat16* o1 = out + (int64_t)s1 * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0)
              o0[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h])) * w0);
            if (v1)
              o1[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h + 1])) * w1);
          }
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
    // Fast exact dequant (needs every e8m0 scale <= 128, checked at load):
    // nibble aligned to bits 12-15 of each half, placed as v * 2^-126, times
    // the folded scale 2^(S-1) = 2^(S-127) * 2^126 (bf16 exponent S + 126).
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
            auto pl = [](uint32_t t) {
              const uint32_t c = (t >> 6) & 0x01C001C0u;
              uint32_t r;
              asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(t), "r"(0x80008000u), "r"(c));
              return r;
            };
            const uint32_t a0 = b2u(__hmul2(u2b(pl(qj << 12)), sl));  // nib0/nib4, col g
            const uint32_t a1 = b2u(__hmul2(u2b(pl(qj << 4)), sh));   // nib2/nib6, col g+8
            const uint32_t a2 = b2u(__hmul2(u2b(pl(qj << 8)), sl));   // nib1/nib5, col g
            const uint32_t a3 = b2u(__hmul2(u2b(pl(qj)), sh));        // nib3/nib7, col g+8
            mma_bf16(acc[u][j], a0, a1, a2, a3, cur.x[i][0], cur.x[i][1]);
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

// h[slot][n] = bf16(bf16(silu(min(G, L))) * clamp(U, -L, L)), G/U = bf16(sum_ks part).
template <int KS>
__global__ void mxfp4_dec_act(const float* __restrict__ part, const int32_t* __restrict__ ids,
                              int n_slots, int Nh, float limit, __nv_bfloat16* __restrict__ h) {
  const int slot = blockIdx.x;
  if (ids[slot] < 0) return;
  const int n = threadIdx.x * 4;
  if (n >= Nh) return;
  const int64_t row = (int64_t)n_slots * 2 * Nh;
  const float* p = part + (int64_t)slot * 2 * Nh + n;
  float4 gv[KS], uv[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    gv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row));
    uv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row + Nh));
  }
  float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    g[0] += gv[ks].x; g[1] += gv[ks].y; g[2] += gv[ks].z; g[3] += gv[ks].w;
    u[0] += uv[ks].x; u[1] += uv[ks].y; u[2] += uv[ks].z; u[3] += uv[ks].w;
  }
  __nv_bfloat16 r[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float gf = fminf(__bfloat162float(__float2bfloat16(g[i])), limit);
    const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(u[i])), limit), -limit);
    const float sl = __bfloat162float(__float2bfloat16(gf / (1.0f + expf(-gf))));
    r[i] = __float2bfloat16(sl * uf);
  }
  *reinterpret_cast<uint2*>(h + (int64_t)slot * Nh + n) = *reinterpret_cast<const uint2*>(r);
}

// w13 partials summed to bf16 only (for the bit-exact activation path / tests).
template <int KS>
__global__ void mxfp4_dec_sum(const float* __restrict__ part, int n_slots, int N,
                              __nv_bfloat16* __restrict__ y) {
  const int slot = blockIdx.x;
  const int64_t row = (int64_t)n_slots * N;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float a = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ks++) a += part[ks * row + (int64_t)slot * N + n];
    y[(int64_t)slot * N + n] = __float2bfloat16(a);
  }
}

template <int R, int CR, int TW, int KK, int NN, bool W13>
static int64_t launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                      at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                      at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int cap,
                      at::Tensor& ctr, at::Tensor& out) {
  constexpr int T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  constexpr int KS = KK / 16 / R;
  auto kern = mxfp4_dec_gemm<R, CR, TW, KK, NN, W13>;
  static const int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM);
    return n > 0 ? n : 1;
  }();
  if (W13) {
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.numel() >= KS * n_slots * NN,
                "w13 partial buffer");
  } else {
    TORCH_CHECK(KS == 1 && out.scalar_type() == at::kBFloat16 && out.numel() >= n_slots * NN,
                "w2 output");
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blk = std::min<int64_t>(sorted.numel() / 8, n_slots);
  const int64_t max_items = max_blk * KS * (NN / 64 / TW);
  const int64_t grid = std::min<int64_t>(
      (max_items + WARPS - 1) / WARPS,
      (int64_t)sms * (cap > 0 ? std::min(cap, occ) : occ));
  if (grid > 0)
    kern<<<grid, T, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)a.data_ptr(), a.stride(0), (const int4*)w.data_ptr(),
        (const uint8_t*)s.data_ptr(), sorted.data_ptr<int32_t>(), eids.data_ptr<int32_t>(),
        ntpp.data_ptr<int32_t>(), topk_w.data_ptr<float>(), (int)topk, (int)n_slots,
        ctr.data_ptr<int32_t>() + (W13 ? 0 : 1), ctr.data_ptr<int32_t>() + (W13 ? 1 : 0),
        W13 ? out.data_ptr<float>() : nullptr,
        W13 ? nullptr : (__nv_bfloat16*)out.data_ptr());
  return KS;
}

}  // namespace dsv4_mxfp4_decode

using namespace dsv4_mxfp4_decode;

// w13: part [KS, n_slots, N] fp32 (returns KS);  w2: out [n_slots, N] bf16 (returns 1).
// cfg & 15: (CR, TW) variant; cfg >> 4: CTA-per-SM cap (0 = occupancy).
int64_t dsv4_mxfp4_gemm(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                        at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                        at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int64_t K,
                        int64_t N, bool w13, int64_t cfg, at::Tensor& ctr, at::Tensor& out) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.stride(1) == 1 && a.stride(0) % 2 == 0,
              "a: bf16, unit inner stride, even row stride");
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && s.element_size() == 1,
              "weights contiguous, e8m0 byte scales");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && sorted.scalar_type() == at::kInt &&
                  eids.scalar_type() == at::kInt && ntpp.scalar_type() == at::kInt,
              "routing dtypes");
  TORCH_CHECK(ctr.scalar_type() == at::kInt && ctr.numel() >= 2 && ctr.is_contiguous(),
              "ctr: 2 int32 work counters");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  const int cap = (int)(cfg >> 4);
#define DSV4_CASE(WW, KK, NN, R, CFG, CR, TW)                                             \
  if (w13 == WW && K == KK && N == NN && (cfg & 15) == CFG)                              \
    return launch<R, CR, TW, KK, NN, WW>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots, \
                                         cap, ctr, out);
  // w13: K=5120 hidden, N=2*2304; KS = 4 (R = 80 k16 rows)
  DSV4_CASE(true, 5120, 4608, 80, 0, 4, 1)
  DSV4_CASE(true, 5120, 4608, 80, 1, 4, 2)
  DSV4_CASE(true, 5120, 4608, 80, 2, 8, 1)
  DSV4_CASE(true, 5120, 4608, 80, 3, 2, 1)
  DSV4_CASE(true, 5120, 4608, 80, 4, 16, 1)
  DSV4_CASE(true, 5120, 4608, 80, 5, 8, 2)
  // w2: K=2304, N=5120 hidden; one k-slice (R = 144)
  DSV4_CASE(false, 2304, 5120, 144, 0, 4, 1)
  DSV4_CASE(false, 2304, 5120, 144, 1, 4, 2)
  DSV4_CASE(false, 2304, 5120, 144, 2, 8, 1)
  DSV4_CASE(false, 2304, 5120, 144, 3, 2, 1)
  DSV4_CASE(false, 2304, 5120, 144, 4, 16, 1)
  DSV4_CASE(false, 2304, 5120, 144, 5, 8, 2)
#undef DSV4_CASE
  TORCH_CHECK(false, "unsupported shape/cfg: w13=", w13, " K=", K, " N=", N, " cfg=", cfg);
}

void dsv4_mxfp4_act(at::Tensor const& part, at::Tensor const& ids, int64_t ksplit,
                    int64_t n_slots, int64_t Nh, double limit, at::Tensor& h) {
  const at::cuda::OptionalCUDAGuard guard(part.device());
  TORCH_CHECK(ksplit == 4 && Nh % 4 == 0 && Nh / 4 <= 1024, "act: ksplit 4, Nh % 4 == 0");
  if (n_slots > 0)
    mxfp4_dec_act<4><<<n_slots, (int)(Nh / 4), 0, at::cuda::getCurrentCUDAStream()>>>(
        part.data_ptr<float>(), ids.data_ptr<int32_t>(), (int)n_slots, (int)Nh, (float)limit,
        (__nv_bfloat16*)h.data_ptr());
}

void dsv4_mxfp4_sum(at::Tensor const& part, int64_t ksplit, int64_t n_slots, int64_t N,
                    at::Tensor& y) {
  const at::cuda::OptionalCUDAGuard guard(part.device());
  TORCH_CHECK(ksplit == 4, "sum: ksplit 4");
  if (n_slots > 0)
    mxfp4_dec_sum<4><<<n_slots, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        part.data_ptr<float>(), (int)n_slots, (int)N, (__nv_bfloat16*)y.data_ptr());
}

int64_t dsv4_mxfp4_abi() { return 1; }

TORCH_LIBRARY(_dsv4_moe_C, m) {
  m.def(
      "gemm(Tensor a, Tensor w, Tensor s, Tensor sorted, Tensor eids, Tensor ntpp, "
      "Tensor topk_w, int topk, int n_slots, int K, int N, bool w13, int cfg, "
      "Tensor(c!) ctr, Tensor(o!) out) -> int");
  m.def("act(Tensor part, Tensor ids, int ksplit, int n_slots, int Nh, float limit, "
        "Tensor(h!) h) -> ()");
  m.def("sum(Tensor part, int ksplit, int n_slots, int N, Tensor(y!) y) -> ()");
  m.def("abi() -> int");
}

TORCH_LIBRARY_IMPL(_dsv4_moe_C, CUDA, m) {
  m.impl("gemm", &dsv4_mxfp4_gemm);
  m.impl("act", &dsv4_mxfp4_act);
  m.impl("sum", &dsv4_mxfp4_sum);
}
TORCH_LIBRARY_IMPL(_dsv4_moe_C, CompositeExplicitAutograd, m) {
  m.impl("abi", &dsv4_mxfp4_abi);
}
