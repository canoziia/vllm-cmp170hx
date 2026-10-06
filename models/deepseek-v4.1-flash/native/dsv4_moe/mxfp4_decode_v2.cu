// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * MXFP4 MoE decode kernels, v2 (DeepSeek V4.1 Flash, sm_80).  Library
 * _dsv4_moe2 (same .so as _dsv4_moe_C; v1 ops untouched).
 *
 * Same work distribution / Marlin weight layout / cp.async ring as v1
 * (mxfp4_decode.cu). New, each selectable per call:
 *
 *  dq = 0  v1 dequant (nibble as v*2^-126, bf16 HMUL2 by folded scale
 *          2^(S-1); needs S <= 128).  Bitwise identical to v1.
 *  dq = 1  no per-weight multiply. The nibble goes to the tensor core as the
 *          exact bf16 v*2^-126 (+-0.5 is a bf16 subnormal: tensor cores take
 *          subnormal inputs; the GPU test probes this bit-exactly).  The
 *          activations are pre-scaled once per k16 row by 2^100 (exact,
 *          |x| < 2^28), each 32-k scale group is accumulated by two chained
 *          mmas into a fresh fp32 tmp and folded into acc with one FFMA by the
 *          fp32 power of two 2^(S-127) (exact: the product only changes the
 *          exponent), epilogue multiplies by 2^26 (exact).  Per 8 weights:
 *          ~15 int + 2 FFMA (+0.25 HMUL2 on x)  instead of ~17 int + 4 HMUL2.
 *          Valid for 1 <= S <= 254 (S = 0 -> contribution 0 instead of
 *          < 2^-120 relative; S = 255 is NaN in e8m0).
 *          Summation: fp32 tensor-core chains of 32 k, then FFMA in fp32 ->
 *          different (shorter-chain) rounding than v1/Marlin; see fp64 test.
 *  dq = 2  dq 1 with the k-pairs (nib0,nib4) and (nib1,nib5) extracted by
 *          and + multiply-by-constant + and (3 ops) instead of shl + shr +
 *          and + lop3 (4 ops): (Y<<12)+(Y<<6) never carries.  Same values as
 *          dq 1 (bitwise identical outputs); faster only if the compiler's
 *          IMAD/LEA is cheaper than the shift pair on this part (bench it).
 *
 *  fuse (w13): the warp that writes the last of the 2*KS split-K partials of
 *          a (8-slot block, gate tile, matching up tile) pair sums them in the
 *          fixed ks order and applies clamp-silu-mul (code identical to v1
 *          act => bitwise identical h), no separate act/sum kernel.
 *  fuse (w2):  the warp that finishes the last block of a 64*TW column tile
 *          sums the topk rows of every token (moe_sum) for that tile, in one
 *          of three orders (0: fp32 sequential, 1: torch reduce vt0=4
 *          ((x0+x4)+(x1+x5))+x2)+x3, 2: bf16 sequential like the vLLM
 *          moe_sum kernel); the GPU test tells which one is bitwise equal to
 *          the deployed reduction.
 *  Last-arriver detection uses per-tile int counters in a caller-provided
 *  zeroed buffer that the last arriver resets to 0 (CUDA-graph safe,
 *  deterministic: data never goes through atomics; summation order fixed).
 *
 *  align: single-CTA CUDA version of fast_det_align (same outputs).
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace dsv4_mxfp4_v2 {

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
// d = A*B (zero accumulator)
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
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

// sign -> bit 15/31, e1 e0 m -> bits 8..6 / 24..22 of the nibble at bits 12..15 / 28..31
__device__ __forceinline__ uint32_t pl(uint32_t t) {
  const uint32_t c = (t >> 6) & 0x01C001C0u;
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(t), "r"(0x80008000u), "r"(c));
  return r;
}

// a0 = nib0/nib4 (row g, k 2c..), a1 = nib2/nib6 (row g+8), a2 = nib1/nib5 (row g, k+8),
// a3 = nib3/nib7 (row g+8, k+8); every half = bf16 bits of v * 2^-126.
template <int DQ>
__device__ __forceinline__ void deq_raw(uint32_t q, uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                        uint32_t& a3) {
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
constexpr uint32_t XPRE = 0x71807180u;  // bf16x2 (2^100, 2^100)
constexpr float EPI = 67108864.0f;       // 2^26 = 2^126 / 2^100
constexpr int SYNC_W2 = 0;               // w2 tile counters  [0, 128)
constexpr int SYNC_W13 = 128;            // w13 pair counters [128, 128 + blocks * pairs)

template <int CR, int TW>
struct Frags {
  uint32_t x[CR][2];
  uint2 s[CR / 2][TW];
};

template <int DQ, int R, int CR, int TW, int K, int N, bool W13>
__global__ void __launch_bounds__(WARPS * 32)
    mxfp4_dec2_gemm(const __nv_bfloat16* __restrict__ A, int64_t a_stride,
                    const int4* __restrict__ W, const uint8_t* __restrict__ S,
                    const int32_t* __restrict__ sorted, const int32_t* __restrict__ eids,
                    const int32_t* __restrict__ ntpp, const float* __restrict__ topk_w,
                    int topk, int n_slots, int* __restrict__ ctr, int* __restrict__ ctr_other,
                    float* __restrict__ part, __nv_bfloat16* __restrict__ out, int fuse,
                    float limit, int sum_order, int* __restrict__ sync,
                    __nv_bfloat16* __restrict__ aux) {
  static_assert(CR % 2 == 0 && R % CR == 0, "chunks hold whole 32-k scale groups");
  static_assert(K % (16 * R) == 0 && N % (64 * TW) == 0, "shape");
  constexpr int CH = R / CR;
  constexpr int TG = N / 64 / TW;
  constexpr int KS = K / 16 / R;
  constexpr int SLOT = CR * TW * 32;
  constexpr int NH = N / 2;
  constexpr int HP = TG / 2;  // w13: gate/up tile pairs
  static_assert(!W13 || (NH % (64 * TW) == 0), "w13 gate/up tiles must pair up");
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

  // ---- fused w13 tail: fixed-order split-K sum + clamp-silu-mul (== v1 act) ----
  auto act_pair = [&](int b, int pair) {
    constexpr int C4 = 16 * TW;  // float4 per slot row in one tile
    const int64_t row = (int64_t)n_slots * N;
    const bool ok = eids[b] >= 0;
    for (int task = lane; task < 8 * C4; task += 32) {
      const int r = task / C4, c4 = task - r * C4;
      const int slot = sorted[b * 8 + r];
      if (!ok || slot >= n_slots) continue;
      const int n = pair * 64 * TW + 4 * c4;
      const float* p = part + (int64_t)slot * N + n;
      float4 gv[KS], uv[KS];
#pragma unroll
      for (int ks = 0; ks < KS; ks++) {
        gv[ks] = __ldcg(reinterpret_cast<const float4*>(p + ks * row));
        uv[ks] = __ldcg(reinterpret_cast<const float4*>(p + ks * row + NH));
      }
      float gg[4] = {0.f, 0.f, 0.f, 0.f}, uu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < KS; ks++) {
        gg[0] += gv[ks].x; gg[1] += gv[ks].y; gg[2] += gv[ks].z; gg[3] += gv[ks].w;
        uu[0] += uv[ks].x; uu[1] += uv[ks].y; uu[2] += uv[ks].z; uu[3] += uv[ks].w;
      }
      __nv_bfloat16 res[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float gf = fminf(__bfloat162float(__float2bfloat16(gg[i])), limit);
        const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(uu[i])), limit), -limit);
        const float sl = __bfloat162float(__float2bfloat16(gf / (1.0f + expf(-gf))));
        res[i] = __float2bfloat16(sl * uf);
      }
      *reinterpret_cast<uint2*>(aux + (int64_t)slot * NH + n) =
          *reinterpret_cast<const uint2*>(res);
    }
  };

  // ---- fused w2 tail: per-token sum over the topk rows of one column tile ----
  auto sum_tile = [&](int tg) {
    constexpr int P2 = 32 * TW;  // bf16 pairs per tile row
    const int M = n_slots / topk;
    for (int task = lane; task < M * P2; task += 32) {
      const int m = task / P2, pp = task - m * P2;
      const int n = tg * 64 * TW + 2 * pp;
      const __nv_bfloat16* src = out + (int64_t)m * topk * N + n;
      float2 v[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        if (k < topk) {
          const uint32_t raw = __ldcg(reinterpret_cast<const unsigned int*>(src + (int64_t)k * N));
          v[k] = __bfloat1622float2(u2b(raw));
        } else {
          v[k] = make_float2(0.f, 0.f);
        }
      }
      float rx, ry;
      if (sum_order == 1) {  // torch reduce: 4 accumulators, tail, sequential combine
        float ax[4] = {0.f, 0.f, 0.f, 0.f}, ay[4] = {0.f, 0.f, 0.f, 0.f};
        int i = 0;
#pragma unroll
        for (int blk = 0; blk < 2; blk++) {
          if (i + 3 < topk) {
#pragma unroll
            for (int t = 0; t < 4; t++) { ax[t] += v[blk * 4 + t].x; ay[t] += v[blk * 4 + t].y; }
            i += 4;
          }
        }
#pragma unroll
        for (int t = 0; t < 4; t++) {
          if (i + t < topk) {
            // i is 0 or 4 here (topk <= 8)
            const float2 vv = i == 0 ? v[t] : v[4 + t];
            ax[t] += vv.x; ay[t] += vv.y;
          }
        }
        rx = ax[0]; ry = ay[0];
#pragma unroll
        for (int t = 1; t < 4; t++) { rx = rx + ax[t]; ry = ry + ay[t]; }
      } else if (sum_order == 2) {  // bf16 accumulator, like moe_sum_kernel<bf16>
        rx = 0.f; ry = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++)
          if (k < topk) {
            rx = __bfloat162float(__float2bfloat16(rx + v[k].x));
            ry = __bfloat162float(__float2bfloat16(ry + v[k].y));
          }
      } else {  // fp32 sequential
        rx = 0.f; ry = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++)
          if (k < topk) { rx += v[k].x; ry += v[k].y; }
      }
      *reinterpret_cast<__nv_bfloat162*>(aux + (int64_t)m * N + n) =
          __floats2bfloat162_rn(rx, ry);
    }
  };

  // returns true in all lanes of the warp that arrived last (and resets the counter)
  auto arrive = [&](int* cp, int need) {
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) {
      last = atomicAdd(cp, 1) == need - 1;
      if (last) *cp = 0;
    }
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last) __threadfence();
    return last != 0;
  };

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
      if (fuse) {
        const int pair = (t0 / TW) % HP;
        if (arrive(sync + SYNC_W13 + b * HP + pair, 2 * KS)) act_pair(b, pair);
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
      if (fuse) {
        const int tg = t0 / TW;
        if (arrive(sync + SYNC_W2 + tg, nblk * KS)) sum_tile(tg);
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
    if constexpr (DQ == 0) {
      // v1 dequant (bitwise identical): v * 2^-126 times folded bf16 scale 2^(S-1).
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
      // Raw nibbles into the tensor core, scale per 32-k group applied in fp32.
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
          // fp32 2^(S-127): row g uses bytes 0/1, row g+8 bytes 2/3 (after [0,2,1,3]).
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

template <int DQ, int R, int CR, int TW, int KK, int NN, bool W13>
static int64_t launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                      at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                      at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int cap,
                      bool fuse, double limit, int64_t sum_order, at::Tensor& ctr,
                      at::Tensor& sync, at::Tensor& out, at::Tensor& aux) {
  constexpr int T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  constexpr int KS = KK / 16 / R;
  constexpr int TG = NN / 64 / TW;
  auto kern = mxfp4_dec2_gemm<DQ, R, CR, TW, KK, NN, W13>;
  static const int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM);
    return n > 0 ? n : 1;
  }();
  const int64_t max_blk = std::min<int64_t>(sorted.numel() / 8, n_slots);
  if (W13) {
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.numel() >= KS * n_slots * NN,
                "w13 partial buffer");
    if (fuse)
      TORCH_CHECK(aux.scalar_type() == at::kBFloat16 && aux.is_contiguous() &&
                      aux.numel() >= n_slots * (NN / 2) &&
                      sync.numel() >= SYNC_W13 + max_blk * (TG / 2),
                  "fused act: h [n_slots, N/2] bf16 and sync counters");
  } else {
    TORCH_CHECK(KS == 1 && out.scalar_type() == at::kBFloat16 && out.numel() >= n_slots * NN,
                "w2 output");
    if (fuse)
      TORCH_CHECK(aux.scalar_type() == at::kBFloat16 && aux.is_contiguous() && topk <= 8 &&
                      n_slots % topk == 0 && aux.numel() >= (n_slots / topk) * NN &&
                      TG <= SYNC_W13 - SYNC_W2 && sync.numel() >= SYNC_W13,
                  "fused sum: y [M, N] bf16, topk <= 8, sync counters");
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_items = max_blk * KS * TG;
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
        W13 ? nullptr : (__nv_bfloat16*)out.data_ptr(), fuse ? 1 : 0, (float)limit,
        (int)sum_order, fuse ? sync.data_ptr<int32_t>() : nullptr,
        fuse ? (__nv_bfloat16*)aux.data_ptr() : nullptr);
  return KS;
}

// ---------------- deterministic MoE align (== fast_det_align / torch det path) -------------
template <typename IdT>
__global__ void __launch_bounds__(1024)
    det_align_kernel(const IdT* __restrict__ ids, int S, int E, int BS,
                     int32_t* __restrict__ sorted, int cap, int32_t* __restrict__ eids, int nb,
                     int32_t* __restrict__ ntpp) {
  __shared__ int cnt[1024];
  __shared__ int sblk[1024];
  __shared__ int sid[512];
  __shared__ int wsum[32];
  const int tid = threadIdx.x;
  cnt[tid] = 0;
  for (int i = tid; i < S; i += blockDim.x) {
    const int64_t e = (int64_t)ids[i];
    sid[i] = (e >= 0 && e < E) ? (int)e : -1;
  }
  __syncthreads();
  for (int i = tid; i < S; i += blockDim.x)
    if (sid[i] >= 0) atomicAdd(&cnt[sid[i]], 1);
  __syncthreads();
  const int myblk = tid < E ? (cnt[tid] + BS - 1) / BS : 0;
  // exclusive block-wide scan of myblk
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
    wsum[lane] = v;  // inclusive
  }
  __syncthreads();
  const int excl = inc - myblk + (wid > 0 ? wsum[wid - 1] : 0);
  sblk[tid] = excl;
  const int total_blk = wsum[31];
  const int total = total_blk * BS;
  for (int p = tid; p < cap; p += blockDim.x) sorted[p] = S;
  for (int b = tid; b < nb; b += blockDim.x)
    if (b * BS >= total) eids[b] = -1;
  __syncthreads();
  for (int i = tid; i < S; i += blockDim.x) {
    const int e = sid[i];
    if (e < 0) continue;
    int rank = 0;
    for (int j = 0; j < i; j++) rank += sid[j] == e;
    sorted[sblk[e] * BS + rank] = i;
  }
  if (tid < E)
    for (int k = 0; k < myblk; k++)
      if (excl + k < nb) eids[excl + k] = tid;
  if (tid == 0) *ntpp = total;
}

}  // namespace dsv4_mxfp4_v2

using namespace dsv4_mxfp4_v2;

// w13: part [KS, n_slots, N] fp32 (+ fused: aux = h [n_slots, N/2]) -> returns KS.
// w2 : out [n_slots, N] bf16 (+ fused: aux = y [n_slots/topk, N]) -> returns 1.
// cfg & 15: (CR, TW) variant as v1; cfg >> 4: CTA-per-SM cap.  dq: 0 / 1 / 2.
int64_t dsv4_mxfp4_gemm2(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                         at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                         at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int64_t K,
                         int64_t N, bool w13, int64_t cfg, int64_t dq, bool fuse, double limit,
                         int64_t sum_order, at::Tensor& ctr, at::Tensor& sync, at::Tensor& out,
                         at::Tensor& aux) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.stride(1) == 1 && a.stride(0) % 2 == 0,
              "a: bf16, unit inner stride, even row stride");
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && s.element_size() == 1,
              "weights contiguous, e8m0 byte scales");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && sorted.scalar_type() == at::kInt &&
                  eids.scalar_type() == at::kInt && ntpp.scalar_type() == at::kInt,
              "routing dtypes");
  TORCH_CHECK(ctr.scalar_type() == at::kInt && ctr.numel() >= 2 && ctr.is_contiguous(),
              "ctr: 2 int32 work counters");
  TORCH_CHECK(!fuse || (sync.scalar_type() == at::kInt && sync.is_contiguous()),
              "sync: int32 counters (zeroed once)");
  TORCH_CHECK(sum_order >= 0 && sum_order <= 2, "sum_order 0..2");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  const int cap = (int)(cfg >> 4);
#define DSV4_CASE2(WW, KK, NN, R, CFG, CR, TW)                                                \
  if (w13 == WW && K == KK && N == NN && (cfg & 15) == CFG) {                                 \
    if (dq == 0)                                                                              \
      return launch<0, R, CR, TW, KK, NN, WW>(a, w, s, sorted, eids, ntpp, topk_w, topk,      \
                                              n_slots, cap, fuse, limit, sum_order, ctr,     \
                                              sync, out, aux);                                \
    if (dq == 1)                                                                              \
      return launch<1, R, CR, TW, KK, NN, WW>(a, w, s, sorted, eids, ntpp, topk_w, topk,      \
                                              n_slots, cap, fuse, limit, sum_order, ctr,     \
                                              sync, out, aux);                                \
    if (dq == 2)                                                                              \
      return launch<2, R, CR, TW, KK, NN, WW>(a, w, s, sorted, eids, ntpp, topk_w, topk,      \
                                              n_slots, cap, fuse, limit, sum_order, ctr,     \
                                              sync, out, aux);                                \
  }
  DSV4_CASE2(true, 5120, 4608, 80, 0, 4, 1)
  DSV4_CASE2(true, 5120, 4608, 80, 1, 4, 2)
  DSV4_CASE2(true, 5120, 4608, 80, 2, 8, 1)
  DSV4_CASE2(true, 5120, 4608, 80, 3, 2, 1)
  DSV4_CASE2(true, 5120, 4608, 80, 4, 16, 1)
  DSV4_CASE2(true, 5120, 4608, 80, 5, 8, 2)
  DSV4_CASE2(false, 2304, 5120, 144, 0, 4, 1)
  DSV4_CASE2(false, 2304, 5120, 144, 1, 4, 2)
  DSV4_CASE2(false, 2304, 5120, 144, 2, 8, 1)
  DSV4_CASE2(false, 2304, 5120, 144, 3, 2, 1)
  DSV4_CASE2(false, 2304, 5120, 144, 4, 16, 1)
  DSV4_CASE2(false, 2304, 5120, 144, 5, 8, 2)
#undef DSV4_CASE2
  TORCH_CHECK(false, "unsupported shape/cfg/dq: w13=", w13, " K=", K, " N=", N, " cfg=", cfg,
              " dq=", dq);
}

void dsv4_det_align(at::Tensor const& ids, int64_t E, int64_t BS, at::Tensor& sorted,
                    at::Tensor& eids, at::Tensor& ntpp) {
  TORCH_CHECK(ids.is_contiguous() && (ids.scalar_type() == at::kInt ||
                                      ids.scalar_type() == at::kLong), "ids int32/int64");
  TORCH_CHECK(ids.numel() <= 512 && E > 0 && E <= 1024 && BS > 0, "align: S <= 512, E <= 1024");
  TORCH_CHECK(sorted.scalar_type() == at::kInt && eids.scalar_type() == at::kInt &&
                  ntpp.scalar_type() == at::kInt && sorted.is_contiguous() &&
                  eids.is_contiguous(), "align outputs int32 contiguous");
  const at::cuda::OptionalCUDAGuard guard(ids.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (ids.scalar_type() == at::kInt)
    det_align_kernel<int32_t><<<1, 1024, 0, st>>>(
        ids.data_ptr<int32_t>(), (int)ids.numel(), (int)E, (int)BS, sorted.data_ptr<int32_t>(),
        (int)sorted.numel(), eids.data_ptr<int32_t>(), (int)eids.numel(), ntpp.data_ptr<int32_t>());
  else
    det_align_kernel<int64_t><<<1, 1024, 0, st>>>(
        ids.data_ptr<int64_t>(), (int)ids.numel(), (int)E, (int)BS, sorted.data_ptr<int32_t>(),
        (int)sorted.numel(), eids.data_ptr<int32_t>(), (int)eids.numel(), ntpp.data_ptr<int32_t>());
}

int64_t dsv4_mxfp4_abi2() { return 1; }

TORCH_LIBRARY(_dsv4_moe2, m) {
  m.def(
      "gemm(Tensor a, Tensor w, Tensor s, Tensor sorted, Tensor eids, Tensor ntpp, "
      "Tensor topk_w, int topk, int n_slots, int K, int N, bool w13, int cfg, int dq, "
      "bool fuse, float limit, int sum_order, Tensor(c!) ctr, Tensor(d!) sync, "
      "Tensor(o!) out, Tensor(x!) aux) -> int");
  m.def("align(Tensor ids, int E, int BS, Tensor(a!) sorted, Tensor(b!) eids, "
        "Tensor(c!) ntpp) -> ()");
  m.def("abi() -> int");
}

TORCH_LIBRARY_IMPL(_dsv4_moe2, CUDA, m) {
  m.impl("gemm", &dsv4_mxfp4_gemm2);
  m.impl("align", &dsv4_det_align);
}
TORCH_LIBRARY_IMPL(_dsv4_moe2, CompositeExplicitAutograd, m) {
  m.impl("abi", &dsv4_mxfp4_abi2);
}
