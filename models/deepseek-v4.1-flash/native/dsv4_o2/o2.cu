// SPDX-License-Identifier: Apache-2.0
/*
 * O2: dense MXFP8 (e4m3 + e8m0 group-32) decode GEMM for sm_80, reading the
 * vLLM Marlin 8-bit layout unchanged (prepare_mxfp8_layer_for_marlin).
 *
 * Layout (verified in v1): per k16 row the weights are 16N bytes; 64-column
 * tile t occupies 1 KiB at t*1024; lane l (g=l/4, c=l%4) owns 32 contiguous
 * bytes = 8 words, word j = bytes (k 2c, 2c+8, 2c+1, 2c+9) of column g+8j.
 * For the 16-column subtile jj: a0=(b0,b2) of word 2jj, a2=(b1,b3) of word
 * 2jj, a1/a3 the same of word 2jj+1.  Scales: e8m0 bytes [K/32][N]; lane g
 * owns the 8 bytes at 8g of each 64-col block; word j's scale is byte
 * {0,2,1,3}[j%4] of the first (j<4) or second dword.
 *
 * Decomposition: warp-granular stream-K.  unit = (64-col tile, GPU k32
 * groups), ordered tile-major; the T = grid*W warps get equal contiguous
 * unit ranges (<= 3 tile segments each).  Each warp streams its units
 * through a private ST-stage cp.async ring (weights + e8m0 + x rows), no CTA
 * barrier in the main loop.  cp.async copies are 512 B contiguous per
 * instruction (lane-strided 16 B cp.async.cg halves the bandwidth on this
 * GPU) and swizzled so the per-lane LDS.128 are conflict free.  The main
 * loop is unrolled by ST (compile-time ring slots), all addresses are
 * pointer increments, segment ends are rare branches.
 * Epilogue (once per CTA, after the loop): per-warp partial segments are
 * summed in smem in warp order; tiles fully inside the CTA are written
 * directly; boundary tiles go through fp32 scratch + a ticket (fence.acq_rel
 * + atom.acq_rel by one thread); the last CTA sums the CTA partials in CTA
 * order (all loads issued before the adds) and re-arms the ticket.
 * => deterministic, no data atomics, graph-replay safe.  The ticket array
 * (int32 >= N/64, zero-initialised once) must not be shared by kernels
 * running concurrently on different streams.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace o2 {

__device__ __forceinline__ uint32_t lop3_or_and(uint32_t a, uint32_t m, uint32_t b) {
  uint32_t r;  // (a & m) | b
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(b));
  return r;
}
// bytes 0 and 2 of w -> bf16x2 (v * 2^-120): 2 IMAD.SHL + 2 LOP3
__device__ __forceinline__ uint32_t fp8_lo(uint32_t w) {
  return lop3_or_and(w << 8, 0x80008000u, (w << 4) & 0x07F007F0u);
}
// bytes 1 and 3 of w -> bf16x2 (v * 2^-120)
template <int HIV>
__device__ __forceinline__ uint32_t fp8_hi(uint32_t w) {
  if (HIV == 0) return lop3_or_and(w, 0x80008000u, (w >> 4) & 0x07F007F0u);  // SHF + 2 LOP3
  // (w & 0x7f007f00) >> 4 on the FMA pipe (IMAD.HI)
  return lop3_or_and(w, 0x80008000u, __umulhi(w & 0x7F007F00u, 1u << 28));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  uint32_t r;  // bf16x2 a*b (+ -0): HFMA2.BF16_V2, exact for power-of-two b
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, int n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t* d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct Params {
  const __nv_bfloat16* x;
  int64_t lda;
  int M;
  const uint4* w;
  const uint8_t* s;
  int K, N;
  __nv_bfloat16* out;
  int64_t ldo;
  float* part;
  int* cnt;
  unsigned long long* dbg;  // optional per-warp timestamps [T][4] (ns)
  int sync;  // experiment: ticket synchronisation variant
  int64_t part_n;  // scratch floats available
};

// 32-bit stream-K index math (U*T < 2^31 checked on the host): 64-bit
// division is a long software sequence on sm_80.
__host__ __device__ __forceinline__ int ustart(int i, int U, int T) {
  return (int)((unsigned)(i * U) / (unsigned)T);
}
__device__ __forceinline__ int owner_of(int u, int U, int T) {
  // largest i with ustart(i) <= u
  int i = (int)(((unsigned)u * (unsigned)T + (unsigned)T - 1u) / (unsigned)U);
  if (i >= T) i = T - 1;
  while (i > 0 && ustart(i, U, T) > u) i--;
  while (i + 1 < T && ustart(i + 1, U, T) <= u) i++;
  return i;
}

// MODE 0: full GEMM.  MODE 1: cp.async only (roofline).  MODE 2: + smem
// reads + dequant (xor checksum, no mma).  MODE 3: full compute, no output.
template <int TB, int W, int ST, int MODE, int GPU, int HIV>
__global__ void __launch_bounds__(W * 32) o2_gemm(const Params p) {
  constexpr int TOK = TB * 8;
  constexpr int SW = GPU * (132 + 32 * TB);  // uint4 per warp per stage
  constexpr int OFF_S = 128 * GPU;           // scales (uint4 index within stage)
  constexpr int OFF_X = 132 * GPU;           // x rows
  extern __shared__ __align__(128) uint4 smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c4 = lane & 3;
  const int N = p.N, M = p.M;
  const int GU = (p.K >> 5) / GPU;  // units per tile
  const int U = (N >> 6) * GU;
  const int T = gridDim.x * W;
  const int gw = blockIdx.x * W + warp;
  const int ua = ustart(gw, U, T), ub = ustart(gw + 1, U, T);
  if (p.dbg && lane == 0) { unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); p.dbg[gw * 16 + 0] = gtime(); p.dbg[gw * 16 + 7] = smid; }
  if (MODE != 0 && ua >= ub) return;

  uint4* const wring = smem + warp * ST * SW;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(wring);

  // ---- per-lane constant smem offsets (bytes)
  int q0 = lane, q1 = 32 + lane;
  const uint32_t d_w0 = (q0 ^ ((q0 >> 3) & 1)) * 16, d_w1 = (q1 ^ ((q1 >> 3) & 1)) * 16;
  const uint32_t d_s = (OFF_S + lane) * 16;  // + gi*64 (lanes < 4)
  const uint32_t d_x = (OFF_X + g * 4 + (c4 ^ ((g >> 1) & 3))) * 16;  // + gi*TB*512 + bb*512
  const int sw = (lane >> 2) & 1;
  const int r_w0 = (2 * lane) ^ sw, r_w1 = (2 * lane + 1) ^ sw;  // uint4 index within a row
  const uint32_t r_x = (OFF_X + (lane & 7) * 4 + ((lane >> 3) ^ (((lane & 7) >> 1) & 3))) * 16;

  // ---- load iterator (pointer increments)
  const int64_t rowN = N;  // uint4 per k16 row
  int l_t = ua / GU, l_ug = ua - l_t * GU;
  int l_left = ub - ua;
  const uint4* l_pw = p.w + (int64_t)l_ug * (2 * GPU) * rowN + l_t * 64 + lane;
  const uint8_t* l_ps = p.s + (int64_t)l_ug * GPU * N + l_t * 64 + lane * 16;
  int xb_n[TB];
  const __nv_bfloat16* l_px[TB];
#pragma unroll
  for (int b = 0; b < TB; b++) {
    const int row = b * 8 + g;
    xb_n[b] = row < M ? 16 : 0;
    l_px[b] = p.x + (int64_t)(row < M ? row : 0) * p.lda + l_ug * GPU * 32 + c4 * 8;
  }
  auto issue = [&](const int slot) {
    if (l_left > 0) {
      const uint32_t dst = ring + slot * SW * 16;
#pragma unroll
      for (int r = 0; r < 2 * GPU; r++) {
        cp16(dst + r * 1024 + d_w0, l_pw + r * rowN);
        cp16(dst + r * 1024 + d_w1, l_pw + r * rowN + 32);
      }
      if (lane < 4) {
#pragma unroll
        for (int gi = 0; gi < GPU; gi++) cp16(dst + d_s + gi * 64, l_ps + gi * N);
      }
#pragma unroll
      for (int gi = 0; gi < GPU; gi++)
#pragma unroll
        for (int b = 0; b < TB; b++) cp16z(dst + d_x + (gi * TB + b) * 512, l_px[b] + gi * 32, xb_n[b]);
      l_left--;
      if (++l_ug == GU) {  // next tile (rare)
        l_ug = 0;
        l_t++;
        l_pw = p.w + l_t * 64 + lane;
        l_ps = p.s + l_t * 64 + lane * 16;
#pragma unroll
        for (int b = 0; b < TB; b++) l_px[b] -= (GU - 1) * GPU * 32;
      } else {
        l_pw += 2 * GPU * rowN;
        l_ps += GPU * N;
#pragma unroll
        for (int b = 0; b < TB; b++) l_px[b] += GPU * 32;
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) issue(s);

  // ---- compute state
  float acc[TB][4][4];
#pragma unroll
  for (int i = 0; i < TB; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;
  bool has_l = false;
  int c_t = ua / GU;
  const int c_ug0 = ua - c_t * GU;
  bool seg_from0 = c_ug0 == 0;
  int n = ub - ua;
  int seg_left = min(GU - c_ug0, n);
  uint32_t chk = 0;

  auto compute = [&](const int slot) {
    const uint4* stp = wring + slot * SW;
    const uint32_t st = ring + slot * SW * 16;
    if (MODE == 1) return;
#pragma unroll
    for (int gi = 0; gi < GPU; gi++) {
      const uint2 sv = reinterpret_cast<const uint2*>(stp + OFF_S + gi * 4)[g];
      uint32_t sd[8];  // duplicated bf16x2 folded scales 2^(S-7) per word j
      {
        const uint32_t e[2] = {sv.x, sv.y};
        constexpr uint32_t sel[4] = {0x4040u, 0x4242u, 0x4141u, 0x4343u};
#pragma unroll
        for (int j = 0; j < 8; j++) sd[j] = __byte_perm(e[j >> 2], 0, sel[j & 3]) * 128u + 0x3c003c00u;
      }
      uint32_t xb[TB][4];
      if (MODE == 0 || MODE == 3) {
#pragma unroll
        for (int bb = 0; bb < TB; bb++) ldsm4(st + r_x + (gi * TB + bb) * 512, xb[bb]);
      }
      uint4 wv[2][2];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wv[i][0] = stp[(gi * 2 + i) * 64 + r_w0];
        wv[i][1] = stp[(gi * 2 + i) * 64 + r_w1];
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint32_t wd[8] = {wv[i][0].x, wv[i][0].y, wv[i][0].z, wv[i][0].w,
                                wv[i][1].x, wv[i][1].y, wv[i][1].z, wv[i][1].w};
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const uint32_t qa = wd[2 * jj], qb = wd[2 * jj + 1];
          const uint32_t a0 = hmul2u(fp8_lo(qa), sd[2 * jj]);
          const uint32_t a2 = hmul2u(fp8_hi<HIV>(qa), sd[2 * jj]);
          const uint32_t a1 = hmul2u(fp8_lo(qb), sd[2 * jj + 1]);
          const uint32_t a3 = hmul2u(fp8_hi<HIV>(qb), sd[2 * jj + 1]);
          if (MODE == 2) {
            chk ^= a0 ^ (a1 + a2) ^ a3;
          } else {
#pragma unroll
            for (int bb = 0; bb < TB; bb++)
              mma_bf16(acc[bb][jj], a0, a1, a2, a3, xb[bb][2 * i], xb[bb][2 * i + 1]);
          }
        }
      }
    }
  };

  while (n > 0) {
#pragma unroll
    for (int s = 0; s < ST; s++) {
      if (n > 0) {
        cp_wait<ST - 2>();
        __syncwarp();
        issue((s + ST - 1) % ST);
        compute(s);
        n--;
        if (--seg_left == 0) {  // segment end (rare branch)
          if (MODE == 0 || MODE >= 4) {
            // whole tile in one segment? (started at unit 0 and reached the tile end)
            const bool tile_end = (n == 0) ? ((ub - c_t * GU) == GU) : true;
            if (seg_from0 && tile_end) {
#pragma unroll
              for (int bb = 0; bb < TB; bb++)
#pragma unroll
                for (int jj = 0; jj < 4; jj++)
#pragma unroll
                  for (int v = 0; v < 4; v++) {
                    const int col = 16 * jj + 8 * (v >> 1) + g, tok = bb * 8 + 2 * c4 + (v & 1);
                    if (tok < M)
                      p.out[(int64_t)tok * p.ldo + c_t * 64 + col] = __float2bfloat16(acc[bb][jj][v]);
                  }
            } else if (!seg_from0) {
              // head segment of a tile owned by an earlier warp: publish the raw
              // fragment (lane-native layout) with plain stores; the slot was
              // pre-filled with the sentinel NaN 0xFFFFFFFF, which FADD/HMMA never
              // produce, so every float4 is self-validating (no fence/atomic).
              float4* dst = reinterpret_cast<float4*>(p.part) + ((int64_t)gw * 32 + lane) * (4 * TB);
#pragma unroll
              for (int bb = 0; bb < TB; bb++)
#pragma unroll
                for (int jj = 0; jj < 4; jj++)
                  __stcg(dst + bb * 4 + jj, make_float4(acc[bb][jj][0], acc[bb][jj][1], acc[bb][jj][2], acc[bb][jj][3]));
              if (p.dbg && lane == 0) { p.dbg[gw * 16 + 6] = 1000 + c_t; p.dbg[gw * 16 + 8] = gtime(); p.dbg[gw * 16 + 9] = __float_as_uint(acc[0][0][0]); }
            } else {
              has_l = true;  // owner of a partial tail segment: collect after the loop
            }
          } else if (MODE == 3) {
#pragma unroll
            for (int i = 0; i < TB; i++)
#pragma unroll
              for (int j = 0; j < 4; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) chk ^= __float_as_uint(acc[i][j][v]);
          }
          if (n > 0) {
#pragma unroll
            for (int i = 0; i < TB; i++)
#pragma unroll
              for (int j = 0; j < 4; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;
            c_t++;
            seg_from0 = true;
            seg_left = min(GU, n);
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (p.dbg && lane == 0) p.dbg[gw * 16 + 1] = gtime();
  if (MODE == 0 && p.sync == 6) return;
  if (MODE == 4) {
#pragma unroll
    for (int i = 0; i < TB; i++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int v = 0; v < 4; v++) chk ^= __float_as_uint(acc[i][j][v]);
    chk ^= has_l;
  }
  if (MODE != 0 && MODE != 5) {
    if (chk == 0x12345678u) p.out[0] = __float2bfloat16(1.f);
    return;
  }
  if (!has_l) {
    if (p.dbg && lane == 0) p.dbg[gw * 16 + 2] = gtime();
    return;
  }
  // ---- owner of the tail tile: add the contributors' fragments in warp order
  {
    const int t = c_t;
    const int w1 = owner_of(t * GU + GU - 1, U, T);
    if (p.dbg && lane == 0) { p.dbg[gw * 16 + 3] = t; p.dbg[gw * 16 + 4] = w1; p.dbg[gw * 16 + 5] = U * 100000ll + T; }
    // Contributors are consumed in batches of NB: all their fragment loads are
    // issued in one round (one L2 round trip per batch instead of per
    // contributor), re-polled until no component is the sentinel, then summed
    // in warp order (deterministic) and the slots re-armed.
    constexpr int NB = TB == 1 ? 8 : 1;
    constexpr int F4 = TB * 4;
    for (int cb = gw + 1; cb <= w1; cb += NB) {
      const int nb = min(NB, w1 - cb + 1);
      uint4 u[NB][F4];
      bool pending;
      int ns = 32;
      do {
#pragma unroll
        for (int k = 0; k < NB; k++)
          if (k < nb) {
            const uint4* src = reinterpret_cast<const uint4*>(p.part) + ((int64_t)(cb + k) * 32 + lane) * F4;
#pragma unroll
            for (int i = 0; i < F4; i++)
              asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                           : "=r"(u[k][i].x), "=r"(u[k][i].y), "=r"(u[k][i].z), "=r"(u[k][i].w)
                           : "l"(src + i) : "memory");
          }
        pending = false;
#pragma unroll
        for (int k = 0; k < NB; k++)
          if (k < nb)
#pragma unroll
            for (int i = 0; i < F4; i++)
              pending |= (u[k][i].x == 0xFFFFFFFFu) | (u[k][i].y == 0xFFFFFFFFu) |
                         (u[k][i].z == 0xFFFFFFFFu) | (u[k][i].w == 0xFFFFFFFFu);
        if (pending && TB > 1) {
          __nanosleep(ns);
          if (ns < 256) ns *= 2;
        }
      } while (pending);
#pragma unroll
      for (int k = 0; k < NB; k++)
        if (k < nb) {
#pragma unroll
          for (int bb = 0; bb < TB; bb++)
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
              const uint4 q = u[k][bb * 4 + jj];
              acc[bb][jj][0] += __uint_as_float(q.x); acc[bb][jj][1] += __uint_as_float(q.y);
              acc[bb][jj][2] += __uint_as_float(q.z); acc[bb][jj][3] += __uint_as_float(q.w);
            }
          uint4* dst = reinterpret_cast<uint4*>(p.part) + ((int64_t)(cb + k) * 32 + lane) * F4;
#pragma unroll
          for (int i = 0; i < F4; i++) __stcg(dst + i, make_uint4(~0u, ~0u, ~0u, ~0u));
        }
    }
#pragma unroll
    for (int bb = 0; bb < TB; bb++)
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const int col = 16 * jj + 8 * (v >> 1) + g, tok = bb * 8 + 2 * c4 + (v & 1);
          if (tok < M) p.out[(int64_t)tok * p.ldo + t * 64 + col] = __float2bfloat16(acc[bb][jj][v]);
        }
  }
  if (p.dbg && lane == 0) p.dbg[gw * 16 + 2] = gtime();
}

template <int TB, int W, int ST, int MODE, int GPU, int HIV>
static void launch(const Params& p, int ctas, cudaStream_t stream) {
  constexpr int SW = GPU * (132 + 32 * TB);
  constexpr int SMEM0 = W * ST * SW * 16;
  constexpr int SMEMR = W * 2 * 64 * TB * 8 * 4 + W * 2 * 4 + 16;
  constexpr int SMEM = SMEM0;
  auto kern = o2_gemm<TB, W, ST, MODE, GPU, HIV>;
  static int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, W * 32, SMEM);
    return n;
  }();
  TORCH_CHECK(occ > 0, "config does not fit");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (ctas == 0) {
    ctas = occ * sms;  // one full wave: identical warps (and work) per SM
  } else if (ctas < 0) {  // experiments: -ctas = target k32 groups per warp
    const int64_t U = (int64_t)(p.N / 64) * (p.K / 32 / GPU);
    int64_t t = (U * GPU + (-ctas) - 1) / (-ctas);
    int64_t c = (t + W - 1) / W;
    ctas = (int)std::max<int64_t>(1, std::min<int64_t>(c, (int64_t)occ * sms));
  }
  const int64_t Uu = (int64_t)(p.N / 64) * (p.K / 32 / GPU);
  if ((int64_t)ctas * W > Uu) ctas = (int)(Uu / W);  // >= 1 unit per warp (strictly increasing ranges)
  TORCH_CHECK(ctas >= 1 && Uu * ctas * W < (1ll << 31), "U*T overflow / too small");
  TORCH_CHECK(ctas <= occ * sms, "grid must be co-resident (owners spin on contributors)");
  TORCH_CHECK(p.part_n >= (int64_t)ctas * W * 32 * 16 * TB, "scratch too small");
  kern<<<ctas, W * 32, SMEM, stream>>>(p);
}

// ============================================================================
// O3: M = 9..48 variant.  CTA = W warps on W ADJACENT 64-col tiles at the same
// k-range; the activation chunk of every stage is cp.async'ed ONCE into smem by
// the whole CTA and read by all W warps with ldmatrix (x L2 traffic / W).
// One __syncthreads per stage (x is loaded cooperatively).  Stream-K over
// (tile-group, k-unit) at CTA granularity; per-warp fragments are reduced with
// the same fence-free sentinel owner/contributor scheme as O2 (slot = cta*W+warp).
// ============================================================================
template <int TB, int W, int ST, int MODE, int GPU>
__global__ void __launch_bounds__(W * 32) o3_gemm(const Params p) {
  constexpr int TOK = TB * 8;
  constexpr int WS = 128 * GPU;          // uint4 weights per warp per stage
  constexpr int SS = 4 * GPU;            // uint4 scales per warp per stage
  constexpr int XS = GPU * TB * 32;      // uint4 x per stage (shared by the CTA)
  constexpr int STG = W * (WS + SS) + XS;
  constexpr int NT = W * 32;
  constexpr int QN = (XS + NT - 1) / NT;  // x chunks per thread per stage
  extern __shared__ __align__(128) uint4 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, c4 = lane & 3;
  const int N = p.N, M = p.M;
  const int GU = (p.K >> 5) / GPU;
  const int TG = (N >> 6) / W;
  const int U = TG * GU;
  const int T = gridDim.x;
  const int cta = blockIdx.x;
  const int ua = ustart(cta, U, T), ub = ustart(cta + 1, U, T);
  const int sid = cta * W + warp;  // partial slot
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  if (p.dbg && lane == 0) p.dbg[sid * 16 + 0] = gtime();

  // ---- per-thread constant offsets
  const int q0 = lane, q1 = 32 + lane;
  const uint32_t d_w0 = (warp * WS + (q0 ^ ((q0 >> 3) & 1))) * 16;
  const uint32_t d_w1 = (warp * WS + (q1 ^ ((q1 >> 3) & 1))) * 16;
  const uint32_t d_s = (W * WS + warp * SS + lane) * 16;  // lanes < 4, + gi*64
  const int sw = (lane >> 2) & 1;
  const int r_w0 = warp * WS + ((2 * lane) ^ sw), r_w1 = warp * WS + ((2 * lane + 1) ^ sw);
  const uint32_t r_x = (W * (WS + SS) + (lane & 7) * 4 + ((lane >> 3) ^ (((lane & 7) >> 1) & 3))) * 16;
  // x chunk assignment (fixed per thread)
  uint32_t xdst[QN];
  const __nv_bfloat16* xsrc[QN];
  int xn[QN];
#pragma unroll
  for (int k = 0; k < QN; k++) {
    const int q = tid + k * NT;
    const int qq = q < XS ? q : 0;
    const int gi = qq / (TB * 32), rem = qq % (TB * 32);
    const int row = rem >> 2, c = rem & 3;
    xdst[k] = (W * (WS + SS) + gi * TB * 32 + row * 4 + (c ^ ((row >> 1) & 3))) * 16;
    xn[k] = (q < XS && row < M) ? 16 : 0;
    xsrc[k] = p.x + (int64_t)(row < M ? row : 0) * p.lda + gi * 32 + c * 8;
  }

  // ---- load iterator
  const int64_t rowN = N;
  int l_tg = ua / GU, l_ug = ua - l_tg * GU;
  int l_left = ub - ua;
  const uint4* l_pw = p.w + (int64_t)l_ug * (2 * GPU) * rowN + (l_tg * W + warp) * 64 + lane;
  const uint8_t* l_ps = p.s + (int64_t)l_ug * GPU * N + (l_tg * W + warp) * 64 + lane * 16;
  int l_xo = l_ug * GPU * 32;  // element offset along K
  auto issue = [&](const int slot) {
    if (l_left > 0) {
      const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
      for (int r = 0; r < 2 * GPU; r++) {
        cp16(st + r * 1024 + d_w0 - 0 * 16, l_pw + r * rowN);
        cp16(st + r * 1024 + d_w1, l_pw + r * rowN + 32);
      }
      if (lane < 4) {
#pragma unroll
        for (int gi = 0; gi < GPU; gi++) cp16(st + d_s + gi * 64, l_ps + gi * N);
      }
#pragma unroll
      for (int k = 0; k < QN; k++)
        if (k * NT + tid < XS) cp16z(st + xdst[k], xsrc[k] + l_xo, xn[k]);
      l_left--;
      if (++l_ug == GU) {
        l_ug = 0;
        l_tg++;
        l_pw = p.w + (l_tg * W + warp) * 64 + lane;
        l_ps = p.s + (l_tg * W + warp) * 64 + lane * 16;
        l_xo = 0;
      } else {
        l_pw += 2 * GPU * rowN;
        l_ps += GPU * N;
        l_xo += GPU * 32;
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; s++) issue(s);

  float acc[TB][4][4];
#pragma unroll
  for (int i = 0; i < TB; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;
  bool has_l = false;
  int c_tg = ua / GU;
  const int c_ug0 = ua - c_tg * GU;
  bool seg_from0 = c_ug0 == 0;
  int n = ub - ua;
  int seg_left = min(GU - c_ug0, n);
  uint32_t chk = 0;

  auto compute = [&](const int slot) {
    const uint4* stp = smem + slot * STG;
    const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
    for (int gi = 0; gi < GPU; gi++) {
      const uint2 sv = reinterpret_cast<const uint2*>(stp + W * WS + warp * SS + gi * 4)[g];
      uint32_t sd[8];
      {
        const uint32_t e[2] = {sv.x, sv.y};
        constexpr uint32_t sel[4] = {0x4040u, 0x4242u, 0x4141u, 0x4343u};
#pragma unroll
        for (int j = 0; j < 8; j++) sd[j] = __byte_perm(e[j >> 2], 0, sel[j & 3]) * 128u + 0x3c003c00u;
      }
      uint32_t xb[TB][4];
#pragma unroll
      for (int bb = 0; bb < TB; bb++) ldsm4(st + r_x + (gi * TB + bb) * 512, xb[bb]);
      uint4 wv[2][2];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wv[i][0] = stp[(gi * 2 + i) * 64 + r_w0];
        wv[i][1] = stp[(gi * 2 + i) * 64 + r_w1];
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint32_t wd[8] = {wv[i][0].x, wv[i][0].y, wv[i][0].z, wv[i][0].w,
                                wv[i][1].x, wv[i][1].y, wv[i][1].z, wv[i][1].w};
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const uint32_t qa = wd[2 * jj], qb = wd[2 * jj + 1];
          const uint32_t a0 = hmul2u(fp8_lo(qa), sd[2 * jj]);
          const uint32_t a2 = hmul2u(fp8_hi<0>(qa), sd[2 * jj]);
          const uint32_t a1 = hmul2u(fp8_lo(qb), sd[2 * jj + 1]);
          const uint32_t a3 = hmul2u(fp8_hi<0>(qb), sd[2 * jj + 1]);
#pragma unroll
          for (int bb = 0; bb < TB; bb++)
            mma_bf16(acc[bb][jj], a0, a1, a2, a3, xb[bb][2 * i], xb[bb][2 * i + 1]);
        }
      }
    }
  };

  while (n > 0) {
#pragma unroll
    for (int s = 0; s < ST; s++) {
      if (n > 0) {
        cp_wait<ST - 2>();
        __syncthreads();  // stage s visible to all warps; slot (s-1) free for reuse
        issue((s + ST - 1) % ST);
        compute(s);
        n--;
        if (--seg_left == 0) {
          const int tile = c_tg * W + warp;
          if (MODE == 0) {
            const bool tile_end = (n == 0) ? ((ub - c_tg * GU) == GU) : true;
            if (seg_from0 && tile_end) {
              if (p.sync != 9)
#pragma unroll
              for (int bb = 0; bb < TB; bb++)
#pragma unroll
                for (int jj = 0; jj < 4; jj++)
#pragma unroll
                  for (int v = 0; v < 4; v++) {
                    const int col = 16 * jj + 8 * (v >> 1) + g, tok = bb * 8 + 2 * c4 + (v & 1);
                    if (tok < M) p.out[(int64_t)tok * p.ldo + tile * 64 + col] = __float2bfloat16(acc[bb][jj][v]);
                  }
            } else if (!seg_from0) {
              if (p.dbg && lane == 0) p.dbg[sid * 16 + 3] = gtime();
              float4* dst = reinterpret_cast<float4*>(p.part) + ((int64_t)sid * 32 + lane) * (4 * TB);
#pragma unroll
              for (int bb = 0; bb < TB; bb++)
#pragma unroll
                for (int jj = 0; jj < 4; jj++)
                  __stcg(dst + bb * 4 + jj, make_float4(acc[bb][jj][0], acc[bb][jj][1], acc[bb][jj][2], acc[bb][jj][3]));
            } else {
              has_l = true;
            }
          } else {
#pragma unroll
            for (int i = 0; i < TB; i++)
#pragma unroll
              for (int j = 0; j < 4; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) chk ^= __float_as_uint(acc[i][j][v]);
          }
          if (n > 0) {
#pragma unroll
            for (int i = 0; i < TB; i++)
#pragma unroll
              for (int j = 0; j < 4; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;
            c_tg++;
            seg_from0 = true;
            seg_left = min(GU, n);
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (p.dbg && lane == 0) p.dbg[sid * 16 + 1] = gtime();
  if (MODE != 0) {
    if (chk == 0x12345678u) p.out[0] = __float2bfloat16(1.f);
    return;
  }
  if (p.sync == 9) return;
  if (!has_l) { if (p.dbg && lane == 0) p.dbg[sid * 16 + 2] = gtime(); return; }
  // ---- owner: add the contributing CTAs' fragments for this warp's tile
  const int tile = c_tg * W + warp;
  const int c1 = owner_of(c_tg * GU + GU - 1, U, T);
  constexpr int F4 = TB * 4;
  for (int cc = cta + 1; cc <= c1; cc++) {
    const uint4* src = reinterpret_cast<const uint4*>(p.part) + ((int64_t)(cc * W + warp) * 32 + lane) * F4;
    // cheap wait: lane 0 polls ONE word (lane 31's last float4) with back-off,
    // so waiting owners do not flood L2; then load + verify the whole fragment
    // (every component; re-polled in the rare case other lanes' stores lag).
    if (lane == 0) {
      const uint32_t* flag = reinterpret_cast<const uint32_t*>(
          reinterpret_cast<const uint4*>(p.part) + ((int64_t)(cc * W + warp) * 32 + 31) * F4 + F4 - 1) + 3;
      int ns = 64;
      for (;;) {
        uint32_t fv;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(fv) : "l"(flag) : "memory");
        if (fv != 0xFFFFFFFFu) break;
        __nanosleep(ns);
        if (ns < 1024) ns *= 2;
      }
    }
    __syncwarp();
    uint4 u[F4];
    bool pending;
    int ns = 32;
    do {
#pragma unroll
      for (int i = 0; i < F4; i++)
        asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(u[i].x), "=r"(u[i].y), "=r"(u[i].z), "=r"(u[i].w) : "l"(src + i) : "memory");
      pending = false;
#pragma unroll
      for (int i = 0; i < F4; i++)
        pending |= (u[i].x == 0xFFFFFFFFu) | (u[i].y == 0xFFFFFFFFu) | (u[i].z == 0xFFFFFFFFu) |
                   (u[i].w == 0xFFFFFFFFu);
      if (pending) {
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
      }
    } while (pending);
#pragma unroll
    for (int bb = 0; bb < TB; bb++)
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const uint4 q = u[bb * 4 + jj];
        acc[bb][jj][0] += __uint_as_float(q.x); acc[bb][jj][1] += __uint_as_float(q.y);
        acc[bb][jj][2] += __uint_as_float(q.z); acc[bb][jj][3] += __uint_as_float(q.w);
      }
    uint4* dst = reinterpret_cast<uint4*>(p.part) + ((int64_t)(cc * W + warp) * 32 + lane) * F4;
#pragma unroll
    for (int i = 0; i < F4; i++) __stcg(dst + i, make_uint4(~0u, ~0u, ~0u, ~0u));
  }
#pragma unroll
  for (int bb = 0; bb < TB; bb++)
#pragma unroll
    for (int jj = 0; jj < 4; jj++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int col = 16 * jj + 8 * (v >> 1) + g, tok = bb * 8 + 2 * c4 + (v & 1);
        if (tok < M) p.out[(int64_t)tok * p.ldo + tile * 64 + col] = __float2bfloat16(acc[bb][jj][v]);
      }
  if (p.dbg && lane == 0) { p.dbg[sid * 16 + 2] = gtime(); p.dbg[sid * 16 + 4] = c1 - cta; }
}

template <int TB, int W, int ST, int MODE, int GPU>
static void launch3(const Params& p, int ctas, cudaStream_t stream) {
  constexpr int STG = W * (128 * GPU + 4 * GPU) + GPU * TB * 32;
  constexpr int SMEM = ST * STG * 16;
  auto kern = o3_gemm<TB, W, ST, MODE, GPU>;
  static int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, W * 32, SMEM);
    return n;
  }();
  TORCH_CHECK(occ > 0, "o3 config does not fit");
  TORCH_CHECK((p.N / 64) % W == 0, "o3: N/64 must be a multiple of W");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t U = (int64_t)(p.N / 64 / W) * (p.K / 32 / GPU);
  if (ctas == 0) ctas = occ * sms;
  else if (ctas < 0) ctas = (int)std::max<int64_t>(1, std::min<int64_t>((U + (-ctas) - 1) / (-ctas), (int64_t)occ * sms));
  if (ctas > U) ctas = (int)U;
  TORCH_CHECK(ctas >= 1 && U * ctas < (1ll << 31), "o3: U*T overflow");
  TORCH_CHECK(ctas <= occ * sms, "o3: grid must be co-resident (owners spin on contributors)");
  TORCH_CHECK(p.part_n >= (int64_t)ctas * W * 32 * 16 * TB, "o3: scratch too small");
  kern<<<ctas, W * 32, SMEM, stream>>>(p);
}

// ============================================================================
// O4: M = 9..64, static split-K with CTA-shared activations.
// grid = (TG tile-groups) x S K-slices.  CTA (tg, s) = W warps on the W adjacent
// 64-col tiles of tile-group tg over k-units [s*GU/S, (s+1)*GU/S).  The x chunk
// of every stage is cp.async'ed once into smem and read by all warps with
// ldmatrix.  S == 1: bf16 written directly.  S > 1: each warp stores its fp32
// fragment (lane-native, plain stores), and o4_reduce sums the S slices in
// slice order (deterministic, no atomics, no spinning, graph safe).
// ============================================================================
template <int TB, int W, int ST, int GPU, bool PART>
__global__ void __launch_bounds__(W * 32) o4_gemm(const Params p, const int S, const int NCH) {
  constexpr int TOK = TB * 8;
  constexpr int WS = 128 * GPU, SS = 4 * GPU, XS = GPU * TB * 32;
  constexpr int STG = W * (WS + SS) + XS;
  constexpr int NT = W * 32;
  constexpr int QN = (XS + NT - 1) / NT;
  extern __shared__ __align__(128) uint4 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, c4 = lane & 3;
  const int N = p.N;
  // 64-row chunks for M > TOK: chunk index fastest so CTAs sharing a weight
  // slice run together (weights re-read hit L2)
  const int ch = blockIdx.x % NCH;
  const int M = min(TOK, p.M - ch * TOK);  // rows of this chunk
  const __nv_bfloat16* const X = p.x + (int64_t)ch * TOK * p.lda;
  const int GU = (p.K >> 5) / GPU;
  // slice index fastest: concurrently running CTAs sit at different K offsets
  // (same-slice CTAs in lockstep over the same rows were ~50% slower: DRAM
  // channel/row concentration)
  const int s = blockIdx.x / NCH, tg = blockIdx.y;
  const int u0 = s * GU / S, u1 = (s + 1) * GU / S;
  const int tile = tg * W + warp;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);

  const int q0 = lane, q1 = 32 + lane;
  const uint32_t d_w0 = (warp * WS + (q0 ^ ((q0 >> 3) & 1))) * 16;
  const uint32_t d_w1 = (warp * WS + (q1 ^ ((q1 >> 3) & 1))) * 16;
  const uint32_t d_s = (W * WS + warp * SS + lane) * 16;
  const int sw = (lane >> 2) & 1;
  const int r_w0 = warp * WS + ((2 * lane) ^ sw), r_w1 = warp * WS + ((2 * lane + 1) ^ sw);
  const uint32_t r_x = (W * (WS + SS) + (lane & 7) * 4 + ((lane >> 3) ^ (((lane & 7) >> 1) & 3))) * 16;
  uint32_t xdst[QN];
  const __nv_bfloat16* xsrc[QN];
  int xn[QN];
#pragma unroll
  for (int k = 0; k < QN; k++) {
    const int q = tid + k * NT;
    const int qq = q < XS ? q : 0;
    const int gi = qq / (TB * 32), rem = qq % (TB * 32);
    const int row = rem >> 2, c = rem & 3;
    xdst[k] = (W * (WS + SS) + gi * TB * 32 + row * 4 + (c ^ ((row >> 1) & 3))) * 16;
    xn[k] = (q < XS && row < M) ? 16 : 0;
    xsrc[k] = X + (int64_t)(row < M ? row : 0) * p.lda + (u0 * GPU + gi) * 32 + c * 8;
  }
  const int64_t rowN = N;
  int l_left = u1 - u0;
  const uint4* l_pw = p.w + (int64_t)u0 * (2 * GPU) * rowN + tile * 64 + lane;
  const uint8_t* l_ps = p.s + (int64_t)u0 * GPU * N + tile * 64 + lane * 16;
  int l_xo = 0;
  auto issue = [&](const int slot) {
    if (l_left > 0) {
      const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
      for (int r = 0; r < 2 * GPU; r++) {
        cp16(st + r * 1024 + d_w0, l_pw + r * rowN);
        cp16(st + r * 1024 + d_w1, l_pw + r * rowN + 32);
      }
      if (lane < 4) {
#pragma unroll
        for (int gi = 0; gi < GPU; gi++) cp16(st + d_s + gi * 64, l_ps + gi * N);
      }
#pragma unroll
      for (int k = 0; k < QN; k++)
        if (k * NT + tid < XS) cp16z(st + xdst[k], xsrc[k] + l_xo, xn[k]);
      l_left--;
      l_pw += 2 * GPU * rowN;
      l_ps += GPU * N;
      l_xo += GPU * 32;
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < ST - 1; i++) issue(i);

  float acc[TB][4][4];
#pragma unroll
  for (int i = 0; i < TB; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;

  auto compute = [&](const int slot) {
    const uint4* stp = smem + slot * STG;
    const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
    for (int gi = 0; gi < GPU; gi++) {
      const uint2 sv = reinterpret_cast<const uint2*>(stp + W * WS + warp * SS + gi * 4)[g];
      uint32_t sd[8];
      {
        const uint32_t e[2] = {sv.x, sv.y};
        constexpr uint32_t sel[4] = {0x4040u, 0x4242u, 0x4141u, 0x4343u};
#pragma unroll
        for (int j = 0; j < 8; j++) sd[j] = __byte_perm(e[j >> 2], 0, sel[j & 3]) * 128u + 0x3c003c00u;
      }
      uint32_t xb[TB][4];
#pragma unroll
      for (int bb = 0; bb < TB; bb++) ldsm4(st + r_x + (gi * TB + bb) * 512, xb[bb]);
      uint4 wv[2][2];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wv[i][0] = stp[(gi * 2 + i) * 64 + r_w0];
        wv[i][1] = stp[(gi * 2 + i) * 64 + r_w1];
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint32_t wd[8] = {wv[i][0].x, wv[i][0].y, wv[i][0].z, wv[i][0].w,
                                wv[i][1].x, wv[i][1].y, wv[i][1].z, wv[i][1].w};
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const uint32_t qa = wd[2 * jj], qb = wd[2 * jj + 1];
          const uint32_t a0 = hmul2u(fp8_lo(qa), sd[2 * jj]);
          const uint32_t a2 = hmul2u(fp8_hi<0>(qa), sd[2 * jj]);
          const uint32_t a1 = hmul2u(fp8_lo(qb), sd[2 * jj + 1]);
          const uint32_t a3 = hmul2u(fp8_hi<0>(qb), sd[2 * jj + 1]);
#pragma unroll
          for (int bb = 0; bb < TB; bb++)
            mma_bf16(acc[bb][jj], a0, a1, a2, a3, xb[bb][2 * i], xb[bb][2 * i + 1]);
        }
      }
    }
  };

  int n = u1 - u0;
  while (n > 0) {
#pragma unroll
    for (int i = 0; i < ST; i++) {
      if (n > 0) {
        cp_wait<ST - 2>();
        __syncthreads();
        issue((i + ST - 1) % ST);
        compute(i);
        n--;
      }
    }
  }
  cp_wait<0>();
  if (p.sync == 9) {  // debug: no output (checksum only)
    uint32_t c = 0;
#pragma unroll
    for (int bb = 0; bb < TB; bb++)
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int v = 0; v < 4; v++) c ^= __float_as_uint(acc[bb][jj][v]);
    if (c == 0x12345678u) p.out[0] = __float2bfloat16(1.f);
    return;
  }
  // ---- epilogue: transpose the warp's [TOK x 64] tile through smem, then
  // write whole rows (256 B fp32 / 128 B bf16 per warp store) -- the lane-native
  // fragment stores touched 32 lines with 16 B each and cost ~15-20 us.
  __syncthreads();  // all warps done with the pipeline smem
  float* tw = reinterpret_cast<float*>(smem) + warp * TOK * 68;
#pragma unroll
  for (int bb = 0; bb < TB; bb++)
#pragma unroll
    for (int jj = 0; jj < 4; jj++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int col = 16 * jj + 8 * (v >> 1) + g, tok = bb * 8 + 2 * c4 + (v & 1);
        tw[tok * 68 + col] = acc[bb][jj][v];
      }
  __syncwarp();
  for (int row = 0; row < M; row++) {
    const float2 v = *reinterpret_cast<const float2*>(tw + row * 68 + 2 * lane);
    if (PART) {
      __stcg(reinterpret_cast<float2*>(p.part + ((int64_t)s * p.M + ch * TOK + row) * N + tile * 64) + lane, v);
    } else {
      *reinterpret_cast<__nv_bfloat162*>(p.out + (int64_t)(ch * TOK + row) * p.ldo + tile * 64 + 2 * lane) =
          __floats2bfloat162_rn(v.x, v.y);
    }
  }
}

// out[m][n] = bf16( sum_{s=0..S-1} part[s][m][n] ), fixed slice order (deterministic).
__global__ void __launch_bounds__(256) o4_reduce(const float4* __restrict__ part, __nv_bfloat16* __restrict__ out,
                                                 int64_t ldo, int M, int N, int S) {
  const int n4 = N >> 2;
  const int64_t per = (int64_t)M * n4;
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= per) return;
  float4 a = __ldcg(part + i);
  for (int s = 1; s < S; s++) {
    const float4 b = __ldcg(part + s * per + i);
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  }
  const int m = (int)(i / n4), c = (int)(i % n4) * 4;
  __nv_bfloat162 lo = __floats2bfloat162_rn(a.x, a.y), hi = __floats2bfloat162_rn(a.z, a.w);
  uint2 pk = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
  *reinterpret_cast<uint2*>(out + (int64_t)m * ldo + c) = pk;
}

template <int TB, int W, int ST, int GPU>
static void launch4(const Params& p, int S, cudaStream_t stream) {
  constexpr int STG = W * (128 * GPU + 4 * GPU) + GPU * TB * 32;
  constexpr int SMEMP = ST * STG * 16, SMEMT = W * TB * 8 * 68 * 4;
  constexpr int SMEM = SMEMP > SMEMT ? SMEMP : SMEMT;
  TORCH_CHECK((p.N / 64) % W == 0, "o4: N/64 must be a multiple of W");
  const int GU = p.K / 32 / GPU;
  TORCH_CHECK(S >= 1 && S <= GU, "o4: bad split");
  const int TG = p.N / 64 / W;
  const int NCH = (p.M + TB * 8 - 1) / (TB * 8);
  dim3 grid(S * NCH, TG);
  TORCH_CHECK(p.ldo % 4 == 0 && reinterpret_cast<uintptr_t>(p.out) % 8 == 0, "o4: out alignment");
  if (S == 1) {
    auto kern = o4_gemm<TB, W, ST, GPU, false>;
    static bool init = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); return true; }();
    (void)init;
    kern<<<grid, W * 32, SMEM, stream>>>(p, S, NCH);
  } else {
    TORCH_CHECK(p.part_n >= (int64_t)S * p.M * p.N, "o4: scratch too small");
    TORCH_CHECK(p.ldo % 4 == 0 && reinterpret_cast<uintptr_t>(p.out) % 8 == 0, "o4: out alignment");
    auto kern = o4_gemm<TB, W, ST, GPU, true>;
    static bool init = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); return true; }();
    (void)init;
    kern<<<grid, W * 32, SMEM, stream>>>(p, S, NCH);
    const int64_t per = (int64_t)p.M * (p.N / 4);
    o4_reduce<<<(unsigned)((per + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const float4*>(p.part), p.out, p.ldo, p.M, p.N, S);
  }
}

// ============================================================================
// O5: large M (65..192).  Like O4 (static split-K, CTA-shared activations) but
// TWO warps share each 64-col tile: warp (tile = w>>1, h = w&1) owns the
// 16-col subtiles jj = 2h, 2h+1 -> it dequantises only its half of the words
// (one LDS.128 per row) and its accumulator is 2 x 4 x TB floats, so one pass
// covers TB*8 = up to 128 tokens with ONE dequantisation.  M is processed in
// NCH balanced chunks of <= TB*8 rows (chunk index fastest in the grid).
// ============================================================================
template <int TB, int TW, int ST, int GPU, bool PART>
__global__ void __launch_bounds__(TW * 64) o5_gemm(const Params p, const int S, const int NCH, const int CR) {
  constexpr int W = 2 * TW;  // warps
  constexpr int TOK = TB * 8;
  constexpr int WS = 128 * GPU, SS = 4 * GPU, XS = GPU * TB * 32;
  constexpr int STG = TW * (WS + SS) + XS;
  constexpr int NT = W * 32;
  constexpr int QN = (XS + NT - 1) / NT;
  extern __shared__ __align__(128) uint4 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wt = warp >> 1, h = warp & 1;
  const int g = lane >> 2, c4 = lane & 3;
  const int N = p.N;
  const int ch = blockIdx.x % NCH;
  const int r0 = ch * CR;                        // first row of this chunk
  const int M = min(CR, p.M - r0);               // rows of this chunk (<= TOK)
  const __nv_bfloat16* const X = p.x + (int64_t)r0 * p.lda;
  const int GU = (p.K >> 5) / GPU;
  const int s = blockIdx.x / NCH, tg = blockIdx.y;
  const int u0 = s * GU / S, u1 = (s + 1) * GU / S;
  const int tile = tg * TW + wt;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);

  // weight copy: warp h of the tile copies chunk q = h*32+lane of each row
  const int q = h * 32 + lane;
  const uint32_t d_w = (wt * WS + (q ^ ((q >> 3) & 1))) * 16;
  const uint32_t d_s = (TW * WS + wt * SS + lane) * 16;  // h == 0, lanes < 4
  const int sw = (lane >> 2) & 1;
  const int r_w = wt * WS + ((2 * lane + h) ^ sw);       // this warp's half of the lane's 32 B
  const uint32_t r_x = (TW * (WS + SS) + (lane & 7) * 4 + ((lane >> 3) ^ (((lane & 7) >> 1) & 3))) * 16;
  uint32_t xdst[QN];
  const __nv_bfloat16* xsrc[QN];
  int xn[QN];
#pragma unroll
  for (int k = 0; k < QN; k++) {
    const int qq0 = tid + k * NT;
    const int qq = qq0 < XS ? qq0 : 0;
    const int gi = qq / (TB * 32), rem = qq % (TB * 32);
    const int row = rem >> 2, c = rem & 3;
    xdst[k] = (TW * (WS + SS) + gi * TB * 32 + row * 4 + (c ^ ((row >> 1) & 3))) * 16;
    xn[k] = (qq0 < XS && row < M) ? 16 : 0;
    xsrc[k] = X + (int64_t)(row < M ? row : 0) * p.lda + (u0 * GPU + gi) * 32 + c * 8;
  }
  const int64_t rowN = N;
  int l_left = u1 - u0;
  const uint4* l_pw = p.w + (int64_t)u0 * (2 * GPU) * rowN + tile * 64 + q;
  const uint8_t* l_ps = p.s + (int64_t)u0 * GPU * N + tile * 64 + lane * 16;
  int l_xo = 0;
  auto issue = [&](const int slot) {
    if (l_left > 0) {
      const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
      for (int r = 0; r < 2 * GPU; r++) cp16(st + r * 1024 + d_w, l_pw + r * rowN);
      if (h == 0 && lane < 4) {
#pragma unroll
        for (int gi = 0; gi < GPU; gi++) cp16(st + d_s + gi * 64, l_ps + gi * N);
      }
#pragma unroll
      for (int k = 0; k < QN; k++)
        if (k * NT + tid < XS) cp16z(st + xdst[k], xsrc[k] + l_xo, xn[k]);
      l_left--;
      l_pw += 2 * GPU * rowN;
      l_ps += GPU * N;
      l_xo += GPU * 32;
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < ST - 1; i++) issue(i);

  float acc[TB][2][4];
#pragma unroll
  for (int i = 0; i < TB; i++)
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[i][j][v] = 0.f;

  auto compute = [&](const int slot) {
    const uint4* stp = smem + slot * STG;
    const uint32_t st = sbase + slot * STG * 16;
#pragma unroll
    for (int gi = 0; gi < GPU; gi++) {
      const uint32_t e = reinterpret_cast<const uint32_t*>(stp + TW * WS + wt * SS + gi * 4)[2 * g + h];
      uint32_t sd[4];  // words 4h..4h+3: bytes {0,2,1,3} of this warp's dword
      sd[0] = __byte_perm(e, 0, 0x4040u) * 128u + 0x3c003c00u;
      sd[1] = __byte_perm(e, 0, 0x4242u) * 128u + 0x3c003c00u;
      sd[2] = __byte_perm(e, 0, 0x4141u) * 128u + 0x3c003c00u;
      sd[3] = __byte_perm(e, 0, 0x4343u) * 128u + 0x3c003c00u;
      uint32_t a[2][2][4];  // [row i][jl][a0..a3]
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint4 wv = stp[(gi * 2 + i) * 64 + r_w];
        const uint32_t wd[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int jl = 0; jl < 2; jl++) {
          const uint32_t qa = wd[2 * jl], qb = wd[2 * jl + 1];
          a[i][jl][0] = hmul2u(fp8_lo(qa), sd[2 * jl]);
          a[i][jl][2] = hmul2u(fp8_hi<0>(qa), sd[2 * jl]);
          a[i][jl][1] = hmul2u(fp8_lo(qb), sd[2 * jl + 1]);
          a[i][jl][3] = hmul2u(fp8_hi<0>(qb), sd[2 * jl + 1]);
        }
      }
#pragma unroll
      for (int bb = 0; bb < TB; bb++) {
        uint32_t xb[4];
        ldsm4(st + r_x + (gi * TB + bb) * 512, xb);
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int jl = 0; jl < 2; jl++)
            mma_bf16(acc[bb][jl], a[i][jl][0], a[i][jl][1], a[i][jl][2], a[i][jl][3], xb[2 * i], xb[2 * i + 1]);
      }
    }
  };

  int n = u1 - u0;
  while (n > 0) {
#pragma unroll
    for (int i = 0; i < ST; i++) {
      if (n > 0) {
        cp_wait<ST - 2>();
        __syncthreads();
        issue((i + ST - 1) % ST);
        compute(i);
        n--;
      }
    }
  }
  cp_wait<0>();
  // ---- epilogue: per 8-token block, transpose this warp's [8 x 32] through
  // smem and write 32-col row segments (128 B fp32 / 64 B bf16).
  __syncthreads();
  float* tw = reinterpret_cast<float*>(smem) + warp * 8 * 36;
#pragma unroll
  for (int bb = 0; bb < TB; bb++) {
    if (bb * 8 >= M) break;
#pragma unroll
    for (int jl = 0; jl < 2; jl++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int col = 16 * jl + 8 * (v >> 1) + g, tok = 2 * c4 + (v & 1);
        tw[tok * 36 + col] = acc[bb][jl][v];
      }
    __syncwarp();
#pragma unroll
    for (int rr = 0; rr < 8; rr += 2) {  // 2 rows per instruction: lanes 0-15 row rr, 16-31 row rr+1
      const int row = rr + (lane >> 4), cc = (lane & 15) * 2;
      const int tok = bb * 8 + row;
      if (tok < M) {
        const float2 v = *reinterpret_cast<const float2*>(tw + row * 36 + cc);
        const int64_t col = (int64_t)tile * 64 + h * 32 + cc;
        if (PART)
          __stcg(reinterpret_cast<float2*>(p.part + ((int64_t)s * p.M + r0 + tok) * N + col), v);
        else
          *reinterpret_cast<__nv_bfloat162*>(p.out + (int64_t)(r0 + tok) * p.ldo + col) = __floats2bfloat162_rn(v.x, v.y);
      }
    }
    __syncwarp();
  }
}

template <int TB, int TW, int ST, int GPU>
static void launch5(const Params& p, int S, cudaStream_t stream) {
  constexpr int STG = TW * (128 * GPU + 4 * GPU) + GPU * TB * 32;
  constexpr int SMEMP = ST * STG * 16, SMEMT = TW * 2 * 8 * 36 * 4;
  constexpr int SMEM = SMEMP > SMEMT ? SMEMP : SMEMT;
  TORCH_CHECK((p.N / 64) % TW == 0, "o5: N/64 must be a multiple of TW");
  const int GU = p.K / 32 / GPU;
  TORCH_CHECK(S >= 1 && S <= GU, "o5: bad split");
  TORCH_CHECK(p.ldo % 4 == 0 && reinterpret_cast<uintptr_t>(p.out) % 8 == 0, "o5: out alignment");
  const int NCH = (p.M + TB * 8 - 1) / (TB * 8);
  const int CR = (((p.M + NCH - 1) / NCH) + 7) / 8 * 8;  // balanced chunk rows (multiple of 8)
  dim3 grid(S * NCH, p.N / 64 / TW);
  if (S == 1) {
    auto kern = o5_gemm<TB, TW, ST, GPU, false>;
    static bool init = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); return true; }();
    (void)init;
    kern<<<grid, TW * 64, SMEM, stream>>>(p, S, NCH, CR);
  } else {
    TORCH_CHECK(p.part_n >= (int64_t)S * p.M * p.N, "o5: scratch too small");
    auto kern = o5_gemm<TB, TW, ST, GPU, true>;
    static bool init = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); return true; }();
    (void)init;
    kern<<<grid, TW * 64, SMEM, stream>>>(p, S, NCH, CR);
    const int64_t per = (int64_t)p.M * (p.N / 4);
    o4_reduce<<<(unsigned)((per + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const float4*>(p.part), p.out, p.ldo, p.M, p.N, S);
  }
}

}  // namespace o2

static unsigned long long* g_dbg = nullptr;
static int g_sync = 0;
void o2_set_sync(int64_t v) { g_sync = (int)v; }
void o2_set_dbg(at::Tensor const& d) { g_dbg = d.numel() ? (unsigned long long*)d.data_ptr() : nullptr; }

// cfg = mode*1000 + variant; variant selects (TB,W,ST,GPU,HIV)
void o2_dense(at::Tensor const& x, at::Tensor const& w, at::Tensor const& s, at::Tensor& out,
              at::Tensor& part, at::Tensor& cnt, int64_t cfg, int64_t ctas) {
  using namespace o2;
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: bf16 [M,K], row stride % 8, 16B aligned");
  TORCH_CHECK(w.is_contiguous() && w.numel() * 4 == (int64_t)K * N, "w: Marlin 8-bit [K/16, 4N]");
  TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && s.numel() == (int64_t)K / 32 * N, "s");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0 && M >= 1, "shape: N % 64, K % 64");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.size(0) == M &&
                  out.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "out: bf16 [M,N], row stride % 4, 8B aligned");
  TORCH_CHECK(part.scalar_type() == at::kFloat && cnt.scalar_type() == at::kInt &&
                  cnt.numel() >= N / 64, "part/cnt");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  Params p{(const __nv_bfloat16*)x.data_ptr(), x.stride(0), M, (const uint4*)w.data_ptr(),
           (const uint8_t*)s.data_ptr(), K, N, (__nv_bfloat16*)out.data_ptr(), out.stride(0),
           part.data_ptr<float>(), cnt.data_ptr<int>(), g_dbg, g_sync, part.numel()};
  auto st = at::cuda::getCurrentCUDAStream();
  const int mode = (int)(cfg / 1000), v = (int)(cfg % 1000);
#define L(TB, W, ST, GPU)                                                      \
  do {                                                                         \
    TORCH_CHECK(M <= TB * 8, "M too large for cfg");                           \
    if (mode == 0) launch<TB, W, ST, 0, GPU, 0>(p, (int)ctas, st);             \
    else if (mode == 1) launch<TB, W, ST, 1, GPU, 0>(p, (int)ctas, st);        \
    else if (mode == 2) launch<TB, W, ST, 2, GPU, 0>(p, (int)ctas, st);        \
    else launch<TB, W, ST, 3, GPU, 0>(p, (int)ctas, st);                       \
    return;                                                                    \
  } while (0)
  switch (v) {
    // M <= 8
    case 0: L(1, 4, 4, 1);
    case 1: L(1, 4, 4, 2);
    case 3: L(1, 4, 3, 2);
    case 5: L(1, 4, 6, 1);
    case 7: L(1, 6, 4, 2);
    case 8: L(1, 8, 3, 2);
    case 9: L(1, 4, 6, 2);
    case 11: L(1, 5, 4, 2);
    case 12: L(1, 6, 3, 2);
    // M <= 16 / 48
    case 10: L(2, 4, 4, 1);
    case 20: L(6, 4, 3, 1);
    case 21: L(6, 4, 4, 1);
    case 22: L(6, 2, 4, 1);
    case 23: L(6, 2, 3, 2);
    case 24: L(6, 4, 2, 2);
  }
#undef L
  TORCH_CHECK(false, "bad cfg");
}


// O3 (M <= 48): cfg = mode*1000 + variant (mode 0 full, 3 no-output ablation)
void o3_dense(at::Tensor const& x, at::Tensor const& w, at::Tensor const& s, at::Tensor& out,
              at::Tensor& part, int64_t cfg, int64_t ctas) {
  using namespace o2;
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: bf16 [M,K], row stride % 8, 16B aligned");
  TORCH_CHECK(w.is_contiguous() && w.numel() * 4 == (int64_t)K * N, "w: Marlin 8-bit [K/16, 4N]");
  TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && s.numel() == (int64_t)K / 32 * N, "s");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0 && M >= 1, "shape: N % 64, K % 64");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.size(0) == M, "out");
  TORCH_CHECK(part.scalar_type() == at::kFloat, "part");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  Params p{(const __nv_bfloat16*)x.data_ptr(), x.stride(0), M, (const uint4*)w.data_ptr(),
           (const uint8_t*)s.data_ptr(), K, N, (__nv_bfloat16*)out.data_ptr(), out.stride(0),
           part.data_ptr<float>(), nullptr, g_dbg, g_sync, part.numel()};
  auto st = at::cuda::getCurrentCUDAStream();
  const int mode = (int)(cfg / 1000), v = (int)(cfg % 1000);
#define L3(TB, W, ST, GPU)                                                     \
  do {                                                                         \
    TORCH_CHECK(M <= TB * 8, "M too large for cfg");                           \
    if (mode == 0) launch3<TB, W, ST, 0, GPU>(p, (int)ctas, st);               \
    else launch3<TB, W, ST, 3, GPU>(p, (int)ctas, st);                         \
    return;                                                                    \
  } while (0)
  switch (v) {
    case 21: L3(2, 4, 3, 2);
    case 60: L3(6, 4, 4, 1);
  }
#undef L3
  TORCH_CHECK(false, "bad o3 cfg");
}


// O4 (M <= 64, static split-K + shared activations): cfg = TB*10 + variant, S = K slices
void o4_dense(at::Tensor const& x, at::Tensor const& w, at::Tensor const& s, at::Tensor& out,
              at::Tensor& part, int64_t cfg, int64_t S) {
  using namespace o2;
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: bf16 [M,K], row stride % 8, 16B aligned");
  TORCH_CHECK(w.is_contiguous() && w.numel() * 4 == (int64_t)K * N, "w: Marlin 8-bit [K/16, 4N]");
  TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && s.numel() == (int64_t)K / 32 * N, "s");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0 && M >= 1, "shape: N % 64, K % 64");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.size(0) == M, "out");
  TORCH_CHECK(part.scalar_type() == at::kFloat, "part");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  Params p{(const __nv_bfloat16*)x.data_ptr(), x.stride(0), M, (const uint4*)w.data_ptr(),
           (const uint8_t*)s.data_ptr(), K, N, (__nv_bfloat16*)out.data_ptr(), out.stride(0),
           part.data_ptr<float>(), nullptr, nullptr, g_sync, part.numel()};
  auto st = at::cuda::getCurrentCUDAStream();
#define L4(TB, W, ST, GPU)                                   \
  do {                                                       \
    TORCH_CHECK(M <= TB * 8 || TB == 8, "M too large for cfg (M > 64 needs TB=8 chunks)"); \
    launch4<TB, W, ST, GPU>(p, (int)S, st);                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                          \
    return;                                                  \
  } while (0)
  switch ((int)cfg) {
    case 20: L4(2, 4, 4, 1);
    case 21: L4(2, 4, 3, 2);
    case 22: L4(2, 8, 3, 1);
    case 30: L4(3, 4, 4, 1);
    case 31: L4(3, 4, 3, 2);
    case 32: L4(3, 8, 3, 1);
    case 40: L4(4, 4, 4, 1);
    case 41: L4(4, 4, 3, 2);
    case 42: L4(4, 8, 3, 1);
    case 60: L4(6, 4, 4, 1);
    case 61: L4(6, 4, 3, 2);
    case 62: L4(6, 8, 3, 1);
    case 80: L4(8, 4, 4, 1);
    case 81: L4(8, 4, 3, 2);
    case 82: L4(8, 8, 3, 1);
  }
#undef L4
  TORCH_CHECK(false, "bad o4 cfg");
}


// O5 (large M, two warps per tile): cfg = TB*10 + variant, S = K slices
void o5_dense(at::Tensor const& x, at::Tensor const& w, at::Tensor const& s, at::Tensor& out,
              at::Tensor& part, int64_t cfg, int64_t S) {
  using namespace o2;
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: bf16 [M,K], row stride % 8, 16B aligned");
  TORCH_CHECK(w.is_contiguous() && w.numel() * 4 == (int64_t)K * N, "w: Marlin 8-bit [K/16, 4N]");
  TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && s.numel() == (int64_t)K / 32 * N, "s");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0 && M >= 1, "shape: N % 64, K % 64");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.size(0) == M, "out");
  TORCH_CHECK(part.scalar_type() == at::kFloat, "part");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  Params p{(const __nv_bfloat16*)x.data_ptr(), x.stride(0), M, (const uint4*)w.data_ptr(),
           (const uint8_t*)s.data_ptr(), K, N, (__nv_bfloat16*)out.data_ptr(), out.stride(0),
           part.data_ptr<float>(), nullptr, nullptr, 0, part.numel()};
  auto st = at::cuda::getCurrentCUDAStream();
#define L5(TB, TW, ST, GPU)                                  \
  do {                                                       \
    launch5<TB, TW, ST, GPU>(p, (int)S, st);                 \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                          \
    return;                                                  \
  } while (0)
  switch ((int)cfg) {
    case 80: L5(8, 4, 3, 1);
    case 81: L5(8, 4, 4, 1);
    case 82: L5(8, 2, 4, 1);
    case 120: L5(12, 4, 3, 1);
    case 121: L5(12, 4, 4, 1);
    case 122: L5(12, 2, 4, 1);
    case 160: L5(16, 4, 3, 1);
    case 161: L5(16, 4, 4, 1);
    case 162: L5(16, 2, 4, 1);
    case 83: L5(8, 1, 4, 1);
    case 123: L5(12, 1, 4, 1);
    case 124: L5(12, 2, 3, 1);
    case 125: L5(12, 1, 6, 1);
  }
#undef L5
  TORCH_CHECK(false, "bad o5 cfg");
}

TORCH_LIBRARY_FRAGMENT(_o2, m) {
  m.def("set_dbg(Tensor d) -> ()");
  m.def("dense5(Tensor x, Tensor w, Tensor s, Tensor(o!) out, Tensor(p!) part, int cfg, int S) -> ()");
  m.def("dense4(Tensor x, Tensor w, Tensor s, Tensor(o!) out, Tensor(p!) part, int cfg, int S) -> ()");
  m.def("dense3(Tensor x, Tensor w, Tensor s, Tensor(o!) out, Tensor(p!) part, int cfg, int ctas) -> ()");
  m.def("set_sync(int v) -> ()", &o2_set_sync);
  m.def("dense(Tensor x, Tensor w, Tensor s, Tensor(o!) out, Tensor(p!) part, Tensor(c!) cnt, int cfg, int ctas) -> ()");
}
TORCH_LIBRARY_IMPL(_o2, CUDA, m) {
  m.impl("dense", &o2_dense);
  m.impl("dense3", &o3_dense);
  m.impl("dense4", &o4_dense);
  m.impl("dense5", &o5_dense);
  m.impl("set_dbg", &o2_set_dbg);
}
