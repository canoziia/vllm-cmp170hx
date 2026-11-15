// DeepSeek V4.1 sparse-indexer prefill on SM80 (CMP 170HX): fused
// tensor-core MQA logits + exact chunk-max-pruned top-k + fast candidate
// writer (fused 8-col block scores + smem radix top-2048) + per-row compacted
// candidate reader.
//
// Bitwise contract (see REPORT.md):
//  * dsv41_idx_logits reproduces the incumbent Triton _fp8_mqa_logits_kernel
//    (BLOCK_H=32, D=128, FACTOR_K_SCALE=1) value-for-value inside [ks, ke):
//    same bf16 operands, same mma.m16n8k16 k-order (0..127 ascending from a
//    zero accumulator), same epilogue tree as the Triton PTX:
//      v = rn(relu(s[g+8]) * w[g+8]); v = fma(relu(s[g]), w[g], v);
//      v = fma(relu(s[16+g]), w[16+g], v); v = fma(relu(s[24+g]), w[24+g], v);
//      butterfly add over lanes xor 16, 8, 4;  out = rn(v * k_scale)
//    (the xor reduce-scatter used here forms exactly the same sums).
//    Optional candidate flags reproduce apply_candidate_mask.
//  * dsv41_idx_logits_writer additionally emits per-8-column NaN-propagating
//    block maxima for aligned rows, consumed by dsv41_idx_cand_topk to match
//    select_candidate_blocks (_block_scores_kernel + scores.topk +
//    _store_candidates_kernel) without re-reading the FP32 logits from HBM.
//  * dsv41_idx_reader_compact stages only each row's active 8-key candidate
//    blocks via cp.async, computing bit-identical logits and exact top-k.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace {

constexpr int kH = 32;
constexpr int kD = 128;
constexpr int kTN = 32;            // keys per tile == chunk width
constexpr int kWarps = 8;          // query rows per CTA
constexpr int kStages = 4;
constexpr int kTileBytes = kTN * kD * 2;  // 8 KiB bf16

__device__ __forceinline__ uint32_t f2key(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ uint32_t f2key_cand(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t mag = u & 0x7FFFFFFFu;
  uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  if (mag == 0u) k = 0x80000000u;
  if (mag > 0x7F800000u) k = 0xFFFFFFFFu;
  return k;
}
__device__ __forceinline__ float relu_ptx(float x) {
  float y;
  asm("max.f32 %0, %1, 0f00000000;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fmax_nan(float a, float b) {
  float y;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// e4m3fn -> bf16 via a 256-entry table built on the host from torch's own
// conversion (so NaN handling etc. matches q.to(torch.bfloat16) bit for bit).
__global__ void decode_fp8_kernel(const uint8_t* __restrict__ in, uint16_t* __restrict__ out,
                                  const uint16_t* __restrict__ lut, int64_t n16) {
  __shared__ uint16_t t[256];
  t[threadIdx.x] = lut[threadIdx.x];
  __syncthreads();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n16; i += (int64_t)gridDim.x * blockDim.x) {
    uint4 v = reinterpret_cast<const uint4*>(in)[i];
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[2 * j] = (uint32_t)t[w[j] & 255] | ((uint32_t)t[(w[j] >> 8) & 255] << 16);
      o[2 * j + 1] = (uint32_t)t[(w[j] >> 16) & 255] | ((uint32_t)t[w[j] >> 24] << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out) + 2 * i;
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// ---------------------------------------------------------------------------
struct LogitsArgs {
  const __nv_bfloat16* q;  // [R, 32, 128] contiguous rows
  int64_t q_s0;
  const __nv_bfloat16* k;  // [N, 128] contiguous
  const float* ksc;        // [N]
  const float* w;
  int64_t w_s0, w_s1;
  const int* ks;
  const int* ke;
  float* out;
  int64_t out_s0;
  uint32_t* mx;  // [R, ceil(N/32)]
  int64_t mx_s0;
  uint32_t* bkeys;  // optional [R, nblocks] order-preserving block keys for writer fast path
  int64_t bk_s0;
  const uint8_t* flags;  // optional [R, nblocks+1]
  int64_t fl_s0;
  int nblocks;
  int cand_bs;  // candidate block size (8)
  int R, N, kpc;  // kpc: keys per CTA (multiple of 32)
};

// One 32-key sub-tile worth of MMAs: acc = q[32x128] x K[sub]^T, k ascending.
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

template <bool kMask = false>
__device__ __forceinline__ void tile_mma(float (&acc)[2][4][4], const uint32_t (&A)[2][8][4], uint32_t st,
                                         int sub, int lm_mat, int lm_r, uint32_t ntmask = 0xFu) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  if (kMask && ntmask != 0xFu) {
    // sparse candidate tile: only the 8-key n-tiles holding a kept column
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      if (!((ntmask >> nt) & 1u)) continue;
#pragma unroll
      for (int kc = 0; kc < 8; ++kc) {
        uint32_t b0, b1;
        const int key = sub * kTN + nt * 8 + lm_r;
        const int ch = kc * 2 + (lm_mat & 1);
        ldsm_x2(st + key * 256 + ((ch ^ (key & 7)) << 4), b0, b1);
        mma16816(acc[0][nt], A[0][kc], b0, b1);
        mma16816(acc[1][nt], A[1][kc], b0, b1);
      }
    }
    return;
  }
#pragma unroll
  for (int kc = 0; kc < 8; ++kc) {
    uint32_t b[4][2];
#pragma unroll
    for (int hp = 0; hp < 2; ++hp) {
      const int key = sub * kTN + (hp * 2 + (lm_mat >> 1)) * 8 + lm_r;
      const int ch = kc * 2 + (lm_mat & 1);
      ldsm_x4(st + key * 256 + ((ch ^ (key & 7)) << 4), b[2 * hp][0], b[2 * hp][1], b[2 * hp + 1][0],
              b[2 * hp + 1][1]);
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma16816(acc[mt][nt], A[mt][kc], b[nt][0], b[nt][1]);
  }
}

struct Pend {
  int tb;        // first key of the sub-tile
  bool present;  // sub-tile inside the CTA range
  bool need;     // MMAs were run for it
  bool keep;     // per-lane candidate keep flag (kFlags)
  uint32_t ntmask;  // 8-key n-tiles holding a kept column (kFlags)
};

// Exact head-weighted ReLU reduction tree matching Triton PTX.
__device__ __forceinline__ float head_reduce_epi(const float (&acc)[2][4][4], float w0, float w1, float w2,
                                                 float w3, int lane) {
  const int hi16 = (lane >> 4) & 1, hi8 = (lane >> 3) & 1, hi4 = (lane >> 2) & 1;
  float v[4][2];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float x = __fmul_rn(relu_ptx(acc[0][nt][2 + j]), w1);
      x = __fmaf_rn(relu_ptx(acc[0][nt][j]), w0, x);
      x = __fmaf_rn(relu_ptx(acc[1][nt][j]), w2, x);
      x = __fmaf_rn(relu_ptx(acc[1][nt][2 + j]), w3, x);
      v[nt][j] = x;
    }
  float k1[2][2];
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float mine = hi16 ? v[2 + q][j] : v[q][j];
      float send = hi16 ? v[q][j] : v[2 + q][j];
      k1[q][j] = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, send, 16));
    }
  float k2[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float mine = hi8 ? k1[1][j] : k1[0][j];
    float send = hi8 ? k1[0][j] : k1[1][j];
    k2[j] = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, send, 8));
  }
  float mine = hi4 ? k2[1] : k2[0];
  float send = hi4 ? k2[0] : k2[1];
  return __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, send, 4));
}

// Branch-free epilogue (same tree as the Triton PTX); stores logits + chunk max
// (and optional 8-column NaN-propagating block keys when kWriteBlocks == true).
template <bool kFlags, int kDbg = 0, bool kWriteBlocks = false>
__device__ __forceinline__ void tile_epi(const float (&acc)[2][4][4], const Pend& p, float w0, float w1, float w2,
                                         float w3, int lane, int my_col, int rks, int rke, int nb_col,
                                         const LogitsArgs& a, float* orow, uint32_t* mrow,
                                         uint32_t* bkrow) {
  float outv = head_reduce_epi(acc, w0, w1, w2, w3, lane);
  const int n = p.tb + my_col;
  const bool inN = n < a.N;
  const float sc = __ldg(a.ksc + (inN ? n : 0));
  outv = __fmul_rn(outv, sc);
  bool valid = p.need && n >= rks && n < rke;
  if (kFlags) valid = valid && p.keep;
  outv = valid ? outv : -INFINITY;
  if (kDbg != 4 && p.present && inN) orow[n] = outv;
  if (kDbg == 4 && outv == 1.2345f) orow[n] = outv;
  if (kDbg == 3) return;
  const uint32_t key = __reduce_max_sync(0xffffffffu, f2key(outv));  // redux.sync.max.u32
  if (p.present && lane == 0) mrow[p.tb / kTN] = key;
  if constexpr (kWriteBlocks) {
    // 100% branch-free predicated instructions so tile_mma + tile_epi remain one basic block!
    uint32_t uk = (valid && inN) ? f2key_cand(outv) : 0x007FFFFFu;
    uk = max(uk, __shfl_xor_sync(0xffffffffu, uk, 4));
    uk = max(uk, __shfl_xor_sync(0xffffffffu, uk, 2));
    uk = max(uk, __shfl_xor_sync(0xffffffffu, uk, 1));
    const int blk = ((p.tb - rks) >> 3) + (lane >> 3);
    if (p.present && ((rks & 7) == 0) && ((lane & 7) == 0) && blk >= 0 && blk < nb_col) bkrow[blk] = uk;
  }
}

constexpr int kSub = 2;                       // 32-key sub-tiles per pipeline stage
constexpr int kStageBytes = kSub * kTileBytes;  // 16 KiB

template <bool kFlags, int kDbg = 0, bool kWriteBlocks = false>
__global__ void __launch_bounds__(kWarps * 32, 1) logits_kernel(const LogitsArgs a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_lo[kWarps], s_hi[kWarps];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row = blockIdx.x * kWarps + warp;
  const bool has_row = row < a.R;
  const int rks = has_row ? a.ks[row] : 0;
  const int rke = has_row ? a.ke[row] : 0;
  const int n_begin = blockIdx.y * a.kpc;
  const int n_end = min(a.N, n_begin + a.kpc);
  if (lane == 0) {
    bool live = has_row && rke > rks && rke > n_begin && rks < n_end;
    s_lo[warp] = live ? max(rks, n_begin) : INT_MAX;
    s_hi[warp] = live ? min(rke, n_end) : INT_MIN;
  }
  __syncthreads();
  int lo = INT_MAX, hi = INT_MIN;
#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    lo = min(lo, s_lo[i]);
    hi = max(hi, s_hi[i]);
  }
  if (lo >= hi) return;  // CTA-uniform
  const int t0 = lo / kTN, t1 = (hi + kTN - 1) / kTN;
  const int nst = (t1 - t0 + kSub - 1) / kSub;
  const bool live_row = has_row && rke > rks && rke > lo && rks < hi;

  uint32_t A[2][8][4];
  float w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
  if (live_row) {
    const uint32_t* qr = reinterpret_cast<const uint32_t*>(a.q + row * a.q_s0);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kc = 0; kc < 8; ++kc) {
        const int h0 = mt * 16 + g, d0 = kc * 16 + 2 * t;
        A[mt][kc][0] = __ldg(qr + (h0 * kD + d0) / 2);
        A[mt][kc][1] = __ldg(qr + ((h0 + 8) * kD + d0) / 2);
        A[mt][kc][2] = __ldg(qr + (h0 * kD + d0 + 8) / 2);
        A[mt][kc][3] = __ldg(qr + ((h0 + 8) * kD + d0 + 8) / 2);
      }
    const float* wr = a.w + row * a.w_s0;
    w0 = wr[g * a.w_s1];
    w1 = wr[(g + 8) * a.w_s1];
    w2 = wr[(16 + g) * a.w_s1];
    w3 = wr[(24 + g) * a.w_s1];
  } else {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kc = 0; kc < 8; ++kc)
#pragma unroll
        for (int i = 0; i < 4; ++i) A[mt][kc][i] = 0u;
  }

  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  auto issue = [&](int s) {
    if (s < nst) {
      const int key0 = (t0 + s * kSub) * kTN;
      const uint32_t st = sbase + (s % kStages) * kStageBytes;
#pragma unroll
      for (int j = 0; j < kSub * 2; ++j) {
        const int p = tid + j * kWarps * 32;
        const int key = p >> 4, ch = p & 15;
        const int gk = key0 + key;
        const bool v = gk < a.N;
        const __nv_bfloat16* src = a.k + (int64_t)(v ? gk : 0) * kD + ch * 8;
        cp_async16(st + key * 256 + ((ch ^ (key & 7)) << 4), src, v);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < kStages - 1; ++i) issue(i);

  const int hi16 = (lane >> 4) & 1, hi8 = (lane >> 3) & 1, hi4 = (lane >> 2) & 1;
  const int my_col = (hi16 * 2 + hi8) * 8 + 2 * t + hi4;
  const int lm_mat = lane >> 3, lm_r = lane & 7;
  float* orow = a.out + (int64_t)(has_row ? row : 0) * a.out_s0;
  uint32_t* mrow = a.mx + (int64_t)(has_row ? row : 0) * a.mx_s0;
  uint32_t* bsrow = kWriteBlocks ? a.bkeys + (int64_t)(has_row ? row : 0) * a.bk_s0 : nullptr;
  const int end_eff = min(rke, a.N);
  const int nb_col = (kWriteBlocks && has_row && rke > rks && end_eff > rks)
                         ? min(a.nblocks, (end_eff - rks + 7) >> 3)
                         : 0;
  const uint8_t* frow = kFlags ? a.flags + (int64_t)(has_row ? row : 0) * a.fl_s0 : nullptr;

  auto meta = [&](int j) {
    Pend p;
    p.ntmask = 0xFu;
    p.tb = j * kTN;
    p.present = j < t1;
    p.need = p.present && (rke > p.tb) && (rks < p.tb + kTN) && (rke > rks);
    p.keep = false;
    if (kFlags) {
      const int n = p.tb + my_col;
      if (p.need && n >= rks && n < rke && n < a.N) {
        int blk = (n - rks) / a.cand_bs;
        p.keep = frow[blk] != 0;
        if (!p.keep && n == a.N - 1) p.keep = frow[a.nblocks] != 0;
      }
      const uint32_t bal = __ballot_sync(0xffffffffu, p.keep);
      p.ntmask = 0;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const uint32_t lanes = 0xFFu << (8 * ((nt >> 1) * 2 + (nt & 1)));
        if (bal & lanes) p.ntmask |= 1u << nt;
      }
      p.need = bal != 0;
    }
    return p;
  };

  float accA[2][4][4], accB[2][4][4];
  if (kDbg == 2) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) accA[mt][nt][e] = __uint_as_float(A[mt][nt][e]);
  }
  Pend pa, pb;
  pb.present = false;
  pb.need = false;
  pb.keep = false;
  pb.tb = 0;
  constexpr int kEpiDbg = (kDbg >= 3 ? kDbg : 0);
  for (int s = 0; s < nst; ++s) {
    cp_wait<kStages - 2>();
    __syncthreads();
    issue(s + kStages - 1);
    if (!live_row) continue;
    const uint32_t st = sbase + (s % kStages) * kStageBytes;
    const int j0 = t0 + s * kSub;
    pa = meta(j0);
    if (kDbg == 1) {
      tile_mma(accA, A, st, 0, lm_mat, lm_r);
      tile_mma(accB, A, st, 1, lm_mat, lm_r);
      float z = 0.f;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) z += accA[mt][nt][0] + accB[mt][nt][3];
      if (z == 1.2345f) orow[lane] = z;
      continue;
    }
    if (kDbg == 2) {
      tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
      pb = meta(j0 + 1);
      tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accA, pb, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
      continue;
    }
    if (pa.need) {
      tile_mma<kFlags>(accA, A, st, 0, lm_mat, lm_r, pa.ntmask);
      if (s > 0) tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
    } else if (s > 0) {
      tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
    }
    pb = meta(j0 + 1);
    if (pb.need) {
      tile_mma<kFlags>(accB, A, st, 1, lm_mat, lm_r, pb.ntmask);
      tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
    } else {
      tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
    }
  }
  if ((kDbg == 0 || kDbg >= 3) && live_row) {
    tile_epi<kFlags, kEpiDbg, kWriteBlocks>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, nb_col, a, orow, mrow, bsrow);
  }
  cp_wait<0>();
}

// chunk-max keys for externally produced logits (testing / generic top-k)
__global__ void chunkmax_kernel(const float* __restrict__ logits, int64_t ls0, const int* ks, const int* ke,
                                uint32_t* mx, int64_t ms0, int N) {
  const int row = blockIdx.y;
  const int rks = ks[row], rke = ke[row];
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const int nch = (N + 31) >> 5;
  if (warp >= nch) return;
  const int n = warp * 32 + lane;
  float v = (n >= rks && n < rke && n < N) ? logits[row * ls0 + n] : -INFINITY;
  uint32_t key = f2key(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) key = max(key, __shfl_xor_sync(0xffffffffu, key, o));
  if (lane == 0) mx[row * ms0 + warp] = key;
}

// ---------------------------------------------------------------------------
// Exact per-row top-k with group-max pruning.
constexpr int kTopkThreads = 256;
constexpr int kCandCap = 3072;
constexpr int kListCap = 1024;

struct TopkSmem {
  uint32_t hist[2048];
  uint32_t sel[kListCap];  // group maxima / chunk list, then the output columns
  uint32_t ckey[kCandCap];
  uint32_t ccol[kCandCap];
  typename cub::BlockScan<int, kTopkThreads>::TempStorage scan;
  int cnt, nsel, bin, above, binc;
};

__device__ __forceinline__ void hist_add(uint32_t* hist, uint32_t bin) {
  const unsigned act = __activemask();
  const unsigned m = __match_any_sync(act, bin);
  if ((int)(threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&hist[bin], (uint32_t)__popc(m));
}

template <int kBits, class SmemT, class Visit>
__device__ int radix_step_g(SmemT& S, Visit&& visit, uint32_t prefix, uint32_t mask, int shift, int k,
                            int& above, int& binc) {
  constexpr int nb = 1 << kBits;
  constexpr int kMaxPer = nb / kTopkThreads;
  for (int i = threadIdx.x; i < nb; i += kTopkThreads) S.hist[i] = 0;
  __syncthreads();
  visit([&](uint32_t key) {
    if ((key & mask) == prefix) hist_add(S.hist, (key >> shift) & (nb - 1));
  });
  __syncthreads();
  constexpr int per = kMaxPer;
  int loc[kMaxPer];
  int sum = 0;
#pragma unroll
  for (int j = 0; j < kMaxPer; ++j) {
    loc[j] = 0;
    if (j < per) {
      loc[j] = S.hist[nb - 1 - (threadIdx.x * per + j)];
      sum += loc[j];
    }
  }
  int excl;
  cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(sum, excl);
  int run = excl;
#pragma unroll
  for (int j = 0; j < kMaxPer; ++j) {
    if (j < per) {
      if (run < k && run + loc[j] >= k) {
        S.bin = nb - 1 - (threadIdx.x * per + j);
        S.above = run;
        S.binc = loc[j];
      }
      run += loc[j];
    }
  }
  __syncthreads();
  above = S.above;
  binc = S.binc;
  const int d = S.bin;
  __syncthreads();
  return d;
}

template <class SmemT, class Visit>
__device__ uint32_t radix_kth_g(SmemT& S, Visit&& visit, int k, int& G, int& E) {
  uint32_t prefix = 0, mask = 0;
  int kk = k, above = 0, binc = 0;
  int d = radix_step_g<11>(S, visit, prefix, mask, 21, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 21;
  mask |= 0x7FFu << 21;
  d = radix_step_g<11>(S, visit, prefix, mask, 10, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 10;
  mask |= 0x7FFu << 10;
  d = radix_step_g<10>(S, visit, prefix, mask, 0, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d;
  G = k - kk;
  E = binc;
  return prefix;
}

template <class SmemT, class Visit>
__device__ uint32_t radix_lower_bound_g(SmemT& S, Visit&& visit, int k) {
  int above, binc;
  const int d1 = radix_step_g<11>(S, visit, 0u, 0u, 21, k, above, binc);
  const int d2 = radix_step_g<11>(S, visit, (uint32_t)d1 << 21, 0xFFE00000u, 10, k - above, above, binc);
  return ((uint32_t)d1 << 21) | ((uint32_t)d2 << 10);
}

// Exact selection over a (key, col) stream: fills S.sel[0..K) (unordered).
// Ties at the k-th value keep the lowest columns.
template <class VisitKC>
__device__ void select_exact(TopkSmem& S, VisitKC&& vkc, int K) {
  auto vk = [&](auto f) { vkc([&](uint32_t key, uint32_t) { f(key); }); };
  int G, E;
  const uint32_t T = radix_kth_g(S, vk, K, G, E);
  const int need = K - G;
  uint32_t colT = 0xFFFFFFFFu;
  if (E > need) {
    auto vt = [&](auto f) {
      vkc([&](uint32_t key, uint32_t col) {
        if (key == T) f(~col);
      });
    };
    int g2, e2;
    colT = ~radix_kth_g(S, vt, need, g2, e2);
  }
  if (threadIdx.x == 0) S.nsel = 0;
  __syncthreads();
  vkc([&](uint32_t key, uint32_t col) {
    if (key > T || (key == T && col <= colT)) S.sel[atomicAdd(&S.nsel, 1)] = col;
  });
  __syncthreads();
}

template <int PER>
__device__ void sort_store_t(TopkSmem& S, int K, int KP, int* orow, int rks) {
  const int tid = threadIdx.x;
  uint32_t x[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = j * kTopkThreads + tid;
    x[j] = i < K ? S.sel[i] : 0xFFFFFFFFu;
  }
  __syncthreads();
  for (int size = 2; size <= KP; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      uint32_t y[PER];
      if (stride >= kTopkThreads) {
        const int js = stride / kTopkThreads;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
          uint32_t v = x[j];
#pragma unroll
          for (int jj = 0; jj < PER; ++jj)
            if (jj == (j ^ js)) v = x[jj];
          y[j] = v;
        }
      } else if (stride >= 32) {
#pragma unroll
        for (int j = 0; j < PER; ++j) S.sel[j * kTopkThreads + tid] = x[j];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PER; ++j) y[j] = S.sel[j * kTopkThreads + (tid ^ stride)];
        __syncthreads();
      } else {
#pragma unroll
        for (int j = 0; j < PER; ++j) y[j] = __shfl_xor_sync(0xffffffffu, x[j], stride);
      }
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const int i = j * kTopkThreads + tid;
        const bool up = (i & size) == 0;
        const bool lower = (i & stride) == 0;
        x[j] = (lower == up) ? min(x[j], y[j]) : max(x[j], y[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = j * kTopkThreads + tid;
    if (i < K) orow[i] = (int)x[j] - rks;
  }
}

__device__ void sort_and_store(TopkSmem& S, int K, int* orow, int rks) {
  int KP = kTopkThreads;
  while (KP < K) KP <<= 1;
  const int per = KP / kTopkThreads;
  if (per == 1) sort_store_t<1>(S, K, KP, orow, rks);
  else if (per == 2) sort_store_t<2>(S, K, KP, orow, rks);
  else if (per == 4) sort_store_t<4>(S, K, KP, orow, rks);
  else sort_store_t<8>(S, K, KP, orow, rks);
}

__device__ __forceinline__ void push_cand(TopkSmem& S, bool take, uint32_t key, uint32_t col) {
  const int lane = threadIdx.x & 31;
  const uint32_t bal = __ballot_sync(0xffffffffu, take);
  int b = 0;
  if (lane == 0 && bal) b = atomicAdd(&S.cnt, __popc(bal));
  b = __shfl_sync(0xffffffffu, b, 0);
  const int p = b + __popc(bal & ((1u << lane) - 1u));
  if (take && p < kCandCap) {
    S.ckey[p] = key;
    S.ccol[p] = col;
  }
}

__global__ void __launch_bounds__(kTopkThreads, 4) topk_kernel(const float* __restrict__ logits, int64_t ls0,
                                                               const uint32_t* __restrict__ mx, int64_t ms0,
                                                               const int* __restrict__ ks, const int* __restrict__ ke,
                                                               int* __restrict__ out, int64_t os0, int K,
                                                               int* __restrict__ stats) {
  extern __shared__ __align__(16) uint8_t smraw[];
  TopkSmem& S = *reinterpret_cast<TopkSmem*>(smraw);
  const int row = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int kWarpsT = kTopkThreads / 32;
  const int rks = ks[row], rke = ke[row];
  const int L = rke - rks;
  int* orow = out + row * os0;
  if (L <= K) {
    for (int i = tid; i < K; i += kTopkThreads) orow[i] = i < L ? i : -1;
    return;
  }
  const float* lrow = logits + row * ls0;
  const uint32_t* mrow = mx + row * ms0;
  long long tprof[5];
  tprof[0] = clock64();
  const int c0 = rks >> 5, c1 = (rke + 31) >> 5, nc = c1 - c0;
  const bool chunked = nc >= K;
  const int kG8 = L > 8 * kListCap ? 16 : 8;
  const int ng = chunked ? 0 : min(L / kG8, kListCap);
  bool grouped = !chunked && ng >= K;
  uint32_t tau = 0;
  int nsch = 0;
  if (tid == 0) S.cnt = 0;

  if (chunked) {
    uint32_t* mc = S.ckey;
    const bool mcached = nc <= 2 * kCandCap;
    if (mcached) {
      for (int i0 = 0; i0 < nc; i0 += kTopkThreads * 8) {
        uint32_t v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int i = i0 + u * kTopkThreads + tid;
          v[u] = i < nc ? __ldg(mrow + c0 + i) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int i = i0 + u * kTopkThreads + tid;
          if (i < nc) mc[i] = v[u];
        }
      }
      __syncthreads();
    }
    auto mget = [&](int i) -> uint32_t { return mcached ? mc[i] : __ldg(mrow + c0 + i); };
    auto vmax = [&](auto f) {
      for (int i = tid; i < nc; i += kTopkThreads) f(mget(i));
    };
    tau = radix_lower_bound_g(S, vmax, K);
    int cntl = 0;
    for (int i = tid; i < nc; i += kTopkThreads) cntl += mget(i) >= tau;
    int pos;
    cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(cntl, pos, nsch);
    if (nsch <= kListCap) {
      for (int i = tid; i < nc; i += kTopkThreads)
        if (mget(i) >= tau) S.sel[pos++] = (uint32_t)(c0 + i);
    }
    __syncthreads();
  } else if (grouped) {
    for (int gi = tid; gi < ng; gi += kTopkThreads) {
      const float* p = lrow + rks + gi * kG8;
      uint32_t km = 0;
      for (int u = 0; u < kG8; ++u) km = max(km, f2key(p[u]));
      S.sel[gi] = km;
    }
    __syncthreads();
    auto vg = [&](auto f) {
      for (int gi = tid; gi < ng; gi += kTopkThreads) f(S.sel[gi]);
    };
    tau = radix_lower_bound_g(S, vg, K);
  }
  tprof[1] = clock64();

  auto scan_elems = [&](uint32_t thr, auto g) {
    const int nsch_l = nsch;
    const bool grouped_l = grouped;
    if (chunked) {
      if (nsch_l <= kListCap) {
        const int tot_e = nsch_l * 32;
        for (int e0 = 0; e0 < tot_e; e0 += kTopkThreads * 8) {
          float v[8];
          int col[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int e = e0 + u * kTopkThreads + tid;
            col[u] = e < tot_e ? (int)S.sel[e >> 5] * 32 + lane : -1;
            const bool in = col[u] >= rks && col[u] < rke;
            v[u] = in ? __ldcs(lrow + col[u]) : 0.f;
          }
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const bool in = col[u] >= rks && col[u] < rke;
            const uint32_t key = f2key(v[u]);
            g(in && key >= thr, key, (uint32_t)col[u]);
          }
        }
      } else {
        for (int cb = c0 + warp * 4; cb < c1; cb += kWarpsT * 4) {
          float v[4];
          int col[4];
          bool ok[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int c = cb + u;
            ok[u] = c < c1 && __ldg(mrow + c) >= thr;
            col[u] = c * 32 + lane;
            ok[u] = ok[u] && col[u] >= rks && col[u] < rke;
            v[u] = ok[u] ? lrow[col[u]] : 0.f;
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const uint32_t key = f2key(v[u]);
            g(ok[u] && key >= thr, key, (uint32_t)col[u]);
          }
        }
      }
    } else if (grouped_l) {
      for (int gi0 = 0; gi0 < ng; gi0 += kTopkThreads) {
        const int gi = gi0 + tid;
        const bool g_ok = gi < ng && S.sel[gi] >= thr;
        for (int h = 0; h < kG8; h += 8) {
          float v[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) v[u] = g_ok ? lrow[rks + gi * kG8 + h + u] : -INFINITY;
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const uint32_t key = f2key(v[u]);
            g(g_ok && key >= thr, key, (uint32_t)(rks + gi * kG8 + h + u));
          }
        }
      }
      for (int i0 = ng * kG8; i0 < L; i0 += kTopkThreads) {
        const int i = i0 + tid;
        const bool in = i < L;
        const uint32_t key = in ? f2key(lrow[rks + i]) : 0u;
        g(in && key >= thr, key, (uint32_t)(rks + i));
      }
    } else {
      for (int i0 = 0; i0 < L; i0 += kTopkThreads * 4) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int i = i0 + u * kTopkThreads + tid;
          v[u] = i < L ? lrow[rks + i] : 0.f;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int i = i0 + u * kTopkThreads + tid;
          const uint32_t key = f2key(v[u]);
          g(i < L && key >= thr, key, (uint32_t)(rks + i));
        }
      }
    }
  };
  auto gather = [&](uint32_t thr) {
    if (tid == 0) S.cnt = 0;
    __syncthreads();
    scan_elems(thr, [&](bool take, uint32_t key, uint32_t col) { push_cand(S, take, key, col); });
    __syncthreads();
    const int t = S.cnt;
    __syncthreads();
    return t;
  };

  int mode = chunked ? 1 : 0;
  int total = gather(tau);
  tprof[2] = clock64();
  if (total > kCandCap) {
    mode += 2;
    auto vk = [&](auto f) {
      scan_elems(tau, [&](bool take, uint32_t key, uint32_t) {
        if (take) f(key);
      });
    };
    tau = radix_lower_bound_g(S, vk, K);
    total = gather(tau);
  }
  tprof[3] = clock64();
  if (total <= kCandCap) {
    auto vkc = [&](auto f) {
      for (int i = tid; i < total; i += kTopkThreads) f(S.ckey[i], S.ccol[i]);
    };
    select_exact(S, vkc, K);
  } else {
    mode += 2;
    nsch = kListCap + 1;
    if (grouped) {
      grouped = false;
    }
    __syncthreads();
    auto vkc = [&](auto f) {
      scan_elems(tau, [&](bool take, uint32_t key, uint32_t col) {
        if (take) f(key, col);
      });
    };
    select_exact(S, vkc, K);
  }
  tprof[4] = clock64();
  if (stats && tid == 0) {
#pragma unroll
    for (int ph = 0; ph < 4; ++ph) atomicAdd(stats + 8 + ph, (int)((tprof[ph + 1] - tprof[ph]) >> 4));
    atomicAdd(stats + 0, total);
    atomicMax(stats + 1, total);
    atomicAdd(stats + 2 + mode, 1);
  }
  sort_and_store(S, K, orow, rks);
  if (stats && tid == 0) atomicAdd(stats + 12, (int)((clock64() - tprof[4]) >> 4));
}

// ---------------------------------------------------------------------------
// Writer Fast Path:
//  - cand_topk_reg_kernel: 512-thread register-resident exact radix top-2048
//    for N <= 131,072 (reads bkeys once via uint4, 0 stack frame, 0 spills).
//  - cand_topk_kernel: 4-CTA/SM fallback for N > 131,072 (0 stack, 0 spills).
constexpr int kCandOutCap = 2048;
constexpr int kCandSelCap = 2432;
constexpr int kCandGatherCap = 2688;

struct CandTopkSmem {
  uint32_t hist[2048];
  uint32_t sel[kCandSelCap];
  union {
    struct {
      uint32_t ckey[kCandGatherCap];
      uint32_t ccol[kCandGatherCap];
    };
    uint64_t pack[kCandOutCap];
    uint32_t mc[2 * kCandGatherCap];  // 5376 uint32s -> covers up to N = 172,032
  };
  typename cub::BlockScan<int, kTopkThreads>::TempStorage scan;
  int cnt, nsel, bin, above, binc;
};

__device__ __forceinline__ uint32_t load_cand_bkey(const uint32_t* __restrict__ bkrow,
                                                   const float* __restrict__ lrow,
                                                   int rks, int end_eff, int nb_col, int pin_blk,
                                                   bool has_bkeys, int blk) {
  if (blk == pin_blk) return 0xFF800000u;
  if (blk >= nb_col) return 0x007FFFFFu;
  if (has_bkeys) return __ldg(bkrow + blk);
  const int c0 = rks + blk * 8;
  uint32_t bk = 0x007FFFFFu;
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int c = c0 + u;
    if (c < end_eff) bk = max(bk, f2key_cand(__ldg(lrow + c)));
  }
  return bk;
}

__device__ __forceinline__ void push_cand_w(CandTopkSmem& S, bool take, uint32_t key, uint32_t col) {
  const int lane = threadIdx.x & 31;
  const uint32_t bal = __ballot_sync(0xffffffffu, take);
  int b = 0;
  if (lane == 0 && bal) b = atomicAdd(&S.cnt, __popc(bal));
  b = __shfl_sync(0xffffffffu, b, 0);
  const int p = b + __popc(bal & ((1u << lane) - 1u));
  if (take && p < kCandGatherCap) {
    S.ckey[p] = key;
    S.ccol[p] = col;
  }
}

template <int kBits>
__device__ __forceinline__ void cand_hist_begin(CandTopkSmem& S) {
  constexpr int nb = 1 << kBits;
  for (int i = threadIdx.x; i < nb; i += kTopkThreads) S.hist[i] = 0;
  __syncthreads();
}

template <int kBits>
__device__ __forceinline__ int cand_hist_end(CandTopkSmem& S, int k, int& above, int& binc) {
  constexpr int nb = 1 << kBits;
  constexpr int per = nb / kTopkThreads;
  __syncthreads();
  int loc[per];
  int sum = 0;
#pragma unroll
  for (int j = 0; j < per; ++j) {
    loc[j] = S.hist[nb - 1 - (threadIdx.x * per + j)];
    sum += loc[j];
  }
  int excl = 0;
  cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(sum, excl);
  int run = excl;
#pragma unroll
  for (int j = 0; j < per; ++j) {
    if (run < k && run + loc[j] >= k) {
      S.bin = nb - 1 - (threadIdx.x * per + j);
      S.above = run;
      S.binc = loc[j];
    }
    run += loc[j];
  }
  __syncthreads();
  above = S.above;
  binc = S.binc;
  const int d = S.bin;
  __syncthreads();
  return d;
}

__device__ __forceinline__ uint32_t cand_lb_smem(CandTopkSmem& S, const uint32_t* arr, int n, int k) {
  int above = 0, binc = 0;
  cand_hist_begin<11>(S);
  for (int i = threadIdx.x; i < n; i += kTopkThreads) hist_add(S.hist, (arr[i] >> 21) & 2047u);
  const int d1 = cand_hist_end<11>(S, k, above, binc);
  const uint32_t pref = (uint32_t)d1 << 21;
  cand_hist_begin<11>(S);
  for (int i = threadIdx.x; i < n; i += kTopkThreads) {
    uint32_t v = arr[i];
    if ((v & 0xFFE00000u) == pref) hist_add(S.hist, (v >> 10) & 2047u);
  }
  const int d2 = cand_hist_end<11>(S, k - above, above, binc);
  return pref | ((uint32_t)d2 << 10);
}

__device__ __forceinline__ void cand_select_smem(CandTopkSmem& S, int total, int K) {
  const int tid = threadIdx.x, lane = tid & 31;
  uint32_t prefix = 0u, mask = 0u;
  int kk = K, above = 0, binc = 0;
  cand_hist_begin<11>(S);
  for (int i = tid; i < total; i += kTopkThreads) hist_add(S.hist, (S.ckey[i] >> 21) & 2047u);
  int d = cand_hist_end<11>(S, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 21;
  mask |= 0x7FFu << 21;

  cand_hist_begin<11>(S);
  for (int i = tid; i < total; i += kTopkThreads) {
    uint32_t key = S.ckey[i];
    if ((key & mask) == prefix) hist_add(S.hist, (key >> 10) & 2047u);
  }
  d = cand_hist_end<11>(S, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 10;
  mask |= 0x7FFu << 10;

  cand_hist_begin<10>(S);
  for (int i = tid; i < total; i += kTopkThreads) {
    uint32_t key = S.ckey[i];
    if ((key & mask) == prefix) hist_add(S.hist, key & 1023u);
  }
  d = cand_hist_end<10>(S, kk, above, binc);
  kk -= above;
  const uint32_t T = prefix | (uint32_t)d;
  const int G = K - kk;
  const int need = K - G;
  uint32_t colT = 0xFFFFFFFFu;

  if (binc > need) {
    uint32_t cp = 0u, cm = 0u;
    int ckk = need, cab = 0, cbc = 0;
    cand_hist_begin<11>(S);
    for (int i = tid; i < total; i += kTopkThreads) {
      if (S.ckey[i] == T) hist_add(S.hist, ((~S.ccol[i]) >> 21) & 2047u);
    }
    int cd = cand_hist_end<11>(S, ckk, cab, cbc);
    ckk -= cab;
    cp |= (uint32_t)cd << 21;
    cm |= 0x7FFu << 21;

    cand_hist_begin<11>(S);
    for (int i = tid; i < total; i += kTopkThreads) {
      if (S.ckey[i] == T) {
        uint32_t nc = ~S.ccol[i];
        if ((nc & cm) == cp) hist_add(S.hist, (nc >> 10) & 2047u);
      }
    }
    cd = cand_hist_end<11>(S, ckk, cab, cbc);
    ckk -= cab;
    cp |= (uint32_t)cd << 10;
    cm |= 0x7FFu << 10;

    cand_hist_begin<10>(S);
    for (int i = tid; i < total; i += kTopkThreads) {
      if (S.ckey[i] == T) {
        uint32_t nc = ~S.ccol[i];
        if ((nc & cm) == cp) hist_add(S.hist, nc & 1023u);
      }
    }
    cd = cand_hist_end<10>(S, ckk, cab, cbc);
    colT = ~(cp | (uint32_t)cd);
  }

  if (tid == 0) S.nsel = 0;
  __syncthreads();
  for (int i0 = 0; i0 < total; i0 += kTopkThreads) {
    const int i = i0 + tid;
    bool take = false;
    uint32_t col = 0u;
    if (i < total) {
      uint32_t key = S.ckey[i];
      col = S.ccol[i];
      take = (key > T) || (key == T && col <= colT);
    }
    const uint32_t bal = __ballot_sync(0xffffffffu, take);
    int b = 0;
    if (lane == 0 && bal) b = atomicAdd(&S.nsel, __popc(bal));
    b = __shfl_sync(0xffffffffu, b, 0);
    const int p = b + __popc(bal & ((1u << lane) - 1u));
    if (take && p < kCandOutCap) S.sel[p] = col;
  }
  __syncthreads();
}

__device__ __forceinline__ void cand_select_stream(
    CandTopkSmem& S, const uint32_t* __restrict__ bkrow, const float* __restrict__ lrow,
    int rks, int end_eff, int nb_col, int pin_blk, bool has_bkeys, int nb_scan, uint32_t tau, int K) {
  const int tid = threadIdx.x, lane = tid & 31;
  uint32_t prefix = 0u, mask = 0u;
  int kk = K, above = 0, binc = 0;
  cand_hist_begin<11>(S);
  for (int i = tid; i < nb_scan; i += kTopkThreads) {
    uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
    if (key >= tau) hist_add(S.hist, (key >> 21) & 2047u);
  }
  int d = cand_hist_end<11>(S, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 21;
  mask |= 0x7FFu << 21;

  cand_hist_begin<11>(S);
  for (int i = tid; i < nb_scan; i += kTopkThreads) {
    uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
    if (key >= tau && (key & mask) == prefix) hist_add(S.hist, (key >> 10) & 2047u);
  }
  d = cand_hist_end<11>(S, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 10;
  mask |= 0x7FFu << 10;

  cand_hist_begin<10>(S);
  for (int i = tid; i < nb_scan; i += kTopkThreads) {
    uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
    if (key >= tau && (key & mask) == prefix) hist_add(S.hist, key & 1023u);
  }
  d = cand_hist_end<10>(S, kk, above, binc);
  kk -= above;
  const uint32_t T = prefix | (uint32_t)d;
  const int G = K - kk;
  const int need = K - G;
  uint32_t colT = 0xFFFFFFFFu;

  if (binc > need) {
    uint32_t cp = 0u, cm = 0u;
    int ckk = need, cab = 0, cbc = 0;
    cand_hist_begin<11>(S);
    for (int i = tid; i < nb_scan; i += kTopkThreads) {
      uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
      if (key == T) hist_add(S.hist, ((~(uint32_t)i) >> 21) & 2047u);
    }
    int cd = cand_hist_end<11>(S, ckk, cab, cbc);
    ckk -= cab;
    cp |= (uint32_t)cd << 21;
    cm |= 0x7FFu << 21;

    cand_hist_begin<11>(S);
    for (int i = tid; i < nb_scan; i += kTopkThreads) {
      uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
      if (key == T) {
        uint32_t nc = ~(uint32_t)i;
        if ((nc & cm) == cp) hist_add(S.hist, (nc >> 10) & 2047u);
      }
    }
    cd = cand_hist_end<11>(S, ckk, cab, cbc);
    ckk -= cab;
    cp |= (uint32_t)cd << 10;
    cm |= 0x7FFu << 10;

    cand_hist_begin<10>(S);
    for (int i = tid; i < nb_scan; i += kTopkThreads) {
      uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
      if (key == T) {
        uint32_t nc = ~(uint32_t)i;
        if ((nc & cm) == cp) hist_add(S.hist, nc & 1023u);
      }
    }
    cd = cand_hist_end<10>(S, ckk, cab, cbc);
    colT = ~(cp | (uint32_t)cd);
  }

  if (tid == 0) S.nsel = 0;
  __syncthreads();
  for (int i0 = 0; i0 < nb_scan; i0 += kTopkThreads) {
    const int i = i0 + tid;
    bool take = false;
    if (i < nb_scan) {
      uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
      take = (key > T) || (key == T && (uint32_t)i <= colT);
    }
    const uint32_t bal = __ballot_sync(0xffffffffu, take);
    int b = 0;
    if (lane == 0 && bal) b = atomicAdd(&S.nsel, __popc(bal));
    b = __shfl_sync(0xffffffffu, b, 0);
    const int p = b + __popc(bal & ((1u << lane) - 1u));
    if (take && p < kCandOutCap) S.sel[p] = (uint32_t)i;
  }
  __syncthreads();
}

template <bool kSortOrder>
__device__ __forceinline__ void cand_store_smem(CandTopkSmem& S, const uint32_t* __restrict__ bkrow,
                                                const float* __restrict__ lrow, int rks, int end_eff,
                                                int nb_col, int pin_blk, bool has_bkeys, bool from_sel,
                                                int K_act, int K_out, int KP, int* __restrict__ orow) {
  const int tid = threadIdx.x;
  if constexpr (!kSortOrder) {
    // Deterministic ascending valid block order + trailing -1s (0 local memory!).
    for (int i = tid; i < KP; i += kTopkThreads) {
      uint32_t out_v = 0xFFFFFFFFu;
      if (i < K_act) {
        const uint32_t blk = from_sel ? S.sel[i] : (uint32_t)i;
        const uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, (int)blk);
        if (key > 0x007FFFFFu && key <= 0xFF800000u) out_v = blk;
      }
      S.mc[i] = out_v;
    }
    __syncthreads();
    const int half32 = KP >> 1;
    for (int size = 2; size <= KP; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int p = tid; p < half32; p += kTopkThreads) {
          const int i = ((p / stride) * (stride << 1)) + (p & (stride - 1));
          const int j = i ^ stride;
          const uint32_t a = S.mc[i], b = S.mc[j];
          const bool asc = (i & size) == 0;
          if (asc ? (a > b) : (a < b)) {
            S.mc[i] = b;
            S.mc[j] = a;
          }
        }
        __syncthreads();
      }
    }
    for (int i = tid; i < K_out; i += kTopkThreads) {
      orow[i] = (i < K_act && S.mc[i] != 0xFFFFFFFFu) ? (int)S.mc[i] : -1;
    }
    return;
  }
  // Zero-stack-frame shared-memory 64-bit descending bitonic sort (0 local memory!).
  for (int i = tid; i < KP; i += kTopkThreads) {
    if (i < K_act) {
      const uint32_t blk = from_sel ? S.sel[i] : (uint32_t)i;
      const uint32_t key = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, (int)blk);
      S.pack[i] = ((uint64_t)key << 32) | (uint64_t)(~blk);
    } else {
      S.pack[i] = 0ULL;
    }
  }
  __syncthreads();
  const int half = KP >> 1;
  for (int size = 2; size <= KP; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int p = tid; p < half; p += kTopkThreads) {
        const int i = ((p / stride) * (stride << 1)) + (p & (stride - 1));
        const int j = i ^ stride;
        const uint64_t a = S.pack[i];
        const uint64_t b = S.pack[j];
        const bool desc = (i & size) == 0;
        if (desc ? (a < b) : (a > b)) {
          S.pack[i] = b;
          S.pack[j] = a;
        }
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < K_out; i += kTopkThreads) {
    if (i < K_act) {
      const uint64_t pk = S.pack[i];
      const uint32_t key = (uint32_t)(pk >> 32);
      const int blk = (int)(~(uint32_t)pk);
      orow[i] = (key > 0x007FFFFFu && key <= 0xFF800000u) ? blk : -1;
    } else {
      orow[i] = -1;
    }
  }
}

template <bool kSortOrder>
__global__ void __launch_bounds__(kTopkThreads, 4) cand_topk_kernel(
    const float* __restrict__ logits, int64_t ls0,
    const uint32_t* __restrict__ bkeys, int64_t bk_s0,
    const int* __restrict__ ks, const int* __restrict__ ke,
    int* __restrict__ out, int64_t os0, int64_t os1,
    int N, int nblocks, int K_out, bool bkeys_all_rows) {
  extern __shared__ __align__(16) uint8_t smraw[];
  CandTopkSmem& S = *reinterpret_cast<CandTopkSmem*>(smraw);
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int rks = ks[row], rke = ke[row];
  int* orow_base = out + (int64_t)row * os0;

  const int K_sel = min(K_out, nblocks);
  const int end_eff = min(rke, N);
  int nb_col = (rke > rks && end_eff > rks) ? ((end_eff - rks + 7) >> 3) : 0;
  if (nb_col > nblocks) nb_col = nblocks;
  const int pin_blk = (rke > rks) ? ((rke - rks - 1) >> 3) : -1;
  int nb_scan = nb_col;
  if (pin_blk >= 0 && pin_blk < nblocks && pin_blk + 1 > nb_scan) nb_scan = pin_blk + 1;

  if (nb_scan <= 0 || K_sel <= 0) {
    for (int i = tid; i < K_out; i += kTopkThreads) orow_base[i * os1] = -1;
    return;
  }

  const uint32_t* bkrow = bkeys + (int64_t)row * bk_s0;
  const float* lrow = logits + (int64_t)row * ls0;
  const bool has_bkeys = bkeys_all_rows || ((rks & 7) == 0);
  const int K_act = min(K_sel, nb_scan);
  bool from_sel = false;

  if (nb_scan > K_sel) {
    from_sel = true;
    const int ng2 = (nb_scan + 1) >> 1;
    int grp_w = 1;
    int nsg = kCandSelCap + 1;
    uint32_t tau = 0u;

    if (has_bkeys && ng2 >= K_act && ng2 <= 2 * kCandGatherCap) {
      grp_w = 2;
      for (int g = tid; g < ng2; g += kTopkThreads) {
        uint32_t k0 = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, true, 2 * g);
        uint32_t k1 = (2 * g + 1 < nb_scan) ? load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, true, 2 * g + 1)
                                            : 0x007FFFFFu;
        S.mc[g] = max(k0, k1);
      }
      __syncthreads();
      tau = cand_lb_smem(S, S.mc, ng2, K_act);
      int cntl = 0;
      for (int g = tid; g < ng2; g += kTopkThreads) cntl += (S.mc[g] >= tau);
      int pos = 0;
      cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(cntl, pos, nsg);
      if (nsg <= kCandSelCap) {
        for (int g = tid; g < ng2; g += kTopkThreads)
          if (S.mc[g] >= tau) S.sel[pos++] = (uint32_t)g;
      }
      __syncthreads();
    } else {
      int above = 0, binc = 0;
      cand_hist_begin<11>(S);
      for (int i = tid; i < nb_scan; i += kTopkThreads)
        hist_add(S.hist, (load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i) >> 21) & 2047u);
      const int d1 = cand_hist_end<11>(S, K_act, above, binc);
      const uint32_t pref = (uint32_t)d1 << 21;
      cand_hist_begin<11>(S);
      for (int i = tid; i < nb_scan; i += kTopkThreads) {
        uint32_t v = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
        if ((v & 0xFFE00000u) == pref) hist_add(S.hist, (v >> 10) & 2047u);
      }
      const int d2 = cand_hist_end<11>(S, K_act - above, above, binc);
      tau = pref | ((uint32_t)d2 << 10);
    }

    if (tid == 0) S.cnt = 0;
    __syncthreads();
    if (grp_w == 4 && nsg <= kCandSelCap) {
      const int tot_e = nsg * 4;
      for (int e0 = 0; e0 < tot_e; e0 += kTopkThreads * 4) {
        int blk[4];
        uint32_t k[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int e = e0 + u * kTopkThreads + tid;
          blk[u] = (e < tot_e) ? ((int)S.sel[e >> 2] * 4 + (e & 3)) : -1;
          const bool in = (blk[u] >= 0) && (blk[u] < nb_scan);
          k[u] = in ? load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, blk[u]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const bool in = (blk[u] >= 0) && (blk[u] < nb_scan);
          push_cand_w(S, in && k[u] >= tau, k[u], (uint32_t)blk[u]);
        }
      }
    } else if (grp_w == 2 && nsg <= kCandSelCap) {
      const int tot_e = nsg * 2;
      for (int e0 = 0; e0 < tot_e; e0 += kTopkThreads * 4) {
        int blk[4];
        uint32_t k[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int e = e0 + u * kTopkThreads + tid;
          blk[u] = (e < tot_e) ? ((int)S.sel[e >> 1] * 2 + (e & 1)) : -1;
          const bool in = (blk[u] >= 0) && (blk[u] < nb_scan);
          k[u] = in ? load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, blk[u]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const bool in = (blk[u] >= 0) && (blk[u] < nb_scan);
          push_cand_w(S, in && k[u] >= tau, k[u], (uint32_t)blk[u]);
        }
      }
    } else {
      for (int i0 = 0; i0 < nb_scan; i0 += kTopkThreads * 4) {
        int blk[4];
        uint32_t k[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          blk[u] = i0 + u * kTopkThreads + tid;
          const bool in = blk[u] < nb_scan;
          k[u] = in ? load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, blk[u]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const bool in = blk[u] < nb_scan;
          push_cand_w(S, in && k[u] >= tau, k[u], (uint32_t)blk[u]);
        }
      }
    }
    __syncthreads();
    int total = S.cnt;
    __syncthreads();

    if (total > kCandGatherCap) {
      int above = 0, binc = 0;
      cand_hist_begin<11>(S);
      for (int i = tid; i < nb_scan; i += kTopkThreads) {
        uint32_t v = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
        if (v >= tau) hist_add(S.hist, (v >> 21) & 2047u);
      }
      const int d1 = cand_hist_end<11>(S, K_act, above, binc);
      const uint32_t pref = (uint32_t)d1 << 21;
      cand_hist_begin<11>(S);
      for (int i = tid; i < nb_scan; i += kTopkThreads) {
        uint32_t v = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, i);
        if (v >= tau && (v & 0xFFE00000u) == pref) hist_add(S.hist, (v >> 10) & 2047u);
      }
      const int d2 = cand_hist_end<11>(S, K_act - above, above, binc);
      tau = pref | ((uint32_t)d2 << 10);

      if (tid == 0) S.cnt = 0;
      __syncthreads();
      for (int i0 = 0; i0 < nb_scan; i0 += kTopkThreads * 4) {
        int blk[4];
        uint32_t k[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          blk[u] = i0 + u * kTopkThreads + tid;
          const bool in = blk[u] < nb_scan;
          k[u] = in ? load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, blk[u]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const bool in = blk[u] < nb_scan;
          push_cand_w(S, in && k[u] >= tau, k[u], (uint32_t)blk[u]);
        }
      }
      __syncthreads();
      total = S.cnt;
      __syncthreads();
    }

    if (total <= kCandGatherCap) {
      cand_select_smem(S, total, K_act);
    } else {
      cand_select_stream(S, bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, nb_scan, tau, K_act);
    }
  }

  int KP = kTopkThreads;
  while (KP < K_act) KP <<= 1;
  const bool unit_stride = (os1 == 1);
  int* dst = unit_stride ? orow_base : reinterpret_cast<int*>(S.hist);
  cand_store_smem<kSortOrder>(S, bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, from_sel, K_act, unit_stride ? K_out : K_act, KP, dst);
  if (!unit_stride) {
    __syncthreads();
    for (int i = tid; i < K_out; i += kTopkThreads) {
      orow_base[i * os1] = (i < K_act) ? dst[i] : -1;
    }
  }
}

// ---------------------------------------------------------------------------
// Reader Compact Path:
// 1. reader_compact_blocks_kernel: deduplicates & sorts active 8-key candidate
//    blocks per row (matching _candidate_flags_kernel + _mask_candidates_kernel
//    including the width-1 edge rule).
// 2. reader_compact_logits_kernel: stages only active 8-key blocks via cp.async
//    and computes bit-identical logits + 32-element compact chunk maxima.
// 3. reader_compact_topk_kernel: exact pruned radix top-k over compact logits.

__global__ void __launch_bounds__(256, 4) reader_compact_blocks_kernel(
    const int* __restrict__ cands, int64_t cs0, int64_t cs1, int K_cand,
    const int* __restrict__ ks, const int* __restrict__ ke,
    int* __restrict__ comp_blks, int64_t bs0, int max_blks,
    int* __restrict__ comp_ngroups,
    int N, int nblocks, int topk_tokens, bool skip_short_rows) {
  extern __shared__ __align__(16) uint8_t s_flags[];
  __shared__ typename cub::BlockScan<int, 256>::TempStorage scan_smem;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int rks = ks[row], rke = ke[row];
  if (rke <= rks || (skip_short_rows && (rke - rks) <= topk_tokens)) {
    if (tid == 0) comp_ngroups[row] = 0;
    return;
  }
  for (int i = tid; i <= nblocks; i += 256) s_flags[i] = 0;
  __syncthreads();

  const int* crow = cands + (int64_t)row * cs0;
  for (int j = tid; j < K_cand; j += 256) {
    const int b = crow[j * cs1];
    if (b >= 0) {
      const int64_t clamped = ((int64_t)rks + (int64_t)b * 8 >= (int64_t)N) ? (int64_t)nblocks : (int64_t)b;
      if (clamped <= nblocks) s_flags[clamped] = 1;
    }
  }
  __syncthreads();

  int edge_only_blk = -1;
  if (s_flags[nblocks] != 0 && (N - 1) >= rks && (N - 1) < rke) {
    const int eb = (N - 1 - rks) >> 3;
    if (eb >= 0 && eb < nblocks && s_flags[eb] == 0) edge_only_blk = eb;
  }

  const int end_eff = min(rke, N);
  int nb_col = (end_eff > rks) ? ((end_eff - rks + 7) >> 3) : 0;
  if (nb_col > nblocks) nb_col = nblocks;

  int* brow = comp_blks + (int64_t)row * bs0;
  int out_pos = 0;
  for (int b0 = 0; b0 < nb_col; b0 += 256) {
    const int blk = b0 + tid;
    const bool is_full = (blk < nb_col) && (s_flags[blk] != 0);
    const bool is_edge = (blk < nb_col) && (blk == edge_only_blk);
    const int keep = (is_full || is_edge) ? 1 : 0;
    int rel = 0, step_tot = 0;
    cub::BlockScan<int, 256>(scan_smem).ExclusiveSum(keep, rel, step_tot);
    if (keep && (out_pos + rel) < max_blks) {
      brow[out_pos + rel] = is_edge ? (blk | 0x40000000) : blk;
    }
    out_pos += step_tot;
    __syncthreads();
  }
  const int U = min(out_pos, max_blks);
  const int ng = (U + 3) >> 2;
  if (tid < 4 && (U + tid) < ng * 4) brow[U + tid] = -1;
  if (tid == 0) comp_ngroups[row] = ng;
}

struct ReaderCompactArgs {
  const __nv_bfloat16* q;
  int64_t q_s0;
  const __nv_bfloat16* k;
  const float* ksc;
  const float* w;
  int64_t w_s0, w_s1;
  const int* ks;
  const int* ke;
  const int* comp_blks;
  int64_t blk_s0;
  const int* comp_ngroups;
  float* comp_out;
  int64_t cout_s0;
  uint32_t* comp_mx;
  int64_t cmx_s0;
  float* full_out;  // optional [R, N] for bitwise full-logits testing
  int64_t fout_s0;
  int R, N, groups_per_cta;
};

__device__ __forceinline__ void rc_tile_epi(const float (&acc)[2][4][4], int gi, bool present,
                                            float w0, float w1, float w2, float w3, int lane,
                                            int rks, int rke, const ReaderCompactArgs& a,
                                            const int* cblks_row, float* corow, uint32_t* cmrow,
                                            float* forow) {
  if (!present) return;
  float outv = head_reduce_epi(acc, w0, w1, w2, w3, lane);
  const int nt = lane >> 3;
  const int off_in_blk = 2 * (lane & 3) + ((lane >> 2) & 1);
  const int raw_b = __ldg(cblks_row + gi * 4 + nt);
  const int blk = raw_b & 0x3FFFFFFF;
  const bool edge_only = (raw_b & 0x40000000) != 0;
  const int true_col = rks + blk * 8 + off_in_blk;
  bool valid_col = (raw_b >= 0) && (true_col >= rks) && (true_col < rke) && (true_col < a.N);
  if (edge_only) valid_col = valid_col && (true_col == a.N - 1);
  const float sc = __ldg(a.ksc + (valid_col ? true_col : 0));
  outv = __fmul_rn(outv, sc);
  outv = valid_col ? outv : -INFINITY;
  corow[gi * 32 + nt * 8 + off_in_blk] = outv;
  if (forow != nullptr && valid_col) forow[true_col] = outv;
  const uint32_t key = __reduce_max_sync(0xffffffffu, f2key(outv));
  if (lane == 0) cmrow[gi] = key;
}

template <int kRCWarps, int kRCStages>
__global__ void __launch_bounds__(kRCWarps * 32, (kRCWarps == 4 && kRCStages == 2) ? 2 : 1)
reader_compact_logits_kernel(const ReaderCompactArgs a) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int row = blockIdx.x;
  if (row >= a.R) return;
  const int ng = a.comp_ngroups[row];
  const int g_begin = blockIdx.y * a.groups_per_cta;
  const int g_end = min(ng, g_begin + a.groups_per_cta);
  if (g_begin >= g_end) return;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int rks = a.ks[row], rke = a.ke[row];
  const int nst = (g_end - g_begin + kRCWarps - 1) / kRCWarps;

  uint32_t A[2][8][4];
  const uint32_t* qr = reinterpret_cast<const uint32_t*>(a.q + (int64_t)row * a.q_s0);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      const int h0 = mt * 16 + g, d0 = kc * 16 + 2 * t;
      A[mt][kc][0] = __ldg(qr + (h0 * kD + d0) / 2);
      A[mt][kc][1] = __ldg(qr + ((h0 + 8) * kD + d0) / 2);
      A[mt][kc][2] = __ldg(qr + (h0 * kD + d0 + 8) / 2);
      A[mt][kc][3] = __ldg(qr + ((h0 + 8) * kD + d0 + 8) / 2);
    }
  const float* wr = a.w + (int64_t)row * a.w_s0;
  const float w0 = wr[g * a.w_s1];
  const float w1 = wr[(g + 8) * a.w_s1];
  const float w2 = wr[(16 + g) * a.w_s1];
  const float w3 = wr[(24 + g) * a.w_s1];

  constexpr int kRCStageBytes = kRCWarps * kTileBytes;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int* cblks_row = a.comp_blks + (int64_t)row * a.blk_s0;

  auto issue = [&](int s) {
    if (s < nst) {
      const uint32_t st = sbase + (s % kRCStages) * kRCStageBytes;
      const int base_gi = g_begin + s * kRCWarps;
      // Total 8-key blocks in one stage = 4 * kRCWarps; total 16B chunks = 4 * kRCWarps * 128.
      constexpr int kChunks = 4 * kRCWarps * 128;
      constexpr int kThreads = kRCWarps * 32;
#pragma unroll
      for (int it = 0; it < kChunks / kThreads; ++it) {
        const int p = tid + it * kThreads;
        const int bi = p >> 7;                // 0 .. 4*kRCWarps - 1
        const int rem = p & 127;
        const int key_in_blk = rem >> 4;      // 0 .. 7
        const int ch = rem & 15;              // 0 .. 15
        const int gi = base_gi + (bi >> 2);
        const int nt = bi & 3;
        const int raw_b = (gi < g_end) ? __ldg(cblks_row + gi * 4 + nt) : -1;
        const int blk = raw_b & 0x3FFFFFFF;
        const int gk = rks + blk * 8 + key_in_blk;
        const bool v = (raw_b >= 0) && (gk >= 0) && (gk < a.N);
        const int smem_key = bi * 8 + key_in_blk;
        const __nv_bfloat16* src = a.k + (int64_t)(v ? gk : 0) * kD + ch * 8;
        cp_async16(st + smem_key * 256 + ((ch ^ key_in_blk) << 4), src, v);
      }
    }
    cp_commit();
  };

#pragma unroll
  for (int i = 0; i < kRCStages - 1; ++i) issue(i);

  const int lm_mat = lane >> 3, lm_r = lane & 7;
  float* corow = a.comp_out + (int64_t)row * a.cout_s0;
  uint32_t* cmrow = a.comp_mx + (int64_t)row * a.cmx_s0;
  float* forow = a.full_out ? (a.full_out + (int64_t)row * a.fout_s0) : nullptr;

  float accA[2][4][4], accB[2][4][4];
  int prev_gi = 0;
  bool prev_present = false;
  for (int s = 0; s < nst; ++s) {
    cp_wait<kRCStages - 2>();
    __syncthreads();
    issue(s + kRCStages - 1);
    const uint32_t st = sbase + (s % kRCStages) * kRCStageBytes;
    const int gi = g_begin + s * kRCWarps + warp;
    const bool present = gi < g_end;
    if ((s & 1) == 0) {
      if (present) tile_mma<false>(accA, A, st, warp, lm_mat, lm_r);
      if (s > 0) rc_tile_epi(accB, prev_gi, prev_present, w0, w1, w2, w3, lane, rks, rke, a, cblks_row, corow, cmrow, forow);
    } else {
      if (present) tile_mma<false>(accB, A, st, warp, lm_mat, lm_r);
      rc_tile_epi(accA, prev_gi, prev_present, w0, w1, w2, w3, lane, rks, rke, a, cblks_row, corow, cmrow, forow);
    }
    prev_gi = gi;
    prev_present = present;
  }
  if (((nst - 1) & 1) == 0) {
    rc_tile_epi(accA, prev_gi, prev_present, w0, w1, w2, w3, lane, rks, rke, a, cblks_row, corow, cmrow, forow);
  } else {
    rc_tile_epi(accB, prev_gi, prev_present, w0, w1, w2, w3, lane, rks, rke, a, cblks_row, corow, cmrow, forow);
  }
  cp_wait<0>();
}

__global__ void __launch_bounds__(kTopkThreads, 4) reader_compact_topk_kernel(
    const float* __restrict__ comp_logits, int64_t cls0,
    const uint32_t* __restrict__ comp_mx, int64_t cms0,
    const int* __restrict__ comp_blks, int64_t cbs0,
    const int* __restrict__ comp_ngroups,
    const int* __restrict__ ks, const int* __restrict__ ke,
    int* __restrict__ out, int64_t os0, int N, int K) {
  extern __shared__ __align__(16) uint8_t smraw[];
  TopkSmem& S = *reinterpret_cast<TopkSmem*>(smraw);
  const int row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31;
  const int rks = ks[row], rke = ke[row];
  const int L = rke - rks;
  int* orow = out + (int64_t)row * os0;
  if (L <= K) {
    for (int i = tid; i < K; i += kTopkThreads) orow[i] = i < L ? i : -1;
    return;
  }
  const int ng = comp_ngroups[row];
  const int nc_elems = ng * 32;
  const float* clrow = comp_logits + (int64_t)row * cls0;
  const uint32_t* cmrow = comp_mx + (int64_t)row * cms0;
  const int* cblks_row = comp_blks + (int64_t)row * cbs0;

  auto elem_col = [&](int ce) -> uint32_t {
    const int raw_b = __ldg(cblks_row + (ce >> 3));
    return (uint32_t)(rks + (raw_b & 0x3FFFFFFF) * 8 + (ce & 7));
  };

  // Count finite (> -INFINITY) active elements or check if chunked pruning guarantees >= K finite elements.
  uint32_t tau = 0x00800000u;
  int nsch = 0;
  bool chunked = (ng >= K) && (ng <= 2 * kCandCap);
  bool enough_finite = false;

  if (chunked) {
    uint32_t* mc = S.ckey;
    for (int i = tid; i < ng; i += kTopkThreads) mc[i] = __ldg(cmrow + i);
    __syncthreads();
    auto vmax = [&](auto f) {
      for (int i = tid; i < ng; i += kTopkThreads) f(mc[i]);
    };
    uint32_t lb = radix_lower_bound_g(S, vmax, K);
    if (lb >= 0x00800000u) {
      enough_finite = true;
      tau = lb;
      int cntl = 0;
      for (int i = tid; i < ng; i += kTopkThreads) cntl += (mc[i] >= tau);
      int pos = 0;
      cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(cntl, pos, nsch);
      if (nsch <= kListCap) {
        for (int i = tid; i < ng; i += kTopkThreads)
          if (mc[i] >= tau) S.sel[pos++] = (uint32_t)i;
      }
      __syncthreads();
    } else {
      chunked = false;
    }
  }

  if (!enough_finite) {
    int my_fin = 0;
    for (int ce = tid; ce < nc_elems; ce += kTopkThreads) {
      if (f2key(__ldg(clrow + ce)) > 0x007FFFFFu) ++my_fin;
    }
    int pos_unused = 0, tot_fin = 0;
    cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(my_fin, pos_unused, tot_fin);
    __syncthreads();
    if (tot_fin < K) {
      // Fewer than K columns > -INFINITY while L > K: collect all tot_fin finite columns,
      // then fill remaining K - tot_fin with lowest unused columns in [rks, rks + K).
      if (tid == 0) S.nsel = 0;
      for (int i = tid; i < K; i += kTopkThreads) S.hist[i] = 0;
      __syncthreads();
      for (int ce = tid; ce < nc_elems; ce += kTopkThreads) {
        if (f2key(__ldg(clrow + ce)) > 0x007FFFFFu) {
          uint32_t c = elem_col(ce);
          int p = atomicAdd(&S.nsel, 1);
          S.sel[p] = c;
          int rel = (int)c - rks;
          if (rel >= 0 && rel < K) S.hist[rel] = 1;
        }
      }
      __syncthreads();
      const int need_pad = K - tot_fin;
      int run_off = 0;
      for (int d0 = 0; d0 < K; d0 += kTopkThreads) {
        const int d = d0 + tid;
        const int avail = (d < K && S.hist[d] == 0) ? 1 : 0;
        int rel = 0, stot = 0;
        cub::BlockScan<int, kTopkThreads>(S.scan).ExclusiveSum(avail, rel, stot);
        if (avail && (run_off + rel) < need_pad) {
          S.sel[tot_fin + run_off + rel] = (uint32_t)(rks + d);
        }
        run_off += stot;
        __syncthreads();
      }
      sort_and_store(S, K, orow, rks);
      return;
    }
    // tot_fin >= K, but ng < K (or lb < 0x00800000): find 22-bit lower bound directly on finite elements.
    auto vfin = [&](auto f) {
      for (int ce = tid; ce < nc_elems; ce += kTopkThreads) {
        uint32_t k = f2key(__ldg(clrow + ce));
        if (k > 0x007FFFFFu) f(k);
      }
    };
    tau = max(0x00800000u, radix_lower_bound_g(S, vfin, K));
  }

  auto scan_elems = [&](uint32_t thr, auto g) {
    const int nsch_l = nsch;
    if (chunked && nsch_l <= kListCap) {
      const int tot_e = nsch_l * 32;
      for (int e0 = 0; e0 < tot_e; e0 += kTopkThreads * 4) {
        float v[4];
        int ce[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int e = e0 + u * kTopkThreads + tid;
          ce[u] = e < tot_e ? ((int)S.sel[e >> 5] * 32 + lane) : -1;
          v[u] = ce[u] >= 0 ? __ldg(clrow + ce[u]) : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const uint32_t key = f2key(v[u]);
          const bool take = (ce[u] >= 0) && (key >= thr) && (key > 0x007FFFFFu);
          g(take, key, take ? elem_col(ce[u]) : 0u);
        }
      }
    } else {
      for (int ce0 = 0; ce0 < nc_elems; ce0 += kTopkThreads * 4) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int ce = ce0 + u * kTopkThreads + tid;
          v[u] = ce < nc_elems ? __ldg(clrow + ce) : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int ce = ce0 + u * kTopkThreads + tid;
          const uint32_t key = f2key(v[u]);
          const bool take = (ce < nc_elems) && (key >= thr) && (key > 0x007FFFFFu);
          g(take, key, take ? elem_col(ce) : 0u);
        }
      }
    }
  };

  auto gather = [&](uint32_t thr) {
    if (tid == 0) S.cnt = 0;
    __syncthreads();
    scan_elems(thr, [&](bool take, uint32_t key, uint32_t col) { push_cand(S, take, key, col); });
    __syncthreads();
    const int t = S.cnt;
    __syncthreads();
    return t;
  };

  int total = gather(tau);
  if (total > kCandCap) {
    auto vk = [&](auto f) {
      scan_elems(tau, [&](bool take, uint32_t key, uint32_t) {
        if (take) f(key);
      });
    };
    tau = max(0x00800000u, radix_lower_bound_g(S, vk, K));
    total = gather(tau);
  }
  if (total <= kCandCap) {
    auto vkc = [&](auto f) {
      for (int i = tid; i < total; i += kTopkThreads) f(S.ckey[i], S.ccol[i]);
    };
    select_exact(S, vkc, K);
  } else {
    nsch = kListCap + 1;
    __syncthreads();
    auto vkc = [&](auto f) {
      scan_elems(tau, [&](bool take, uint32_t key, uint32_t col) {
        if (take) f(key, col);
      });
    };
    select_exact(S, vkc, K);
  }
  sort_and_store(S, K, orow, rks);
}

// ---------------------------------------------------------------------------
void check_cuda(const at::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda(), n, " must be CUDA"); }

void dsv41_idx_decode_fp8(const at::Tensor& in, const at::Tensor& lut, at::Tensor& out) {
  check_cuda(in, "in");
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "contiguous");
  TORCH_CHECK(in.numel() == out.numel() && out.scalar_type() == at::kBFloat16 && in.element_size() == 1);
  TORCH_CHECK(in.numel() % 16 == 0, "numel % 16");
  TORCH_CHECK(lut.numel() == 256 && lut.element_size() == 2);
  const c10::cuda::CUDAGuard guard(in.device());
  int64_t n16 = in.numel() / 16;
  if (n16 == 0) return;
  int blocks = (int)std::min<int64_t>((n16 + 255) / 256, 74 * 16);
  decode_fp8_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(in.data_ptr()), reinterpret_cast<uint16_t*>(out.data_ptr()),
      reinterpret_cast<const uint16_t*>(lut.data_ptr()), n16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_logits_impl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                           const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx,
                           const c10::optional<at::Tensor>& flags, const c10::optional<at::Tensor>& bkeys,
                           int64_t cand_bs, int64_t kpc, int64_t dbg) {
  check_cuda(q, "q");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kH && q.size(2) == kD && q.stride(2) == 1 && q.stride(1) == kD);
  TORCH_CHECK(k.dim() == 2 && k.size(1) == kD && k.is_contiguous());
  TORCH_CHECK(ksc.scalar_type() == at::kFloat && ksc.is_contiguous() && ksc.numel() >= k.size(0));
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2 && w.size(1) == kH);
  TORCH_CHECK(ks.scalar_type() == at::kInt && ke.scalar_type() == at::kInt && ks.is_contiguous() && ke.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 && out.stride(1) == 1);
  TORCH_CHECK(mx.scalar_type() == at::kInt && mx.dim() == 2 && mx.stride(1) == 1);
  const int R = (int)q.size(0), N = (int)k.size(0);
  TORCH_CHECK(out.size(0) >= R && out.size(1) >= N && mx.size(0) >= R && mx.size(1) >= (N + 31) / 32);
  TORCH_CHECK(kpc > 0 && kpc % kTN == 0);
  if (R == 0 || N == 0) return;
  const c10::cuda::CUDAGuard guard(q.device());
  LogitsArgs a;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.q_s0 = q.stride(0);
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.ksc = ksc.data_ptr<float>();
  a.w = w.data_ptr<float>();
  a.w_s0 = w.stride(0);
  a.w_s1 = w.stride(1);
  a.ks = ks.data_ptr<int>();
  a.ke = ke.data_ptr<int>();
  a.out = out.data_ptr<float>();
  a.out_s0 = out.stride(0);
  a.mx = reinterpret_cast<uint32_t*>(mx.data_ptr<int>());
  a.mx_s0 = mx.stride(0);
  a.bkeys = nullptr;
  a.bk_s0 = 0;
  a.flags = nullptr;
  a.fl_s0 = 0;
  a.nblocks = 0;
  a.cand_bs = (int)cand_bs;
  a.R = R;
  a.N = N;
  a.kpc = (int)kpc;
  const bool has_flags = flags.has_value() && flags->defined();
  const bool has_bkeys = bkeys.has_value() && bkeys->defined();
  if (has_flags) {
    TORCH_CHECK(flags->element_size() == 1 && flags->dim() == 2 && flags->stride(1) == 1 && cand_bs > 0);
    a.flags = reinterpret_cast<const uint8_t*>(flags->data_ptr());
    a.fl_s0 = flags->stride(0);
    a.nblocks = (int)((N + cand_bs - 1) / cand_bs);
    TORCH_CHECK(flags->size(1) >= a.nblocks + 1);
  }
  if (has_bkeys) {
    TORCH_CHECK(!has_flags && cand_bs == 8);
    TORCH_CHECK(bkeys->scalar_type() == at::kInt && bkeys->dim() == 2 && bkeys->stride(1) == 1);
    a.nblocks = (int)((N + 7) / 8);
    TORCH_CHECK(bkeys->size(0) >= R && bkeys->size(1) >= a.nblocks);
    a.bkeys = reinterpret_cast<uint32_t*>(bkeys->data_ptr<int>());
    a.bk_s0 = bkeys->stride(0);
  }
  dim3 grid((R + kWarps - 1) / kWarps, (N + kpc - 1) / kpc);
  TORCH_CHECK(grid.y <= 65535);
  const int smem = kStages * kStageBytes;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dbg == 3) {
    cudaFuncSetAttribute(logits_kernel<false, 3, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 3, false><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 4) {
    cudaFuncSetAttribute(logits_kernel<false, 4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 4, false><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 1) {
    cudaFuncSetAttribute(logits_kernel<false, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 1, false><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 2) {
    cudaFuncSetAttribute(logits_kernel<false, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 2, false><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (has_bkeys) {
    cudaFuncSetAttribute(logits_kernel<false, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 0, true><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (has_flags) {
    cudaFuncSetAttribute(logits_kernel<true, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<true, 0, false><<<grid, kWarps * 32, smem, stream>>>(a);
  } else {
    cudaFuncSetAttribute(logits_kernel<false, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 0, false><<<grid, kWarps * 32, smem, stream>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_logits(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                      const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx,
                      const c10::optional<at::Tensor>& flags, int64_t cand_bs, int64_t kpc) {
  dsv41_idx_logits_impl(q, k, ksc, w, ks, ke, out, mx, flags, c10::nullopt, cand_bs, kpc, 0);
}

void dsv41_idx_logits_writer(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                             const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx,
                             at::Tensor& bkeys, int64_t cand_bs, int64_t kpc) {
  dsv41_idx_logits_impl(q, k, ksc, w, ks, ke, out, mx, c10::nullopt, bkeys, cand_bs, kpc, 0);
}

void dsv41_idx_logits_dbg(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                          const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx,
                          int64_t kpc, int64_t dbg) {
  dsv41_idx_logits_impl(q, k, ksc, w, ks, ke, out, mx, c10::nullopt, c10::nullopt, 0, kpc, dbg);
}

constexpr int kRegThreads = 512;
constexpr int kRegVecs = 8;  // 512 * 8 * 4 = 16,384 blocks (up to N = 131,072)

struct CandRegSmem {
  union {
    uint32_t hist[2048];
    uint64_t pack[kCandOutCap];
  };
  typename cub::BlockScan<int, kRegThreads>::TempStorage scan;
  int nsel, bin, above, binc;
};

template <int kBits>
__device__ __forceinline__ void reg_hist_begin(CandRegSmem& S) {
  constexpr int nb = 1 << kBits;
  for (int i = threadIdx.x; i < nb; i += kRegThreads) S.hist[i] = 0;
  __syncthreads();
}

template <int kBits>
__device__ __forceinline__ int reg_hist_end(CandRegSmem& S, int k, int& above, int& binc) {
  constexpr int nb = 1 << kBits;
  constexpr int per = nb / kRegThreads;
  __syncthreads();
  int loc[per];
  int sum = 0;
#pragma unroll
  for (int j = 0; j < per; ++j) {
    loc[j] = S.hist[nb - 1 - (threadIdx.x * per + j)];
    sum += loc[j];
  }
  int excl = 0;
  cub::BlockScan<int, kRegThreads>(S.scan).ExclusiveSum(sum, excl);
  int run = excl;
#pragma unroll
  for (int j = 0; j < per; ++j) {
    if (run < k && run + loc[j] >= k) {
      S.bin = nb - 1 - (threadIdx.x * per + j);
      S.above = run;
      S.binc = loc[j];
    }
    run += loc[j];
  }
  __syncthreads();
  above = S.above;
  binc = S.binc;
  const int d = S.bin;
  __syncthreads();
  return d;
}

template <int PER>
__device__ __forceinline__ void reg_bitonic_sort_store(CandRegSmem& S, int K_act, int K_out, int* __restrict__ orow) {
  const int tid = threadIdx.x;
  constexpr int KP = PER * kRegThreads;
  uint64_t x[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = j * kRegThreads + tid;
    x[j] = (i < K_act) ? S.pack[i] : 0ULL;
  }
  __syncthreads();
  for (int size = 2; size <= KP; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      uint64_t y[PER];
      if (PER == 4 && stride == 1024) {
        y[0] = x[2]; y[1] = x[3]; y[2] = x[0]; y[3] = x[1];
      } else if (PER >= 2 && stride == 512) {
        y[0] = x[1]; y[1] = x[0];
        if constexpr (PER == 4) { y[2] = x[3]; y[3] = x[2]; }
      } else if (stride >= 32) {
#pragma unroll
        for (int j = 0; j < PER; ++j) S.pack[j * kRegThreads + tid] = x[j];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PER; ++j) y[j] = S.pack[j * kRegThreads + (tid ^ stride)];
        __syncthreads();
      } else {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
          uint32_t lo32 = __shfl_xor_sync(0xffffffffu, (uint32_t)x[j], stride);
          uint32_t hi32 = __shfl_xor_sync(0xffffffffu, (uint32_t)(x[j] >> 32), stride);
          y[j] = ((uint64_t)hi32 << 32) | lo32;
        }
      }
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const int i = j * kRegThreads + tid;
        const bool up = (i & size) == 0;
        const bool lower = (i & stride) == 0;
        x[j] = (lower == up) ? max(x[j], y[j]) : min(x[j], y[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = j * kRegThreads + tid;
    if (i < K_act) {
      const uint32_t key = (uint32_t)(x[j] >> 32);
      const int blk = (int)(~(uint32_t)x[j]);
      orow[i] = (key > 0x007FFFFFu && key <= 0xFF800000u) ? blk : -1;
    } else if (i < K_out) {
      orow[i] = -1;
    }
  }
  for (int i = KP + tid; i < K_out; i += kRegThreads) orow[i] = -1;
}

template <bool kSortOrder>
__global__ void __launch_bounds__(kRegThreads, 1) cand_topk_reg_kernel(
    const float* __restrict__ logits, int64_t ls0,
    const uint32_t* __restrict__ bkeys, int64_t bk_s0,
    const int* __restrict__ ks, const int* __restrict__ ke,
    int* __restrict__ out, int64_t os0, int64_t os1,
    int N, int nblocks, int K_out, bool bkeys_all_rows) {
  extern __shared__ __align__(16) uint8_t smraw[];
  CandRegSmem& S = *reinterpret_cast<CandRegSmem*>(smraw);
  const int row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31;
  const int rks = ks[row], rke = ke[row];
  int* orow_base = out + (int64_t)row * os0;

  const int K_sel = min(K_out, nblocks);
  const int end_eff = min(rke, N);
  int nb_col = (rke > rks && end_eff > rks) ? ((end_eff - rks + 7) >> 3) : 0;
  if (nb_col > nblocks) nb_col = nblocks;
  const int pin_blk = (rke > rks) ? ((rke - rks - 1) >> 3) : -1;
  int nb_scan = nb_col;
  if (pin_blk >= 0 && pin_blk < nblocks && pin_blk + 1 > nb_scan) nb_scan = pin_blk + 1;

  if (nb_scan <= 0 || K_sel <= 0) {
    for (int i = tid; i < K_out; i += kRegThreads) orow_base[i * os1] = -1;
    return;
  }

  const uint32_t* bkrow = bkeys + (int64_t)row * bk_s0;
  const float* lrow = logits + (int64_t)row * ls0;
  const bool has_bkeys = bkeys_all_rows || ((rks & 7) == 0);
  const bool vec_ok = has_bkeys && ((reinterpret_cast<uintptr_t>(bkrow) & 15u) == 0u);
  const int K_act = min(K_sel, nb_scan);

  uint32_t rk[kRegVecs][4];
#pragma unroll
  for (int u = 0; u < kRegVecs; ++u) {
    const int vec_i = u * kRegThreads + tid;
    const int blk0 = vec_i * 4;
    uint32_t arr[4] = {0x007FFFFFu, 0x007FFFFFu, 0x007FFFFFu, 0x007FFFFFu};
    if (blk0 < nb_col) {
      if (vec_ok && blk0 + 4 <= nb_col) {
        uint4 v4 = __ldg(reinterpret_cast<const uint4*>(bkrow) + vec_i);
        arr[0] = v4.x; arr[1] = v4.y; arr[2] = v4.z; arr[3] = v4.w;
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (blk0 + c < nb_col) {
            arr[c] = load_cand_bkey(bkrow, lrow, rks, end_eff, nb_col, pin_blk, has_bkeys, blk0 + c);
          }
        }
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int blk = blk0 + c;
      if (blk == pin_blk) arr[c] = 0xFF800000u;
      rk[u][c] = arr[c];
    }
  }

  if (nb_scan <= K_sel) {
    if constexpr (!kSortOrder) {
      int out_off = 0;
#pragma unroll
      for (int u = 0; u < kRegVecs; ++u) {
        const int blk0 = (u * kRegThreads + tid) * 4;
        if (u * kRegThreads * 4 >= K_act) break;
        bool v_ok[4];
        int my_cnt = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int blk = blk0 + c;
          const uint32_t key = rk[u][c];
          v_ok[c] = (blk < K_act) && (key > 0x007FFFFFu) && (key <= 0xFF800000u);
          my_cnt += v_ok[c] ? 1 : 0;
        }
        int pos = 0, stot = 0;
        cub::BlockScan<int, kRegThreads>(S.scan).ExclusiveSum(my_cnt, pos, stot);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (v_ok[c] && (out_off + pos) < K_out) {
            orow_base[(out_off + pos) * os1] = blk0 + c;
            ++pos;
          }
        }
        out_off += stot;
        __syncthreads();
      }
      for (int i = out_off + tid; i < K_out; i += kRegThreads) orow_base[i * os1] = -1;
      return;
    }
#pragma unroll
    for (int u = 0; u < kRegVecs; ++u) {
      const int blk0 = (u * kRegThreads + tid) * 4;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int blk = blk0 + c;
        if (blk < K_act) S.pack[blk] = ((uint64_t)rk[u][c] << 32) | (uint64_t)(~(uint32_t)blk);
      }
    }
    __syncthreads();
  } else {
    uint32_t prefix = 0u, mask = 0u;
    int kk = K_act, above = 0, binc = 0;

    reg_hist_begin<11>(S);
#pragma unroll
    for (int u = 0; u < kRegVecs; ++u) {
      const int blk0 = (u * kRegThreads + tid) * 4;
      if (blk0 < nb_scan) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (blk0 + c < nb_scan) hist_add(S.hist, (rk[u][c] >> 21) & 2047u);
        }
      }
    }
    int d = reg_hist_end<11>(S, kk, above, binc);
    kk -= above;
    prefix |= (uint32_t)d << 21;
    mask |= 0x7FFu << 21;

    reg_hist_begin<11>(S);
#pragma unroll
    for (int u = 0; u < kRegVecs; ++u) {
      const int blk0 = (u * kRegThreads + tid) * 4;
      if (blk0 < nb_scan) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (blk0 + c < nb_scan && (rk[u][c] & mask) == prefix) hist_add(S.hist, (rk[u][c] >> 10) & 2047u);
        }
      }
    }
    d = reg_hist_end<11>(S, kk, above, binc);
    kk -= above;
    prefix |= (uint32_t)d << 10;
    mask |= 0x7FFu << 10;

    reg_hist_begin<10>(S);
#pragma unroll
    for (int u = 0; u < kRegVecs; ++u) {
      const int blk0 = (u * kRegThreads + tid) * 4;
      if (blk0 < nb_scan) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (blk0 + c < nb_scan && (rk[u][c] & mask) == prefix) hist_add(S.hist, rk[u][c] & 1023u);
        }
      }
    }
    d = reg_hist_end<10>(S, kk, above, binc);
    kk -= above;
    const uint32_t T = prefix | (uint32_t)d;
    const int need = K_act - (K_act - kk);
    uint32_t colT = 0xFFFFFFFFu;

    if (binc > need) {
      uint32_t cp = 0u, cm = 0u;
      int ckk = need, cab = 0, cbc = 0;
      reg_hist_begin<11>(S);
#pragma unroll
      for (int u = 0; u < kRegVecs; ++u) {
        const int blk0 = (u * kRegThreads + tid) * 4;
        if (blk0 < nb_scan) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int blk = blk0 + c;
            if (blk < nb_scan && rk[u][c] == T) hist_add(S.hist, ((~(uint32_t)blk) >> 21) & 2047u);
          }
        }
      }
      int cd = reg_hist_end<11>(S, ckk, cab, cbc);
      ckk -= cab;
      cp |= (uint32_t)cd << 21;
      cm |= 0x7FFu << 21;

      reg_hist_begin<11>(S);
#pragma unroll
      for (int u = 0; u < kRegVecs; ++u) {
        const int blk0 = (u * kRegThreads + tid) * 4;
        if (blk0 < nb_scan) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int blk = blk0 + c;
            const uint32_t nc = ~(uint32_t)blk;
            if (blk < nb_scan && rk[u][c] == T && (nc & cm) == cp) hist_add(S.hist, (nc >> 10) & 2047u);
          }
        }
      }
      cd = reg_hist_end<11>(S, ckk, cab, cbc);
      ckk -= cab;
      cp |= (uint32_t)cd << 10;
      cm |= 0x7FFu << 10;

      reg_hist_begin<10>(S);
#pragma unroll
      for (int u = 0; u < kRegVecs; ++u) {
        const int blk0 = (u * kRegThreads + tid) * 4;
        if (blk0 < nb_scan) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int blk = blk0 + c;
            const uint32_t nc = ~(uint32_t)blk;
            if (blk < nb_scan && rk[u][c] == T && (nc & cm) == cp) hist_add(S.hist, nc & 1023u);
          }
        }
      }
      cd = reg_hist_end<10>(S, ckk, cab, cbc);
      colT = ~(cp | (uint32_t)cd);
    }

    if constexpr (!kSortOrder) {
      // Deterministic BlockScan output in strictly ascending block ID order!
      int out_off = 0;
#pragma unroll
      for (int u = 0; u < kRegVecs; ++u) {
        if (u * kRegThreads * 4 >= nb_scan) break;
        const int blk0 = (u * kRegThreads + tid) * 4;
        bool v_ok[4];
        int my_cnt = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int blk = blk0 + c;
          const uint32_t key = rk[u][c];
          const bool take = (blk < nb_scan) && ((key > T) || (key == T && (uint32_t)blk <= colT));
          v_ok[c] = take && (key > 0x007FFFFFu) && (key <= 0xFF800000u);
          my_cnt += v_ok[c] ? 1 : 0;
        }
        int pos = 0, stot = 0;
        cub::BlockScan<int, kRegThreads>(S.scan).ExclusiveSum(my_cnt, pos, stot);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (v_ok[c] && (out_off + pos) < K_out) {
            orow_base[(out_off + pos) * os1] = blk0 + c;
            ++pos;
          }
        }
        out_off += stot;
        __syncthreads();
      }
      for (int i = out_off + tid; i < K_out; i += kRegThreads) orow_base[i * os1] = -1;
      return;
    }

    if (tid == 0) S.nsel = 0;
    __syncthreads();
#pragma unroll
    for (int u = 0; u < kRegVecs; ++u) {
      const int blk0 = (u * kRegThreads + tid) * 4;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int blk = blk0 + c;
        const uint32_t key = rk[u][c];
        const bool take = (blk < nb_scan) && ((key > T) || (key == T && (uint32_t)blk <= colT));
        const uint32_t bal = __ballot_sync(0xffffffffu, take);
        int b = 0;
        if (lane == 0 && bal) b = atomicAdd(&S.nsel, __popc(bal));
        b = __shfl_sync(0xffffffffu, b, 0);
        const int p = b + __popc(bal & ((1u << lane) - 1u));
        if (take && p < kCandOutCap) {
          S.pack[p] = ((uint64_t)key << 32) | (uint64_t)(~(uint32_t)blk);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kSortOrder) {
    const bool unit_stride = (os1 == 1);
    if (unit_stride) {
      if (K_act <= 512) reg_bitonic_sort_store<1>(S, K_act, K_out, orow_base);
      else if (K_act <= 1024) reg_bitonic_sort_store<2>(S, K_act, K_out, orow_base);
      else reg_bitonic_sort_store<4>(S, K_act, K_out, orow_base);
    } else {
      int* tmp = reinterpret_cast<int*>(S.hist);
      if (K_act <= 512) reg_bitonic_sort_store<1>(S, K_act, K_act, tmp);
      else if (K_act <= 1024) reg_bitonic_sort_store<2>(S, K_act, K_act, tmp);
      else reg_bitonic_sort_store<4>(S, K_act, K_act, tmp);
      __syncthreads();
      for (int i = tid; i < K_out; i += kRegThreads) {
        orow_base[i * os1] = (i < K_act) ? tmp[i] : -1;
      }
    }
  }
}

void dsv41_idx_cand_topk(const at::Tensor& logits, const at::Tensor& bkeys,
                         const at::Tensor& ks, const at::Tensor& ke,
                         at::Tensor& out, int64_t N, int64_t cand_bs, int64_t K,
                         bool bkeys_all_rows, bool sort_order) {
  check_cuda(logits, "logits");
  check_cuda(bkeys, "bkeys");
  check_cuda(out, "out");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1);
  TORCH_CHECK(bkeys.scalar_type() == at::kInt && bkeys.stride(1) == 1);
  TORCH_CHECK(out.scalar_type() == at::kInt && out.dim() == 2 && out.size(1) >= K);
  TORCH_CHECK(cand_bs == 8 && K > 0 && K <= kCandOutCap);
  const int R = (int)out.size(0);
  if (R == 0) return;
  if (N <= 0) {
    out.fill_(-1);
    return;
  }
  const int nblocks = (int)((N + 7) / 8);
  TORCH_CHECK(logits.size(0) >= R && logits.size(1) >= N);
  TORCH_CHECK(bkeys.size(0) >= R && bkeys.size(1) >= nblocks);
  TORCH_CHECK(ks.numel() >= R && ke.numel() >= R);
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (nblocks <= kRegThreads * kRegVecs * 4) {
    const int smem = sizeof(CandRegSmem);
    if (sort_order) {
      cudaFuncSetAttribute(cand_topk_reg_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      cand_topk_reg_kernel<true><<<R, kRegThreads, smem, stream>>>(
          logits.data_ptr<float>(), logits.stride(0),
          reinterpret_cast<const uint32_t*>(bkeys.data_ptr<int>()), bkeys.stride(0),
          ks.data_ptr<int>(), ke.data_ptr<int>(),
          out.data_ptr<int>(), out.stride(0), out.stride(1),
          (int)N, nblocks, (int)K, bkeys_all_rows);
    } else {
      cudaFuncSetAttribute(cand_topk_reg_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      cand_topk_reg_kernel<false><<<R, kRegThreads, smem, stream>>>(
          logits.data_ptr<float>(), logits.stride(0),
          reinterpret_cast<const uint32_t*>(bkeys.data_ptr<int>()), bkeys.stride(0),
          ks.data_ptr<int>(), ke.data_ptr<int>(),
          out.data_ptr<int>(), out.stride(0), out.stride(1),
          (int)N, nblocks, (int)K, bkeys_all_rows);
    }
  } else {
    const int smem = sizeof(CandTopkSmem);
    if (sort_order) {
      cudaFuncSetAttribute(cand_topk_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      cand_topk_kernel<true><<<R, kTopkThreads, smem, stream>>>(
          logits.data_ptr<float>(), logits.stride(0),
          reinterpret_cast<const uint32_t*>(bkeys.data_ptr<int>()), bkeys.stride(0),
          ks.data_ptr<int>(), ke.data_ptr<int>(),
          out.data_ptr<int>(), out.stride(0), out.stride(1),
          (int)N, nblocks, (int)K, bkeys_all_rows);
    } else {
      cudaFuncSetAttribute(cand_topk_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      cand_topk_kernel<false><<<R, kTopkThreads, smem, stream>>>(
          logits.data_ptr<float>(), logits.stride(0),
          reinterpret_cast<const uint32_t*>(bkeys.data_ptr<int>()), bkeys.stride(0),
          ks.data_ptr<int>(), ke.data_ptr<int>(),
          out.data_ptr<int>(), out.stride(0), out.stride(1),
          (int)N, nblocks, (int)K, bkeys_all_rows);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_reader_compact(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc,
                              const at::Tensor& w, const at::Tensor& ks, const at::Tensor& ke,
                              const at::Tensor& cands, at::Tensor& comp_logits, at::Tensor& comp_mx,
                              at::Tensor& comp_blks, at::Tensor& comp_ngroups, at::Tensor& topk_out,
                              int64_t topk_tokens, int64_t cand_bs, int64_t groups_per_cta, int64_t variant,
                              const c10::optional<at::Tensor>& full_out) {
  check_cuda(q, "q");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kH && q.size(2) == kD && q.stride(2) == 1 && q.stride(1) == kD);
  TORCH_CHECK(k.dim() == 2 && k.size(1) == kD && k.is_contiguous());
  TORCH_CHECK(ksc.scalar_type() == at::kFloat && ksc.is_contiguous() && ksc.numel() >= k.size(0));
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2 && w.size(1) == kH);
  TORCH_CHECK(ks.scalar_type() == at::kInt && ke.scalar_type() == at::kInt && ks.is_contiguous() && ke.is_contiguous());
  TORCH_CHECK(cands.scalar_type() == at::kInt && cands.dim() == 2);
  TORCH_CHECK(cand_bs == 8 && topk_tokens > 0 && topk_tokens <= kListCap);
  const int R = (int)q.size(0), N = (int)k.size(0);
  if (R == 0) return;
  if (N == 0) {
    topk_out.slice(1, 0, topk_tokens).fill_(-1);
    return;
  }
  const int nblocks = (N + 7) / 8;
  TORCH_CHECK(nblocks + 1 <= 65536, "nblocks exceeds smem flag capacity");
  const int K_cand = (int)cands.size(1);
  const int max_groups = (std::min(K_cand, nblocks) + 1 + 3) / 4;
  const int max_blks = max_groups * 4;
  TORCH_CHECK(comp_logits.scalar_type() == at::kFloat && comp_logits.dim() == 2 && comp_logits.stride(1) == 1);
  TORCH_CHECK(comp_logits.size(0) >= R && comp_logits.size(1) >= max_groups * 32);
  TORCH_CHECK(comp_mx.scalar_type() == at::kInt && comp_mx.dim() == 2 && comp_mx.stride(1) == 1);
  TORCH_CHECK(comp_mx.size(0) >= R && comp_mx.size(1) >= max_groups);
  TORCH_CHECK(comp_blks.scalar_type() == at::kInt && comp_blks.dim() == 2 && comp_blks.stride(1) == 1);
  TORCH_CHECK(comp_blks.size(0) >= R && comp_blks.size(1) >= max_blks);
  TORCH_CHECK(comp_ngroups.scalar_type() == at::kInt && comp_ngroups.is_contiguous() && comp_ngroups.numel() >= R);
  TORCH_CHECK(topk_out.scalar_type() == at::kInt && topk_out.dim() == 2 && topk_out.stride(1) == 1 && topk_out.size(1) >= topk_tokens);

  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool has_fout = full_out.has_value() && full_out->defined();
  if (has_fout) {
    TORCH_CHECK(full_out->scalar_type() == at::kFloat && full_out->dim() == 2 && full_out->stride(1) == 1);
    TORCH_CHECK(full_out->size(0) >= R && full_out->size(1) >= N);
  }

  // 1. Compact active candidate blocks per row
  const int flag_smem = (nblocks + 1 + 15) & ~15;
  cudaFuncSetAttribute(reader_compact_blocks_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, flag_smem);
  reader_compact_blocks_kernel<<<R, 256, flag_smem, stream>>>(
      cands.data_ptr<int>(), cands.stride(0), cands.stride(1), K_cand,
      ks.data_ptr<int>(), ke.data_ptr<int>(),
      comp_blks.data_ptr<int>(), comp_blks.stride(0), max_blks,
      comp_ngroups.data_ptr<int>(), N, nblocks, (int)topk_tokens, !has_fout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 2. Gather & compute compact logits
  if (max_groups > 0) {
    const int gpc = (int)groups_per_cta;
    TORCH_CHECK(gpc > 0 && gpc % 8 == 0);
    ReaderCompactArgs a;
    a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    a.q_s0 = q.stride(0);
    a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    a.ksc = ksc.data_ptr<float>();
    a.w = w.data_ptr<float>();
    a.w_s0 = w.stride(0);
    a.w_s1 = w.stride(1);
    a.ks = ks.data_ptr<int>();
    a.ke = ke.data_ptr<int>();
    a.comp_blks = comp_blks.data_ptr<int>();
    a.blk_s0 = comp_blks.stride(0);
    a.comp_ngroups = comp_ngroups.data_ptr<int>();
    a.comp_out = comp_logits.data_ptr<float>();
    a.cout_s0 = comp_logits.stride(0);
    a.comp_mx = reinterpret_cast<uint32_t*>(comp_mx.data_ptr<int>());
    a.cmx_s0 = comp_mx.stride(0);
    a.full_out = has_fout ? full_out->data_ptr<float>() : nullptr;
    a.fout_s0 = has_fout ? full_out->stride(0) : 0;
    a.R = R;
    a.N = N;
    a.groups_per_cta = gpc;

    dim3 grid(R, (max_groups + gpc - 1) / gpc);
    (void)variant;
    constexpr int smem = 2 * 4 * kTileBytes;  // 64 KiB, 2 CTAs/SM = 8 warps/SM
    cudaFuncSetAttribute(reader_compact_logits_kernel<4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    reader_compact_logits_kernel<4, 2><<<grid, 4 * 32, smem, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 3. Pruned radix top-k over compact logits
  const int tk_smem = sizeof(TopkSmem);
  cudaFuncSetAttribute(reader_compact_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, tk_smem);
  reader_compact_topk_kernel<<<R, kTopkThreads, tk_smem, stream>>>(
      comp_logits.data_ptr<float>(), comp_logits.stride(0),
      reinterpret_cast<const uint32_t*>(comp_mx.data_ptr<int>()), comp_mx.stride(0),
      comp_blks.data_ptr<int>(), comp_blks.stride(0),
      comp_ngroups.data_ptr<int>(),
      ks.data_ptr<int>(), ke.data_ptr<int>(),
      topk_out.data_ptr<int>(), topk_out.stride(0), N, (int)topk_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_chunkmax(const at::Tensor& logits, const at::Tensor& ks, const at::Tensor& ke, at::Tensor& mx) {
  check_cuda(logits, "logits");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1);
  const int R = (int)logits.size(0), N = (int)logits.size(1);
  if (R == 0 || N == 0) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  int nch = (N + 31) / 32;
  dim3 grid((nch * 32 + 255) / 256, R);
  chunkmax_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), logits.stride(0), ks.data_ptr<int>(), ke.data_ptr<int>(),
      reinterpret_cast<uint32_t*>(mx.data_ptr<int>()), mx.stride(0), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_topk(const at::Tensor& logits, const at::Tensor& mx, const at::Tensor& ks, const at::Tensor& ke,
                    at::Tensor& out, int64_t K, const c10::optional<at::Tensor>& stats) {
  check_cuda(logits, "logits");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1);
  TORCH_CHECK(out.scalar_type() == at::kInt && out.stride(1) == 1 && out.size(1) >= K);
  TORCH_CHECK(K > 0 && K <= kListCap);
  const int R = (int)out.size(0);
  if (R == 0) return;
  TORCH_CHECK(logits.size(0) >= R && mx.size(0) >= R && ks.numel() >= R && ke.numel() >= R);
  const c10::cuda::CUDAGuard guard(logits.device());
  const int smem = sizeof(TopkSmem);
  cudaFuncSetAttribute(topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int* st = (stats.has_value() && stats->defined()) ? stats->data_ptr<int>() : nullptr;
  topk_kernel<<<R, kTopkThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), logits.stride(0), reinterpret_cast<const uint32_t*>(mx.data_ptr<int>()),
      mx.stride(0), ks.data_ptr<int>(), ke.data_ptr<int>(), out.data_ptr<int>(), out.stride(0), (int)K, st);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t dsv41_idx_abi() { return 1; }

}  // namespace

TORCH_LIBRARY(_dsv41_indexer_C, m) {
  m.def("decode_fp8(Tensor inp, Tensor lut, Tensor(a!) out) -> ()");
  m.def(
      "logits(Tensor q, Tensor k, Tensor ksc, Tensor w, Tensor ks, Tensor ke, Tensor(a!) out, Tensor(b!) mx, "
      "Tensor? flags, int cand_bs, int kpc) -> ()");
  m.def(
      "logits_writer(Tensor q, Tensor k, Tensor ksc, Tensor w, Tensor ks, Tensor ke, Tensor(a!) out, Tensor(b!) mx, "
      "Tensor(c!) bkeys, int cand_bs, int kpc) -> ()");
  m.def(
      "cand_topk(Tensor logits, Tensor bkeys, Tensor ks, Tensor ke, Tensor(a!) out, "
      "int N, int cand_bs, int K, bool bkeys_all_rows, bool sort_order) -> ()");
  m.def(
      "reader_compact(Tensor q, Tensor k, Tensor ksc, Tensor w, Tensor ks, Tensor ke, Tensor cands, "
      "Tensor(a!) comp_logits, Tensor(b!) comp_mx, Tensor(c!) comp_blks, Tensor(d!) comp_ngroups, "
      "Tensor(e!) topk_out, int topk_tokens, int cand_bs, int groups_per_cta, int variant, Tensor(f!)? full_out) -> ()");
  m.def(
      "logits_dbg(Tensor q, Tensor k, Tensor ksc, Tensor w, Tensor ks, Tensor ke, Tensor(a!) out, Tensor(b!) mx, "
      "int kpc, int dbg) -> ()");
  m.def("chunkmax(Tensor logits, Tensor ks, Tensor ke, Tensor(a!) mx) -> ()");
  m.def("topk(Tensor logits, Tensor mx, Tensor ks, Tensor ke, Tensor(a!) out, int K, Tensor(b!)? stats) -> ()");
  m.def("abi() -> int");
}
TORCH_LIBRARY_IMPL(_dsv41_indexer_C, CUDA, m) {
  m.impl("decode_fp8", &dsv41_idx_decode_fp8);
  m.impl("logits", &dsv41_idx_logits);
  m.impl("logits_writer", &dsv41_idx_logits_writer);
  m.impl("cand_topk", &dsv41_idx_cand_topk);
  m.impl("reader_compact", &dsv41_idx_reader_compact);
  m.impl("chunkmax", &dsv41_idx_chunkmax);
  m.impl("logits_dbg", &dsv41_idx_logits_dbg);
  m.impl("topk", &dsv41_idx_topk);
}
TORCH_LIBRARY_IMPL(_dsv41_indexer_C, CompositeExplicitAutograd, m) { m.impl("abi", &dsv41_idx_abi); }
