// DeepSeek V4.1 sparse-indexer prefill on SM80 (CMP 170HX): fused
// tensor-core MQA logits + exact chunk-max-pruned top-k.
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
//  * Each 32-column tile also emits the max order-preserving key of the row
//    inside the tile ("chunk max").
//  * dsv41_idx_topk selects, per row, exactly the top-k set of logits[ks,ke)
//    (ties broken by lowest column, output sorted by column, relative to ks;
//    rows with ke-ks <= k emit 0..L-1 then -1 like top_k_per_row_prefill).
//    Pruning: tau = lower edge of the 22-bit radix bin holding the k-th
//    largest chunk max <= k-th largest chunk max <= k-th largest value, so
//    only chunks with max >= tau and values >= tau can be in the top-k.
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
__device__ __forceinline__ float relu_ptx(float x) {
  float y;
  asm("max.f32 %0, %1, 0f00000000;" : "=f"(y) : "f"(x));
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

// Branch-free epilogue (same tree as the Triton PTX); stores logits + chunk max.
template <bool kFlags, int kDbg = 0>
__device__ __forceinline__ void tile_epi(const float (&acc)[2][4][4], const Pend& p, float w0, float w1, float w2,
                                         float w3, int lane, int my_col, int rks, int rke, const LogitsArgs& a,
                                         float* orow, uint32_t* mrow) {
  const int t = lane & 3;
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
  float outv = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, send, 4));
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
}

constexpr int kSub = 2;                       // 32-key sub-tiles per pipeline stage
constexpr int kStageBytes = kSub * kTileBytes;  // 16 KiB

template <bool kFlags, int kDbg = 0>
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
      // lane -> n-tile: my_col / 8 == (lane >> 3) bit-swapped: nt = hi16 * 2 + hi8
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
      tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
      pb = meta(j0 + 1);
      tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accA, pb, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
      continue;
    }
    if (pa.need) {
      tile_mma<kFlags>(accA, A, st, 0, lm_mat, lm_r, pa.ntmask);
      if (s > 0) tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
    } else if (s > 0) {
      tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
    }
    pb = meta(j0 + 1);
    if (pb.need) {
      tile_mma<kFlags>(accB, A, st, 1, lm_mat, lm_r, pb.ntmask);
      tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
    } else {
      tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accA, pa, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
    }
  }
  if ((kDbg == 0 || kDbg >= 3) && live_row) tile_epi<kFlags, (kDbg >= 3 ? kDbg : 0)>(accB, pb, w0, w1, w2, w3, lane, my_col, rks, rke, a, orow, mrow);
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

// Bin d of digit (shift, bits) among keys with (key & mask) == prefix such
// that #keys in higher bins < k <= #keys in bins >= d.
template <int kBits, class Visit>
__device__ int radix_step(TopkSmem& S, Visit&& visit, uint32_t prefix, uint32_t mask, int shift, int k,
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

template <class Visit>
__device__ uint32_t radix_kth(TopkSmem& S, Visit&& visit, int k, int& G, int& E) {
  uint32_t prefix = 0, mask = 0;
  int kk = k, above = 0, binc = 0;
  int d = radix_step<11>(S, visit, prefix, mask, 21, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 21;
  mask |= 0x7FFu << 21;
  d = radix_step<11>(S, visit, prefix, mask, 10, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d << 10;
  mask |= 0x7FFu << 10;
  d = radix_step<10>(S, visit, prefix, mask, 0, kk, above, binc);
  kk -= above;
  prefix |= (uint32_t)d;
  G = k - kk;
  E = binc;
  return prefix;
}

// Lower bound of the k-th largest visited key (lower edge of its 22-bit bin).
template <class Visit>
__device__ uint32_t radix_lower_bound(TopkSmem& S, Visit&& visit, int k) {
  int above, binc;
  const int d1 = radix_step<11>(S, visit, 0u, 0u, 21, k, above, binc);
  const int d2 = radix_step<11>(S, visit, (uint32_t)d1 << 21, 0xFFE00000u, 10, k - above, above, binc);
  return ((uint32_t)d1 << 21) | ((uint32_t)d2 << 10);
}

// Exact selection over a (key, col) stream: fills S.sel[0..K) (unordered).
// Ties at the k-th value keep the lowest columns.
template <class VisitKC>
__device__ void select_exact(TopkSmem& S, VisitKC&& vkc, int K) {
  auto vk = [&](auto f) { vkc([&](uint32_t key, uint32_t) { f(key); }); };
  int G, E;
  const uint32_t T = radix_kth(S, vk, K, G, E);
  const int need = K - G;
  uint32_t colT = 0xFFFFFFFFu;
  if (E > need) {
    auto vt = [&](auto f) {
      vkc([&](uint32_t key, uint32_t col) {
        if (key == T) f(~col);
      });
    };
    int g2, e2;
    colT = ~radix_kth(S, vt, need, g2, e2);
  }
  if (threadIdx.x == 0) S.nsel = 0;
  __syncthreads();
  vkc([&](uint32_t key, uint32_t col) {
    if (key > T || (key == T && col <= colT)) S.sel[atomicAdd(&S.nsel, 1)] = col;
  });
  __syncthreads();
}

// Ascending sort of S.sel[0..K) (K <= 2048), store (col - rks).
// Element i = j * T + tid lives in x[j]: strides < 32 via shuffles, strides
// < T via smem, larger strides are register-local (static indices only).
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

// Append (key, col) for take lanes using warp-aggregated slots.
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
  const bool chunked = nc >= K;  // use the logits kernel's 32-column maxima
  const int kG8 = L > 8 * kListCap ? 16 : 8;  // on-the-fly group size (rows with < K chunks)
  const int ng = chunked ? 0 : min(L / kG8, kListCap);
  bool grouped = !chunked && ng >= K;  // 8-groups computed here
  uint32_t tau = 0;
  int nsch = 0;  // chunked: #listed chunks (> kListCap: not listed)
  if (tid == 0) S.cnt = 0;

  if (chunked) {
    // cache the row's chunk maxima in smem (aliases the candidate buffers)
    uint32_t* mc = S.ckey;  // ckey+ccol are contiguous: 2*kCandCap words
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
    tau = radix_lower_bound(S, vmax, K);
    // ordered list of chunks with max >= tau: warp-strided ballots + one scan
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
    tau = radix_lower_bound(S, vg, K);
  }
  tprof[1] = clock64();

  // Visit every element >= thr that survives the pruning. g(take, key, col) is
  // called warp-uniformly (all 32 lanes), so it may use warp collectives.
  auto scan_elems = [&](uint32_t thr, auto g) {
    const int nsch_l = nsch;        // register snapshots: keep the hot loops
    const bool grouped_l = grouped;  // free of local-memory reads
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
      for (int i0 = ng * kG8; i0 < L; i0 += kTopkThreads) {  // ungrouped tail
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
    // refine the bound on the elements themselves (22-bit bin), then regather
    mode += 2;
    auto vk = [&](auto f) {
      scan_elems(tau, [&](bool take, uint32_t key, uint32_t) {
        if (take) f(key);
      });
    };
    tau = radix_lower_bound(S, vk, K);
    total = gather(tau);
  }
  tprof[3] = clock64();
  if (total <= kCandCap) {
    auto vkc = [&](auto f) {
      for (int i = tid; i < total; i += kTopkThreads) f(S.ckey[i], S.ccol[i]);
    };
    select_exact(S, vkc, K);
  } else {
    mode += 2;  // massive exact ties: stream (exact, slow); S.sel is the output now
    nsch = kListCap + 1;
    if (grouped) {
      grouped = false;  // plain scan of the row (tau still applies)
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
                      const c10::optional<at::Tensor>& flags, int64_t cand_bs, int64_t kpc, int64_t dbg) {
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
  a.flags = nullptr;
  a.fl_s0 = 0;
  a.nblocks = 0;
  a.cand_bs = (int)cand_bs;
  a.R = R;
  a.N = N;
  a.kpc = (int)kpc;
  const bool has_flags = flags.has_value() && flags->defined();
  if (has_flags) {
    TORCH_CHECK(flags->element_size() == 1 && flags->dim() == 2 && flags->stride(1) == 1 && cand_bs > 0);
    a.flags = reinterpret_cast<const uint8_t*>(flags->data_ptr());
    a.fl_s0 = flags->stride(0);
    a.nblocks = (int)((N + cand_bs - 1) / cand_bs);
    TORCH_CHECK(flags->size(1) >= a.nblocks + 1);
  }
  dim3 grid((R + kWarps - 1) / kWarps, (N + kpc - 1) / kpc);
  TORCH_CHECK(grid.y <= 65535);
  const int smem = kStages * kStageBytes;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dbg == 3) {
    cudaFuncSetAttribute(logits_kernel<false, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 3><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 4) {
    cudaFuncSetAttribute(logits_kernel<false, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 4><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 1) {
    cudaFuncSetAttribute(logits_kernel<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 1><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (dbg == 2) {
    cudaFuncSetAttribute(logits_kernel<false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false, 2><<<grid, kWarps * 32, smem, stream>>>(a);
  } else if (has_flags) {
    cudaFuncSetAttribute(logits_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<true><<<grid, kWarps * 32, smem, stream>>>(a);
  } else {
    cudaFuncSetAttribute(logits_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    logits_kernel<false><<<grid, kWarps * 32, smem, stream>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv41_idx_logits(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                      const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx,
                      const c10::optional<at::Tensor>& flags, int64_t cand_bs, int64_t kpc) {
  dsv41_idx_logits_impl(q, k, ksc, w, ks, ke, out, mx, flags, cand_bs, kpc, 0);
}
void dsv41_idx_logits_dbg(const at::Tensor& q, const at::Tensor& k, const at::Tensor& ksc, const at::Tensor& w,
                      const at::Tensor& ks, const at::Tensor& ke, at::Tensor& out, at::Tensor& mx, int64_t kpc, int64_t dbg) {
  dsv41_idx_logits_impl(q, k, ksc, w, ks, ke, out, mx, c10::nullopt, 0, kpc, dbg);
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
      "logits_dbg(Tensor q, Tensor k, Tensor ksc, Tensor w, Tensor ks, Tensor ke, Tensor(a!) out, Tensor(b!) mx, "
      "int kpc, int dbg) -> ()");
  m.def("chunkmax(Tensor logits, Tensor ks, Tensor ke, Tensor(a!) mx) -> ()");
  m.def("topk(Tensor logits, Tensor mx, Tensor ks, Tensor ke, Tensor(a!) out, int K, Tensor(b!)? stats) -> ()");
  m.def("abi() -> int");
}
TORCH_LIBRARY_IMPL(_dsv41_indexer_C, CUDA, m) {
  m.impl("decode_fp8", &dsv41_idx_decode_fp8);
  m.impl("logits", &dsv41_idx_logits);
  m.impl("chunkmax", &dsv41_idx_chunkmax);
  m.impl("logits_dbg", &dsv41_idx_logits_dbg);
  m.impl("topk", &dsv41_idx_topk);
}
TORCH_LIBRARY_IMPL(_dsv41_indexer_C, CompositeExplicitAutograd, m) { m.impl("abi", &dsv41_idx_abi); }
