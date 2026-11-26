// SPDX-License-Identifier: Apache-2.0
// Multi-threaded OpenMP + AVX2 CPU gather & FP8(E4M3FN) x UE8M0 -> BF16
// dequantizer for DeepSeek-V4.1-Flash Engram host-memory tables.
//
// Produces 100% bitwise-identical uint16 BF16 outputs to the SM80 Triton
// `_engram_lookup_kernel` across all 256 x 256 (FP8, UE8M0) byte pairs,
// including NaNs (0x7FFF canonical BF16 NaN), signed zeros, subnormals,
// overflows, and out-of-range / masked indices.

#include <cstdint>
#include <cstring>
#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <xmmintrin.h>
#endif

namespace {

static inline float bits_to_f32(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

static inline uint32_t f32_to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

alignas(64) static float g_e4m3_to_f32[256];

__attribute__((constructor)) static void init_e4m3_lut() {
    for (int u = 0; u < 256; ++u) {
        uint32_t sign = (uint32_t(u & 0x80)) << 24;
        uint32_t exp = (uint32_t(u) >> 3) & 0xFu;
        uint32_t mant = uint32_t(u) & 0x7u;
        uint32_t bits;
        if (exp == 0xFu && mant == 0x7u) {
            bits = 0x7FC00000u;
        } else if (exp == 0u) {
            if (mant == 0u) {
                bits = sign;
            } else {
                float val = float(mant) * (1.0f / 512.0f);
                bits = f32_to_bits(val) | sign;
            }
        } else {
            bits = sign | ((exp + 120u) << 23) | (mant << 20);
        }
        g_e4m3_to_f32[u] = bits_to_f32(bits);
    }
}

static inline uint16_t f32_to_bf16_sm80_bits(float f) {
    uint32_t b = f32_to_bits(f);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) {
        // Match SM80 PTX cvt.rn.bf16.f32 canonical NaN payload (0x7FFF).
        return 0x7FFFu;
    }
    uint32_t round_bias = 0x7FFFu + ((b >> 16) & 1u);
    return static_cast<uint16_t>((b + round_bias) >> 16);
}

#if defined(__AVX2__)
static inline __m256 decode_e4m3_x8_avx2(__m128i bytes8) {
    const __m256i u = _mm256_cvtepu8_epi32(bytes8);
    const __m256i sign = _mm256_slli_epi32(
        _mm256_and_si256(u, _mm256_set1_epi32(0x80)), 24);
    const __m256i exp = _mm256_and_si256(
        _mm256_srli_epi32(u, 3), _mm256_set1_epi32(0xF));
    const __m256i mant = _mm256_and_si256(u, _mm256_set1_epi32(0x7));

    // Normal: sign | ((exp + 120) << 23) | (mant << 20)
    const __m256i norm_exp = _mm256_slli_epi32(
        _mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23);
    const __m256i norm_mant = _mm256_slli_epi32(mant, 20);
    const __m256i norm_bits = _mm256_or_si256(
        sign, _mm256_or_si256(norm_exp, norm_mant));

    // Subnormal / signed zero: (float(mant) * 2^-9) | sign
    const __m256 sub_mag = _mm256_mul_ps(
        _mm256_cvtepi32_ps(mant), _mm256_set1_ps(1.0f / 512.0f));
    const __m256i sub_bits = _mm256_or_si256(
        _mm256_castps_si256(sub_mag), sign);

    const __m256i is_zero_exp = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    __m256i res = _mm256_blendv_epi8(norm_bits, sub_bits, is_zero_exp);

    // NaN: (u & 0x7F) == 0x7F -> 0x7FC00000
    const __m256i u_abs = _mm256_and_si256(u, _mm256_set1_epi32(0x7F));
    const __m256i is_nan = _mm256_cmpeq_epi32(u_abs, _mm256_set1_epi32(0x7F));
    res = _mm256_blendv_epi8(res, _mm256_set1_epi32(0x7FC00000), is_nan);
    return _mm256_castsi256_ps(res);
}

static inline __m256i f32x8_to_bf16_u32_sm80(__m256 v) {
    const __m256i b = _mm256_castps_si256(v);
    const __m256i abs_b = _mm256_and_si256(b, _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i is_nan = _mm256_cmpgt_epi32(
        abs_b, _mm256_set1_epi32(0x7F800000));
    const __m256i lsb = _mm256_and_si256(
        _mm256_srli_epi32(b, 16), _mm256_set1_epi32(1));
    const __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
    const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(b, bias), 16);
    return _mm256_blendv_epi8(rounded, _mm256_set1_epi32(0x7FFF), is_nan);
}

static inline void dequant_block32_avx2(
    const uint8_t* w32,
    uint8_t scale_u8,
    uint16_t* out32
) {
    const __m256 vscale = _mm256_castsi256_ps(
        _mm256_set1_epi32(static_cast<int32_t>(uint32_t(scale_u8) << 23)));

    const __m128i w_lo16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w32));
    const __m128i w_hi16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w32 + 16));

    const __m256 f0 = _mm256_mul_ps(decode_e4m3_x8_avx2(w_lo16), vscale);
    const __m256 f1 = _mm256_mul_ps(
        decode_e4m3_x8_avx2(_mm_srli_si128(w_lo16, 8)), vscale);
    const __m256i u0 = f32x8_to_bf16_u32_sm80(f0);
    const __m256i u1 = f32x8_to_bf16_u32_sm80(f1);
    const __m256i p01 = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(u0, u1), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out32), p01);

    const __m256 f2 = _mm256_mul_ps(decode_e4m3_x8_avx2(w_hi16), vscale);
    const __m256 f3 = _mm256_mul_ps(
        decode_e4m3_x8_avx2(_mm_srli_si128(w_hi16, 8)), vscale);
    const __m256i u2 = f32x8_to_bf16_u32_sm80(f2);
    const __m256i u3 = f32x8_to_bf16_u32_sm80(f3);
    const __m256i p23 = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(u2, u3), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out32 + 16), p23);
}
#endif

}  // namespace

extern "C" void engram_cpu_gather_bf16(
    const uint8_t* __restrict__ weight,
    int64_t w_stride,
    const uint8_t* __restrict__ scales,
    int64_t s_stride,
    const int64_t* __restrict__ local_ids,
    uint16_t* __restrict__ out_bf16,
    int64_t vocab_start,
    int64_t vocab_end,
    int64_t num_tokens,
    int64_t local_heads,
    int64_t local_valid_heads,
    int64_t dim,
    int64_t quant_block,
    int num_threads
) {
    const int64_t total_rows = num_tokens * local_heads;
    if (total_rows <= 0 || dim <= 0) {
        return;
    }
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    if (total_rows < 64) {
        num_threads = 1;
    } else if (total_rows < int64_t(num_threads) * 16) {
        num_threads = static_cast<int>((total_rows + 15) / 16);
        if (num_threads < 1) num_threads = 1;
    }

    constexpr int64_t PREFETCH_DIST = 8;

    #pragma omp parallel num_threads(num_threads)
    {
#if defined(__x86_64__) || defined(_M_X64)
        // Ensure IEEE-754 gradual underflow (FTZ=0, DAZ=0) so subnormal BF16
        // products (e.g. 2^-133 -> 0x0001) match SM80 mul.f32 bit-for-bit even
        // if the host process enabled flush-to-zero.
        const unsigned int saved_mxcsr = _mm_getcsr();
        const bool altered_mxcsr = (saved_mxcsr & 0x8040u) != 0u;
        if (altered_mxcsr) {
            _mm_setcsr(saved_mxcsr & ~0x8040u);
        }
#endif

        // Gather + BF16 dequant in 256-row tiles; each thread handles one
        // contiguous chunk of rows to maximize L2/L3 hit rate and avoid false
        // sharing.
        #pragma omp for schedule(static, 64)
        for (int64_t row = 0; row < total_rows; ++row) {
            // Issue software prefetches ahead to overlap Optane PMem / DRAM
            // latency across per-core Line Fill Buffers (MLP).
            const int64_t pf_row = row + PREFETCH_DIST;
            if (pf_row < total_rows) {
                const int64_t pf_lh = pf_row % local_heads;
                if (pf_lh < local_valid_heads) {
                    const int64_t pf_tok = pf_row / local_heads;
                    const int64_t pf_idx =
                        local_ids[pf_tok * local_valid_heads + pf_lh];
                    if (pf_idx >= vocab_start && pf_idx < vocab_end) {
                        const int64_t pf_local = pf_idx - vocab_start;
                        const uint8_t* pf_w = weight + pf_local * w_stride;
                        for (int64_t off = 0; off < dim; off += 64) {
                            __builtin_prefetch(pf_w + off, 0, 1);
                        }
                        __builtin_prefetch(scales + pf_local * s_stride, 0, 1);
                    }
                }
            }

            uint16_t* row_out = out_bf16 + row * dim;
            const int64_t lh = row % local_heads;
            if (lh >= local_valid_heads) {
                std::memset(row_out, 0, static_cast<size_t>(dim) * sizeof(uint16_t));
                continue;
            }
            const int64_t tok = row / local_heads;
            const int64_t idx = local_ids[tok * local_valid_heads + lh];
            if (idx < vocab_start || idx >= vocab_end) {
                std::memset(row_out, 0, static_cast<size_t>(dim) * sizeof(uint16_t));
                continue;
            }

            const int64_t local = idx - vocab_start;
            const uint8_t* w_row = weight + local * w_stride;
            const uint8_t* s_row = scales + local * s_stride;

#if defined(__AVX2__)
            if (quant_block == 32 && (dim & 31) == 0) {
                const int64_t n_blocks = dim >> 5;
                for (int64_t b = 0; b < n_blocks; ++b) {
                    dequant_block32_avx2(
                        w_row + (b << 5),
                        s_row[b],
                        row_out + (b << 5));
                }
            } else
#endif
            {
                for (int64_t c = 0; c < dim; ++c) {
                    const uint8_t sb = s_row[c / quant_block];
                    const float sf = bits_to_f32(uint32_t(sb) << 23);
                    const float wf = g_e4m3_to_f32[w_row[c]];
                    row_out[c] = f32_to_bf16_sm80_bits(wf * sf);
                }
            }
        }

#if defined(__x86_64__) || defined(_M_X64)
        if (altered_mxcsr) {
            _mm_setcsr(saved_mxcsr);
        }
#endif
    }
}

extern "C" void engram_cpu_gather_fp8(
    const uint8_t* __restrict__ weight,
    int64_t w_stride,
    const uint8_t* __restrict__ scales,
    int64_t s_stride,
    const int64_t* __restrict__ local_ids,
    uint8_t* __restrict__ out_fp8,
    uint8_t* __restrict__ out_scales,
    int64_t vocab_start,
    int64_t vocab_end,
    int64_t num_tokens,
    int64_t local_heads,
    int64_t local_valid_heads,
    int64_t dim,
    int64_t quant_block,
    int num_threads
) {
    const int64_t total_rows = num_tokens * local_heads;
    const int64_t s_cols = dim / quant_block;
    if (total_rows <= 0 || dim <= 0) {
        return;
    }
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    if (total_rows < 64) {
        num_threads = 1;
    } else if (total_rows < int64_t(num_threads) * 16) {
        num_threads = static_cast<int>((total_rows + 15) / 16);
        if (num_threads < 1) num_threads = 1;
    }

    constexpr int64_t PREFETCH_DIST = 8;

    #pragma omp parallel for schedule(static, 64) num_threads(num_threads)
    for (int64_t row = 0; row < total_rows; ++row) {
        const int64_t pf_row = row + PREFETCH_DIST;
        if (pf_row < total_rows) {
            const int64_t pf_lh = pf_row % local_heads;
            if (pf_lh < local_valid_heads) {
                const int64_t pf_tok = pf_row / local_heads;
                const int64_t pf_idx =
                    local_ids[pf_tok * local_valid_heads + pf_lh];
                if (pf_idx >= vocab_start && pf_idx < vocab_end) {
                    const int64_t pf_local = pf_idx - vocab_start;
                    const uint8_t* pf_w = weight + pf_local * w_stride;
                    for (int64_t off = 0; off < dim; off += 64) {
                        __builtin_prefetch(pf_w + off, 0, 1);
                    }
                    __builtin_prefetch(scales + pf_local * s_stride, 0, 1);
                }
            }
        }

        uint8_t* dst_w = out_fp8 + row * dim;
        uint8_t* dst_s = out_scales + row * s_cols;
        const int64_t lh = row % local_heads;
        if (lh >= local_valid_heads) {
            std::memset(dst_w, 0, static_cast<size_t>(dim));
            std::memset(dst_s, 0, static_cast<size_t>(s_cols));
            continue;
        }
        const int64_t tok = row / local_heads;
        const int64_t idx = local_ids[tok * local_valid_heads + lh];
        if (idx < vocab_start || idx >= vocab_end) {
            std::memset(dst_w, 0, static_cast<size_t>(dim));
            std::memset(dst_s, 0, static_cast<size_t>(s_cols));
            continue;
        }

        const int64_t local = idx - vocab_start;
        std::memcpy(dst_w, weight + local * w_stride, static_cast<size_t>(dim));
        std::memcpy(dst_s, scales + local * s_stride, static_cast<size_t>(s_cols));
    }
}

namespace {

static inline void resolve_multi_shard(
    int64_t local,
    const uint8_t* const* weight_ptrs,
    const uint8_t* const* scale_ptrs,
    const int64_t* shard_row_ends,
    int num_shards,
    int64_t w_stride,
    int64_t s_stride,
    const uint8_t*& w_row,
    const uint8_t*& s_row
) {
    int64_t base = 0;
    for (int s = 0; s < num_shards; ++s) {
        const int64_t end = shard_row_ends[s];
        if (local < end || s == num_shards - 1) {
            const int64_t off = local - base;
            w_row = weight_ptrs[s] + off * w_stride;
            s_row = scale_ptrs[s] + off * s_stride;
            return;
        }
        base = end;
    }
    w_row = weight_ptrs[0];
    s_row = scale_ptrs[0];
}

}  // namespace

extern "C" void engram_cpu_gather_bf16_multi(
    const uint8_t* const* __restrict__ weight_ptrs,
    int64_t w_stride,
    const uint8_t* const* __restrict__ scale_ptrs,
    int64_t s_stride,
    const int64_t* __restrict__ shard_row_ends,
    int num_shards,
    const int64_t* __restrict__ local_ids,
    uint16_t* __restrict__ out_bf16,
    int64_t vocab_start,
    int64_t vocab_end,
    int64_t num_tokens,
    int64_t local_heads,
    int64_t local_valid_heads,
    int64_t dim,
    int64_t quant_block,
    int num_threads
) {
    const int64_t total_rows = num_tokens * local_heads;
    if (total_rows <= 0 || dim <= 0 || num_shards <= 0) {
        return;
    }
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    if (total_rows < 64) {
        num_threads = 1;
    } else if (total_rows < int64_t(num_threads) * 16) {
        num_threads = static_cast<int>((total_rows + 15) / 16);
        if (num_threads < 1) num_threads = 1;
    }

    constexpr int64_t PREFETCH_DIST = 8;

    #pragma omp parallel num_threads(num_threads)
    {
#if defined(__x86_64__) || defined(_M_X64)
        const unsigned int saved_mxcsr = _mm_getcsr();
        const bool altered_mxcsr = (saved_mxcsr & 0x8040u) != 0u;
        if (altered_mxcsr) {
            _mm_setcsr(saved_mxcsr & ~0x8040u);
        }
#endif

        #pragma omp for schedule(static, 64)
        for (int64_t row = 0; row < total_rows; ++row) {
            const int64_t pf_row = row + PREFETCH_DIST;
            if (pf_row < total_rows) {
                const int64_t pf_lh = pf_row % local_heads;
                if (pf_lh < local_valid_heads) {
                    const int64_t pf_tok = pf_row / local_heads;
                    const int64_t pf_idx =
                        local_ids[pf_tok * local_valid_heads + pf_lh];
                    if (pf_idx >= vocab_start && pf_idx < vocab_end) {
                        const uint8_t* pf_w = nullptr;
                        const uint8_t* pf_s = nullptr;
                        resolve_multi_shard(
                            pf_idx - vocab_start,
                            weight_ptrs,
                            scale_ptrs,
                            shard_row_ends,
                            num_shards,
                            w_stride,
                            s_stride,
                            pf_w,
                            pf_s);
                        for (int64_t off = 0; off < dim; off += 64) {
                            __builtin_prefetch(pf_w + off, 0, 1);
                        }
                        __builtin_prefetch(pf_s, 0, 1);
                    }
                }
            }

            uint16_t* row_out = out_bf16 + row * dim;
            const int64_t lh = row % local_heads;
            if (lh >= local_valid_heads) {
                std::memset(row_out, 0, static_cast<size_t>(dim) * sizeof(uint16_t));
                continue;
            }
            const int64_t tok = row / local_heads;
            const int64_t idx = local_ids[tok * local_valid_heads + lh];
            if (idx < vocab_start || idx >= vocab_end) {
                std::memset(row_out, 0, static_cast<size_t>(dim) * sizeof(uint16_t));
                continue;
            }

            const uint8_t* w_row = nullptr;
            const uint8_t* s_row = nullptr;
            resolve_multi_shard(
                idx - vocab_start,
                weight_ptrs,
                scale_ptrs,
                shard_row_ends,
                num_shards,
                w_stride,
                s_stride,
                w_row,
                s_row);

#if defined(__AVX2__)
            if (quant_block == 32 && (dim & 31) == 0) {
                const int64_t n_blocks = dim >> 5;
                for (int64_t b = 0; b < n_blocks; ++b) {
                    dequant_block32_avx2(
                        w_row + (b << 5),
                        s_row[b],
                        row_out + (b << 5));
                }
            } else
#endif
            {
                for (int64_t c = 0; c < dim; ++c) {
                    const uint8_t sb = s_row[c / quant_block];
                    const float sf = bits_to_f32(uint32_t(sb) << 23);
                    const float wf = g_e4m3_to_f32[w_row[c]];
                    row_out[c] = f32_to_bf16_sm80_bits(wf * sf);
                }
            }
        }

#if defined(__x86_64__) || defined(_M_X64)
        if (altered_mxcsr) {
            _mm_setcsr(saved_mxcsr);
        }
#endif
    }
}

extern "C" void engram_cpu_gather_fp8_multi(
    const uint8_t* const* __restrict__ weight_ptrs,
    int64_t w_stride,
    const uint8_t* const* __restrict__ scale_ptrs,
    int64_t s_stride,
    const int64_t* __restrict__ shard_row_ends,
    int num_shards,
    const int64_t* __restrict__ local_ids,
    uint8_t* __restrict__ out_fp8,
    uint8_t* __restrict__ out_scales,
    int64_t vocab_start,
    int64_t vocab_end,
    int64_t num_tokens,
    int64_t local_heads,
    int64_t local_valid_heads,
    int64_t dim,
    int64_t quant_block,
    int num_threads
) {
    const int64_t total_rows = num_tokens * local_heads;
    const int64_t s_cols = dim / quant_block;
    if (total_rows <= 0 || dim <= 0 || num_shards <= 0) {
        return;
    }
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    if (total_rows < 64) {
        num_threads = 1;
    } else if (total_rows < int64_t(num_threads) * 16) {
        num_threads = static_cast<int>((total_rows + 15) / 16);
        if (num_threads < 1) num_threads = 1;
    }

    constexpr int64_t PREFETCH_DIST = 8;

    #pragma omp parallel for schedule(static, 64) num_threads(num_threads)
    for (int64_t row = 0; row < total_rows; ++row) {
        const int64_t pf_row = row + PREFETCH_DIST;
        if (pf_row < total_rows) {
            const int64_t pf_lh = pf_row % local_heads;
            if (pf_lh < local_valid_heads) {
                const int64_t pf_tok = pf_row / local_heads;
                const int64_t pf_idx =
                    local_ids[pf_tok * local_valid_heads + pf_lh];
                if (pf_idx >= vocab_start && pf_idx < vocab_end) {
                    const uint8_t* pf_w = nullptr;
                    const uint8_t* pf_s = nullptr;
                    resolve_multi_shard(
                        pf_idx - vocab_start,
                        weight_ptrs,
                        scale_ptrs,
                        shard_row_ends,
                        num_shards,
                        w_stride,
                        s_stride,
                        pf_w,
                        pf_s);
                    for (int64_t off = 0; off < dim; off += 64) {
                        __builtin_prefetch(pf_w + off, 0, 1);
                    }
                    __builtin_prefetch(pf_s, 0, 1);
                }
            }
        }

        uint8_t* dst_w = out_fp8 + row * dim;
        uint8_t* dst_s = out_scales + row * s_cols;
        const int64_t lh = row % local_heads;
        if (lh >= local_valid_heads) {
            std::memset(dst_w, 0, static_cast<size_t>(dim));
            std::memset(dst_s, 0, static_cast<size_t>(s_cols));
            continue;
        }
        const int64_t tok = row / local_heads;
        const int64_t idx = local_ids[tok * local_valid_heads + lh];
        if (idx < vocab_start || idx >= vocab_end) {
            std::memset(dst_w, 0, static_cast<size_t>(dim));
            std::memset(dst_s, 0, static_cast<size_t>(s_cols));
            continue;
        }

        const uint8_t* w_row = nullptr;
        const uint8_t* s_row = nullptr;
        resolve_multi_shard(
            idx - vocab_start,
            weight_ptrs,
            scale_ptrs,
            shard_row_ends,
            num_shards,
            w_stride,
            s_stride,
            w_row,
            s_row);
        std::memcpy(dst_w, w_row, static_cast<size_t>(dim));
        std::memcpy(dst_s, s_row, static_cast<size_t>(s_cols));
    }
}
