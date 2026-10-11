// SPDX-License-Identifier: Apache-2.0
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

static int pread_full(int fd, unsigned char *output, size_t length, int64_t offset) {
    size_t completed = 0;
    while (completed < length) {
        ssize_t result = pread(
            fd,
            output + completed,
            length - completed,
            (off_t)(offset + (int64_t)completed));
        if (result > 0) {
            completed += (size_t)result;
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        return -1;
    }
    return 0;
}

int ple_pread_rows(
    const int *file_descriptors,
    const int64_t *offsets,
    unsigned char *output,
    size_t row_count,
    size_t row_bytes,
    int thread_count) {
    int failures = 0;
    if (thread_count < 1) {
        thread_count = 1;
    }
    if ((size_t)thread_count > row_count && row_count > 0) {
        thread_count = (int)row_count;
    }
#pragma omp parallel for num_threads(thread_count) schedule(static) reduction(+ : failures)
    for (size_t index = 0; index < row_count; ++index) {
        failures += pread_full(
            file_descriptors[index],
            output + index * row_bytes,
            row_bytes,
            offsets[index]) != 0;
    }
    return failures == 0 ? 0 : -failures;
}

static inline int64_t remainder_i64(int64_t value, int64_t modulus) {
    // torch.remainder for a positive modulus: result in [0, modulus).
    int64_t result = value % modulus;
    return result < 0 ? result + modulus : result;
}

// Bit-exact port of Qwen4ExpNGramEmbedding.compute_ngram_ids (CPU path) for
// the PLE offload worker. Per request r and local column j the n-gram window
// is the ngram_context row followed by the request's own tokens; a token
// shifted back by s is replaced by EOS when it lies before the window or
// before the most recent EOS strictly preceding the current position. For
// each n in [2, ngram_size] the mixed hash is the XOR of token[pos - i] *
// multiplier[i] (int64, wrapping), and head h emits remainder(mixed,
// size[h]) + offset[h]. Output: [num_tokens, (ngram_size - 1) * heads].
// Returns 0, or -1 on an invalid layout.
int ple_ngram_ids(
    const int32_t *input_ids,
    int64_t num_tokens,
    const int32_t *query_start_loc,
    int64_t num_reqs,
    const int32_t *ngram_context,
    int64_t context_cols,
    const int64_t *multipliers,
    int64_t ngram_size,
    const int64_t *head_sizes,
    const int64_t *head_offsets,
    int64_t heads_per_ngram,
    int64_t eos_token_id,
    int64_t *output) {
    if (ngram_size < 2 || ngram_size > 8 || context_cols != ngram_size - 1 ||
        heads_per_ngram <= 0 || num_reqs <= 0 || num_tokens < 0 ||
        query_start_loc[num_reqs] > num_tokens) {
        return -1;
    }
    const int64_t heads = (ngram_size - 1) * heads_per_ngram;
    for (int64_t r = 0; r < num_reqs; ++r) {
        const int64_t begin = query_start_loc[r];
        // Rows past query_start_loc[num_reqs] (graph padding) continue the
        // last request, exactly as the torch path packs them.
        const int64_t end = r == num_reqs - 1 ? num_tokens : query_start_loc[r + 1];
        if (begin < 0 || end < begin || end > num_tokens) {
            return -1;
        }
        // Window: context_cols context tokens, then this request's tokens.
        const int64_t window = context_cols + (end - begin);
        int64_t last_eos = -1;  // most recent EOS position strictly before pos
        int64_t prev_eos_inclusive = -1;
        for (int64_t pos = 0; pos < window; ++pos) {
            const int64_t token = pos < context_cols
                ? (int64_t)ngram_context[r * context_cols + pos]
                : (int64_t)input_ids[begin + pos - context_cols];
            last_eos = prev_eos_inclusive;
            if (token == eos_token_id) {
                prev_eos_inclusive = pos;
            }
            if (pos < context_cols) {
                continue;
            }
            const int64_t in_segment = pos - last_eos - 1;
            int64_t shifted[8];
            for (int64_t s = 0; s < ngram_size; ++s) {
                if (s == 0) {
                    shifted[0] = token;
                } else if (pos - s >= 0 && in_segment >= s) {
                    const int64_t src = pos - s;
                    shifted[s] = src < context_cols
                        ? (int64_t)ngram_context[r * context_cols + src]
                        : (int64_t)input_ids[begin + src - context_cols];
                } else {
                    shifted[s] = eos_token_id;
                }
            }
            int64_t *row = output + (begin + pos - context_cols) * heads;
            for (int64_t n = 2; n <= ngram_size; ++n) {
                uint64_t mixed = (uint64_t)shifted[0] * (uint64_t)multipliers[0];
                for (int64_t i = 1; i < n; ++i) {
                    mixed ^= (uint64_t)shifted[i] * (uint64_t)multipliers[i];
                }
                const int64_t base = (n - 2) * heads_per_ngram;
                for (int64_t h = 0; h < heads_per_ngram; ++h) {
                    row[base + h] = remainder_i64((int64_t)mixed, head_sizes[base + h]) +
                        head_offsets[base + h];
                }
            }
        }
    }
    return 0;
}
