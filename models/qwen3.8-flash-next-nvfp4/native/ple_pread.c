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
