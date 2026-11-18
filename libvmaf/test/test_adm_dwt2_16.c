/**
 *
 *  Copyright 2026 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "config.h"
#include "cpu.h"
#include "feature/integer_adm.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#if HAVE_AVX512
#include "feature/x86/adm_avx512.h"
#endif
#endif

typedef void (*adm_dwt2_16_fn)(const uint16_t *src, const adm_dwt_band_t *dst,
                               AdmBuffer *buf, int w, int h, int src_stride,
                               int dst_stride, int inp_size_bits);

typedef enum {
    FILL_RANDOM,     /* pseudo-random values from low to high, inclusive */
    FILL_ONE_HIGH,   /* one row in four at high, the rest at low */
    FILL_ONE_LOW,    /* one row in four at low, the rest at high */
} fill_kind;

typedef struct {
    fill_kind kind;
    int phase;       /* row of the period of four that stands out */
    uint32_t low;
    uint32_t high;
} fill_t;

static int16_t tap4(const int16_t *filter, const int64_t *s, int64_t offset,
                    int shift)
{
    int64_t accum = 0;
    for (int k = 0; k < 4; k++)
        accum += (int64_t)filter[k] * s[k];
    return (int16_t)((accum - offset + ((int64_t)1 << (shift - 1))) >> shift);
}

/* The two passes of adm_dwt2_16() in 64-bit arithmetic. */
static void reference_dwt2_16(const AdmBuffer *buf, const uint16_t *src,
                              int16_t *expected[4], int w, int h, int bpc)
{
    const int w_half = (w + 1) / 2;
    const int h_half = (h + 1) / 2;
    int16_t *tmplo = malloc(2 * w * sizeof(int16_t));
    int16_t *tmphi = tmplo + w;
    const int64_t lo_offset = (int64_t)dwt2_db2_coeffs_lo_sum << (bpc - 1);

    for (int i = 0; i < h_half; i++) {
        for (int j = 0; j < w; j++) {
            int64_t s[4];
            for (int k = 0; k < 4; k++)
                s[k] = src[buf->ind_y[k][i] * w + j];
            tmplo[j] = tap4(dwt2_db2_coeffs_lo, s, lo_offset, bpc);
            tmphi[j] = tap4(dwt2_db2_coeffs_hi, s, 0, bpc);
        }
        for (int j = 0; j < w_half; j++) {
            int64_t lo[4], hi[4];
            for (int k = 0; k < 4; k++) {
                lo[k] = tmplo[buf->ind_x[k][j]];
                hi[k] = tmphi[buf->ind_x[k][j]];
            }
            expected[0][i * w_half + j] = tap4(dwt2_db2_coeffs_lo, lo, 0, 16);
            expected[1][i * w_half + j] = tap4(dwt2_db2_coeffs_hi, lo, 0, 16);
            expected[2][i * w_half + j] = tap4(dwt2_db2_coeffs_lo, hi, 0, 16);
            expected[3][i * w_half + j] = tap4(dwt2_db2_coeffs_hi, hi, 0, 16);
        }
    }
    free(tmplo);
}

static int matches_reference(adm_dwt2_16_fn fn, AdmBuffer *buf,
                             const uint16_t *src, int16_t *expected[4],
                             int w, int h, int bpc, const char *name)
{
    const int w_half = (w + 1) / 2;
    const int h_half = (h + 1) / 2;
    const int stride = (int)(buf->ind_size_x >> 2);
    const adm_dwt_band_t *dst = &buf->ref_dwt2;
    const int16_t *bands[4] = { dst->band_a, dst->band_v, dst->band_h, dst->band_d };

    memset(dst->band_a, 0, h_half * stride * sizeof(int16_t));
    memset(dst->band_v, 0, h_half * stride * sizeof(int16_t));
    memset(dst->band_h, 0, h_half * stride * sizeof(int16_t));
    memset(dst->band_d, 0, h_half * stride * sizeof(int16_t));
    fn(src, dst, buf, w, h, w, stride, bpc);

    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < h_half; i++) {
            for (int j = 0; j < w_half; j++) {
                if (bands[b][i * stride + j] != expected[b][i * w_half + j]) {
                    fprintf(stderr,
                            "%s, %dx%d, %d bits, band %d (%d, %d): %d, expected %d\n",
                            name, w, h, bpc, b, i, j, bands[b][i * stride + j],
                            expected[b][i * w_half + j]);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static void fill_source(uint16_t *src, int w, int h, const fill_t *fill)
{
    uint32_t lcg = 1;
    const uint16_t high = (uint16_t)fill->high;
    const uint16_t low = (uint16_t)fill->low;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint16_t v;
            lcg = lcg * 1664525u + 1013904223u;
            switch (fill->kind) {
            case FILL_ONE_HIGH:
                v = (y % 4 == fill->phase) ? high : low;
                break;
            case FILL_ONE_LOW:
                v = (y % 4 == fill->phase) ? low : high;
                break;
            default:
                v = (uint16_t)(low + (lcg >> 8) % (fill->high - fill->low + 1));
                break;
            }
            src[y * w + x] = v;
        }
    }
}

static int run_case(int w, int h, int bpc, const fill_t *fill)
{
    AdmBuffer buf;
    const int w_half = (w + 1) / 2;
    const int h_half = (h + 1) / 2;
    uint16_t *src = malloc((size_t)w * h * sizeof(uint16_t));
    int16_t *expected[4];
    int ok;

    for (int b = 0; b < 4; b++)
        expected[b] = malloc((size_t)w_half * h_half * sizeof(int16_t));

    fill_source(src, w, h, fill);

    if (adm_buffer_alloc(&buf, w, h)) {
        ok = 0;
        goto out;
    }
    dwt2_src_indices_filt(buf.ind_y, buf.ind_x, w, h);
    reference_dwt2_16(&buf, src, expected, w, h, bpc);

    ok = matches_reference(adm_dwt2_16, &buf, src, expected, w, h, bpc,
                           "adm_dwt2_16");
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (ok && (flags & VMAF_X86_CPU_FLAG_AVX2))
        ok = matches_reference(adm_dwt2_16_avx2, &buf, src, expected, w, h,
                               bpc, "adm_dwt2_16_avx2");
#if HAVE_AVX512
    if (ok && (flags & VMAF_X86_CPU_FLAG_AVX512))
        ok = matches_reference(adm_dwt2_16_avx512, &buf, src, expected, w, h,
                               bpc, "adm_dwt2_16_avx512");
#endif
#endif
    adm_buffer_free(&buf);
out:
    for (int b = 0; b < 4; b++)
        free(expected[b]);
    free(src);
    return ok;
}

/* After the first output column, the AVX2 kernel takes steps of 16 columns
 * over the largest multiple of 16 that fits in the row, and the AVX-512 kernel
 * steps of 32 over the largest multiple of 64; the scalar loop finishes the
 * row. With n = (w + 1) / 2 - 1 equal to 3, 15, 17, 31, 33, 47, 49, 63, 65, 95,
 * 97, 127 and 129 the widths cover a row that is only remainder, rows
 * with and without one, and one or more full steps. Odd and even widths cover
 * both parities of the last column.
 *
 * Not covered: widths with n a positive multiple of 16, where the AVX2 kernel
 * ends its last step on the final output column, and with n a positive
 * multiple of 64 for the AVX-512 kernel. There the last column takes mirrored
 * input in the scalar kernel and unmirrored input in the SIMD one. That
 * horizontal pass difference is independent of the vertical pass tested here. */
static const int widths[] = {
    8, 31, 32, 35, 36, 63, 64, 67, 68, 95, 96, 99, 100, 127, 128, 131, 132,
    191, 192, 195, 196, 255, 256, 259, 260
};
#define NUM_WIDTHS ((int)(sizeof(widths) / sizeof(widths[0])))

/* Rows: mirrored top and bottom, odd and even. */
static const int heights[] = { 6, 7, 26 };
#define NUM_HEIGHTS ((int)(sizeof(heights) / sizeof(heights[0])))

static int run_all_sizes(int bpc, const fill_t *fill)
{
    for (int hi = 0; hi < NUM_HEIGHTS; hi++) {
        for (int wi = 0; wi < NUM_WIDTHS; wi++) {
            if (!run_case(widths[wi], heights[hi], bpc, fill))
                return 0;
        }
    }
    return 1;
}

/* The first three low-pass taps add up to (lo[0] + lo[1] + lo[2]). A sample
 * value v takes the sum of three equal samples past INT32_MAX when
 * (lo[0] + lo[1] + lo[2]) * v does. Pin the value the boundary cases use. */
#define OVERFLOW_SAMPLE 42456

static char *test_adm_dwt2_16_overflow_threshold()
{
    const int64_t taps = (int64_t)dwt2_db2_coeffs_lo[0] +
                         dwt2_db2_coeffs_lo[1] + dwt2_db2_coeffs_lo[2];
    mu_assert("42455 must stay within INT32_MAX in the first three taps",
              taps * (OVERFLOW_SAMPLE - 1) <= INT32_MAX);
    mu_assert("42456 must pass INT32_MAX in the first three taps",
              taps * OVERFLOW_SAMPLE > INT32_MAX);
    return NULL;
}

/* Bright, dark and full-range random 16-bit input, and the extremes: all
 * samples at the maximum is the worst case of the positive taps. */
static char *test_adm_dwt2_16_sample_range()
{
    const fill_t bright = { FILL_RANDOM, 0, 57344, 65535 };
    const fill_t dark = { FILL_RANDOM, 0, 0, 8191 };
    const fill_t full = { FILL_RANDOM, 0, 0, 65535 };
    const fill_t maximum = { FILL_RANDOM, 0, 65535, 65535 };
    const fill_t zero = { FILL_RANDOM, 0, 0, 0 };

    mu_assert("bright 16-bit input differs from the 64-bit reference",
              run_all_sizes(16, &bright));
    mu_assert("dark 16-bit input differs from the 64-bit reference",
              run_all_sizes(16, &dark));
    mu_assert("full-range 16-bit input differs from the 64-bit reference",
              run_all_sizes(16, &full));
    mu_assert("maximum 16-bit input differs from the 64-bit reference",
              run_all_sizes(16, &maximum));
    mu_assert("zero 16-bit input differs from the 64-bit reference",
              run_all_sizes(16, &zero));
    return NULL;
}

/* Constant input on both sides of the first sample value whose three-tap sum
 * passes INT32_MAX, and row patterns that put one row of zero or of the value
 * among rows of the other in every position of the vertical window. The
 * window starts at row 4k + 3 for even and 4k + 1 for odd output rows, so the
 * four phases cover each position for both. */
static char *test_adm_dwt2_16_overflow_boundary()
{
    static const uint32_t values[] = { OVERFLOW_SAMPLE - 1, OVERFLOW_SAMPLE,
                                       65535 };
    const fill_t below = { FILL_RANDOM, 0, OVERFLOW_SAMPLE - 1,
                           OVERFLOW_SAMPLE - 1 };
    const fill_t at = { FILL_RANDOM, 0, OVERFLOW_SAMPLE, OVERFLOW_SAMPLE };

    mu_assert("input of 42455 differs from the 64-bit reference",
              run_all_sizes(16, &below));
    mu_assert("input of 42456 differs from the 64-bit reference",
              run_all_sizes(16, &at));

    for (int phase = 0; phase < 4; phase++) {
        for (int v = 0; v < 3; v++) {
            const fill_t one_high = { FILL_ONE_HIGH, phase, 0, values[v] };
            const fill_t one_low = { FILL_ONE_LOW, phase, 0, values[v] };

            mu_assert("one row of the value among zero rows differs from "
                      "the 64-bit reference",
                      run_all_sizes(16, &one_high));
            mu_assert("one row of zero among rows of the value differs from "
                      "the 64-bit reference",
                      run_all_sizes(16, &one_low));
        }
    }
    return NULL;
}

/* The other bit depths that take this path. */
static char *test_adm_dwt2_16_bit_depths()
{
    static const int depths[] = { 10, 12 };

    for (int d = 0; d < 2; d++) {
        const uint32_t top = (1u << depths[d]) - 1;
        const fill_t full = { FILL_RANDOM, 0, 0, top };
        const fill_t maximum = { FILL_RANDOM, 0, top, top };

        mu_assert("full-range input differs from the 64-bit reference",
                  run_all_sizes(depths[d], &full));
        mu_assert("maximum input differs from the 64-bit reference",
                  run_all_sizes(depths[d], &maximum));
    }
    return NULL;
}

char *run_tests()
{
    vmaf_init_cpu();
    mu_run_test(test_adm_dwt2_16_overflow_threshold);
    mu_run_test(test_adm_dwt2_16_sample_range);
    mu_run_test(test_adm_dwt2_16_overflow_boundary);
    mu_run_test(test_adm_dwt2_16_bit_depths);
    return NULL;
}
