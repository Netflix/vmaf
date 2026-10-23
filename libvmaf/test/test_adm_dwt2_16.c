/**
 *
 *  Copyright 2026 Lusoris
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

#define W 70
#define H 26
#define W_HALF ((W + 1) / 2)
#define H_HALF ((H + 1) / 2)
#define BPC 16

typedef void (*adm_dwt2_16_fn)(const uint16_t *src, const adm_dwt_band_t *dst,
                               AdmBuffer *buf, int w, int h, int src_stride,
                               int dst_stride, int inp_size_bits);

static uint16_t src[W * H];
static int16_t expected[4][W_HALF * H_HALF];

static int16_t tap4(const int16_t *filter, const int64_t *s, int64_t offset,
                    int shift)
{
    int64_t accum = 0;
    for (int k = 0; k < 4; k++)
        accum += (int64_t)filter[k] * s[k];
    return (int16_t)((accum - offset + ((int64_t)1 << (shift - 1))) >> shift);
}

/* The two passes of adm_dwt2_16() in 64-bit arithmetic. */
static void reference_dwt2_16(const AdmBuffer *buf)
{
    static int16_t tmplo[W], tmphi[W];
    const int64_t lo_offset = (int64_t)dwt2_db2_coeffs_lo_sum << (BPC - 1);

    for (int i = 0; i < H_HALF; i++) {
        for (int j = 0; j < W; j++) {
            int64_t s[4];
            for (int k = 0; k < 4; k++)
                s[k] = src[buf->ind_y[k][i] * W + j];
            tmplo[j] = tap4(dwt2_db2_coeffs_lo, s, lo_offset, BPC);
            tmphi[j] = tap4(dwt2_db2_coeffs_hi, s, 0, BPC);
        }
        for (int j = 0; j < W_HALF; j++) {
            int64_t lo[4], hi[4];
            for (int k = 0; k < 4; k++) {
                lo[k] = tmplo[buf->ind_x[k][j]];
                hi[k] = tmphi[buf->ind_x[k][j]];
            }
            expected[0][i * W_HALF + j] = tap4(dwt2_db2_coeffs_lo, lo, 0, 16);
            expected[1][i * W_HALF + j] = tap4(dwt2_db2_coeffs_hi, lo, 0, 16);
            expected[2][i * W_HALF + j] = tap4(dwt2_db2_coeffs_lo, hi, 0, 16);
            expected[3][i * W_HALF + j] = tap4(dwt2_db2_coeffs_hi, hi, 0, 16);
        }
    }
}

static int matches_reference(adm_dwt2_16_fn fn, AdmBuffer *buf, const char *name)
{
    const int stride = (int)(buf->ind_size_x >> 2);
    const adm_dwt_band_t *dst = &buf->ref_dwt2;
    const int16_t *bands[4] = { dst->band_a, dst->band_v, dst->band_h, dst->band_d };

    memset(buf->data_buf, 0, buf->ind_size_x * H_HALF * NUM_BUFS_ADM);
    fn(src, dst, buf, W, H, W, stride, BPC);

    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < H_HALF; i++) {
            for (int j = 0; j < W_HALF; j++) {
                if (bands[b][i * stride + j] != expected[b][i * W_HALF + j]) {
                    fprintf(stderr, "%s, band %d (%d, %d): %d, expected %d\n", name, b,
                            i, j, bands[b][i * stride + j], expected[b][i * W_HALF + j]);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int run_case(uint16_t base, uint32_t range)
{
    AdmBuffer buf;
    uint32_t lcg = 1;

    for (int i = 0; i < W * H; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        src[i] = (uint16_t)(base + (lcg >> 8) % range);
    }

    if (adm_buffer_alloc(&buf, W, H)) return 0;
    dwt2_src_indices_filt(buf.ind_y, buf.ind_x, W, H);
    reference_dwt2_16(&buf);

    int ok = matches_reference(adm_dwt2_16, &buf, "adm_dwt2_16");
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (ok && (flags & VMAF_X86_CPU_FLAG_AVX2))
        ok = matches_reference(adm_dwt2_16_avx2, &buf, "adm_dwt2_16_avx2");
#if HAVE_AVX512
    if (ok && (flags & VMAF_X86_CPU_FLAG_AVX512))
        ok = matches_reference(adm_dwt2_16_avx512, &buf, "adm_dwt2_16_avx512");
#endif
#endif
    adm_buffer_free(&buf);
    return ok;
}

/* Samples of 42456 or more take the low-pass sum of the vertical pass past
 * INT32_MAX when the normalization offset is subtracted last. A 32-bit sum
 * that overflows wraps back into range on the usual targets, so in that case
 * only a build with -fsanitize=undefined fails the bright input. */
static char *test_adm_dwt2_16_sample_range()
{
    mu_assert("bright 16-bit input differs from the 64-bit reference",
              run_case(57344, 8192));
    mu_assert("dark 16-bit input differs from the 64-bit reference",
              run_case(0, 8192));
    mu_assert("full-range 16-bit input differs from the 64-bit reference",
              run_case(0, 65536));
    return NULL;
}

char *run_tests()
{
    vmaf_init_cpu();
    mu_run_test(test_adm_dwt2_16_sample_range);
    return NULL;
}
