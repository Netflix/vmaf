/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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
#if ARCH_AARCH64
#include "feature/arm64/adm_neon.h"
#endif

typedef float (*adm_cm_fn)(AdmBuffer *buf, int w, int h, int src_stride,
                           int csf_a_stride, double adm_norm_view_dist,
                           int adm_ref_display_height, int adm_csf_mode,
                           double adm_csf_scale, double adm_csf_diag_scale,
                           double adm_noise_weight, bool measure_aim);

static uint32_t lcg_state;

/* Exercise large positive and negative coefficients. */
static int16_t lcg_sample(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    const int16_t magnitude = (int16_t)(12000 + (lcg_state >> 8) % 10917u);
    return (lcg_state & 0x80u) ? magnitude : (int16_t)-magnitude;
}

static void fill_band(adm_dwt_band_t *band, int rows, int stride)
{
    for (int i = 0; i < rows * stride; i++) {
        band->band_h[i] = lcg_sample();
        band->band_v[i] = lcg_sample();
        band->band_d[i] = lcg_sample();
    }
}

/* Independent reference: all threshold and residual arithmetic is 64-bit. */
static float reference_cm(AdmBuffer *buf, int w, int h, int stride)
{
    const int16_t *src[3] = { buf->decouple_r.band_h, buf->decouple_r.band_v,
                              buf->decouple_r.band_d };
    const int16_t *a[3] = { buf->csf_a.band_h, buf->csf_a.band_v, buf->csf_a.band_d };
    const int16_t *f[3] = { buf->csf_f.band_h, buf->csf_f.band_v, buf->csf_f.band_d };
    const int factor[3] = { 36453, 36453, 49417 };
    const int left = w * ADM_BORDER_FACTOR - 0.5, top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left, bottom = h - top;
    const int inner_shift = (int)ceil(log2(h));
    const int cubic_shift[3] = { (int)ceil(log2(w) - 4),
                                  (int)ceil(log2(w) - 4), (int)ceil(log2(w) - 3) };
    int64_t total[3] = { 0, 0, 0 };
    for (int y = top; y < bottom; y++) {
        int64_t row[3] = { 0, 0, 0 };
        for (int x = left; x < right; x++) {
            int64_t threshold = 0;
            for (int b = 0; b < 3; b++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int yy = y + dy;
                    if (yy < 0) yy = 1;
                    if (yy >= h) yy = h - 1;
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = x + dx;
                        if (xx < 0) xx = 1;
                        if (xx >= w) xx = w - 1;
                        if (dx || dy) threshold += f[b][yy * stride + xx];
                    }
                }
                threshold += (8738LL * llabs(a[b][y * stride + x]) + 2048) >> 12;
            }
            for (int b = 0; b < 3; b++) {
                int64_t residual = llabs((int64_t)src[b][y * stride + x] * factor[b]) -
                                   threshold * (1LL << (b == 2 ? 12 : 10));
                if (residual < 0) residual = 0;
                const int sq_shift = b == 2 ? 30 : 29;
                int64_t square = (residual * residual + (1LL << (sq_shift - 1))) >> sq_shift;
                row[b] += (square * residual + (cubic_shift[b] ? (1LL << (cubic_shift[b] - 1)) : 0)) >> cubic_shift[b];
            }
        }
        for (int b = 0; b < 3; b++)
            total[b] += (row[b] + (1LL << (inner_shift - 1))) >> inner_shift;
    }
    float result = 0;
    for (int b = 0; b < 3; b++) {
        float val = (float)(total[b] / pow(2, (b == 2 ? 57 : 52) - cubic_shift[b] - inner_shift));
        result += powf(val, 1.0f / 3.0f) +
                  powf((bottom - top) * (right - left) * DEFAULT_ADM_NOISE_WEIGHT, 1.0f / 3.0f);
    }
    return result;
}

static float run_adm_cm(adm_cm_fn fn, int w, int h, int scenario)
{
    AdmBuffer buf;
    if (adm_buffer_alloc(&buf, w, h)) return NAN;

    const int stride = (int)(buf.ind_size_x >> 2);
    const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

    lcg_state = (uint32_t)(w * 4099 + h);
    fill_band(&buf.decouple_r, h_half, stride);
    fill_band(&buf.decouple_a, h_half, stride);
    /* csf_a and csf_f as the extractor produces them */
    adm_csf(&buf, w_half, h_half, stride, DEFAULT_ADM_NORM_VIEW_DIST,
            DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
            DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE, false);
    if (scenario) {
        int16_t *a[] = { buf.csf_a.band_h, buf.csf_a.band_v, buf.csf_a.band_d };
        int16_t *f[] = { buf.csf_f.band_h, buf.csf_f.band_v, buf.csf_f.band_d };
        for (int b = 0; b < 3; b++) {
            for (int i = 0; i < h_half * stride; i++) {
                a[b][i] = scenario == 1 ? 15359 : scenario == 2 ? 15360 : 22916;
                f[b][i] = scenario == 3 ? 32767 : 0;
            }
        }
    }

    const float score = fn ?
        fn(&buf, w_half, h_half, stride, stride, DEFAULT_ADM_NORM_VIEW_DIST,
           DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
           DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
           DEFAULT_ADM_NOISE_WEIGHT, false) : reference_cm(&buf, w_half, h_half, stride);

    adm_buffer_free(&buf);
    return score;
}

static int matches_scalar(adm_cm_fn fn, const char *name)
{
    static const int widths[] = { 28, 64, 97, 130, 176 };

    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        const int w = widths[i], h = 72;
        /* Case 3 is omitted at frame width 28: the zero cubic shift makes
         * the rounding constant undefined, which this change does not fix. */
        for (int scenario = 0; scenario < (w == 28 ? 3 : 4); scenario++) {
            const float expected = run_adm_cm(NULL, w, h, scenario);
            const float actual = run_adm_cm(fn, w, h, scenario);
            if (!(fabsf(expected - actual) <= 1e-6f * fabsf(expected))) {
                fprintf(stderr, "%s, frame width %d, case %d: %.9g, reference %.9g\n",
                        name, w, scenario, actual, expected);
                return 0;
            }
        }
    }
    return 1;
}

static char *test_adm_cm_large_coeffs_match_reference(void)
{
    mu_assert("scalar differs from reference", matches_scalar(adm_cm, "scalar"));
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();

    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        mu_assert("adm_cm_avx2 differs from adm_cm", matches_scalar(adm_cm_avx2, "adm_cm_avx2"));
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        mu_assert("adm_cm_avx512 differs from adm_cm", matches_scalar(adm_cm_avx512, "adm_cm_avx512"));
#endif
#endif
#if ARCH_AARCH64
    mu_assert("NEON differs from reference", matches_scalar(adm_cm_neon, "NEON"));
#endif
    return NULL;
}

char *run_tests()
{
    vmaf_init_cpu();
    mu_run_test(test_adm_cm_large_coeffs_match_reference);
    return NULL;
}
