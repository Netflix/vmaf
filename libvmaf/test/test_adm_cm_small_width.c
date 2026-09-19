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

typedef float (*adm_cm_fn)(AdmBuffer *buf, int w, int h, int src_stride,
                           int csf_a_stride, double adm_norm_view_dist,
                           int adm_ref_display_height, int adm_csf_mode,
                           double adm_csf_scale, double adm_csf_diag_scale,
                           double adm_noise_weight, bool measure_aim);

static uint32_t lcg_state;

static int16_t lcg_sample(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int16_t)((lcg_state >> 8) % 801u) - 400;
}

static void fill_band(adm_dwt_band_t *band, int rows, int stride)
{
    for (int i = 0; i < rows * stride; i++) {
        band->band_h[i] = lcg_sample();
        band->band_v[i] = lcg_sample();
        band->band_d[i] = lcg_sample();
    }
}

/* As adm_csf() leaves it: csf_f holds 1/30 of the magnitude of csf_a. */
static void fill_csf_f(adm_dwt_band_t *csf_f, const adm_dwt_band_t *csf_a,
                       int rows, int stride)
{
    for (int i = 0; i < rows * stride; i++) {
        csf_f->band_h[i] = (int16_t)(abs(csf_a->band_h[i]) / 30);
        csf_f->band_v[i] = (int16_t)(abs(csf_a->band_v[i]) / 30);
        csf_f->band_d[i] = (int16_t)(abs(csf_a->band_d[i]) / 30);
    }
}

static float run_adm_cm(adm_cm_fn fn, int w, int h)
{
    AdmBuffer buf;
    if (adm_buffer_alloc(&buf, w, h)) return NAN;

    const int stride = (int)(buf.ind_size_x >> 2);
    const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

    lcg_state = (uint32_t)(w * 4099 + h);
    fill_band(&buf.decouple_r, h_half, stride);
    fill_band(&buf.csf_a, h_half, stride);
    fill_csf_f(&buf.csf_f, &buf.csf_a, h_half, stride);

    const float score =
        fn(&buf, w_half, h_half, stride, stride, DEFAULT_ADM_NORM_VIEW_DIST,
           DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
           DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
           DEFAULT_ADM_NOISE_WEIGHT, false);

    adm_buffer_free(&buf);
    return score;
}

static int matches_scalar(adm_cm_fn fn, const char *name)
{
    /* Frame widths 17 to 32 give a scale 0 band of 9 to 16 samples, for which
     * the cube shift in adm_cm() is 0. 40 and 64 are controls. */
    static const int widths[] = { 17, 18, 20, 23, 24, 27, 31, 32, 40, 64 };

    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        const int w = widths[i], h = 36;
        const float expected = run_adm_cm(adm_cm, w, h);
        const float actual = run_adm_cm(fn, w, h);
        if (!(fabsf(expected - actual) <= 1e-6f * fabsf(expected))) {
            fprintf(stderr, "%s, frame width %d: %.9g, scalar %.9g\n",
                    name, w, actual, expected);
            return 0;
        }
    }
    return 1;
}

static char *test_adm_cm_small_width_matches_scalar()
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();

    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        mu_assert("adm_cm_avx2 differs from adm_cm", matches_scalar(adm_cm_avx2, "adm_cm_avx2"));
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        mu_assert("adm_cm_avx512 differs from adm_cm", matches_scalar(adm_cm_avx512, "adm_cm_avx512"));
#endif
#endif
    return NULL;
}

char *run_tests()
{
    vmaf_init_cpu();
    mu_run_test(test_adm_cm_small_width_matches_scalar);
    return NULL;
}
