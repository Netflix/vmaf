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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "config.h"
#include "cpu.h"
#include "feature/integer_adm.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#if HAVE_AVX512
#include "feature/x86/adm_avx512.h"
#endif
#elif ARCH_AARCH64
#include "feature/arm64/adm_neon.h"
#endif

typedef void (*adm_dwt2_8_fn)(const uint8_t *src, const adm_dwt_band_t *dst,
                               AdmBuffer *buf, int w, int h, int src_stride,
                               int dst_stride);
typedef void (*adm_dwt2_16_fn)(const uint16_t *src, const adm_dwt_band_t *dst,
                                AdmBuffer *buf, int w, int h, int src_stride,
                                int dst_stride, int inp_size_bits);
typedef void (*adm_decouple_fn)(AdmBuffer *buf, int w, int h, int stride,
                                 double adm_enhn_gain_limit,
                                 int32_t *adm_div_lookup);
typedef void (*adm_csf_fn)(AdmBuffer *buf, int w, int h, int stride,
                            double adm_norm_view_dist,
                            int adm_ref_display_height, int adm_csf_mode,
                            double adm_csf_scale, double adm_csf_diag_scale,
                            bool measure_aim);
typedef float (*adm_cm_fn)(AdmBuffer *buf, int w, int h, int src_stride,
                            int csf_a_stride, double adm_norm_view_dist,
                            int adm_ref_display_height, int adm_csf_mode,
                            double adm_csf_scale, double adm_csf_diag_scale,
                            double adm_noise_weight, bool measure_aim);
typedef void (*adm_dwt2_s123_fn)(const int32_t *i4_ref_scale,
                                  const int32_t *i4_curr_dis, AdmBuffer *buf,
                                  int w, int h, int ref_stride,
                                  int dis_stride, int dst_stride, int scale);
typedef float (*adm_csf_den_scale_fn)(const adm_dwt_band_t *src, int w,
                                       int h, int src_stride,
                                       double adm_norm_view_dist,
                                       int adm_ref_display_height,
                                       int adm_csf_mode, double adm_csf_scale,
                                       double adm_csf_diag_scale,
                                       double adm_noise_weight);
typedef float (*adm_csf_den_s123_fn)(const i4_adm_dwt_band_t *src, int scale,
                                      int w, int h, int src_stride,
                                      double adm_norm_view_dist,
                                      int adm_ref_display_height,
                                      int adm_csf_mode, double adm_csf_scale,
                                      double adm_csf_diag_scale,
                                      double adm_noise_weight);
typedef void (*i4_adm_csf_fn)(AdmBuffer *buf, int scale, int w, int h,
                               int stride, double adm_norm_view_dist,
                               int adm_ref_display_height, int adm_csf_mode,
                               double adm_csf_scale, double adm_csf_diag_scale,
                               bool measure_aim);
typedef float (*i4_adm_cm_fn)(AdmBuffer *buf, int w, int h, int src_stride,
                               int csf_a_stride, int scale,
                               double adm_norm_view_dist,
                               int adm_ref_display_height, int adm_csf_mode,
                               double adm_csf_scale, double adm_csf_diag_scale,
                               double adm_noise_weight, bool measure_aim);

static adm_dwt2_8_fn get_dwt2_8(unsigned cpu_flags, int w)
{
    adm_dwt2_8_fn fn = adm_dwt2_8;
#if ARCH_X86
    if ((cpu_flags & VMAF_X86_CPU_FLAG_AVX2) && !(w % 8))
        fn = adm_dwt2_8_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_dwt2_8_avx512;
#endif
#elif ARCH_AARCH64
    if ((cpu_flags & VMAF_ARM_CPU_FLAG_NEON) && !(w % 8))
        fn = adm_dwt2_8_neon;
#endif
    return fn;
}

static adm_dwt2_16_fn get_dwt2_16(unsigned cpu_flags)
{
#if ARCH_X86
    adm_dwt2_16_fn fn = adm_dwt2_16;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_dwt2_16_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_dwt2_16_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_decouple_fn get_decouple(unsigned cpu_flags)
{
#if ARCH_X86
    adm_decouple_fn fn = adm_decouple;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_decouple_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_decouple_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_csf_fn get_csf(unsigned cpu_flags)
{
#if ARCH_X86
    adm_csf_fn fn = adm_csf;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_csf_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_csf_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_cm_fn get_cm(unsigned cpu_flags)
{
#if ARCH_X86
    adm_cm_fn fn = adm_cm;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_cm_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_cm_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_decouple_fn get_decouple_s123(unsigned cpu_flags)
{
#if ARCH_X86
    adm_decouple_fn fn = adm_decouple_s123;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_decouple_s123_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_decouple_s123_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_dwt2_s123_fn get_dwt2_s123_combined(unsigned cpu_flags)
{
#if ARCH_X86
    adm_dwt2_s123_fn fn = adm_dwt2_s123_combined;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_dwt2_s123_combined_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_dwt2_s123_combined_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_csf_den_scale_fn get_csf_den_scale(unsigned cpu_flags)
{
#if ARCH_X86
    adm_csf_den_scale_fn fn = adm_csf_den_scale;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_csf_den_scale_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_csf_den_scale_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static adm_csf_den_s123_fn get_csf_den_s123(unsigned cpu_flags)
{
#if ARCH_X86
    adm_csf_den_s123_fn fn = adm_csf_den_s123;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = adm_csf_den_s123_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = adm_csf_den_s123_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static i4_adm_csf_fn get_i4_csf(unsigned cpu_flags)
{
#if ARCH_X86
    i4_adm_csf_fn fn = i4_adm_csf;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = i4_adm_csf_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = i4_adm_csf_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static i4_adm_cm_fn get_i4_cm(unsigned cpu_flags)
{
#if ARCH_X86
    i4_adm_cm_fn fn = i4_adm_cm;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = i4_adm_cm_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = i4_adm_cm_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static void fill_band(int16_t *band, int rows, int stride)
{
    if (!band) return;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < stride; j++)
            band[i * stride + j] =
                (int16_t) ((checkasm_rand_uint32() % 16001) - 8000);
}

static void copy_band(int16_t *dst, const int16_t *src, int rows, int stride)
{
    if (!dst || !src) return;
    memcpy(dst, src, (size_t) rows * stride * sizeof(int16_t));
}

static void check2d_band(const int16_t *a, const int16_t *b, int cols,
                          int rows, int stride, const char *name)
{
    if (!a || !b) return;
    checkasm_check2d(int16_t, a, stride, b, stride, cols, rows, name);
}

static void fill_band_i32(int32_t *band, int rows, int stride)
{
    if (!band) return;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < stride; j++)
            band[i * stride + j] =
                (int32_t) ((checkasm_rand_uint32() % 16001) - 8000);
}

static void copy_band_i32(int32_t *dst, const int32_t *src, int rows,
                           int stride)
{
    if (!dst || !src) return;
    memcpy(dst, src, (size_t) rows * stride * sizeof(int32_t));
}

static void check2d_band_i32(const int32_t *a, const int32_t *b, int cols,
                              int rows, int stride, const char *name)
{
    if (!a || !b) return;
    checkasm_check2d(int32_t, a, stride, b, stride, cols, rows, name);
}

static const struct { int w, h; } dwt2_sizes[] = {
    { 16, 16 },
    { 32, 24 },
    { 64, 48 },
    { 20, 18 },
};

static void check_adm_dwt2(void)
{
    for (size_t i = 0; i < sizeof(dwt2_sizes) / sizeof(*dwt2_sizes); i++) {
        const int w = dwt2_sizes[i].w, h = dwt2_sizes[i].h;
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        AdmBuffer buf;
        if (adm_buffer_alloc(&buf, w, h)) continue;
        dwt2_src_indices_filt(buf.ind_y, buf.ind_x, w, h);
        const int dst_stride = (int) (buf.ind_size_x >> 2);

        {
            checkasm_declare(void, const uint8_t *, const adm_dwt_band_t *,
                              AdmBuffer *, int, int, int, int);
            uint8_t *src = malloc((size_t) h * w);
            for (int i2 = 0; i2 < h * w; i2++)
                src[i2] = (uint8_t) checkasm_rand_uint32();

            if (checkasm_check_func(get_dwt2_8(checkasm_get_cpu_flags(), w),
                                     "adm_dwt2_8_%dx%d", w, h))
            {
                checkasm_call_ref(src, &buf.ref_dwt2, &buf, w, h, w,
                                   dst_stride);
                checkasm_call_new(src, &buf.dis_dwt2, &buf, w, h, w,
                                   dst_stride);

                check2d_band(buf.ref_dwt2.band_a, buf.dis_dwt2.band_a,
                             w_half, h_half, dst_stride, "band_a");
                check2d_band(buf.ref_dwt2.band_h, buf.dis_dwt2.band_h,
                             w_half, h_half, dst_stride, "band_h");
                check2d_band(buf.ref_dwt2.band_v, buf.dis_dwt2.band_v,
                             w_half, h_half, dst_stride, "band_v");
                check2d_band(buf.ref_dwt2.band_d, buf.dis_dwt2.band_d,
                             w_half, h_half, dst_stride, "band_d");

                checkasm_bench_new(src, &buf.dis_dwt2, &buf, w, h, w,
                                    dst_stride);
            }
            free(src);
        }

        {
            checkasm_declare(void, const uint8_t *, const adm_dwt_band_t *,
                              AdmBuffer *, int, int, int, int, int);
            uint16_t *src = malloc((size_t) h * w * sizeof(uint16_t));
            for (int i2 = 0; i2 < h * w; i2++)
                src[i2] = (uint16_t) checkasm_rand_uint32() & 0x3ff;

            if (checkasm_check_func(get_dwt2_16(checkasm_get_cpu_flags()),
                                     "adm_dwt2_16_%dx%d", w, h))
            {
                checkasm_call_ref((const uint8_t *) src, &buf.ref_dwt2, &buf,
                                   w, h, w, dst_stride, 10);
                checkasm_call_new((const uint8_t *) src, &buf.dis_dwt2, &buf,
                                   w, h, w, dst_stride, 10);

                check2d_band(buf.ref_dwt2.band_a, buf.dis_dwt2.band_a,
                             w_half, h_half, dst_stride, "band_a");
                check2d_band(buf.ref_dwt2.band_h, buf.dis_dwt2.band_h,
                             w_half, h_half, dst_stride, "band_h");
                check2d_band(buf.ref_dwt2.band_v, buf.dis_dwt2.band_v,
                             w_half, h_half, dst_stride, "band_v");
                check2d_band(buf.ref_dwt2.band_d, buf.dis_dwt2.band_d,
                             w_half, h_half, dst_stride, "band_d");

                checkasm_bench_new((const uint8_t *) src, &buf.dis_dwt2, &buf,
                                    w, h, w, dst_stride, 10);
            }
            free(src);
        }

        adm_buffer_free(&buf);
    }
}

static const struct { int w, h; } post_dwt_sizes[] = {
    { 16, 16 },
    { 33, 21 },
    { 65, 49 },
    { 32, 20 },
    { 64, 48 },
};

static void check_adm_decouple(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        fill_band(buf_c.ref_dwt2.band_a, h_half, stride);
        fill_band(buf_c.ref_dwt2.band_h, h_half, stride);
        fill_band(buf_c.ref_dwt2.band_v, h_half, stride);
        fill_band(buf_c.ref_dwt2.band_d, h_half, stride);
        fill_band(buf_c.dis_dwt2.band_a, h_half, stride);
        fill_band(buf_c.dis_dwt2.band_h, h_half, stride);
        fill_band(buf_c.dis_dwt2.band_v, h_half, stride);
        fill_band(buf_c.dis_dwt2.band_d, h_half, stride);

        copy_band(buf_a.ref_dwt2.band_a, buf_c.ref_dwt2.band_a, h_half, stride);
        copy_band(buf_a.ref_dwt2.band_h, buf_c.ref_dwt2.band_h, h_half, stride);
        copy_band(buf_a.ref_dwt2.band_v, buf_c.ref_dwt2.band_v, h_half, stride);
        copy_band(buf_a.ref_dwt2.band_d, buf_c.ref_dwt2.band_d, h_half, stride);
        copy_band(buf_a.dis_dwt2.band_a, buf_c.dis_dwt2.band_a, h_half, stride);
        copy_band(buf_a.dis_dwt2.band_h, buf_c.dis_dwt2.band_h, h_half, stride);
        copy_band(buf_a.dis_dwt2.band_v, buf_c.dis_dwt2.band_v, h_half, stride);
        copy_band(buf_a.dis_dwt2.band_d, buf_c.dis_dwt2.band_d, h_half, stride);

        checkasm_declare(void, AdmBuffer *, int, int, int, double, int32_t *);

        if (checkasm_check_func(get_decouple(checkasm_get_cpu_flags()),
                                 "adm_decouple_%dx%d", w, h))
        {
            checkasm_call_ref(&buf_c, w_half, h_half, stride,
                               DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);
            checkasm_call_new(&buf_a, w_half, h_half, stride,
                               DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);

            check2d_band(buf_c.decouple_r.band_h, buf_a.decouple_r.band_h,
                         w_half, h_half, stride, "decouple_r.band_h");
            check2d_band(buf_c.decouple_r.band_v, buf_a.decouple_r.band_v,
                         w_half, h_half, stride, "decouple_r.band_v");
            check2d_band(buf_c.decouple_r.band_d, buf_a.decouple_r.band_d,
                         w_half, h_half, stride, "decouple_r.band_d");
            check2d_band(buf_c.decouple_a.band_h, buf_a.decouple_a.band_h,
                         w_half, h_half, stride, "decouple_a.band_h");
            check2d_band(buf_c.decouple_a.band_v, buf_a.decouple_a.band_v,
                         w_half, h_half, stride, "decouple_a.band_v");
            check2d_band(buf_c.decouple_a.band_d, buf_a.decouple_a.band_d,
                         w_half, h_half, stride, "decouple_a.band_d");

            checkasm_bench_new(&buf_a, w_half, h_half, stride,
                                DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_csf(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        fill_band(buf_c.decouple_r.band_h, h_half, stride);
        fill_band(buf_c.decouple_r.band_v, h_half, stride);
        fill_band(buf_c.decouple_r.band_d, h_half, stride);
        fill_band(buf_c.decouple_a.band_h, h_half, stride);
        fill_band(buf_c.decouple_a.band_v, h_half, stride);
        fill_band(buf_c.decouple_a.band_d, h_half, stride);

        copy_band(buf_a.decouple_r.band_h, buf_c.decouple_r.band_h, h_half, stride);
        copy_band(buf_a.decouple_r.band_v, buf_c.decouple_r.band_v, h_half, stride);
        copy_band(buf_a.decouple_r.band_d, buf_c.decouple_r.band_d, h_half, stride);
        copy_band(buf_a.decouple_a.band_h, buf_c.decouple_a.band_h, h_half, stride);
        copy_band(buf_a.decouple_a.band_v, buf_c.decouple_a.band_v, h_half, stride);
        copy_band(buf_a.decouple_a.band_d, buf_c.decouple_a.band_d, h_half, stride);

        checkasm_declare(void, AdmBuffer *, int, int, int, double, int,
                          int, double, double, bool);

        for (int aim = 0; aim <= 1; aim++) {
            if (checkasm_check_func(get_csf(checkasm_get_cpu_flags()),
                                     "adm_csf_%dx%d_aim%d", w, h, aim))
            {
                checkasm_call_ref(&buf_c, w_half, h_half, stride,
                                   DEFAULT_ADM_NORM_VIEW_DIST,
                                   DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                   DEFAULT_ADM_CSF_MODE, DEFAULT_ADM_CSF_SCALE,
                                   DEFAULT_ADM_CSF_DIAG_SCALE, (bool) aim);
                checkasm_call_new(&buf_a, w_half, h_half, stride,
                                   DEFAULT_ADM_NORM_VIEW_DIST,
                                   DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                   DEFAULT_ADM_CSF_MODE, DEFAULT_ADM_CSF_SCALE,
                                   DEFAULT_ADM_CSF_DIAG_SCALE, (bool) aim);

                check2d_band(buf_c.csf_a.band_h, buf_a.csf_a.band_h, w_half,
                             h_half, stride, "csf_a.band_h");
                check2d_band(buf_c.csf_a.band_v, buf_a.csf_a.band_v, w_half,
                             h_half, stride, "csf_a.band_v");
                check2d_band(buf_c.csf_a.band_d, buf_a.csf_a.band_d, w_half,
                             h_half, stride, "csf_a.band_d");
                check2d_band(buf_c.csf_f.band_h, buf_a.csf_f.band_h, w_half,
                             h_half, stride, "csf_f.band_h");
                check2d_band(buf_c.csf_f.band_v, buf_a.csf_f.band_v, w_half,
                             h_half, stride, "csf_f.band_v");
                check2d_band(buf_c.csf_f.band_d, buf_a.csf_f.band_d, w_half,
                             h_half, stride, "csf_f.band_d");

                checkasm_bench_new(&buf_a, w_half, h_half, stride,
                                    DEFAULT_ADM_NORM_VIEW_DIST,
                                    DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                    DEFAULT_ADM_CSF_MODE,
                                    DEFAULT_ADM_CSF_SCALE,
                                    DEFAULT_ADM_CSF_DIAG_SCALE, (bool) aim);
            }
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_cm(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int h_half = (h + 1) / 2;
        const int w_half = (w + 1) / 2;

        adm_dwt_band_t *c_bands[4] = { &buf_c.decouple_r, &buf_c.decouple_a,
                                        &buf_c.csf_f, &buf_c.csf_a };
        adm_dwt_band_t *a_bands[4] = { &buf_a.decouple_r, &buf_a.decouple_a,
                                        &buf_a.csf_f, &buf_a.csf_a };
        for (int b = 0; b < 4; b++) {
            fill_band(c_bands[b]->band_h, h_half, stride);
            fill_band(c_bands[b]->band_v, h_half, stride);
            fill_band(c_bands[b]->band_d, h_half, stride);
            copy_band(a_bands[b]->band_h, c_bands[b]->band_h, h_half, stride);
            copy_band(a_bands[b]->band_v, c_bands[b]->band_v, h_half, stride);
            copy_band(a_bands[b]->band_d, c_bands[b]->band_d, h_half, stride);
        }

        checkasm_declare(float, AdmBuffer *, int, int, int, int, double, int,
                          int, double, double, double, bool);

        for (int aim = 0; aim <= 1; aim++) {
            if (checkasm_check_func(get_cm(checkasm_get_cpu_flags()),
                                     "adm_cm_%dx%d_aim%d", w, h, aim))
            {
                const float ref = checkasm_call_ref(
                    &buf_c, w_half, h_half, stride, stride,
                    DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);
                const float new = checkasm_call_new(
                    &buf_a, w_half, h_half, stride, stride,
                    DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);

                const float tol = 1e-4f * (fabsf(ref) + 1.0f);
                if (fabsf(ref - new) > tol) {
                    if (checkasm_fail())
                        fprintf(stderr, "expected %f, got %f\n", ref, new);
                }

                checkasm_bench_new(&buf_a, w_half, h_half, stride, stride,
                                    DEFAULT_ADM_NORM_VIEW_DIST,
                                    DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                    DEFAULT_ADM_CSF_MODE,
                                    DEFAULT_ADM_CSF_SCALE,
                                    DEFAULT_ADM_CSF_DIAG_SCALE,
                                    DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);
            }
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_dwt2_s123(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        dwt2_src_indices_filt(buf_c.ind_y, buf_c.ind_x, w, h);
        dwt2_src_indices_filt(buf_a.ind_y, buf_a.ind_x, w, h);
        const int stride = (int) (buf_c.ind_size_x >> 2);
        /* The bands hold (w + 1) / 2 samples per row, but the source planes
           hold w. integer_compute_adm() passes the stride of the previous
           scale, which is at least w, for both. */
        const int src_stride = stride * 2;
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        int32_t *i4_ref = malloc((size_t) h * src_stride * sizeof(int32_t));
        int32_t *i4_dis = malloc((size_t) h * src_stride * sizeof(int32_t));
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < src_stride; c++) {
                i4_ref[r * src_stride + c] =
                    (int32_t) ((checkasm_rand_uint32() % 16001) - 8000);
                i4_dis[r * src_stride + c] =
                    (int32_t) ((checkasm_rand_uint32() % 16001) - 8000);
            }
        }

        checkasm_declare(void, const int32_t *, const int32_t *, AdmBuffer *,
                          int, int, int, int, int, int);

        for (int scale = 1; scale <= 3; scale++) {
            if (checkasm_check_func(
                    get_dwt2_s123_combined(checkasm_get_cpu_flags()),
                    "adm_dwt2_s123_%dx%d_scale%d", w, h, scale))
            {
                checkasm_call_ref(i4_ref, i4_dis, &buf_c, w, h, src_stride,
                                   src_stride, stride, scale);
                checkasm_call_new(i4_ref, i4_dis, &buf_a, w, h, src_stride,
                                   src_stride, stride, scale);

                check2d_band_i32(buf_c.i4_ref_dwt2.band_a,
                                 buf_a.i4_ref_dwt2.band_a, w_half, h_half,
                                 stride, "i4_ref_dwt2.band_a");
                check2d_band_i32(buf_c.i4_ref_dwt2.band_h,
                                 buf_a.i4_ref_dwt2.band_h, w_half, h_half,
                                 stride, "i4_ref_dwt2.band_h");
                check2d_band_i32(buf_c.i4_ref_dwt2.band_v,
                                 buf_a.i4_ref_dwt2.band_v, w_half, h_half,
                                 stride, "i4_ref_dwt2.band_v");
                check2d_band_i32(buf_c.i4_ref_dwt2.band_d,
                                 buf_a.i4_ref_dwt2.band_d, w_half, h_half,
                                 stride, "i4_ref_dwt2.band_d");
                check2d_band_i32(buf_c.i4_dis_dwt2.band_a,
                                 buf_a.i4_dis_dwt2.band_a, w_half, h_half,
                                 stride, "i4_dis_dwt2.band_a");
                check2d_band_i32(buf_c.i4_dis_dwt2.band_h,
                                 buf_a.i4_dis_dwt2.band_h, w_half, h_half,
                                 stride, "i4_dis_dwt2.band_h");
                check2d_band_i32(buf_c.i4_dis_dwt2.band_v,
                                 buf_a.i4_dis_dwt2.band_v, w_half, h_half,
                                 stride, "i4_dis_dwt2.band_v");
                check2d_band_i32(buf_c.i4_dis_dwt2.band_d,
                                 buf_a.i4_dis_dwt2.band_d, w_half, h_half,
                                 stride, "i4_dis_dwt2.band_d");

                checkasm_bench_new(i4_ref, i4_dis, &buf_a, w, h, src_stride,
                                    src_stride, stride, scale);
            }
        }

        free(i4_ref);
        free(i4_dis);
        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_decouple_s123(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        fill_band_i32(buf_c.i4_ref_dwt2.band_a, h_half, stride);
        fill_band_i32(buf_c.i4_ref_dwt2.band_h, h_half, stride);
        fill_band_i32(buf_c.i4_ref_dwt2.band_v, h_half, stride);
        fill_band_i32(buf_c.i4_ref_dwt2.band_d, h_half, stride);
        fill_band_i32(buf_c.i4_dis_dwt2.band_a, h_half, stride);
        fill_band_i32(buf_c.i4_dis_dwt2.band_h, h_half, stride);
        fill_band_i32(buf_c.i4_dis_dwt2.band_v, h_half, stride);
        fill_band_i32(buf_c.i4_dis_dwt2.band_d, h_half, stride);

        copy_band_i32(buf_a.i4_ref_dwt2.band_a, buf_c.i4_ref_dwt2.band_a,
                      h_half, stride);
        copy_band_i32(buf_a.i4_ref_dwt2.band_h, buf_c.i4_ref_dwt2.band_h,
                      h_half, stride);
        copy_band_i32(buf_a.i4_ref_dwt2.band_v, buf_c.i4_ref_dwt2.band_v,
                      h_half, stride);
        copy_band_i32(buf_a.i4_ref_dwt2.band_d, buf_c.i4_ref_dwt2.band_d,
                      h_half, stride);
        copy_band_i32(buf_a.i4_dis_dwt2.band_a, buf_c.i4_dis_dwt2.band_a,
                      h_half, stride);
        copy_band_i32(buf_a.i4_dis_dwt2.band_h, buf_c.i4_dis_dwt2.band_h,
                      h_half, stride);
        copy_band_i32(buf_a.i4_dis_dwt2.band_v, buf_c.i4_dis_dwt2.band_v,
                      h_half, stride);
        copy_band_i32(buf_a.i4_dis_dwt2.band_d, buf_c.i4_dis_dwt2.band_d,
                      h_half, stride);

        checkasm_declare(void, AdmBuffer *, int, int, int, double, int32_t *);

        if (checkasm_check_func(get_decouple_s123(checkasm_get_cpu_flags()),
                                 "adm_decouple_s123_%dx%d", w, h))
        {
            checkasm_call_ref(&buf_c, w_half, h_half, stride,
                               DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);
            checkasm_call_new(&buf_a, w_half, h_half, stride,
                               DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);

            check2d_band_i32(buf_c.i4_decouple_r.band_h,
                             buf_a.i4_decouple_r.band_h, w_half, h_half,
                             stride, "i4_decouple_r.band_h");
            check2d_band_i32(buf_c.i4_decouple_r.band_v,
                             buf_a.i4_decouple_r.band_v, w_half, h_half,
                             stride, "i4_decouple_r.band_v");
            check2d_band_i32(buf_c.i4_decouple_r.band_d,
                             buf_a.i4_decouple_r.band_d, w_half, h_half,
                             stride, "i4_decouple_r.band_d");
            check2d_band_i32(buf_c.i4_decouple_a.band_h,
                             buf_a.i4_decouple_a.band_h, w_half, h_half,
                             stride, "i4_decouple_a.band_h");
            check2d_band_i32(buf_c.i4_decouple_a.band_v,
                             buf_a.i4_decouple_a.band_v, w_half, h_half,
                             stride, "i4_decouple_a.band_v");
            check2d_band_i32(buf_c.i4_decouple_a.band_d,
                             buf_a.i4_decouple_a.band_d, w_half, h_half,
                             stride, "i4_decouple_a.band_d");

            checkasm_bench_new(&buf_a, w_half, h_half, stride,
                                DEFAULT_ADM_ENHN_GAIN_LIMIT, div_lookup);
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_csf_den(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf;
        if (adm_buffer_alloc(&buf, w, h)) continue;
        const int stride = (int) (buf.ind_size_x >> 2);
        const int h_half = (h + 1) / 2;

        fill_band(buf.ref_dwt2.band_h, h_half, stride);
        fill_band(buf.ref_dwt2.band_v, h_half, stride);
        fill_band(buf.ref_dwt2.band_d, h_half, stride);
        fill_band_i32(buf.i4_ref_dwt2.band_h, h_half, stride);
        fill_band_i32(buf.i4_ref_dwt2.band_v, h_half, stride);
        fill_band_i32(buf.i4_ref_dwt2.band_d, h_half, stride);

        {
            checkasm_declare(float, const adm_dwt_band_t *, int, int, int,
                              double, int, int, double, double, double);

            if (checkasm_check_func(
                    get_csf_den_scale(checkasm_get_cpu_flags()),
                    "adm_csf_den_scale_%dx%d", w, h))
            {
                const float ref = checkasm_call_ref(
                    &buf.ref_dwt2, w, h, stride, DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT);
                const float new = checkasm_call_new(
                    &buf.ref_dwt2, w, h, stride, DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT);

                const float tol = 1e-4f * (fabsf(ref) + 1.0f);
                if (fabsf(ref - new) > tol) {
                    if (checkasm_fail())
                        fprintf(stderr, "expected %f, got %f\n", ref, new);
                }

                checkasm_bench_new(&buf.ref_dwt2, w, h, stride,
                                    DEFAULT_ADM_NORM_VIEW_DIST,
                                    DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                    DEFAULT_ADM_CSF_MODE,
                                    DEFAULT_ADM_CSF_SCALE,
                                    DEFAULT_ADM_CSF_DIAG_SCALE,
                                    DEFAULT_ADM_NOISE_WEIGHT);
            }
        }

        checkasm_declare(float, const i4_adm_dwt_band_t *, int, int, int,
                          int, double, int, int, double, double, double);

        for (int scale = 1; scale <= 3; scale++) {
            if (checkasm_check_func(
                    get_csf_den_s123(checkasm_get_cpu_flags()),
                    "adm_csf_den_s123_%dx%d_scale%d", w, h, scale))
            {
                const float ref = checkasm_call_ref(
                    &buf.i4_ref_dwt2, scale, w, h, stride,
                    DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT);
                const float new = checkasm_call_new(
                    &buf.i4_ref_dwt2, scale, w, h, stride,
                    DEFAULT_ADM_NORM_VIEW_DIST,
                    DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                    DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                    DEFAULT_ADM_NOISE_WEIGHT);

                const float tol = 1e-4f * (fabsf(ref) + 1.0f);
                if (fabsf(ref - new) > tol) {
                    if (checkasm_fail())
                        fprintf(stderr, "expected %f, got %f\n", ref, new);
                }

                checkasm_bench_new(&buf.i4_ref_dwt2, scale, w, h, stride,
                                    DEFAULT_ADM_NORM_VIEW_DIST,
                                    DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                    DEFAULT_ADM_CSF_MODE,
                                    DEFAULT_ADM_CSF_SCALE,
                                    DEFAULT_ADM_CSF_DIAG_SCALE,
                                    DEFAULT_ADM_NOISE_WEIGHT);
            }
        }

        adm_buffer_free(&buf);
    }
}

static void check_adm_i4_csf(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int w_half = (w + 1) / 2, h_half = (h + 1) / 2;

        fill_band_i32(buf_c.i4_decouple_r.band_h, h_half, stride);
        fill_band_i32(buf_c.i4_decouple_r.band_v, h_half, stride);
        fill_band_i32(buf_c.i4_decouple_r.band_d, h_half, stride);
        fill_band_i32(buf_c.i4_decouple_a.band_h, h_half, stride);
        fill_band_i32(buf_c.i4_decouple_a.band_v, h_half, stride);
        fill_band_i32(buf_c.i4_decouple_a.band_d, h_half, stride);

        copy_band_i32(buf_a.i4_decouple_r.band_h, buf_c.i4_decouple_r.band_h,
                      h_half, stride);
        copy_band_i32(buf_a.i4_decouple_r.band_v, buf_c.i4_decouple_r.band_v,
                      h_half, stride);
        copy_band_i32(buf_a.i4_decouple_r.band_d, buf_c.i4_decouple_r.band_d,
                      h_half, stride);
        copy_band_i32(buf_a.i4_decouple_a.band_h, buf_c.i4_decouple_a.band_h,
                      h_half, stride);
        copy_band_i32(buf_a.i4_decouple_a.band_v, buf_c.i4_decouple_a.band_v,
                      h_half, stride);
        copy_band_i32(buf_a.i4_decouple_a.band_d, buf_c.i4_decouple_a.band_d,
                      h_half, stride);

        checkasm_declare(void, AdmBuffer *, int, int, int, int, double, int,
                          int, double, double, bool);

        for (int scale = 1; scale <= 3; scale++) {
            for (int aim = 0; aim <= 1; aim++) {
                if (checkasm_check_func(
                        get_i4_csf(checkasm_get_cpu_flags()),
                        "i4_adm_csf_%dx%d_scale%d_aim%d", w, h, scale, aim))
                {
                    checkasm_call_ref(&buf_c, scale, w_half, h_half, stride,
                                       DEFAULT_ADM_NORM_VIEW_DIST,
                                       DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                       DEFAULT_ADM_CSF_MODE,
                                       DEFAULT_ADM_CSF_SCALE,
                                       DEFAULT_ADM_CSF_DIAG_SCALE, (bool) aim);
                    checkasm_call_new(&buf_a, scale, w_half, h_half, stride,
                                       DEFAULT_ADM_NORM_VIEW_DIST,
                                       DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                       DEFAULT_ADM_CSF_MODE,
                                       DEFAULT_ADM_CSF_SCALE,
                                       DEFAULT_ADM_CSF_DIAG_SCALE, (bool) aim);

                    check2d_band_i32(buf_c.i4_csf_a.band_h,
                                     buf_a.i4_csf_a.band_h, w_half, h_half,
                                     stride, "i4_csf_a.band_h");
                    check2d_band_i32(buf_c.i4_csf_a.band_v,
                                     buf_a.i4_csf_a.band_v, w_half, h_half,
                                     stride, "i4_csf_a.band_v");
                    check2d_band_i32(buf_c.i4_csf_a.band_d,
                                     buf_a.i4_csf_a.band_d, w_half, h_half,
                                     stride, "i4_csf_a.band_d");
                    check2d_band_i32(buf_c.i4_csf_f.band_h,
                                     buf_a.i4_csf_f.band_h, w_half, h_half,
                                     stride, "i4_csf_f.band_h");
                    check2d_band_i32(buf_c.i4_csf_f.band_v,
                                     buf_a.i4_csf_f.band_v, w_half, h_half,
                                     stride, "i4_csf_f.band_v");
                    check2d_band_i32(buf_c.i4_csf_f.band_d,
                                     buf_a.i4_csf_f.band_d, w_half, h_half,
                                     stride, "i4_csf_f.band_d");

                    checkasm_bench_new(&buf_a, scale, w_half, h_half, stride,
                                        DEFAULT_ADM_NORM_VIEW_DIST,
                                        DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                        DEFAULT_ADM_CSF_MODE,
                                        DEFAULT_ADM_CSF_SCALE,
                                        DEFAULT_ADM_CSF_DIAG_SCALE,
                                        (bool) aim);
                }
            }
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

static void check_adm_i4_cm(void)
{
    for (size_t i = 0; i < sizeof(post_dwt_sizes) / sizeof(*post_dwt_sizes);
         i++)
    {
        const int w = post_dwt_sizes[i].w, h = post_dwt_sizes[i].h;

        AdmBuffer buf_c, buf_a;
        if (adm_buffer_alloc(&buf_c, w, h)) continue;
        if (adm_buffer_alloc(&buf_a, w, h)) {
            adm_buffer_free(&buf_c);
            continue;
        }
        const int stride = (int) (buf_c.ind_size_x >> 2);
        const int h_half = (h + 1) / 2;
        const int w_half = (w + 1) / 2;

        i4_adm_dwt_band_t *c_bands[4] = { &buf_c.i4_decouple_r,
                                           &buf_c.i4_decouple_a,
                                           &buf_c.i4_csf_f, &buf_c.i4_csf_a };
        i4_adm_dwt_band_t *a_bands[4] = { &buf_a.i4_decouple_r,
                                           &buf_a.i4_decouple_a,
                                           &buf_a.i4_csf_f, &buf_a.i4_csf_a };
        for (int b = 0; b < 4; b++) {
            fill_band_i32(c_bands[b]->band_h, h_half, stride);
            fill_band_i32(c_bands[b]->band_v, h_half, stride);
            fill_band_i32(c_bands[b]->band_d, h_half, stride);
            copy_band_i32(a_bands[b]->band_h, c_bands[b]->band_h, h_half,
                          stride);
            copy_band_i32(a_bands[b]->band_v, c_bands[b]->band_v, h_half,
                          stride);
            copy_band_i32(a_bands[b]->band_d, c_bands[b]->band_d, h_half,
                          stride);
        }

        checkasm_declare(float, AdmBuffer *, int, int, int, int, int, double,
                          int, int, double, double, double, bool);

        for (int scale = 1; scale <= 3; scale++) {
            for (int aim = 0; aim <= 1; aim++) {
                if (checkasm_check_func(
                        get_i4_cm(checkasm_get_cpu_flags()),
                        "i4_adm_cm_%dx%d_scale%d_aim%d", w, h, scale, aim))
                {
                    const float ref = checkasm_call_ref(
                        &buf_c, w_half, h_half, stride, stride, scale,
                        DEFAULT_ADM_NORM_VIEW_DIST,
                        DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                        DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                        DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);
                    const float new = checkasm_call_new(
                        &buf_a, w_half, h_half, stride, stride, scale,
                        DEFAULT_ADM_NORM_VIEW_DIST,
                        DEFAULT_ADM_REF_DISPLAY_HEIGHT, DEFAULT_ADM_CSF_MODE,
                        DEFAULT_ADM_CSF_SCALE, DEFAULT_ADM_CSF_DIAG_SCALE,
                        DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);

                    const float tol = 1e-4f * (fabsf(ref) + 1.0f);
                    if (fabsf(ref - new) > tol) {
                        if (checkasm_fail())
                            fprintf(stderr, "expected %f, got %f\n", ref,
                                    new);
                    }

                    checkasm_bench_new(&buf_a, w_half, h_half, stride, stride,
                                        scale, DEFAULT_ADM_NORM_VIEW_DIST,
                                        DEFAULT_ADM_REF_DISPLAY_HEIGHT,
                                        DEFAULT_ADM_CSF_MODE,
                                        DEFAULT_ADM_CSF_SCALE,
                                        DEFAULT_ADM_CSF_DIAG_SCALE,
                                        DEFAULT_ADM_NOISE_WEIGHT, (bool) aim);
                }
            }
        }

        adm_buffer_free(&buf_c);
        adm_buffer_free(&buf_a);
    }
}

void checkasm_check_adm(void)
{
    check_adm_dwt2();
    checkasm_report("adm_dwt2");

    check_adm_decouple();
    checkasm_report("adm_decouple");

    check_adm_csf();
    checkasm_report("adm_csf");

    check_adm_cm();
    checkasm_report("adm_cm");

    check_adm_dwt2_s123();
    checkasm_report("adm_dwt2_s123");

    check_adm_decouple_s123();
    checkasm_report("adm_decouple_s123");

    check_adm_csf_den();
    checkasm_report("adm_csf_den");

    check_adm_i4_csf();
    checkasm_report("i4_adm_csf");

    check_adm_i4_cm();
    checkasm_report("i4_adm_cm");
}
