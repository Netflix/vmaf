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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "config.h"
#include "cpu.h"
#include "mem.h"
#include "picture.h"
#include "feature/cambi.h"

#if ARCH_X86
#include "feature/x86/cambi_avx2.h"
#endif

typedef void (*derivative_fn)(const uint16_t *image_data,
                               uint16_t *derivative_buffer, int width,
                               int height, int row, int stride);

static derivative_fn get_derivative(unsigned cpu_flags)
{
#if ARCH_X86
    derivative_fn fn = get_derivative_data_for_row;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = get_derivative_data_for_row_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

#define MAX_WIDTH  256
#define MAX_HEIGHT 32

static const struct { int w, h; } sizes[] = {
    { 4,   4  },
    { 16,  16 },
    { 173, 9  },
    { MAX_WIDTH, MAX_HEIGHT },
};

static void check_get_derivative_data_for_row(void)
{
    CHECKASM_ALIGN(uint16_t image[MAX_WIDTH * MAX_HEIGHT]);
    CHECKASM_ALIGN(uint16_t deriv_c[MAX_WIDTH]);
    CHECKASM_ALIGN(uint16_t deriv_a[MAX_WIDTH]);

    checkasm_declare(void, const uint16_t *, uint16_t *, int, int, int, int);

    if (checkasm_check_func(get_derivative(checkasm_get_cpu_flags()),
                             "get_derivative_data_for_row"))
    {
        INITIALIZE_BUF(image);

        for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
            const int w = sizes[i].w, h = sizes[i].h;

            for (int row = 0; row < h; row += (h > 4 ? h / 3 + 1 : 1)) {
                CLEAR_BUF(deriv_c);
                CLEAR_BUF(deriv_a);

                checkasm_call_ref(image, deriv_c, w, h, row, w);
                checkasm_call_new(image, deriv_a, w, h, row, w);

                char name[64];
                snprintf(name, sizeof(name), "%dx%d_row%d", w, h, row);
                checkasm_check1d(uint16_t, deriv_c, deriv_a, w, name);
            }
        }

        checkasm_bench_new(image, deriv_a, MAX_WIDTH, MAX_HEIGHT, 0,
                            MAX_WIDTH);
    }
}

typedef void (*decimate_fn)(VmafPicture *image, unsigned width,
                             unsigned height);
typedef void (*filter_mode_fn)(const VmafPicture *image, int width,
                                int height, uint16_t *buffer);
typedef void (*calc_c_values_fn)(VmafPicture *pic, const VmafPicture *mask_pic,
                                  float *c_values, uint16_t *histograms,
                                  uint16_t window_size,
                                  const uint16_t num_diffs,
                                  const uint16_t *tvi_for_diff,
                                  uint16_t vlt_luma, const int *diff_weights,
                                  const int *all_diffs, int width, int height);

static decimate_fn get_decimate(unsigned cpu_flags)
{
#if ARCH_X86
    decimate_fn fn = decimate;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = decimate_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static filter_mode_fn get_filter_mode(unsigned cpu_flags)
{
#if ARCH_X86
    filter_mode_fn fn = filter_mode;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = filter_mode_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static calc_c_values_fn get_calc_c_values(unsigned cpu_flags)
{
#if ARCH_X86
    calc_c_values_fn fn = calculate_c_values;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = calculate_c_values_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static const struct { unsigned w, h; } pic_sizes[] = {
    { 32, 32 },
    { 64, 48 },
    { 173, 65 },
};

static void check_decimate(void)
{
    checkasm_declare(void, VmafPicture *, unsigned, unsigned);

    if (!checkasm_check_func(get_decimate(checkasm_get_cpu_flags()),
                              "decimate"))
        return;

    for (size_t i = 0; i < sizeof(pic_sizes) / sizeof(*pic_sizes); i++) {
        const unsigned out_w = pic_sizes[i].w, out_h = pic_sizes[i].h;

        VmafPicture pic_c, pic_a;
        if (vmaf_picture_alloc(&pic_c, VMAF_PIX_FMT_YUV400P, 10, out_w * 2,
                                out_h * 2))
            continue;
        if (vmaf_picture_alloc(&pic_a, VMAF_PIX_FMT_YUV400P, 10, out_w * 2,
                                out_h * 2))
        {
            vmaf_picture_unref(&pic_c);
            continue;
        }

        uint16_t *dc = pic_c.data[0], *da = pic_a.data[0];
        const ptrdiff_t stride_c = pic_c.stride[0] / sizeof(uint16_t);
        const ptrdiff_t stride_a = pic_a.stride[0] / sizeof(uint16_t);
        for (unsigned r = 0; r < out_h * 2; r++) {
            for (unsigned c = 0; c < out_w * 2; c++) {
                const uint16_t v = (uint16_t) checkasm_rand_uint32() & 0x3ff;
                dc[r * stride_c + c] = v;
                da[r * stride_a + c] = v;
            }
        }

        checkasm_call_ref(&pic_c, out_w, out_h);
        checkasm_call_new(&pic_a, out_w, out_h);

        checkasm_check2d(uint16_t, dc, stride_c, da, stride_a, out_w, out_h,
                          "decimate");

        checkasm_bench_new(&pic_a, out_w, out_h);

        vmaf_picture_unref(&pic_c);
        vmaf_picture_unref(&pic_a);
    }
}

static void check_filter_mode(void)
{
    checkasm_declare(void, const VmafPicture *, int, int, uint16_t *);

    if (!checkasm_check_func(get_filter_mode(checkasm_get_cpu_flags()),
                              "filter_mode"))
        return;

    for (size_t i = 0; i < sizeof(pic_sizes) / sizeof(*pic_sizes); i++) {
        const unsigned w = pic_sizes[i].w, h = pic_sizes[i].h;

        VmafPicture pic_c, pic_a;
        if (vmaf_picture_alloc(&pic_c, VMAF_PIX_FMT_YUV400P, 10, w, h))
            continue;
        if (vmaf_picture_alloc(&pic_a, VMAF_PIX_FMT_YUV400P, 10, w, h)) {
            vmaf_picture_unref(&pic_c);
            continue;
        }

        uint16_t *dc = pic_c.data[0], *da = pic_a.data[0];
        const ptrdiff_t stride_c = pic_c.stride[0] / sizeof(uint16_t);
        const ptrdiff_t stride_a = pic_a.stride[0] / sizeof(uint16_t);
        for (unsigned r = 0; r < h; r++) {
            for (unsigned c = 0; c < w; c++) {
                const uint16_t v = (uint16_t) checkasm_rand_uint32() & 0x3ff;
                dc[r * stride_c + c] = v;
                da[r * stride_a + c] = v;
            }
        }

        uint16_t *buf_c = malloc(3 * w * sizeof(uint16_t));
        uint16_t *buf_a = malloc(3 * w * sizeof(uint16_t));

        checkasm_call_ref(&pic_c, (int) w, (int) h, buf_c);
        checkasm_call_new(&pic_a, (int) w, (int) h, buf_a);

        checkasm_check2d(uint16_t, dc, stride_c, da, stride_a, w, h,
                          "filter_mode");

        checkasm_bench_new(&pic_a, (int) w, (int) h, buf_a);

        free(buf_c);
        free(buf_a);
        vmaf_picture_unref(&pic_c);
        vmaf_picture_unref(&pic_a);
    }
}

static void check_calculate_c_values(void)
{
    checkasm_declare(void, VmafPicture *, const VmafPicture *, float *,
                      uint16_t *, uint16_t, uint16_t, const uint16_t *,
                      uint16_t, const int *, const int *, int, int);

    if (!checkasm_check_func(get_calc_c_values(checkasm_get_cpu_flags()),
                              "calculate_c_values"))
        return;

    const uint16_t num_diffs = 4;
    uint16_t *diffs_to_consider = NULL;
    int *diff_weights = NULL, *all_diffs = NULL;
    if (set_contrast_arrays(num_diffs, &diffs_to_consider, &diff_weights,
                             &all_diffs))
        return;

    VmafLumaRange luma_range;
    VmafEOTF eotf;
    vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    vmaf_luminance_init_eotf(&eotf, "bt1886");

    uint16_t *tvi_for_diff = malloc(num_diffs * sizeof(uint16_t));
    for (int d = 0; d < num_diffs; d++) {
        tvi_for_diff[d] = (uint16_t) (get_tvi_for_diff(diffs_to_consider[d],
                                                        0.019, 10, luma_range,
                                                        eotf) +
                                       num_diffs);
    }
    const uint16_t vlt_luma = (uint16_t) get_vlt_luma(0.0, luma_range, eotf);

    const uint16_t window_size = 15;
    const int v_lo_signed = (int) vlt_luma - 3 * (int) num_diffs + 1;
    const uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t) v_lo_signed : 0;
    const uint16_t v_band_size =
        tvi_for_diff[num_diffs - 1] + 1 - v_band_base;

    for (size_t i = 0; i < sizeof(pic_sizes) / sizeof(*pic_sizes); i++) {
        const unsigned w = pic_sizes[i].w, h = pic_sizes[i].h;

        VmafPicture pic, mask_pic;
        if (vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV400P, 10, w, h))
            continue;
        if (vmaf_picture_alloc(&mask_pic, VMAF_PIX_FMT_YUV400P, 10, w, h)) {
            vmaf_picture_unref(&pic);
            continue;
        }

        uint16_t *image = pic.data[0], *mask = mask_pic.data[0];
        const ptrdiff_t stride = pic.stride[0] / sizeof(uint16_t);
        const ptrdiff_t mask_stride = mask_pic.stride[0] / sizeof(uint16_t);
        for (unsigned r = 0; r < h; r++) {
            for (unsigned c = 0; c < w; c++) {
                image[r * stride + c] =
                    (uint16_t) checkasm_rand_uint32() & 0x3ff;
                mask[r * mask_stride + c] =
                    (uint16_t) (checkasm_rand_uint32() & 1);
            }
        }

        float *c_values_c = malloc((size_t) w * h * sizeof(float));
        float *c_values_a = malloc((size_t) w * h * sizeof(float));
        uint16_t *hist_c =
            malloc((size_t) w * v_band_size * sizeof(uint16_t));
        uint16_t *hist_a =
            malloc((size_t) w * v_band_size * sizeof(uint16_t));

        checkasm_call_ref(&pic, &mask_pic, c_values_c, hist_c, window_size,
                           num_diffs, tvi_for_diff, vlt_luma, diff_weights,
                           all_diffs, (int) w, (int) h);
        checkasm_call_new(&pic, &mask_pic, c_values_a, hist_a, window_size,
                           num_diffs, tvi_for_diff, vlt_luma, diff_weights,
                           all_diffs, (int) w, (int) h);

        for (unsigned r = 0; r < h; r++) {
            for (unsigned c = 0; c < w; c++) {
                const float ref = c_values_c[r * w + c];
                const float new = c_values_a[r * w + c];
                if (ref != new) {
                    if (checkasm_fail())
                        fprintf(stderr,
                                "%ux%u (%u,%u): expected %f, got %f\n", w, h,
                                r, c, ref, new);
                }
            }
        }

        checkasm_bench_new(&pic, &mask_pic, c_values_a, hist_a, window_size,
                            num_diffs, tvi_for_diff, vlt_luma, diff_weights,
                            all_diffs, (int) w, (int) h);

        free(c_values_c);
        free(c_values_a);
        free(hist_c);
        free(hist_a);
        vmaf_picture_unref(&pic);
        vmaf_picture_unref(&mask_pic);
    }

    free(tvi_for_diff);
    aligned_free(diffs_to_consider);
    aligned_free(diff_weights);
    aligned_free(all_diffs);
}

void checkasm_check_cambi(void)
{
    check_get_derivative_data_for_row();
    checkasm_report("get_derivative_data_for_row");

    check_decimate();
    checkasm_report("decimate");

    check_filter_mode();
    checkasm_report("filter_mode");

    check_calculate_c_values();
    checkasm_report("calculate_c_values");
}
