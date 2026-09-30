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

typedef void (*compute_dp_row_fn)(uint32_t *dp_curr, const uint32_t *dp_prev,
                                   const uint16_t *deriv, int width,
                                   int pad_size, bool deriv_valid);
typedef void (*compute_mask_row_fn)(uint16_t *mask_row,
                                     const uint32_t *dp_bottom,
                                     const uint32_t *dp_top, int width,
                                     int pad_size, uint32_t mask_index);
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

static compute_dp_row_fn get_compute_dp_row(unsigned cpu_flags)
{
#if ARCH_X86
    compute_dp_row_fn fn = compute_dp_row;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = compute_dp_row_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

static compute_mask_row_fn get_compute_mask_row(unsigned cpu_flags)
{
#if ARCH_X86
    compute_mask_row_fn fn = compute_mask_row;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = compute_mask_row_avx2;
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

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

#define DP_MAX_WIDTH 256
#define DP_MAX_PAD   8
#define DP_MAX_LEN   (DP_MAX_WIDTH + 2 * DP_MAX_PAD + 1)

static const struct { int width, pad_size; } dp_row_sizes[] = {
    { 4,           1 },
    { 8,           3 },
    { 16,          3 },
    { 37,          3 },
    { 173,         3 },
    { DP_MAX_WIDTH, 7 },
};

static void check_compute_dp_row(void)
{
    CHECKASM_ALIGN(uint32_t dp_prev[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint32_t dp_curr_c[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint32_t dp_curr_a[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint16_t deriv[DP_MAX_WIDTH]);

    checkasm_declare(void, uint32_t *, const uint32_t *, const uint16_t *,
                      int, int, bool);

    if (!checkasm_check_func(get_compute_dp_row(checkasm_get_cpu_flags()),
                              "compute_dp_row"))
        return;

    INITIALIZE_BUF(dp_prev);
    INITIALIZE_BUF(deriv);

    for (size_t i = 0; i < sizeof(dp_row_sizes) / sizeof(*dp_row_sizes); i++) {
        const int width = dp_row_sizes[i].width;
        const int pad_size = dp_row_sizes[i].pad_size;

        for (int pass = 0; pass < 2; pass++) {
            const bool deriv_valid = pass == 0;

            CLEAR_BUF(dp_curr_c);
            CLEAR_BUF(dp_curr_a);

            checkasm_call_ref(dp_curr_c, dp_prev, deriv, width, pad_size,
                               deriv_valid);
            checkasm_call_new(dp_curr_a, dp_prev, deriv, width, pad_size,
                               deriv_valid);

            char name[64];
            snprintf(name, sizeof(name), "w%d_pad%d_valid%d", width,
                      pad_size, deriv_valid);
            checkasm_check1d(uint32_t, dp_curr_c, dp_curr_a,
                              width + 2 * pad_size + 1, name);
        }
    }

    checkasm_bench_new(dp_curr_a, dp_prev, deriv, DP_MAX_WIDTH, DP_MAX_PAD,
                        true);
}

static void build_dp_row_pair(uint32_t *dp_top, uint32_t *dp_bottom,
                               int width, int pad_size)
{
    CHECKASM_ALIGN(uint32_t dp_zero[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint16_t deriv1[DP_MAX_WIDTH]);
    CHECKASM_ALIGN(uint16_t deriv2[DP_MAX_WIDTH]);

    CLEAR_BUF(dp_zero);
    checkasm_clear(dp_top, DP_MAX_LEN * sizeof(*dp_top));
    checkasm_clear(dp_bottom, DP_MAX_LEN * sizeof(*dp_bottom));
    INITIALIZE_BUF(deriv1);
    INITIALIZE_BUF(deriv2);
    for (int j = 0; j < width; j++) {
        deriv1[j] &= 0xff;
        deriv2[j] &= 0xff;
    }

    compute_dp_row(dp_top, dp_zero, deriv1, width, pad_size, true);
    compute_dp_row(dp_bottom, dp_top, deriv2, width, pad_size, true);
}

static void check_compute_mask_row(void)
{
    CHECKASM_ALIGN(uint32_t dp_bottom[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint32_t dp_top[DP_MAX_LEN]);
    CHECKASM_ALIGN(uint16_t mask_row_c[DP_MAX_WIDTH]);
    CHECKASM_ALIGN(uint16_t mask_row_a[DP_MAX_WIDTH]);

    checkasm_declare(void, uint16_t *, const uint32_t *, const uint32_t *,
                      int, int, uint32_t);

    if (!checkasm_check_func(get_compute_mask_row(checkasm_get_cpu_flags()),
                              "compute_mask_row"))
        return;

    const uint32_t mask_index = 32;

    for (size_t i = 0; i < sizeof(dp_row_sizes) / sizeof(*dp_row_sizes); i++) {
        const int width = dp_row_sizes[i].width;
        const int pad_size = dp_row_sizes[i].pad_size;

        build_dp_row_pair(dp_top, dp_bottom, width, pad_size);

        CLEAR_BUF(mask_row_c);
        CLEAR_BUF(mask_row_a);

        checkasm_call_ref(mask_row_c, dp_bottom, dp_top, width, pad_size,
                           mask_index);
        checkasm_call_new(mask_row_a, dp_bottom, dp_top, width, pad_size,
                           mask_index);

        char name[64];
        snprintf(name, sizeof(name), "w%d_pad%d", width, pad_size);
        checkasm_check1d(uint16_t, mask_row_c, mask_row_a, width, name);
    }

    build_dp_row_pair(dp_top, dp_bottom, DP_MAX_WIDTH, DP_MAX_PAD);
    checkasm_bench_new(mask_row_a, dp_bottom, dp_top, DP_MAX_WIDTH,
                        DP_MAX_PAD, mask_index);
}

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

    check_compute_dp_row();
    checkasm_report("compute_dp_row");

    check_compute_mask_row();
    checkasm_report("compute_mask_row");

    check_decimate();
    checkasm_report("decimate");

    check_filter_mode();
    checkasm_report("filter_mode");

    check_calculate_c_values();
    checkasm_report("calculate_c_values");
}
