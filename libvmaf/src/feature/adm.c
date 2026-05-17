/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "mem.h"
#include "adm_options.h"
#include "adm_tools.h"
#include "offset.h"

typedef adm_dwt_band_t_s adm_dwt_band_t;

#define adm_dwt2 adm_dwt2_s
#define adm_dwt2_lo adm_dwt2_lo_s
#define adm_decouple adm_decouple_s
#define adm_csf adm_csf_s
#define adm_cm_thresh adm_cm_thresh_s
#define adm_cm adm_cm_s
#define adm_sum_cube adm_sum_cube_s
#define offset_image offset_image_s

#define adm_csf_den_scale adm_csf_den_scale_s
#define dwt2_src_indices_filt dwt2_src_indices_filt_s

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static char *init_dwt_band(adm_dwt_band_t *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = (float *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_h = (float *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_v = (float *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_d = (float *)(void *)data_top;
    data_top += buf_sz_one;
    return data_top;
}

UNUSED_FUNCTION
static char *init_dwt_band_d(adm_dwt_band_t_d *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = (double *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_h = (double *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_v = (double *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_d = (double *)(void *)data_top;
    data_top += buf_sz_one;
    return data_top;
}

static char *init_dwt_band_hvd(adm_dwt_band_t *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = NULL;
    band->band_h = (float *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_v = (float *)(void *)data_top;
    data_top += buf_sz_one;
    band->band_d = (float *)(void *)data_top;
    data_top += buf_sz_one;
    return data_top;
}

int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride,
                double *score, double *score_num, double *score_den, double *scores,
                double border_factor, double adm_enhn_gain_limit, double adm_norm_view_dist,
                int adm_ref_display_height, int adm_csf_mode, double luminance_level,
                double adm_csf_scale, double adm_csf_diag_scale, double adm_noise_weight,
                int adm_bypass_cm, double adm_p_norm, double *score_aim, double adm_f1s0,
                double adm_f1s1, double adm_f1s2, double adm_f1s3, double adm_f2s0, double adm_f2s1,
                double adm_f2s2, double adm_f2s3, int adm_skip_aim_scale, bool adm_skip_scale0)
{
#ifdef ADM_OPT_SINGLE_PRECISION
    double numden_limit = 1e-2 * (w * h) / (1920.0 * 1080.0);
#else
    double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
#endif
    float *data_buf = 0;
    char *data_top;

    char *ind_buf_y = 0, *buf_y_orig = 0;
    char *ind_buf_x = 0, *buf_x_orig = 0;
    int *ind_y[4], *ind_x[4];

    float *ref_scale;
    float *dis_scale;

    adm_dwt_band_t ref_dwt2;
    adm_dwt_band_t dis_dwt2;

    adm_dwt_band_t decouple_r;
    adm_dwt_band_t decouple_a;

    adm_dwt_band_t csf_a;
    adm_dwt_band_t csf_f; //Store filtered coeffs

    const float *curr_ref_scale = ref;
    const float *curr_dis_scale = dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int orig_h = h;

    int buf_stride = ALIGN_CEIL(((w + 1) / 2) * sizeof(float));
    size_t buf_sz_one = (size_t)buf_stride * ((h + 1) / 2);

    int ind_size_y = ALIGN_CEIL(((h + 1) / 2) * sizeof(int));
    int ind_size_x = ALIGN_CEIL(((w + 1) / 2) * sizeof(int));

    double num = 0;
    double den = 0;
    double aim_num = 0;
    double aim_den = 0;

    int scale;
    int ret = 1;

    // Code optimized to save on multiple buffer copies
    // hence the reduction in the number of buffers required from 35 to 17
#define NUM_BUFS_ADM 20
    if (SIZE_MAX / buf_sz_one < NUM_BUFS_ADM) {
        printf("error: SIZE_MAX / buf_sz_one < NUM_BUFS_ADM, buf_sz_one = %zu.\n", buf_sz_one);
        fflush(stdout);
        goto fail;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * NUM_BUFS_ADM, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for data_buf.\n");
        fflush(stdout);
        goto fail;
    }

    data_top = (char *)data_buf;

    data_top = init_dwt_band(&ref_dwt2, data_top, buf_sz_one);
    data_top = init_dwt_band(&dis_dwt2, data_top, buf_sz_one);
    data_top = init_dwt_band_hvd(&decouple_r, data_top, buf_sz_one);
    data_top = init_dwt_band_hvd(&decouple_a, data_top, buf_sz_one);
    data_top = init_dwt_band_hvd(&csf_a, data_top, buf_sz_one);
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores) — ADR-0418 upstream-parity (4dcc2f7c)
    data_top = init_dwt_band_hvd(&csf_f, data_top, buf_sz_one);

    if (!(buf_y_orig = aligned_malloc(ind_size_y * 4, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for ind_buf_y.\n");
        fflush(stdout);
        goto fail;
    }
    ind_buf_y = buf_y_orig;
    ind_y[0] = (int *)ind_buf_y;
    ind_buf_y += ind_size_y;
    ind_y[1] = (int *)ind_buf_y;
    ind_buf_y += ind_size_y;
    ind_y[2] = (int *)ind_buf_y;
    ind_buf_y += ind_size_y;
    ind_y[3] = (int *)ind_buf_y;
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores) — ADR-0418 upstream-parity
    ind_buf_y += ind_size_y;

    if (!(buf_x_orig = aligned_malloc(ind_size_x * 4, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for ind_buf_x.\n");
        fflush(stdout);
        goto fail;
    }
    ind_buf_x = buf_x_orig;
    ind_x[0] = (int *)ind_buf_x;
    ind_buf_x += ind_size_x;
    ind_x[1] = (int *)ind_buf_x;
    ind_buf_x += ind_size_x;
    ind_x[2] = (int *)ind_buf_x;
    ind_buf_x += ind_size_x;
    ind_x[3] = (int *)ind_buf_x;
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores) — ADR-0418 upstream-parity
    ind_buf_x += ind_size_x;

    for (scale = 0; scale < 4; ++scale) {
#ifdef ADM_OPT_DEBUG_DUMP
        char pathbuf[256];
#endif
        float num_scale = 0.0;
        float den_scale = 0.0;
        float aim_num_scale = 0.0;

        dwt2_src_indices_filt(ind_y, ind_x, w, h);
        if ((scale == 0) && (adm_skip_scale0)) {
            if (adm_dwt2_lo(curr_ref_scale, &ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride,
                            buf_stride) < 0)
                goto fail;
            if (adm_dwt2_lo(curr_dis_scale, &dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride,
                            buf_stride) < 0)
                goto fail;

            w = (w + 1) / 2;
            h = (h + 1) / 2;
            den_scale = 1e-10; // avoid divide by zero
        } else {
            if (adm_dwt2(curr_ref_scale, &ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride,
                         buf_stride) < 0)
                goto fail;
            if (adm_dwt2(curr_dis_scale, &dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride,
                         buf_stride) < 0)
                goto fail;

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h, buf_stride,
                         buf_stride, buf_stride, buf_stride, border_factor, adm_enhn_gain_limit);

            den_scale = adm_csf_den_scale(
                &ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor, adm_norm_view_dist,
                adm_ref_display_height, adm_csf_mode, luminance_level, adm_csf_scale,
                adm_csf_diag_scale, adm_noise_weight, adm_p_norm, adm_f1s0, adm_f1s1, adm_f1s2,
                adm_f1s3, adm_f2s0, adm_f2s1, adm_f2s2, adm_f2s3);

            adm_csf(&decouple_a, &csf_a, &csf_f, orig_h, scale, w, h, buf_stride, buf_stride,
                    border_factor, adm_norm_view_dist, adm_ref_display_height, adm_csf_mode,
                    luminance_level, adm_csf_scale, adm_csf_diag_scale, adm_f1s0, adm_f1s1,
                    adm_f1s2, adm_f1s3, adm_f2s0, adm_f2s1, adm_f2s2, adm_f2s3);

            num_scale =
                adm_cm(&decouple_r, &csf_f, &csf_a, w, h, buf_stride, buf_stride, buf_stride,
                       border_factor, scale, adm_norm_view_dist, adm_ref_display_height,
                       adm_csf_mode, luminance_level, adm_csf_scale, adm_csf_diag_scale,
                       adm_noise_weight, adm_bypass_cm, adm_p_norm, adm_f1s0, adm_f1s1, adm_f1s2,
                       adm_f1s3, adm_f2s0, adm_f2s1, adm_f2s2, adm_f2s3);

            adm_csf(&decouple_r, &csf_f, &csf_a, orig_h, scale, w, h, buf_stride, buf_stride,
                    border_factor, adm_norm_view_dist, adm_ref_display_height, adm_csf_mode,
                    luminance_level, adm_csf_scale, adm_csf_diag_scale, adm_f1s0, adm_f1s1,
                    adm_f1s2, adm_f1s3, adm_f2s0, adm_f2s1, adm_f2s2, adm_f2s3);

            aim_num_scale = adm_cm(
                &decouple_a, &csf_a, &csf_f, w, h, buf_stride, buf_stride, buf_stride,
                border_factor, scale, adm_norm_view_dist, adm_ref_display_height, adm_csf_mode,
                luminance_level, adm_csf_scale, adm_csf_diag_scale, 0.0, adm_bypass_cm, adm_p_norm,
                adm_f1s0, adm_f1s1, adm_f1s2, adm_f1s3, adm_f2s0, adm_f2s1, adm_f2s2, adm_f2s3);

#ifdef ADM_OPT_DEBUG_DUMP
            snprintf(pathbuf, sizeof(pathbuf), "stage/ref[%d]_a.yuv", scale);
            write_image(pathbuf, ref_dwt2.band_a, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/ref[%d]_h.yuv", scale);
            write_image(pathbuf, ref_dwt2.band_h, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/ref[%d]_v.yuv", scale);
            write_image(pathbuf, ref_dwt2.band_v, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/ref[%d]_d.yuv", scale);
            write_image(pathbuf, ref_dwt2.band_d, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/dis[%d]_a.yuv", scale);
            write_image(pathbuf, dis_dwt2.band_a, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/dis[%d]_h.yuv", scale);
            write_image(pathbuf, dis_dwt2.band_h, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/dis[%d]_v.yuv", scale);
            write_image(pathbuf, dis_dwt2.band_v, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/dis[%d]_d.yuv", scale);
            write_image(pathbuf, dis_dwt2.band_d, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/r[%d]_h.yuv", scale);
            write_image(pathbuf, decouple_r.band_h, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/r[%d]_v.yuv", scale);
            write_image(pathbuf, decouple_r.band_v, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/r[%d]_d.yuv", scale);
            write_image(pathbuf, decouple_r.band_d, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/a[%d]_h.yuv", scale);
            write_image(pathbuf, decouple_a.band_h, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/a[%d]_v.yuv", scale);
            write_image(pathbuf, decouple_a.band_v, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/a[%d]_d.yuv", scale);
            write_image(pathbuf, decouple_a.band_d, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/csf_a[%d]_h.yuv", scale);
            write_image(pathbuf, csf_a.band_h, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/csf_a[%d]_v.yuv", scale);
            write_image(pathbuf, csf_a.band_v, w, h, buf_stride, sizeof(float));

            snprintf(pathbuf, sizeof(pathbuf), "stage/csf_a[%d]_d.yuv", scale);
            write_image(pathbuf, csf_a.band_d, w, h, buf_stride, sizeof(float));

#endif
        }

        num += num_scale;
        den += den_scale;
        if (adm_skip_aim_scale != scale) {
            aim_den += den_scale;
            aim_num += aim_num_scale;
        }

        ref_scale = ref_dwt2.band_a;
        dis_scale = dis_dwt2.band_a;

        curr_ref_scale = ref_scale;
        curr_dis_scale = dis_scale;

        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;

#ifdef ADM_OPT_DEBUG_DUMP
        PRINTF("num: %f\n", num);
        PRINTF("den: %f\n", den);
#endif
        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
    }

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0) {
        *score = 1.0f;
    } else {
        // normalize AIM score by the DLM denominator and clip values larger than 1
        *score_aim = MIN(aim_num / aim_den, 1.0f);
        *score = num / den;
    }
    *score_num = num;
    *score_den = den;

    ret = 0;

fail:
    aligned_free(data_buf);
    aligned_free(buf_y_orig);
    aligned_free(buf_x_orig);
    return ret;
}
