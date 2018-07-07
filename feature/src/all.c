/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
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
#include <stdlib.h>
#include <string.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "psnr_tools.h"
#include "motion_tools.h"
#include "all_options.h"
#include "vif_options.h"
#include "adm_options.h"

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s
#define convolution_f32_c  convolution_f32_c_s
#define offset_image       offset_image_s
#define FILTER_5           FILTER_5_s
int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor);
int compute_ansnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_psnr, double peak, double psnr_max);
int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores);
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int all(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    double score2 = 0;
    double scores[4*2];
    double score_num = 0;
    double score_den = 0;
    double score_psnr = 0;
    float *ref_buf = 0;
    float *dis_buf = 0;

    float *prev_blur_buf = 0;
    float *blur_buf = 0;
    float *next_ref_buf = 0;
    float *next_dis_buf = 0;
    float *next_blur_buf = 0;
    float *temp_buf = 0;

    size_t data_sz;
    int stride;
    double peak;
    double psnr_max;
    int ret = 1;
    bool next_frame_read;
    int global_frm_idx = 0; // map to thread_data->frm_idx in combo.c

    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        goto fail_or_end;
    }

    stride = ALIGN_CEIL(w * sizeof(float));

    if ((size_t)h > SIZE_MAX / stride)
    {
        goto fail_or_end;
    }

    ret = psnr_constants(fmt, &peak, &psnr_max);
    if (ret)
    {
        printf("error: unknown format %s.\n", fmt);
        fflush(stdout);
        goto fail_or_end;
    }

    data_sz = (size_t)stride * h;

    if (!(ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(prev_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_blur_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for blur_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_ref_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_dis_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_blur_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    // use temp_buf for convolution_f32_c, and fread u and v
    if (!(temp_buf = aligned_malloc(data_sz * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    int frm_idx = -1;
    while (1)
    {
        // the next frame
        frm_idx = global_frm_idx;
        global_frm_idx++;

        if (frm_idx == 0)
        {
            ret = read_frame(ref_buf, dis_buf, temp_buf, stride, user_data);
            if (ret == 1)
            {
                goto fail_or_end;
            }
            if (ret == 2)
            {
                break;
            }

            // ===============================================================
            // offset pixel by OPT_RANGE_PIXEL_OFFSET
            // ===============================================================
            offset_image(ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
            offset_image(dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);

            // ===============================================================
            // filter
            // apply filtering (to eliminate effects film grain)
            // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
            // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
            // ===============================================================
            convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));

        }

        ret = read_frame(next_ref_buf, next_dis_buf, temp_buf, stride, user_data);
        if (ret == 1)
        {
            goto fail_or_end;
        }
        if (ret == 2)
        {
            next_frame_read = false;
        }
        else
        {
            next_frame_read = true;
        }

        // ===============================================================
        // offset pixel by OPT_RANGE_PIXEL_OFFSET
        // ===============================================================
        if (next_frame_read)
        {
            offset_image(next_ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
            offset_image(next_dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        }

        // ===============================================================
        // filter
        // apply filtering (to eliminate effects film grain)
        // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
        // ===============================================================
        if (next_frame_read)
        {
            convolution_f32_c(FILTER_5, 5, next_ref_buf, next_blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));
        }

        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("adm: %d %f\n", frm_idx, score);
        printf("adm_num: %d %f\n", frm_idx, score_num);
        printf("adm_den: %d %f\n", frm_idx, score_den);
        for(int scale=0;scale<4;scale++){
            printf("adm_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            printf("adm_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
        }
        fflush(stdout);

        /* =========== ansnr ============== */
        ret = compute_ansnr(ref_buf, dis_buf, w, h, stride, stride, &score, &score_psnr, peak, psnr_max);

        if (ret)
        {
            printf("error: compute_ansnr failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        printf("ansnr: %d %f\n", frm_idx, score);
        printf("anpsnr: %d %f\n", frm_idx, score_psnr);
        fflush(stdout);

        /* =========== motion ============== */

        // compute
        if (frm_idx == 0)
        {
            score = 0.0;
            score2 = 0.0;
        }
        else
        {
            if ((ret = compute_motion(prev_blur_buf, blur_buf, w, h, stride, stride, &score)))
            {
                printf("error: compute_motion (prev) failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }

            if (next_frame_read)
            {
                if ((ret = compute_motion(blur_buf, next_blur_buf, w, h, stride, stride, &score2)))
                {
                    printf("error: compute_motion (next) failed.\n");
                    fflush(stdout);
                    goto fail_or_end;
                }
                score2 = MIN(score, score2);
            }
            else
            {
                score2 = score;
            }
        }

        // print
        printf("motion: %d %f\n", frm_idx, score);
        printf("motion2: %d %f\n", frm_idx, score2);
        fflush(stdout);

        /* =========== vif ============== */

        if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores)))
        {
            printf("error: compute_vif failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("vif: %d %f\n", frm_idx, score);
        printf("vif_num: %d %f\n", frm_idx, score_num);
        printf("vif_den: %d %f\n", frm_idx, score_den);
        for(int scale=0;scale<4;scale++){
            printf("vif_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            printf("vif_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
        }
        fflush(stdout);

        // copy to prev_buf
        memcpy(prev_blur_buf, blur_buf, data_sz);
        memcpy(ref_buf, next_ref_buf, data_sz);
        memcpy(dis_buf, next_dis_buf, data_sz);
        memcpy(blur_buf, next_blur_buf, data_sz);

        if (!next_frame_read)
        {
            break;
        }

    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(dis_buf);

    aligned_free(prev_blur_buf);
    aligned_free(next_ref_buf);
    aligned_free(next_dis_buf);
    aligned_free(next_blur_buf);

    aligned_free(blur_buf);
    aligned_free(temp_buf);

    return ret;
}

int all_color(int (*read_yuv_frame)(float *ref_data_y, float *ref_data_u, float *ref_data_v, float *dis_data_y, float *dis_data_u, float *dis_data_v, 
    int stride_byte_y, void *s, int w_u, int w_v, int h_u, int h_v, int stride_byte_u, int stride_byte_v), void *user_data, int w_y, int h_y, const char *fmt)
{
    double score_y = 0;
    double score_u = 0;
    double score_u_anchor = 0;
    double score_v = 0;

    double score2_y = 0;
    double score2_u = 0;
    double score2_v = 0;

    double scores_y[4*2];
    double scores_u[4*2];
    double scores_v[4*2];

    double score_num_y = 0;
    double score_num_u = 0;
    double score_num_v = 0;

    double score_den_y = 0;
    double score_den_u = 0;
    double score_den_v = 0;

    double score_psnr_y = 0;
    double score_psnr_u = 0;
    double score_psnr_v = 0;

    float *ref_buf_y = 0;
    float *ref_buf_u = 0;
    float *ref_buf_v = 0;

    float *dis_buf_y = 0;
    float *dis_buf_u = 0;
    float *dis_buf_v = 0;

    float *prev_blur_buf_y = 0;
    float *prev_blur_buf_u = 0;
    float *prev_blur_buf_v = 0;

    float *blur_buf_y = 0;
    float *blur_buf_u = 0;
    float *blur_buf_v = 0;

    float *next_ref_buf_y = 0;
    float *next_ref_buf_u = 0;
    float *next_ref_buf_v = 0;

    float *next_dis_buf_y = 0;
    float *next_dis_buf_u = 0;
    float *next_dis_buf_v = 0;

    float *next_blur_buf_y = 0;
    float *next_blur_buf_u = 0;
    float *next_blur_buf_v = 0;

    float *temp_buf_y = 0;
    float *temp_buf_u = 0;
    float *temp_buf_v = 0;

    int w_u, w_v, h_u, h_v;

    size_t data_sz_y, data_sz_u, data_sz_v;
    int stride_y, stride_u, stride_v;
    double peak;
    double psnr_max;
    int ret = 1;
    bool next_frame_read;
    int global_frm_idx = 0; // map to thread_data->frm_idx in combo.c

    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
    {
        w_u = w_y / 2;
        w_v = w_y / 2;
        h_u = h_y / 2;
        h_v = h_y / 2;
    }
    else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le"))
    {
        // need to double check this
        w_u = w_y / 2;
        w_v = w_y / 2;
        h_u = h_y;
        h_v = h_y;
    }
    else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le"))
    {
        w_u = w_y;
        w_v = w_y;
        h_u = h_y;
        h_v = h_y;
    }
    else
    {
        printf("error: unknown format %s.\n", fmt);
        fflush(stdout);
        goto fail_or_end;
    }

    if (w_y <= 0 || h_y <= 0 || (size_t)w_y > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        goto fail_or_end;
    }

    if (w_u <= 0 || h_u <= 0 || (size_t)w_u > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        goto fail_or_end;
    }

    if (w_v <= 0 || h_v <= 0 || (size_t)w_v > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        goto fail_or_end;
    }

    stride_y = ALIGN_CEIL(w_y * sizeof(float));
    stride_u = ALIGN_CEIL(w_u * sizeof(float));
    stride_v = ALIGN_CEIL(w_v * sizeof(float));

    printf("widths: %d, %d, %d\n", w_y, w_u, w_v);

    if ((size_t)h_y > SIZE_MAX / stride_y)
    {
        goto fail_or_end;
    }

    if ((size_t)h_u > SIZE_MAX / stride_u)
    {
        goto fail_or_end;
    }

    if ((size_t)h_v > SIZE_MAX / stride_v)
    {
        goto fail_or_end;
    }

    ret = psnr_constants(fmt, &peak, &psnr_max);
    if (ret)
    {
        printf("error: unknown format %s.\n", fmt);
        fflush(stdout);
        goto fail_or_end;
    }

    data_sz_y = (size_t)stride_y * h_y;
    data_sz_u = (size_t)stride_u * h_u;
    data_sz_v = (size_t)stride_v * h_v;

    // Y

    if (!(ref_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(prev_blur_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_blur_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(blur_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for blur_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_ref_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_ref_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_dis_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_dis_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_blur_buf_y = aligned_malloc(data_sz_y, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_blur_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    // U

    if (!(ref_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(prev_blur_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_blur_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(blur_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for blur_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_ref_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_ref_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_dis_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_dis_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_blur_buf_u = aligned_malloc(data_sz_u, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_blur_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    // V

    if (!(ref_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(prev_blur_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_blur_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(blur_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for blur_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_ref_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_ref_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_dis_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_dis_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(next_blur_buf_v = aligned_malloc(data_sz_v, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_blur_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    // use temp_buf for convolution_f32_c
    if (!(temp_buf_y = aligned_malloc(data_sz_y * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf_y.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(temp_buf_u = aligned_malloc(data_sz_u * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf_u.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(temp_buf_v = aligned_malloc(data_sz_v * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf_v.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    int frm_idx = -1;
    while (1)
    {
        // the next frame
        frm_idx = global_frm_idx;
        global_frm_idx++;

        if (frm_idx == 0)
        {
            ret = read_yuv_frame(ref_buf_y, ref_buf_u, ref_buf_v, dis_buf_y, dis_buf_u, dis_buf_v, stride_y, user_data, w_u, w_v, h_u, h_v, stride_u, stride_v);
            if (ret == 1)
            {
                goto fail_or_end;
            }
            if (ret == 2)
            {
                break;
            }

            // ===============================================================
            // offset pixel by OPT_RANGE_PIXEL_OFFSET
            // ===============================================================
            offset_image(ref_buf_y, OPT_RANGE_PIXEL_OFFSET, w_y, h_y, stride_y);
            offset_image(dis_buf_y, OPT_RANGE_PIXEL_OFFSET, w_y, h_y, stride_y);

            offset_image(ref_buf_u, OPT_RANGE_PIXEL_OFFSET, w_u, h_u, stride_u);
            offset_image(dis_buf_u, OPT_RANGE_PIXEL_OFFSET, w_u, h_u, stride_u);

            offset_image(ref_buf_v, OPT_RANGE_PIXEL_OFFSET, w_v, h_v, stride_v);
            offset_image(dis_buf_v, OPT_RANGE_PIXEL_OFFSET, w_v, h_v, stride_v);

            // ===============================================================
            // filter
            // apply filtering (to eliminate effects film grain)
            // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
            // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
            // ===============================================================
            convolution_f32_c(FILTER_5, 5, ref_buf_y, blur_buf_y, temp_buf_y, w_y, h_y, stride_y / sizeof(float), stride_y / sizeof(float));
            convolution_f32_c(FILTER_5, 5, ref_buf_u, blur_buf_u, temp_buf_u, w_u, h_u, stride_u / sizeof(float), stride_u / sizeof(float));
            convolution_f32_c(FILTER_5, 5, ref_buf_v, blur_buf_v, temp_buf_v, w_v, h_v, stride_v / sizeof(float), stride_v / sizeof(float));

        }

        ret = read_yuv_frame(next_ref_buf_y, next_ref_buf_u, next_ref_buf_v, next_dis_buf_y, next_dis_buf_u, next_dis_buf_v, 
            stride_y, user_data, w_u, w_v, h_u, h_v, stride_u, stride_v);

        if (ret == 1)
        {
            goto fail_or_end;
        }
        if (ret == 2)
        {
            next_frame_read = false;
        }
        else
        {
            next_frame_read = true;
        }

        // ===============================================================
        // offset pixel by OPT_RANGE_PIXEL_OFFSET
        // ===============================================================
        if (next_frame_read)
        {
            offset_image(next_ref_buf_y, OPT_RANGE_PIXEL_OFFSET, w_y, h_y, stride_y);
            offset_image(next_dis_buf_y, OPT_RANGE_PIXEL_OFFSET, w_y, h_y, stride_y);

            offset_image(next_ref_buf_u, OPT_RANGE_PIXEL_OFFSET, w_u, h_u, stride_u);
            offset_image(next_dis_buf_u, OPT_RANGE_PIXEL_OFFSET, w_u, h_u, stride_u);

            offset_image(next_ref_buf_v, OPT_RANGE_PIXEL_OFFSET, w_v, h_v, stride_v);
            offset_image(next_dis_buf_v, OPT_RANGE_PIXEL_OFFSET, w_v, h_v, stride_v);
        }

        // ===============================================================
        // filter
        // apply filtering (to eliminate effects film grain)
        // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
        // ===============================================================
        if (next_frame_read)
        {
            convolution_f32_c(FILTER_5, 5, next_ref_buf_y, next_blur_buf_y, temp_buf_y, w_y, h_y, stride_y / sizeof(float), stride_y / sizeof(float));
            convolution_f32_c(FILTER_5, 5, next_ref_buf_u, next_blur_buf_u, temp_buf_u, w_u, h_u, stride_u / sizeof(float), stride_u / sizeof(float));
            convolution_f32_c(FILTER_5, 5, next_ref_buf_v, next_blur_buf_v, temp_buf_v, w_v, h_v, stride_v / sizeof(float), stride_v / sizeof(float));
        }

        /* =========== Y ============== */
        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf_y, dis_buf_y, w_y, h_y, stride_y, stride_y, &score_y, &score_num_y, &score_den_y, scores_y, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm_y failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("adm_y: %d %f\n", frm_idx, score_y);
        printf("adm_num_y: %d %f\n", frm_idx, score_num_y);
        printf("adm_den_y: %d %f\n", frm_idx, score_den_y);
        for(int scale=0;scale<4;scale++){
            printf("adm_num_scale%d_y: %d %f\n", scale, frm_idx, scores_y[2*scale]);
            printf("adm_den_scale%d_y: %d %f\n", scale, frm_idx, scores_y[2*scale+1]);
        }
        fflush(stdout);

        /* =========== ansnr ============== */
        ret = compute_ansnr(ref_buf_y, dis_buf_y, w_y, h_y, stride_y, stride_y, &score_y, &score_psnr_y, peak, psnr_max);

        if (ret)
        {
            printf("error: compute_ansnr_y failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        printf("ansnr_y: %d %f\n", frm_idx, score_y);
        printf("anpsnr_y: %d %f\n", frm_idx, score_psnr_y);
        fflush(stdout);

        /* =========== motion ============== */

        // compute
        if (frm_idx == 0)
        {
            score_y = 0.0;
            score2_y = 0.0;
        }
        else
        {
            if ((ret = compute_motion(prev_blur_buf_y, blur_buf_y, w_y, h_y, stride_y, stride_y, &score_y)))
            {
                printf("error: compute_motion_y (prev) failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }

            if (next_frame_read)
            {
                if ((ret = compute_motion(blur_buf_y, next_blur_buf_y, w_y, h_y, stride_y, stride_y, &score2_y)))
                {
                    printf("error: compute_motion_y (next) failed.\n");
                    fflush(stdout);
                    goto fail_or_end;
                }
                score2_y = MIN(score_y, score2_y);
            }
            else
            {
                score2_y = score_y;
            }
        }

        // print
        printf("motion_y: %d %f\n", frm_idx, score_y);
        printf("motion2_y: %d %f\n", frm_idx, score2_y);
        fflush(stdout);

        /* =========== vif ============== */

        if ((ret = compute_vif(ref_buf_y, dis_buf_y, w_y, h_y, stride_y, stride_y, &score_y, &score_num_y, &score_den_y, scores_y)))
        {
            printf("error: compute_vif_y failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("vif_y: %d %f\n", frm_idx, score_y);
        printf("vif_num_y: %d %f\n", frm_idx, score_num_y);
        printf("vif_den_y: %d %f\n", frm_idx, score_den_y);
        for(int scale=0;scale<4;scale++){
            printf("vif_num_scale%d_y: %d %f\n", scale, frm_idx, scores_y[2*scale]);
            printf("vif_den_scale%d_y: %d %f\n", scale, frm_idx, scores_y[2*scale+1]);
        }
        fflush(stdout);

        // copy to prev_buf
        memcpy(prev_blur_buf_y, blur_buf_y, data_sz_y);
        memcpy(ref_buf_y, next_ref_buf_y, data_sz_y);
        memcpy(dis_buf_y, next_dis_buf_y, data_sz_y);
        memcpy(blur_buf_y, next_blur_buf_y, data_sz_y);

        /* =========== U ============== */
        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf_u, dis_buf_u, w_u, h_u, stride_u, stride_u, &score_u, &score_num_u, &score_den_u, scores_u, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm_u failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("adm_u: %d %f\n", frm_idx, score_u);
        printf("adm_num_u: %d %f\n", frm_idx, score_num_u);
        printf("adm_den_u: %d %f\n", frm_idx, score_den_u);
        for(int scale=0;scale<4;scale++){
            printf("adm_num_scale%d_u: %d %f\n", scale, frm_idx, scores_u[2*scale]);
            printf("adm_den_scale%d_u: %d %f\n", scale, frm_idx, scores_u[2*scale+1]);
        }
        fflush(stdout);

        /* =========== ansnr ============== */
        ret = compute_ansnr(ref_buf_u, dis_buf_u, w_u, h_u, stride_u, stride_u, &score_u, &score_psnr_u, peak, psnr_max);

        if (ret)
        {
            printf("error: compute_ansnr_u failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        printf("ansnr_u: %d %f\n", frm_idx, score_u);
        printf("anpsnr_u: %d %f\n", frm_idx, score_psnr_u);
        fflush(stdout);

        /* =========== motion ============== */

        // compute
        if (frm_idx == 0)
        {
            score_u = 0.0;
            score2_u = 0.0;
        }
        else
        {
            if ((ret = compute_motion(prev_blur_buf_u, blur_buf_u, w_u, h_u, stride_u, stride_u, &score_u)))
            {
                printf("error: compute_motion_u (prev) failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }

            if (next_frame_read)
            {
                if ((ret = compute_motion(blur_buf_u, next_blur_buf_u, w_u, h_u, stride_u, stride_u, &score2_u)))
                {
                    printf("error: compute_motion_u (next) failed.\n");
                    fflush(stdout);
                    goto fail_or_end;
                }
                score2_u = MIN(score_u, score2_u);
            }
            else
            {
                score2_u = score_u;
            }
        }

        // print
        printf("motion_u: %d %f\n", frm_idx, score_u);
        printf("motion2_u: %d %f\n", frm_idx, score2_u);
        fflush(stdout);

        /* =========== vif ============== */

        if ((ret = compute_vif(ref_buf_u, dis_buf_u, w_u, h_u, stride_u, stride_u, &score_u, &score_num_u, &score_den_u, scores_u)))
        {
            printf("error: compute_vif_u failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("vif_u: %d %f\n", frm_idx, score_u);
        printf("vif_num_u: %d %f\n", frm_idx, score_num_u);
        printf("vif_den_u: %d %f\n", frm_idx, score_den_u);
        for(int scale=0;scale<4;scale++){
            printf("vif_num_scale%d_u: %d %f\n", scale, frm_idx, scores_u[2*scale]);
            printf("vif_den_scale%d_u: %d %f\n", scale, frm_idx, scores_u[2*scale+1]);
        }
        fflush(stdout);

        compute_motion(ref_buf_u, next_ref_buf_u, w_u, h_u, stride_u, stride_u, &score_u_anchor);

        // copy to prev_buf
        memcpy(prev_blur_buf_u, blur_buf_u, data_sz_u);
        memcpy(ref_buf_u, next_ref_buf_u, data_sz_u);
        memcpy(dis_buf_u, next_dis_buf_u, data_sz_u);
        memcpy(blur_buf_u, next_blur_buf_u, data_sz_u);

        /* =========== V ============== */
        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf_v, dis_buf_v, w_v, h_v, stride_v, stride_v, &score_v, &score_num_v, &score_den_v, scores_v, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm_v failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("adm_v: %d %f\n", frm_idx, score_v);
        printf("adm_num_v: %d %f\n", frm_idx, score_num_v);
        printf("adm_den_v: %d %f\n", frm_idx, score_den_v);
        for(int scale=0;scale<4;scale++){
            printf("adm_num_scale%d_v: %d %f\n", scale, frm_idx, scores_v[2*scale]);
            printf("adm_den_scale%d_v: %d %f\n", scale, frm_idx, scores_v[2*scale+1]);
        }
        fflush(stdout);

        /* =========== ansnr ============== */
        ret = compute_ansnr(ref_buf_v, dis_buf_v, w_v, h_v, stride_v, stride_v, &score_v, &score_psnr_v, peak, psnr_max);

        if (ret)
        {
            printf("error: compute_ansnr_v failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        printf("ansnr_v: %d %f\n", frm_idx, score_v);
        printf("anpsnr_v: %d %f\n", frm_idx, score_psnr_v);
        fflush(stdout);

        /* =========== motion ============== */

        // compute
        if (frm_idx == 0)
        {
            score_v = 0.0;
            score2_v = 0.0;
        }
        else
        {
            if ((ret = compute_motion(prev_blur_buf_v, blur_buf_v, w_v, h_v, stride_v, stride_v, &score_v)))
            {
                printf("error: compute_motion_v (prev) failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }

            if (next_frame_read)
            {
                if ((ret = compute_motion(blur_buf_v, next_blur_buf_v, w_v, h_v, stride_v, stride_v, &score2_v)))
                {
                    printf("error: compute_motion_v (next) failed.\n");
                    fflush(stdout);
                    goto fail_or_end;
                }
                score2_v = MIN(score_v, score2_v);
            }
            else
            {
                score2_v = score_v;
            }
        }

        // print
        printf("motion_v: %d %f\n", frm_idx, score_v);
        printf("motion2_v: %d %f\n", frm_idx, score2_v);
        fflush(stdout);

        /* =========== vif ============== */

        if ((ret = compute_vif(ref_buf_v, dis_buf_v, w_v, h_v, stride_v, stride_v, &score_v, &score_num_v, &score_den_v, scores_v)))
        {
            printf("error: compute_vif_v failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("vif_v: %d %f\n", frm_idx, score_v);
        printf("vif_num_v: %d %f\n", frm_idx, score_num_v);
        printf("vif_den_v: %d %f\n", frm_idx, score_den_v);
        for(int scale=0;scale<4;scale++){
            printf("vif_num_scale%d_v: %d %f\n", scale, frm_idx, scores_v[2*scale]);
            printf("vif_den_scale%d_v: %d %f\n", scale, frm_idx, scores_v[2*scale+1]);
        }
        fflush(stdout);

        // copy to prev_buf
        memcpy(prev_blur_buf_v, blur_buf_v, data_sz_v);
        memcpy(ref_buf_v, next_ref_buf_v, data_sz_v);
        memcpy(dis_buf_v, next_dis_buf_v, data_sz_v);
        memcpy(blur_buf_v, next_blur_buf_v, data_sz_v);

        if (!next_frame_read)
        {
            break;
        }

    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf_y);
    aligned_free(dis_buf_y);

    aligned_free(prev_blur_buf_y);
    aligned_free(next_ref_buf_y);
    aligned_free(next_dis_buf_y);
    aligned_free(next_blur_buf_y);

    aligned_free(blur_buf_y);
    aligned_free(temp_buf_y);

    aligned_free(ref_buf_u);
    aligned_free(dis_buf_u);

    aligned_free(prev_blur_buf_u);
    aligned_free(next_ref_buf_u);
    aligned_free(next_dis_buf_u);
    aligned_free(next_blur_buf_u);

    aligned_free(blur_buf_u);
    aligned_free(temp_buf_u);

    aligned_free(ref_buf_v);
    aligned_free(dis_buf_v);

    aligned_free(prev_blur_buf_v);
    aligned_free(next_ref_buf_v);
    aligned_free(next_dis_buf_v);
    aligned_free(next_blur_buf_v);

    aligned_free(blur_buf_v);
    aligned_free(temp_buf_v);

    return ret;
}
