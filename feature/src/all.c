/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
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
