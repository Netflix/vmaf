/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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

int all(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    double scores[4*2];
    double score_num = 0;
    double score_den = 0;
    double score_psnr = 0;
    float *ref_buf = 0;
    float *dis_buf = 0;

    float *prev_blur_buf = 0;
    float *blur_buf = 0;
    float *temp_buf = 0;

    size_t data_sz;
    int stride;
    int ret = 1;

    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        goto fail_or_end;
    }

    stride = ALIGN_CEIL(w * sizeof(float));

    if ((size_t)h > SIZE_MAX / stride)
    {
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

    // prev_blur_buf, blur_buf for motion only
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

    // use temp_buf for convolution_f32_c, and fread u and v
    if (!(temp_buf = aligned_malloc(data_sz * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    int frm_idx = 0;
    while (1)
    {
        ret = read_frame(ref_buf, dis_buf, temp_buf, stride, user_data);

        if(ret == 1){
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

        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("adm: %d %f\n", frm_idx, score);
        fflush(stdout);
        printf("adm_num: %d %f\n", frm_idx, score_num);
        fflush(stdout);
        printf("adm_den: %d %f\n", frm_idx, score_den);
        fflush(stdout);
        for(int scale=0;scale<4;scale++){
            printf("adm_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            printf("adm_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
        }

        /* =========== ansnr ============== */
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            // max psnr 60.0 for 8-bit per Ioannis
            ret = compute_ansnr(ref_buf, dis_buf, w, h, stride, stride, &score, &score_psnr, 255.0, 60.0);
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            // 10 bit gets normalized to 8 bit, peak is 1023 / 4.0 = 255.75
            // max psnr 72.0 for 10-bit per Ioannis
            ret = compute_ansnr(ref_buf, dis_buf, w, h, stride, stride, &score, &score_psnr, 255.75, 72.0);
        }
        else
        {
            printf("error: unknown format %s.\n", fmt);
            fflush(stdout);
            goto fail_or_end;
        }
        if (ret)
        {
            printf("error: compute_ansnr failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        printf("ansnr: %d %f\n", frm_idx, score);
        fflush(stdout);
        printf("anpsnr: %d %f\n", frm_idx, score_psnr);
        fflush(stdout);

        /* =========== motion ============== */

        // filter
        // apply filtering (to eliminate effects film grain)
        // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
        convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));

        // compute
        if (frm_idx == 0)
        {
            score = 0.0;
        }
        else
        {
            if ((ret = compute_motion(prev_blur_buf, blur_buf, w, h, stride, stride, &score)))
            {
                printf("error: compute_motion failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }
        }

        // copy to prev_buf
        memcpy(prev_blur_buf, blur_buf, data_sz);

        // print
        printf("motion: %d %f\n", frm_idx, score);
        fflush(stdout);

        /* =========== vif ============== */
        // compute vif last, because its input ref/dis might be offset by -128

        if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores)))
        {
            printf("error: compute_vif failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        printf("vif: %d %f\n", frm_idx, score);
        fflush(stdout);
        printf("vif_num: %d %f\n", frm_idx, score_num);
        fflush(stdout);
        printf("vif_den: %d %f\n", frm_idx, score_den);
        fflush(stdout);
        for(int scale=0;scale<4;scale++){
            printf("vif_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            printf("vif_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
        }

        frm_idx++;
    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(dis_buf);

    aligned_free(prev_blur_buf);
    aligned_free(blur_buf);
    aligned_free(temp_buf);

    return ret;
}
