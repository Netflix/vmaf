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
#include <stdbool.h>
#include <math.h>

#include "motion_options.h"
#include "common/alloc.h"
#include "common/file_io.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "motion_tools.h"

#define read_image_b  read_image_b2s
#define read_image_w  read_image_w2s
#define convolution_f32_c convolution_f32_c_s
#define FILTER_5           FILTER_5_s
#define offset_image       offset_image_s

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(float) bytes)
 */
float vmaf_image_sad_c(const float *img1, const float *img2, int width, int height, int img1_stride, int img2_stride)
{
    float accum = (float)0.0;

    for (int i = 0; i < height; ++i) {
                float accum_line = (float)0.0;
        for (int j = 0; j < width; ++j) {
            float img1px = img1[i * img1_stride + j];
            float img2px = img2[i * img2_stride + j];

            accum_line += fabs(img1px - img2px);
        }
                accum += accum_line;
    }

    return (float) (accum / (width * height));
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score)
{

    if (ref_stride % sizeof(float) != 0)
    {
        printf("error: ref_stride %% sizeof(float) != 0, ref_stride = %d, sizeof(float) = %zu.\n", ref_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(float) != 0)
    {
        printf("error: dis_stride %% sizeof(float) != 0, dis_stride = %d, sizeof(float) = %zu.\n", dis_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    // stride for vmaf_image_sad_c is in terms of (sizeof(float) bytes)
    *score = vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(float), dis_stride / sizeof(float));

    return 0;

fail:
    return 1;
}

int motion(int (*read_noref_frame)(float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    double score2 = 0;
    float *ref_buf = 0;
    float *prev_blur_buf = 0;
    float *blur_buf = 0;
    float *next_ref_buf = 0;
    float *next_blur_buf = 0;
    float *temp_buf = 0;
    size_t data_sz;
    int stride;
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

    data_sz = (size_t)stride * h;

    if (!(ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_buf.\n");
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
    if (!(next_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for next_blur_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(temp_buf = aligned_malloc(data_sz, MAX_ALIGN)))
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
            ret = read_noref_frame(ref_buf, temp_buf, stride, user_data);
            if(ret == 1)
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

            // ===============================================================
            // filter
            // apply filtering (to eliminate effects film grain)
            // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
            // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
            // ===============================================================
            convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));
        }

        ret = read_noref_frame(next_ref_buf, temp_buf, stride, user_data);
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

        memcpy(prev_blur_buf, blur_buf, data_sz);
        memcpy(ref_buf, next_ref_buf, data_sz);
        memcpy(blur_buf, next_blur_buf, data_sz);

        if (!next_frame_read)
        {
            break;
        }
    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(prev_blur_buf);
    aligned_free(blur_buf);
    aligned_free(next_ref_buf);
    aligned_free(next_blur_buf);
    aligned_free(temp_buf);

    return ret;
}
