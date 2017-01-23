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
#include <stdbool.h>
#include <math.h>

#include "motion_options.h"
#include "common/alloc.h"
#include "common/file_io.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "motion_tools.h"

#ifdef MOTION_OPT_SINGLE_PRECISION
    typedef float number_t;
    #define read_image_b  read_image_b2s
    #define read_image_w  read_image_w2s
    #define convolution_f32_c convolution_f32_c_s
    #define FILTER_5           FILTER_5_s

#else
    typedef double number_t;
    #define read_image_b  read_image_b2d
    #define read_image_w  read_image_w2d
    #define convolution_f32_c convolution_f32_c_d
    #define FILTER_5           FILTER_5_d
#endif

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(number_t) bytes)
 */
number_t vmaf_image_sad_c(const number_t *img1, const number_t *img2, int width, int height, int img1_stride, int img2_stride)
{
    number_t accum = (number_t)0.0;

    for (int i = 0; i < height; ++i) {
                number_t accum_line = (number_t)0.0;
        for (int j = 0; j < width; ++j) {
            number_t img1px = img1[i * img1_stride + j];
            number_t img2px = img2[i * img2_stride + j];

            accum_line += fabs(img1px - img2px);
        }
                accum += accum_line;
    }

    return (number_t) (accum / (width * height));
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion(const number_t *ref, const number_t *dis, int w, int h, int ref_stride, int dis_stride, double *score)
{

    if (ref_stride % sizeof(number_t) != 0)
    {
        printf("error: ref_stride %% sizeof(number_t) != 0, ref_stride = %d, sizeof(number_t) = %lu.\n", ref_stride, sizeof(number_t));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(number_t) != 0)
    {
        printf("error: dis_stride %% sizeof(number_t) != 0, dis_stride = %d, sizeof(number_t) = %lu.\n", dis_stride, sizeof(number_t));
        fflush(stdout);
        goto fail;
    }
    // stride for vmaf_image_sad_c is in terms of (sizeof(number_t) bytes)
    *score = vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(number_t), dis_stride / sizeof(number_t));

    return 0;

fail:
    return 1;
}

int motion(const char *ref_path, int w, int h, const char *fmt)
{
    double score = 0;
    FILE *ref_rfile = 0;
    number_t *ref_buf = 0;
    number_t *prev_blur_buf = 0;
    number_t *blur_buf = 0;
    number_t *temp_buf = 0;
    size_t data_sz;
    int stride;
    int ret = 1;

    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(number_t))
    {
        goto fail_or_end;
    }

    stride = ALIGN_CEIL(w * sizeof(number_t));

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
    if (!(temp_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(ref_rfile = fopen(ref_path, "rb")))
    {
        printf("error: fopen ref_path %s failed.\n", ref_path);
        fflush(stdout);
        goto fail_or_end;
    }

    size_t offset;
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
    {
        if ((w * h) % 2 != 0)
        {
            printf("error: (w * h) %% 2 != 0, w = %d, h = %d.\n", w, h);
            fflush(stdout);
            goto fail_or_end;
        }
        offset = w * h / 2;
    }
    else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le"))
    {
        offset = w * h;
    }
    else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le"))
    {
        offset = w * h * 2;
    }
    else
    {
        printf("error: unknown format %s.\n", fmt);
        fflush(stdout);
        goto fail_or_end;
    }

    int frm_idx = 0;
    while (1)
    {
        // ref read y
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            ret = read_image_b(ref_rfile, ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            ret = read_image_w(ref_rfile, ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        }
        else
        {
            printf("error: unknown format %s.\n", fmt);
            fflush(stdout);
            goto fail_or_end;
        }
        if (ret)
        {
            if (feof(ref_rfile))
            {
                ret = 0; // OK if end of file
            }
            goto fail_or_end;
        }

        // filter
        // apply filtering (to eliminate effects film grain)
        // stride input to convolution_f32_c is in terms of (sizeof(number_t) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(number_t)), stride divides sizeof(number_t)
        convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(number_t), stride / sizeof(number_t));

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

        // ref skip u and v
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            if (fread(temp_buf, 1, offset, ref_rfile) != (size_t)offset)
            {
                printf("error: ref fread u and v failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            if (fread(temp_buf, 2, offset, ref_rfile) != (size_t)offset)
            {
                printf("error: ref fread u and v failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }
        }
        else
        {
            printf("error: unknown format %s.\n", fmt);
            fflush(stdout);
            goto fail_or_end;
        }

        frm_idx++;
    }

    ret = 0;

fail_or_end:
    if (ref_rfile)
    {
        fclose(ref_rfile);
    }
    aligned_free(ref_buf);
    aligned_free(prev_blur_buf);
    aligned_free(blur_buf);
    aligned_free(temp_buf);

    return ret;
}
