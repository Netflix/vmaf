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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "moment_options.h"

#define read_image_b  read_image_b2s
#define read_image_w  read_image_w2s

int compute_1st_moment(const float *pic, int w, int h, int stride, double *score)
{
    double cum = 0;
    float pic_;

    int stride_ = stride / sizeof(float);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            pic_ = pic[i * stride_ + j];
            cum += pic_;
        }
    }
    cum /= (w * h);

    *score = cum;

    return 0;
}

int compute_2nd_moment(const float *pic, int w, int h, int stride, double *score)
{
    double cum = 0;
    float pic_;

    int stride_ = stride / sizeof(float);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            pic_ = pic[i * stride_ + j];
            cum += pic_ * pic_;
        }
    }
    cum /= (w * h);

    *score = cum;

    return 0;
}

int moment(int (*read_noref_frame)(float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt, int order)
{
    double score = 0;
    float *pic_buf = 0;
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

    if (!(pic_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for pic_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(temp_buf = aligned_malloc(data_sz * 2, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for temp_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    int frm_idx = 0;
    while (1)
    {
        ret = read_noref_frame(pic_buf, temp_buf, stride, user_data);

        if(ret == 1){
            goto fail_or_end;
        }
        if (ret == 2)
        {
            break;
        }

        // compute
        if (order == 1)
        {
            ret = compute_1st_moment(pic_buf, w, h, stride, &score);
            printf("1stmoment: %d %f\n", frm_idx, score);
            fflush(stdout);
        }
        else if (order == 2)
        {
            ret = compute_1st_moment(pic_buf, w, h, stride, &score);
            printf("1stmoment: %d %f\n", frm_idx, score);
            fflush(stdout);

            ret = compute_2nd_moment(pic_buf, w, h, stride, &score);
            printf("2ndmoment: %d %f\n", frm_idx, score);
            fflush(stdout);
        }
        else
        {
            printf("error: unknown order %d.\n", order);
            fflush(stdout);
            goto fail_or_end;
        }
        if (ret)
        {
            printf("error: compute_moment with order %d failed.\n", order);
            fflush(stdout);
            goto fail_or_end;
        }

        frm_idx++;
    }

    ret = 0;

fail_or_end:

    aligned_free(pic_buf);
    aligned_free(temp_buf);

    return ret;
}
