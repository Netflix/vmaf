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
#include "psnr_tools.h"
#include "psnr_options.h"

#define read_image_b  read_image_b2s
#define read_image_w  read_image_w2s

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int compute_psnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double peak, double psnr_max)
{
    double noise_ = 0;

    int ref_stride_ = ref_stride / sizeof(float);
    int dis_stride_ = dis_stride / sizeof(float);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            float ref_ = ref[i * ref_stride_ + j];
            float dis_ = dis[i * dis_stride_ + j];
            float diff = ref_ - dis_;
            noise_ += diff * diff;
        }
    }
    noise_ /= (w * h);

    double eps = 1e-10;
    *score = MIN(10 * log10(peak * peak / MAX(noise_, eps)), psnr_max);

    return 0;
}

int psnr(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    float *ref_buf = 0;
    float *dis_buf = 0;
    float *temp_buf = 0;
    size_t data_sz;
    int stride;
    double peak;
    double psnr_max;
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

        if (ret == 1)
        {
            goto fail_or_end;
        }
        if (ret == 2)
        {
            break;
        }

        // compute
        ret = compute_psnr(ref_buf, dis_buf, w, h, stride, stride, &score, peak, psnr_max);

        if (ret)
        {
            printf("error: compute_psnr failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        // print
        printf("psnr: %d %f\n", frm_idx, score);
        fflush(stdout);

        frm_idx++;
    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(dis_buf);
    aligned_free(temp_buf);

    return ret;
}


