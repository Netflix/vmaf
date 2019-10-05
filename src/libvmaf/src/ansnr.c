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
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "psnr_tools.h"
#include "ansnr_options.h"
#include "ansnr_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s
#define ansnr_filter1d_ref ansnr_filter1d_ref_s
#define ansnr_filter1d_dis ansnr_filter1d_dis_s
#define ansnr_filter2d_ref ansnr_filter2d_ref_s
#define ansnr_filter2d_dis ansnr_filter2d_dis_s
#define ansnr_filter1d     ansnr_filter1d_s
#define ansnr_filter2d     ansnr_filter2d_s
#define ansnr_mse          ansnr_mse_s
#define offset_image       offset_image_s

int compute_ansnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_psnr, double peak, double psnr_max)
{
    float *data_buf = 0;
    char *data_top;

    float *ref_filtr;
    float *filtd;

    float sig, noise;

#ifdef ANSNR_OPT_NORMALIZE
    float noise_min;
#endif

    int buf_stride = ALIGN_CEIL(w * sizeof(float));
    size_t buf_sz_one = (size_t)buf_stride * h;

    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 2)
    {
        goto fail;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * 2, MAX_ALIGN)))
    {
        goto fail;
    }

    data_top = (char *)data_buf;

    ref_filtr = (float *)data_top; data_top += buf_sz_one;
    filtd = (float *)data_top;

#ifdef ANSNR_OPT_FILTER_1D
    ansnr_filter1d(ansnr_filter1d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter1d_ref_width);
    ansnr_filter1d(ansnr_filter1d_dis, dis, filtd, w, h, dis_stride, buf_stride, ansnr_filter1d_dis_width);
#else
    ansnr_filter2d(ansnr_filter2d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter2d_ref_width);
    ansnr_filter2d(ansnr_filter2d_dis, dis, filtd, w, h, dis_stride, buf_stride, ansnr_filter2d_dis_width);
#endif

#ifdef ANSNR_OPT_DEBUG_DUMP
    write_image("stage/ref_filtr.bin", ref_filtr, w, h, buf_stride, sizeof(float));
    write_image("stage/dis_filtd.bin", filtd, w, h, buf_stride, sizeof(float));
#endif

    ansnr_mse(ref_filtr, filtd, &sig, &noise, w, h, buf_stride, buf_stride);

#ifdef ANSNR_OPT_NORMALIZE
# ifdef ANSNR_OPT_FILTER_1D
    ansnr_filter1d(ansnr_filter1d_dis, ref, filtd, w, h, ref_stride, buf_stride, ansnr_filter1d_dis_width);
# else
    ansnr_filter2d(ansnr_filter2d_dis, ref, filtd, w, h, ref_stride, buf_stride, ansnr_filter2d_dis_width);
# endif
# ifdef ANSNR_OPT_DEBUG_DUMP
    write_image("stage/ref_filtd.bin", filtd, w, h, buf_stride, sizeof(float));
# endif
    ansnr_mse(ref_filtr, filtd, 0, &noise_min, w, h, buf_stride, buf_stride);
    *score = 10.0 * log10(noise / (noise - noise_min));
#else
    *score = noise==0 ? psnr_max : 10.0 * log10(sig / noise);
#endif

    double eps = 1e-10;
    *score_psnr = MIN(10 * log10(peak * peak * w * h / MAX(noise, eps)), psnr_max);

    ret = 0;
fail:
    aligned_free(data_buf);
    return ret;
}

int ansnr(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    double score_psnr = 0;
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

        // ===============================================================
        // offset pixel by OPT_RANGE_PIXEL_OFFSET
        // ===============================================================
        offset_image(ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        offset_image(dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);

        // compute
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

        frm_idx++;
    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(dis_buf);
    aligned_free(temp_buf);

    return ret;
}
