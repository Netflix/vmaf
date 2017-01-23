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
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "ansnr_options.h"
#include "ansnr_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef ANSNR_OPT_SINGLE_PRECISION
  typedef float number_t;

  #define read_image_b       read_image_b2s
  #define read_image_w       read_image_w2s
  #define ansnr_filter1d_ref ansnr_filter1d_ref_s
  #define ansnr_filter1d_dis ansnr_filter1d_dis_s
  #define ansnr_filter2d_ref ansnr_filter2d_ref_s
  #define ansnr_filter2d_dis ansnr_filter2d_dis_s
  #define ansnr_filter1d     ansnr_filter1d_s
  #define ansnr_filter2d     ansnr_filter2d_s
  #define ansnr_mse          ansnr_mse_s
#else
  typedef double number_t;

  #define read_image_b       read_image_b2d
  #define read_image_w       read_image_w2d
  #define ansnr_filter1d_ref ansnr_filter1d_ref_d
  #define ansnr_filter1d_dis ansnr_filter1d_dis_d
  #define ansnr_filter2d_ref ansnr_filter2d_ref_d
  #define ansnr_filter2d_dis ansnr_filter2d_dis_d
  #define ansnr_filter1d     ansnr_filter1d_d
  #define ansnr_filter2d     ansnr_filter2d_d
  #define ansnr_mse          ansnr_mse_d
#endif

int compute_ansnr(const number_t *ref, const number_t *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_psnr, double peak, double psnr_max)
{
    number_t *data_buf = 0;
    char *data_top;

    number_t *ref_filtr;
    number_t *ref_filtd;
    number_t *dis_filtd;

    number_t sig, noise;

#ifdef ANSNR_OPT_NORMALIZE
    number_t noise_min;
#endif

    int buf_stride = ALIGN_CEIL(w * sizeof(number_t));
    size_t buf_sz_one = (size_t)buf_stride * h;

    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 3)
    {
        goto fail;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * 3, MAX_ALIGN)))
    {
        goto fail;
    }

    data_top = (char *)data_buf;

    ref_filtr = (number_t *)data_top; data_top += buf_sz_one;
    ref_filtd = (number_t *)data_top; data_top += buf_sz_one;
    dis_filtd = (number_t *)data_top; data_top += buf_sz_one;

#ifdef ANSNR_OPT_FILTER_1D
    ansnr_filter1d(ansnr_filter1d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter1d_ref_width);
    ansnr_filter1d(ansnr_filter1d_dis, ref, ref_filtd, w, h, ref_stride, buf_stride, ansnr_filter1d_dis_width);
    ansnr_filter1d(ansnr_filter1d_dis, dis, dis_filtd, w, h, dis_stride, buf_stride, ansnr_filter1d_dis_width);
#else
    ansnr_filter2d(ansnr_filter2d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter2d_ref_width);
    ansnr_filter2d(ansnr_filter2d_dis, ref, ref_filtd, w, h, ref_stride, buf_stride, ansnr_filter2d_dis_width);
    ansnr_filter2d(ansnr_filter2d_dis, dis, dis_filtd, w, h, dis_stride, buf_stride, ansnr_filter2d_dis_width);
#endif

#ifdef ANSNR_OPT_DEBUG_DUMP
    write_image("stage/ref_filtr.bin", ref_filtr, w, h, buf_stride, sizeof(number_t));
    write_image("stage/ref_filtd.bin", ref_filtd, w, h, buf_stride, sizeof(number_t));
    write_image("stage/dis_filtd.bin", dis_filtd, w, h, buf_stride, sizeof(number_t));
#endif

    ansnr_mse(ref_filtr, dis_filtd, &sig, &noise, w, h, buf_stride, buf_stride);

#ifdef ANSNR_OPT_NORMALIZE
    ansnr_mse(ref_filtr, ref_filtd, 0, &noise_min, w, h, buf_stride, buf_stride);
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

int ansnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt)
{
    double score = 0;
    double score_psnr = 0;
    number_t *ref_buf = 0;
    number_t *dis_buf = 0;
    number_t *temp_buf = 0;
    FILE *ref_rfile = 0;
    FILE *dis_rfile = 0;
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

    if (!(ref_rfile = fopen(ref_path, "rb")))
    {
        printf("error: fopen ref_path %s failed.\n", ref_path);
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_rfile = fopen(dis_path, "rb")))
    {
        printf("error: fopen dis_path %s failed.\n", dis_path);
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
        // read ref y
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

        // read dis y
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            ret = read_image_b(dis_rfile, dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            ret = read_image_w(dis_rfile, dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        }
        else
        {
            printf("error: unknown format %s.\n", fmt);
            fflush(stdout);
            goto fail_or_end;
        }
        if (ret)
        {
            if (feof(dis_rfile))
            {
                ret = 0; // OK if end of file
            }
            goto fail_or_end;
        }

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

        // dis skip u and v
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            if (fread(temp_buf, 1, offset, dis_rfile) != (size_t)offset)
            {
                printf("error: dis fread u and v failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            if (fread(temp_buf, 2, offset, dis_rfile) != (size_t)offset)
            {
                printf("error: dis fread u and v failed.\n");
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
    if (dis_rfile)
    {
        fclose(dis_rfile);
    }
    aligned_free(ref_buf);
    aligned_free(dis_buf);
    aligned_free(temp_buf);

    return ret;
}
