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
#include <string.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "adm_options.h"
#include "adm_tools.h"

#ifdef ADM_OPT_SINGLE_PRECISION
  typedef float number_t;
  typedef adm_dwt_band_t_s adm_dwt_band_t;

  #define read_image_b  read_image_b2s
  #define read_image_w  read_image_w2s
  #define adm_dwt2      adm_dwt2_s
  #define adm_decouple  adm_decouple_s
  #define adm_csf       adm_csf_s
  #define adm_cm_thresh adm_cm_thresh_s
  #define adm_cm        adm_cm_s
  #define adm_sum_cube  adm_sum_cube_s
#else
  typedef double number_t;
  typedef adm_dwt_band_t_d adm_dwt_band_t;

  #define read_image_b  read_image_b2d
  #define read_image_w  read_image_w2d
  #define adm_dwt2      adm_dwt2_d
  #define adm_decouple  adm_decouple_d
  #define adm_csf       adm_csf_d
  #define adm_cm_thresh adm_cm_thresh_d
  #define adm_cm        adm_cm_d
  #define adm_sum_cube  adm_sum_cube_d
#endif

static char *init_dwt_band(adm_dwt_band_t *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = (number_t *)data_top; data_top += buf_sz_one;
    band->band_h = (number_t *)data_top; data_top += buf_sz_one;
    band->band_v = (number_t *)data_top; data_top += buf_sz_one;
    band->band_d = (number_t *)data_top; data_top += buf_sz_one;
    return data_top;
}

int compute_adm(const number_t *ref, const number_t *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor)
{
#ifdef ADM_OPT_SINGLE_PRECISION
    double numden_limit = 1e-2 * (w * h) / (1920.0 * 1080.0);
#else
    double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
#endif
    number_t *data_buf = 0;
    char *data_top;

    number_t *ref_scale;
    number_t *dis_scale;

    adm_dwt_band_t ref_dwt2;
    adm_dwt_band_t dis_dwt2;

    adm_dwt_band_t decouple_r;
    adm_dwt_band_t decouple_a;

    adm_dwt_band_t csf_o;
    adm_dwt_band_t csf_r;
    adm_dwt_band_t csf_a;

    number_t *mta;

    adm_dwt_band_t cm_r;

    const number_t *curr_ref_scale = ref;
    const number_t *curr_dis_scale = dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int orig_h = h;

    int buf_stride = ALIGN_CEIL(((w + 1) / 2) * sizeof(number_t));
    size_t buf_sz_one = (size_t)buf_stride * ((h + 1) / 2);

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 35)
    {
        printf("error: SIZE_MAX / buf_sz_one < 35, buf_sz_one = %lu.\n", buf_sz_one);
        fflush(stdout);
        goto fail;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * 35, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for data_buf.\n");
        fflush(stdout);
        goto fail;
    }

    data_top = (char *)data_buf;

    ref_scale = (number_t *)data_top; data_top += buf_sz_one;
    dis_scale = (number_t *)data_top; data_top += buf_sz_one;

    data_top = init_dwt_band(&ref_dwt2, data_top, buf_sz_one);
    data_top = init_dwt_band(&dis_dwt2, data_top, buf_sz_one);
    data_top = init_dwt_band(&decouple_r, data_top, buf_sz_one);
    data_top = init_dwt_band(&decouple_a, data_top, buf_sz_one);
    data_top = init_dwt_band(&csf_o, data_top, buf_sz_one);
    data_top = init_dwt_band(&csf_r, data_top, buf_sz_one);
    data_top = init_dwt_band(&csf_a, data_top, buf_sz_one);

    mta = (number_t *)data_top; data_top += buf_sz_one;

    data_top = init_dwt_band(&cm_r, data_top, buf_sz_one);

    for (scale = 0; scale < 4; ++scale) {
#ifdef ADM_OPT_DEBUG_DUMP
        char pathbuf[256];
#endif
        float num_scale = 0.0;
        float den_scale = 0.0;

        adm_dwt2(curr_ref_scale, &ref_dwt2, w, h, curr_ref_stride, buf_stride);
        adm_dwt2(curr_dis_scale, &dis_dwt2, w, h, curr_dis_stride, buf_stride);

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride);

        adm_csf(&ref_dwt2, &csf_o, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_r, &csf_r, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_a, &csf_a, orig_h, scale, w, h, buf_stride, buf_stride);

        adm_cm_thresh(&csf_a, mta, w, h, buf_stride, buf_stride);
        adm_cm(&csf_r, &cm_r, mta, w, h, buf_stride, buf_stride, buf_stride);

#ifdef ADM_OPT_DEBUG_DUMP
        sprintf(pathbuf, "stage/ref[%d]_a.yuv", scale);
        write_image(pathbuf, ref_dwt2.band_a, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/ref[%d]_h.yuv", scale);
        write_image(pathbuf, ref_dwt2.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/ref[%d]_v.yuv", scale);
        write_image(pathbuf, ref_dwt2.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/ref[%d]_d.yuv", scale);
        write_image(pathbuf, ref_dwt2.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/dis[%d]_a.yuv", scale);
        write_image(pathbuf, dis_dwt2.band_a, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/dis[%d]_h.yuv", scale);
        write_image(pathbuf, dis_dwt2.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/dis[%d]_v.yuv", scale);
        write_image(pathbuf, dis_dwt2.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/dis[%d]_d.yuv", scale);
        write_image(pathbuf, dis_dwt2.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/r[%d]_h.yuv", scale);
        write_image(pathbuf, decouple_r.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/r[%d]_v.yuv", scale);
        write_image(pathbuf, decouple_r.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/r[%d]_d.yuv", scale);
        write_image(pathbuf, decouple_r.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/a[%d]_h.yuv", scale);
        write_image(pathbuf, decouple_a.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/a[%d]_v.yuv", scale);
        write_image(pathbuf, decouple_a.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/a[%d]_d.yuv", scale);
        write_image(pathbuf, decouple_a.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_o[%d]_h.yuv", scale);
        write_image(pathbuf, csf_o.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_o[%d]_v.yuv", scale);
        write_image(pathbuf, csf_o.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_o[%d]_d.yuv", scale);
        write_image(pathbuf, csf_o.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_r[%d]_h.yuv", scale);
        write_image(pathbuf, csf_r.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_r[%d]_v.yuv", scale);
        write_image(pathbuf, csf_r.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_r[%d]_d.yuv", scale);
        write_image(pathbuf, csf_r.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_a[%d]_h.yuv", scale);
        write_image(pathbuf, csf_a.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_a[%d]_v.yuv", scale);
        write_image(pathbuf, csf_a.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/csf_a[%d]_d.yuv", scale);
        write_image(pathbuf, csf_a.band_d, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/mta[%d].yuv", scale);
        write_image(pathbuf, mta, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/cm_r[%d]_h.yuv", scale);
        write_image(pathbuf, cm_r.band_h, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/cm_r[%d]_v.yuv", scale);
        write_image(pathbuf, cm_r.band_v, w, h, buf_stride, sizeof(number_t));

        sprintf(pathbuf, "stage/cm_r[%d]_d.yuv", scale);
        write_image(pathbuf, cm_r.band_d, w, h, buf_stride, sizeof(number_t));
#endif
        num_scale += adm_sum_cube(cm_r.band_h, w, h, buf_stride, border_factor);
        num_scale += adm_sum_cube(cm_r.band_v, w, h, buf_stride, border_factor);
        num_scale += adm_sum_cube(cm_r.band_d, w, h, buf_stride, border_factor);

        den_scale += adm_sum_cube(csf_o.band_h, w, h, buf_stride, border_factor);
        den_scale += adm_sum_cube(csf_o.band_v, w, h, buf_stride, border_factor);
        den_scale += adm_sum_cube(csf_o.band_d, w, h, buf_stride, border_factor);

        num += num_scale;
        den += den_scale;

        /* Copy DWT2 approximation band to buffer for next scale. */
        adm_buffer_copy(ref_dwt2.band_a, ref_scale, w * sizeof(number_t), h, buf_stride, buf_stride);
        adm_buffer_copy(dis_dwt2.band_a, dis_scale, w * sizeof(number_t), h, buf_stride, buf_stride);

        curr_ref_scale = ref_scale;
        curr_dis_scale = dis_scale;
        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;
#ifdef ADM_OPT_DEBUG_DUMP
        printf("num: %f\n", num);
        printf("den: %f\n", den);
#endif
        scores[2*scale+0] = num_scale;
        scores[2*scale+1] = den_scale;
    }

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0)
    {
        *score = 1.0f;
    }
    else
    {
        *score = num / den;
    }
    *score_num = num;
    *score_den = den;

    ret = 0;

fail:
    aligned_free(data_buf);
    return ret;
}

int adm(const char *ref_path, const char *dis_path, int w, int h, const char *fmt)
{
    double score = 0;
    double score_num = 0;
    double score_den = 0;
    double scores[2*4];
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
        printf("error: fopen ref_path %s failed\n", ref_path);
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

        // compute
        if ((ret = compute_adm(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores, ADM_BORDER_FACTOR)))
        {
            printf("error: compute_adm failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        // print
        printf("adm: %d %f\n", frm_idx, score);
        fflush(stdout);
        printf("adm_num: %d %f\n", frm_idx, score_num);
        fflush(stdout);
        printf("adm_den: %d %f\n", frm_idx, score_den);
        fflush(stdout);
        for(int scale=0;scale<4;scale++) {
            printf("adm_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            fflush(stdout);
            printf("adm_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
            fflush(stdout);
        }

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
