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
#include "vif_options.h"
#include "vif_tools.h"

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s
#define vif_filter1d_table vif_filter1d_table_s
#define vif_filter1d       vif_filter1d_s
#define vif_filter2d_table vif_filter2d_table_s
#define vif_filter2d       vif_filter2d_s
#define vif_dec2           vif_dec2_s
#define vif_sum            vif_sum_s
#define vif_xx_yy_xy       vif_xx_yy_xy_s
#define vif_statistic      vif_statistic_s
#define offset_image       offset_image_s

int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores)
{
    float *data_buf = 0;
    char *data_top;

    float *ref_scale;
    float *dis_scale;
    float *ref_sq;
    float *dis_sq;
    float *ref_dis;

    float *mu1;
    float *mu2;
    float *mu1_sq;
    float *mu2_sq;
    float *mu1_mu2;
    float *ref_sq_filt;
    float *dis_sq_filt;
    float *ref_dis_filt;
    float *num_array;
    float *den_array;
    float *tmpbuf;

    /* Offset pointers to adjust for convolution border handling. */
    float *mu1_adj = 0;
    float *mu2_adj = 0;

#ifdef VIF_OPT_DEBUG_DUMP
    float *mu1_sq_adj;
    float *mu2_sq_adj;
    float *mu1_mu2_adj;
    float *ref_sq_filt_adj;
    float *dis_sq_filt_adj;
    float *ref_dis_filt_adj = 0;
#endif

    float *num_array_adj = 0;
    float *den_array_adj = 0;

    /* Special handling of first scale. */
    const float *curr_ref_scale = ref;
    const float *curr_dis_scale = dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int buf_stride = ALIGN_CEIL(w * sizeof(float));
    size_t buf_sz_one = (size_t)buf_stride * h;

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 15)
    {
        printf("error: SIZE_MAX / buf_sz_one < 15, buf_sz_one = %zu.\n", buf_sz_one);
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * 16, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for data_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    data_top = (char *)data_buf;

    ref_scale = (float *)data_top; data_top += buf_sz_one;
    dis_scale = (float *)data_top; data_top += buf_sz_one;
    ref_sq    = (float *)data_top; data_top += buf_sz_one;
    dis_sq    = (float *)data_top; data_top += buf_sz_one;
    ref_dis   = (float *)data_top; data_top += buf_sz_one;
    mu1          = (float *)data_top; data_top += buf_sz_one;
    mu2          = (float *)data_top; data_top += buf_sz_one;
    mu1_sq       = (float *)data_top; data_top += buf_sz_one;
    mu2_sq       = (float *)data_top; data_top += buf_sz_one;
    mu1_mu2      = (float *)data_top; data_top += buf_sz_one;
    ref_sq_filt  = (float *)data_top; data_top += buf_sz_one;
    dis_sq_filt  = (float *)data_top; data_top += buf_sz_one;
    ref_dis_filt = (float *)data_top; data_top += buf_sz_one;
    num_array    = (float *)data_top; data_top += buf_sz_one;
    den_array    = (float *)data_top; data_top += buf_sz_one;
    tmpbuf    = (float *)data_top; data_top += buf_sz_one;

    for (scale = 0; scale < 4; ++scale)
    {
#ifdef VIF_OPT_DEBUG_DUMP
        char pathbuf[256];
#endif

#ifdef VIF_OPT_FILTER_1D
        const float *filter = vif_filter1d_table[scale];
        int filter_width       = vif_filter1d_width[scale];
#else
        const float *filter = vif_filter2d_table[scale];
        int filter_width       = vif_filter2d_width[scale];
#endif

#ifdef VIF_OPT_HANDLE_BORDERS
        int buf_valid_w = w;
        int buf_valid_h = h;

  #define ADJUST(x) x
#else
        int filter_adj  = filter_width / 2;
        int buf_valid_w = w - filter_adj * 2;
        int buf_valid_h = h - filter_adj * 2;

  #define ADJUST(x) ((float *)((char *)(x) + filter_adj * buf_stride + filter_adj * sizeof(float)))
#endif

        if (scale > 0)
        {
#ifdef VIF_OPT_FILTER_1D
            vif_filter1d(filter, curr_ref_scale, mu1, tmpbuf, w, h, curr_ref_stride, buf_stride, filter_width);
            vif_filter1d(filter, curr_dis_scale, mu2, tmpbuf, w, h, curr_dis_stride, buf_stride, filter_width);
#else
            vif_filter2d(filter, curr_ref_scale, mu1, w, h, curr_ref_stride, buf_stride, filter_width);
            vif_filter2d(filter, curr_dis_scale, mu2, w, h, curr_dis_stride, buf_stride, filter_width);
#endif
            mu1_adj = ADJUST(mu1);
            mu2_adj = ADJUST(mu2);

            vif_dec2(mu1_adj, ref_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);
            vif_dec2(mu2_adj, dis_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);

            w  = buf_valid_w / 2;
            h  = buf_valid_h / 2;
#ifdef VIF_OPT_HANDLE_BORDERS
            buf_valid_w = w;
            buf_valid_h = h;
#else
            buf_valid_w = w - filter_adj * 2;
            buf_valid_h = h - filter_adj * 2;
#endif
            curr_ref_scale = ref_scale;
            curr_dis_scale = dis_scale;

            curr_ref_stride = buf_stride;
            curr_dis_stride = buf_stride;
        }

#ifdef VIF_OPT_FILTER_1D
        vif_filter1d(filter, curr_ref_scale, mu1, tmpbuf, w, h, curr_ref_stride, buf_stride, filter_width);
        vif_filter1d(filter, curr_dis_scale, mu2, tmpbuf, w, h, curr_dis_stride, buf_stride, filter_width);
#else
        vif_filter2d(filter, curr_ref_scale, mu1, w, h, curr_ref_stride, buf_stride, filter_width);
        vif_filter2d(filter, curr_dis_scale, mu2, w, h, curr_dis_stride, buf_stride, filter_width);
#endif
        vif_xx_yy_xy(mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, w, h, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride);

        vif_xx_yy_xy(curr_ref_scale, curr_dis_scale, ref_sq, dis_sq, ref_dis, w, h, curr_ref_stride, curr_dis_stride, buf_stride, buf_stride, buf_stride);
#ifdef VIF_OPT_FILTER_1D
        vif_filter1d(filter, ref_sq, ref_sq_filt, tmpbuf, w, h, buf_stride, buf_stride, filter_width);
        vif_filter1d(filter, dis_sq, dis_sq_filt, tmpbuf, w, h, buf_stride, buf_stride, filter_width);
        vif_filter1d(filter, ref_dis, ref_dis_filt, tmpbuf, w, h, buf_stride, buf_stride, filter_width);
#else
        vif_filter2d(filter, ref_sq, ref_sq_filt, w, h, buf_stride, buf_stride, filter_width);
        vif_filter2d(filter, dis_sq, dis_sq_filt, w, h, buf_stride, buf_stride, filter_width);
        vif_filter2d(filter, ref_dis, ref_dis_filt, w, h, buf_stride, buf_stride, filter_width);
#endif
        vif_statistic(mu1_sq, mu2_sq, mu1_mu2, ref_sq_filt, dis_sq_filt, ref_dis_filt, num_array, den_array,
                      w, h, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride);

        mu1_adj = ADJUST(mu1);
        mu2_adj = ADJUST(mu2);

#ifdef VIF_OPT_DEBUG_DUMP
        mu1_sq_adj  = ADJUST(mu1_sq);
        mu2_sq_adj  = ADJUST(mu2_sq);
        mu1_mu2_adj = ADJUST(mu1_mu2);

        ref_sq_filt_adj  = ADJUST(ref_sq_filt);
        dis_sq_filt_adj  = ADJUST(dis_sq_filt);
        ref_dis_filt_adj = ADJUST(ref_dis_filt);
#endif

        num_array_adj = ADJUST(num_array);
        den_array_adj = ADJUST(den_array);
#undef ADJUST

#ifdef VIF_OPT_DEBUG_DUMP
        sprintf(pathbuf, "stage/ref[%d].bin", scale);
        write_image(pathbuf, curr_ref_scale, w, h, curr_ref_stride, sizeof(float));

        sprintf(pathbuf, "stage/dis[%d].bin", scale);
        write_image(pathbuf, curr_dis_scale, w, h, curr_dis_stride, sizeof(float));

        sprintf(pathbuf, "stage/mu1[%d].bin", scale);
        write_image(pathbuf, mu1_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/mu2[%d].bin", scale);
        write_image(pathbuf, mu2_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/mu1_sq[%d].bin", scale);
        write_image(pathbuf, mu1_sq_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/mu2_sq[%d].bin", scale);
        write_image(pathbuf, mu2_sq_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/mu1_mu2[%d].bin", scale);
        write_image(pathbuf, mu1_mu2_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/ref_sq_filt[%d].bin", scale);
        write_image(pathbuf, ref_sq_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/dis_sq_filt[%d].bin", scale);
        write_image(pathbuf, dis_sq_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/ref_dis_filt[%d].bin", scale);
        write_image(pathbuf, ref_dis_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/num_array[%d].bin", scale);
        write_image(pathbuf, num_array_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/den_array[%d].bin", scale);
        write_image(pathbuf, den_array_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));
#endif

        num = vif_sum(num_array_adj, buf_valid_w, buf_valid_h, buf_stride);
        den = vif_sum(den_array_adj, buf_valid_w, buf_valid_h, buf_stride);

        scores[2*scale] = num;
        scores[2*scale+1] = den;

#ifdef VIF_OPT_DEBUG_DUMP
        printf("num[%d]: %e\n", scale, num);
        printf("den[%d]: %e\n", scale, den);
#endif
    }

    *score_num = 0.0;
    *score_den = 0.0;
    for (scale = 0; scale < 4; ++scale)
    {
        *score_num += scores[2*scale];
        *score_den += scores[2*scale+1];
    }
    if (*score_den == 0.0)
    {
        *score = 1.0f;
    }
    else
    {
        *score = (*score_num) / (*score_den);
    }

    ret = 0;
fail_or_end:
    aligned_free(data_buf);
    return ret;
}

int vif(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    double scores[4*2];
    double score_num = 0;
    double score_den = 0;
    float *ref_buf = 0;
    float *dis_buf = 0;
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

        // compute
        if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores)))
        {
            printf("error: compute_vif failed.\n");
            fflush(stdout);
            goto fail_or_end;
        }

        // print
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
    aligned_free(temp_buf);

    return ret;
}
