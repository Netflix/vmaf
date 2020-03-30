/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
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

#include "mem.h"
#include "common/convolution.h"
#include "offset.h"
#include "vif_options.h"
#include "integer_vif_tools.h"
#include "common/macros.h"

#define integer_vif_filter1d_table      integer_vif_filter1d_table_s
#define integer_vif_filter1d_rdCombine  integer_vif_filter1d_rdCombine_s
#define integer_vif_filter1d_combined   integer_vif_filter1d_combined_s
#define integer_vif_dec2                integer_vif_dec2_s
#define integer_vif_xx_yy_xy            integer_vif_xx_yy_xy_s
#define integer_vif_statistic           integer_vif_statistic_s

int integer_compute_vif(const int16_t *ref, const int16_t *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, int inp_size_bits)
{
    int32_t *data_buf = 0;
    char *data_top;

    int16_t *integer_ref_scale;
    int16_t *integer_dis_scale;

    int16_t *integer_mu1;
    int16_t *integer_mu2;

    int32_t *integer_mu1_32;
    int32_t *integer_mu2_32;

    int32_t *integer_ref_sq_filt;
    int32_t *integer_dis_sq_filt;
    int32_t *integer_ref_dis_filt;

    int32_t *tmp_mu1;
    int32_t *tmp_mu2;
    int32_t *tmp_ref;
    int32_t *tmp_dis;
    int32_t *tmp_ref_dis;
    int32_t *tmp_ref_convol;
    int32_t *tmp_dis_convol;

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

    int16_t *integer_curr_ref_scale = ref;
    int16_t *integer_curr_dis_scale = dis;
    
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int buf_stride = ALIGN_CEIL(w * sizeof(int32_t));
    size_t buf_sz_one = (size_t)buf_stride * h;

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    //size of tmp buffers used in function
    size_t buf_sz_one_tmp = 0;
    buf_sz_one_tmp = ((size_t)buf_stride * h * 7);

    if (SIZE_MAX / buf_sz_one < 15)
    {
        printf("error: SIZE_MAX / buf_sz_one < 15, buf_sz_one = %zu.\n", buf_sz_one);
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(data_buf = aligned_malloc(buf_sz_one * 9 + buf_sz_one_tmp, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for data_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    data_top = (char *)data_buf;

    num_array    = (float *)data_top; data_top += buf_sz_one;
    den_array    = (float *)data_top; data_top += buf_sz_one;
    integer_ref_scale = (int16_t *)data_top; data_top += buf_sz_one/2;
    integer_dis_scale = (int16_t *)data_top; data_top += buf_sz_one/2;

    integer_mu1 = (int16_t *)data_top; data_top += buf_sz_one/2;
    integer_mu2 = (int16_t *)data_top; data_top += buf_sz_one/2;

    integer_mu1_32 = (int32_t *)data_top; data_top += buf_sz_one;
    integer_mu2_32 = (int32_t *)data_top; data_top += buf_sz_one;

    integer_ref_sq_filt = (int32_t *)data_top; data_top += buf_sz_one;
    integer_dis_sq_filt = (int32_t *)data_top; data_top += buf_sz_one;
    integer_ref_dis_filt = (int32_t *)data_top; data_top += buf_sz_one;
  
    /*
    tmp pointers are used to pass data in VIF function and it make easy for one time free to improve performance
    */
    tmp_ref_convol = (int32_t *)data_top; data_top += buf_stride;
    tmp_dis_convol = (int32_t *)data_top; data_top += buf_stride;

    tmp_mu1 = (int32_t *)data_top; data_top += buf_stride;
    tmp_mu2 = (int32_t *)data_top; data_top += buf_stride;
    tmp_ref = (int32_t *)data_top; data_top += buf_stride;
    tmp_dis = (int32_t *)data_top; data_top += buf_stride;
    tmp_ref_dis = (int32_t *)data_top; data_top += buf_stride;

    for (scale = 0; scale < 4; ++scale)
    {
#ifdef VIF_OPT_DEBUG_DUMP
        char pathbuf[256];
#endif
        const uint16_t *integer_filter = integer_vif_filter1d_table[scale];
        int filter_width       = vif_filter1d_width[scale];

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
            /*
            Combined functionality for both reference and distorted frame in single function
            */
            integer_vif_filter1d_rdCombine(integer_filter, integer_curr_ref_scale, integer_curr_dis_scale, integer_mu1, integer_mu2, w, h, curr_ref_stride, curr_dis_stride, buf_stride, filter_width, scale - 1, tmp_ref_convol, tmp_dis_convol, inp_size_bits);

            integer_vif_dec2(integer_mu1, integer_ref_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);
            integer_vif_dec2(integer_mu2, integer_dis_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);
            w  = buf_valid_w / 2;
            h  = buf_valid_h / 2;
#ifdef VIF_OPT_HANDLE_BORDERS
            buf_valid_w = w;
            buf_valid_h = h;
#else
            buf_valid_w = w - filter_adj * 2;
            buf_valid_h = h - filter_adj * 2;
#endif
            integer_curr_ref_scale = integer_ref_scale;
            integer_curr_dis_scale = integer_dis_scale;
            
            curr_ref_stride = buf_stride;
            curr_dis_stride = buf_stride;
        }
        /*
            Combined functionality for both reference and distorted frame in single function
            for integer_vif_filter1d, integer_vif_filter1d_sq and integer_vif_filter1d_ref_dis function
        */
        integer_vif_filter1d_combined(integer_filter, integer_curr_ref_scale, integer_curr_dis_scale,
            integer_ref_sq_filt, integer_dis_sq_filt, integer_ref_dis_filt, integer_mu1_32, integer_mu2_32, w, h,
            buf_stride, buf_stride, filter_width, scale, tmp_mu1, tmp_mu2, tmp_ref, tmp_dis, tmp_ref_dis, inp_size_bits);

        integer_vif_statistic(integer_mu1_32, integer_mu2_32, integer_ref_sq_filt, integer_dis_sq_filt, integer_ref_dis_filt, num_array, den_array, w, h, buf_stride, scale);

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

		num = *num_array;
		den = *den_array;

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