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
#include <string.h>
#include <math.h>

#include "mem.h"
#include "offset.h"
#include "vif_options.h"
#include "vif_tools.h"

#define ALMOST_EQUAL(x,c) fabs((x)-(c))<1.0e-4

/**
 * Note: stride is in terms of bytes
 */
void apply_frame_differencing(const float *current_frame, const float *previous_frame, float *frame_difference, int width, int height, int stride)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            frame_difference[i * stride + j] = current_frame[i * stride + j] - previous_frame[i * stride + j];
        }
    }
}

int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride,
        double *score, double *score_num, double *score_den, double *scores,
        double vif_enhn_gain_limit, double vif_kernelscale)
{
    float *data_buf = 0;
    char *data_top;

    float *ref_scale;
    float *dis_scale;

    float *mu1;
    float *mu2;
    float *ref_sq_filt;
    float *dis_sq_filt;
    float *ref_dis_filt;
    float *tmpbuf;

    const float *filter;
    int filter_width;

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

    /* Special handling of first scale. */
    const float *curr_ref_scale = ref;
    const float *curr_dis_scale = dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int buf_stride = ALIGN_CEIL(w * sizeof(float));
    size_t buf_sz_one = (size_t)buf_stride * h;

    int scale;
    int ret = 1;

    int kernelscale_index = -1;
    if (ALMOST_EQUAL(vif_kernelscale, 1.0)) {
        kernelscale_index = vif_kernelscale_1;
    } else if (ALMOST_EQUAL(vif_kernelscale, 1.0/2)) {
        kernelscale_index = vif_kernelscale_1o2;
    } else if (ALMOST_EQUAL(vif_kernelscale, 3.0/2)) {
        kernelscale_index = vif_kernelscale_3o2;
    } else if (ALMOST_EQUAL(vif_kernelscale, 2.0)) {
        kernelscale_index = vif_kernelscale_2;
    } else if (ALMOST_EQUAL(vif_kernelscale, 2.0/3)) {
        kernelscale_index = vif_kernelscale_2o3;
    } else if (ALMOST_EQUAL(vif_kernelscale, 2.4/1.0)) {
        kernelscale_index = vif_kernelscale_24o10;
    } else if (ALMOST_EQUAL(vif_kernelscale, 360/97.0)) {
        kernelscale_index = vif_kernelscale_360o97;
    } else if (ALMOST_EQUAL(vif_kernelscale, 4.0/3.0)) {
        kernelscale_index = vif_kernelscale_4o3;
    } else if (ALMOST_EQUAL(vif_kernelscale, 3.5/3.0)) {
        kernelscale_index = vif_kernelscale_3d5o3;
    } else if (ALMOST_EQUAL(vif_kernelscale, 3.75/3.0)) {
        kernelscale_index = vif_kernelscale_3d75o3;
    } else if (ALMOST_EQUAL(vif_kernelscale, 4.25/3.0)) {
        kernelscale_index = vif_kernelscale_4d25o3;
    } else {
        printf("error: vif_kernelscale can only be 0.5, 1.0, 1.5, 2.0, 2.0/3, 2.4, 360/97, 4.0/3.0, 3.5/3.0, 3.75/3.0, 4.25/3.0 for now, but is %f\n", vif_kernelscale);
        fflush(stdout);
        goto fail_or_end;
    }

    // Code optimized to save on multiple buffer copies
    // hence the reduction in the number of buffers required from 15 to 8
#define VIF_BUF_CNT 8
    if (SIZE_MAX / buf_sz_one < VIF_BUF_CNT)
    {
        printf("error: SIZE_MAX / buf_sz_one < VIF_BUF_CNT, buf_sz_one = %zu.\n", buf_sz_one);
        fflush(stdout);
        goto fail_or_end;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * VIF_BUF_CNT, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for data_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    data_top = (char *)data_buf;

    ref_scale = (float *)data_top; data_top += buf_sz_one;
    dis_scale = (float *)data_top; data_top += buf_sz_one;
    mu1 = (float *)data_top; data_top += buf_sz_one;
    mu2 = (float *)data_top; data_top += buf_sz_one;
    ref_sq_filt = (float *)data_top; data_top += buf_sz_one;
    dis_sq_filt = (float *)data_top; data_top += buf_sz_one;
    ref_dis_filt = (float *)data_top; data_top += buf_sz_one;
    tmpbuf = (float *)data_top; data_top += buf_sz_one;

    for (scale = 0; scale < 4; ++scale)
    {
#ifdef VIF_OPT_DEBUG_DUMP
        char pathbuf[256];
#endif

        filter = vif_filter1d_table_s[kernelscale_index][scale];
        filter_width = vif_filter1d_width[kernelscale_index][scale];

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
            vif_filter1d_s(filter, curr_ref_scale, mu1, tmpbuf, w, h, curr_ref_stride, buf_stride, filter_width);
            vif_filter1d_s(filter, curr_dis_scale, mu2, tmpbuf, w, h, curr_dis_stride, buf_stride, filter_width);

            mu1_adj = ADJUST(mu1);
            mu2_adj = ADJUST(mu2);

            vif_dec2_s(mu1_adj, ref_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);
            vif_dec2_s(mu2_adj, dis_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);

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

        vif_filter1d_s(filter, curr_ref_scale, mu1, tmpbuf, w, h, curr_ref_stride, buf_stride, filter_width);
        vif_filter1d_s(filter, curr_dis_scale, mu2, tmpbuf, w, h, curr_dis_stride, buf_stride, filter_width);

        // Code optimized by adding intrinsic code for the functions,
        // vif_filter1d_sq and vif_filter1d_sq
        vif_filter1d_sq_s(filter, curr_ref_scale, ref_sq_filt, tmpbuf, w, h, curr_ref_stride, buf_stride, filter_width);
        vif_filter1d_sq_s(filter, curr_dis_scale, dis_sq_filt, tmpbuf, w, h, curr_dis_stride, buf_stride, filter_width);
        vif_filter1d_xy_s(filter, curr_ref_scale, curr_dis_scale, ref_dis_filt, tmpbuf, w, h, curr_ref_stride, curr_dis_stride, buf_stride, filter_width);

        float num, den;
        vif_statistic_s(mu1, mu2, ref_sq_filt, dis_sq_filt, ref_dis_filt, &num, &den,
            w, h, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride, vif_enhn_gain_limit);
        mu1_adj = ADJUST(mu1);
        mu2_adj = ADJUST(mu2);

#ifdef VIF_OPT_DEBUG_DUMP
        ref_sq_filt_adj  = ADJUST(ref_sq_filt);
        dis_sq_filt_adj  = ADJUST(dis_sq_filt);
        ref_dis_filt_adj = ADJUST(ref_dis_filt);
#endif

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

        sprintf(pathbuf, "stage/ref_sq_filt[%d].bin", scale);
        write_image(pathbuf, ref_sq_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/dis_sq_filt[%d].bin", scale);
        write_image(pathbuf, dis_sq_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));

        sprintf(pathbuf, "stage/ref_dis_filt[%d].bin", scale);
        write_image(pathbuf, ref_dis_filt_adj, buf_valid_w, buf_valid_h, buf_stride, sizeof(float));
#endif

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

int vifdiff(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt)
{
    (void)fmt;

    double score = 0;
    double scores[4 * 2];
    double score_num = 0;
    double score_den = 0;
    float *ref_buf = 0;
    float *ref_diff_buf = 0;
    float *prev_ref_buf = 0;
    float *dis_buf = 0;
    float *dis_diff_buf = 0;
    float *prev_dis_buf = 0;
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
    if (!(ref_diff_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for ref_diff_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(prev_ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_ref_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(dis_diff_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for dis_diff_buf.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (!(prev_dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for prev_dis_buf.\n");
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
        offset_image_s(ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        offset_image_s(dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);

        if (frm_idx > 0)
        {
            apply_frame_differencing(ref_buf, prev_ref_buf, ref_diff_buf, w, h, stride / sizeof(float));
            apply_frame_differencing(dis_buf, prev_dis_buf, dis_diff_buf, w, h, stride / sizeof(float));
        }

        // copy the current frame to the previous frame buffer to have it available for next time you apply frame differencing
        memcpy(prev_ref_buf, ref_buf, data_sz);
        memcpy(prev_dis_buf, dis_buf, data_sz);

        // Pay attention to extracting T-VIF for first frame. Since we are doing subtracting the previous frame from the current frame,
        // we cannot apply T-VIF differencing for the first video frame. Therefore we initialize with a default value (e.g. 0 for num and something
        // very small for den, e.g. 1e-5). Why not difference the other way (next frame minus current frame)? Because the current choice will give us
        // unreliable scores for an earlier video frame, rather than the latest one. This might be better for video quality calculations, since recency effects
        // places more weight on later frames.
        if (frm_idx == 0)
        {
            score = 0.0;
            score_num = 0.0;
            score_den = 0.0;
            for(int scale = 0; scale < 4; scale++){
                scores[2 * scale] = 0.0;
                scores[2 * scale + 1] = 0.0 + 1e-5;
            }
        }
        else
        {
            // compute
            if ((ret = compute_vif(ref_diff_buf, dis_diff_buf, w, h, stride, stride,
                    &score, &score_num, &score_den, scores,
                    DEFAULT_VIF_ENHN_GAIN_LIMIT,
                    DEFAULT_VIF_KERNELSCALE)))
            {
                printf("error: compute_vifdiff failed.\n");
                fflush(stdout);
                goto fail_or_end;
            }
        }

        // print
        printf("vifdiff: %d %f\n", frm_idx, score);
        fflush(stdout);
        printf("vifdiff_num: %d %f\n", frm_idx, score_num);
        fflush(stdout);
        printf("vifdiff_den: %d %f\n", frm_idx, score_den);
        fflush(stdout);
        for(int scale=0;scale<4;scale++){
            printf("vifdiff_num_scale%d: %d %f\n", scale, frm_idx, scores[2*scale]);
            printf("vifdiff_den_scale%d: %d %f\n", scale, frm_idx, scores[2*scale+1]);
        }

        frm_idx++;
    }

    ret = 0;

fail_or_end:

    aligned_free(ref_buf);
    aligned_free(ref_diff_buf);
    aligned_free(prev_ref_buf);
    aligned_free(dis_buf);
    aligned_free(dis_diff_buf);
    aligned_free(prev_dis_buf);
    aligned_free(temp_buf);

    return ret;
}
