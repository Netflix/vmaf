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
#include <math.h>

#include "common/alloc.h"
#include "common/file_io.h"
#include "motion_tools.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "iqa/ssim_tools.h"
#include "darray.h"
#include "adm_options.h"

typedef float number_t;

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s
#define convolution_f32_c  convolution_f32_c_s
#define offset_image       offset_image_s
#define FILTER_5           FILTER_5_s
int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor);
#ifdef COMPUTE_ANSNR
int compute_ansnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_psnr, double peak, double psnr_max);
#endif
int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores);
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);
int compute_psnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double peak, double psnr_max);
int compute_ssim(const number_t *ref, const number_t *cmp, int w, int h, int ref_stride, int cmp_stride, double *score, double *l_score, double *c_score, double *s_score);
int compute_ms_ssim(const number_t *ref, const number_t *cmp, int w, int h, int ref_stride, int cmp_stride, double *score, double* l_scores, double* c_scores, double* s_scores);

int combo(const char *ref_path, const char *dis_path, int w, int h, const char *fmt,
        DArray *adm_num_array,
        DArray *adm_den_array,
        DArray *adm_num_scale0_array,
        DArray *adm_den_scale0_array,
        DArray *adm_num_scale1_array,
        DArray *adm_den_scale1_array,
        DArray *adm_num_scale2_array,
        DArray *adm_den_scale2_array,
        DArray *adm_num_scale3_array,
        DArray *adm_den_scale3_array,
        DArray *motion_array,
        DArray *vif_num_scale0_array,
        DArray *vif_den_scale0_array,
        DArray *vif_num_scale1_array,
        DArray *vif_den_scale1_array,
        DArray *vif_num_scale2_array,
        DArray *vif_den_scale2_array,
        DArray *vif_num_scale3_array,
        DArray *vif_den_scale3_array,
        DArray *vif_array,
        DArray *psnr_array,
        DArray *ssim_array,
        DArray *ms_ssim_array,
        char *errmsg
        )
{
    double score = 0;
    double scores[4*2];
    double score_num = 0;
    double score_den = 0;
    double l_score = 0, c_score = 0, s_score = 0;
    double l_scores[SCALES], c_scores[SCALES], s_scores[SCALES];

#ifdef COMPUTE_ANSNR
    double score_psnr = 0;
#endif

    number_t *ref_buf = 0;
    number_t *dis_buf = 0;
    number_t *prev_blur_buf = 0;
    number_t *blur_buf = 0;
    number_t *temp_buf = 0;

    FILE *ref_rfile = 0;
    FILE *dis_rfile = 0;
    size_t data_sz;
    int stride;
    int ret = 1;

    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(number_t))
    {
        sprintf(errmsg, "wrong width %d or height %d.\n", w, h);
        goto fail_or_end;
    }

    stride = ALIGN_CEIL(w * sizeof(number_t));

    if ((size_t)h > SIZE_MAX / stride)
    {
        sprintf(errmsg, "height %d too large.\n", h);
        goto fail_or_end;
    }

    data_sz = (size_t)stride * h;

    if (!(ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for ref_buf.\n");
        goto fail_or_end;
    }
    if (!(dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for dis_buf.\n");
        goto fail_or_end;
    }

    // prev_blur_buf, blur_buf for motion only
    if (!(prev_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for prev_blur_buf.\n");
        goto fail_or_end;
    }
    if (!(blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for blur_buf.\n");
        goto fail_or_end;
    }

    // use temp_buf for convolution_f32_c, and fread u and v
    if (!(temp_buf = aligned_malloc(data_sz * 2, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for temp_buf.\n");
        goto fail_or_end;
    }

    if (!(ref_rfile = fopen(ref_path, "rb")))
    {
        sprintf(errmsg, "fopen ref_path %s failed.\n", ref_path);
        goto fail_or_end;
    }
    if (!(dis_rfile = fopen(dis_path, "rb")))
    {
        sprintf(errmsg, "fopen dis_path %s failed.\n", dis_path);
        goto fail_or_end;
    }

    size_t offset;
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
    {
        if ((w * h) % 2 != 0)
        {
            sprintf(errmsg, "(w * h) %% 2 != 0, w = %d, h = %d.\n", w, h);
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
        sprintf(errmsg, "unknown format %s.\n", fmt);
        goto fail_or_end;
    }

    int frm_idx = 0;
    while (1)
    {

        // read ref y
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            ret = read_image_b(ref_rfile, ref_buf, 0, w, h, stride);
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            ret = read_image_w(ref_rfile, ref_buf, 0, w, h, stride);
        }
        else
        {
            sprintf(errmsg, "unknown format %s.\n", fmt);
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
            ret = read_image_b(dis_rfile, dis_buf, 0, w, h, stride);
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            ret = read_image_w(dis_rfile, dis_buf, 0, w, h, stride);
        }
        else
        {
            sprintf(errmsg, "unknown format %s.\n", fmt);
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

#ifdef PRINT_PROGRESS
        printf("frame: %d, ", frm_idx);
#endif

        // ===============================================================
        // for the PSNR, SSIM and MS-SSIM, offset are 0 - do them first
        // ===============================================================

        if (psnr_array != NULL)
        {
            /* =========== psnr ============== */
            if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
            {
                // max psnr 60.0 for 8-bit per Ioannis
                ret = compute_psnr(ref_buf, dis_buf, w, h, stride, stride, &score, 255.0, 60.0);
            }
            else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
            {
                // 10 bit gets normalized to 8 bit, peak is 1023 / 4.0 = 255.75
                // max psnr 72.0 for 10-bit per Ioannis
                ret = compute_psnr(ref_buf, dis_buf, w, h, stride, stride, &score, 255.75, 72.0);
            }
            else
            {
                sprintf(errmsg, "unknown format %s.\n", fmt);
                goto fail_or_end;
            }
            if (ret)
            {
                sprintf(errmsg, "compute_psnr failed.\n");
                goto fail_or_end;
            }

#ifdef PRINT_PROGRESS
            printf("psnr: %.3f, ", score);
#endif
            insert_array(psnr_array, score);
        }

        if (ssim_array != NULL)
        {

            /* =========== ssim ============== */
            if ((ret = compute_ssim(ref_buf, dis_buf, w, h, stride, stride, &score, &l_score, &c_score, &s_score)))
            {
                sprintf(errmsg, "compute_ssim failed.\n");
                goto fail_or_end;
            }

#ifdef PRINT_PROGRESS
            printf("ssim: %.3f, ", score);
#endif

            insert_array(ssim_array, score);
        }

        if (ms_ssim_array != NULL)
        {
            /* =========== ms-ssim ============== */
            if ((ret = compute_ms_ssim(ref_buf, dis_buf, w, h, stride, stride, &score, l_scores, c_scores, s_scores)))
            {
                sprintf(errmsg, "compute_ms_ssim failed.\n");
                goto fail_or_end;
            }

#ifdef PRINT_PROGRESS
            printf("ms_ssim: %.3f, ", score);
#endif

            insert_array(ms_ssim_array, score);
        }

        // ===============================================================
        // for the rest, offset pixel by OPT_RANGE_PIXEL_OFFSET
        // ===============================================================

        offset_image(ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        offset_image(dis_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);

        /* =========== adm ============== */
        if ((ret = compute_adm(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores, ADM_BORDER_FACTOR)))
        {
            sprintf(errmsg, "compute_adm failed.\n");
            goto fail_or_end;
        }

#ifdef PRINT_PROGRESS
        printf("adm: %.3f, ", score);
        printf("adm_num: %.3f, ", score_num);
        printf("adm_den: %.3f, ", score_den);
        printf("adm_num_scale0: %.3f, ", scores[0]);
        printf("adm_den_scale0: %.3f, ", scores[1]);
        printf("adm_num_scale1: %.3f, ", scores[2]);
        printf("adm_den_scale1: %.3f, ", scores[3]);
        printf("adm_num_scale2: %.3f, ", scores[4]);
        printf("adm_den_scale2: %.3f, ", scores[5]);
        printf("adm_num_scale3: %.3f, ", scores[6]);
        printf("adm_den_scale3: %.3f, ", scores[7]);
#endif

        insert_array(adm_num_array, score_num);
        insert_array(adm_den_array, score_den);
        insert_array(adm_num_scale0_array, scores[0]);
        insert_array(adm_den_scale0_array, scores[1]);
        insert_array(adm_num_scale1_array, scores[2]);
        insert_array(adm_den_scale1_array, scores[3]);
        insert_array(adm_num_scale2_array, scores[4]);
        insert_array(adm_den_scale2_array, scores[5]);
        insert_array(adm_num_scale3_array, scores[6]);
        insert_array(adm_den_scale3_array, scores[7]);

#ifdef COMPUTE_ANSNR

        /* =========== ansnr ============== */
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
            sprintf(errmsg, "unknown format %s.\n", fmt);
            goto fail_or_end;
        }
        if (ret)
        {
            sprintf(errmsg, "compute_ansnr failed.\n");
            goto fail_or_end;
        }

#ifdef PRINT_PROGRESS
        printf("ansnr: %.3f, ", score);
        printf("anpsnr: %.3f, ", score_psnr);
#endif

#endif

        /* =========== motion ============== */

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
                sprintf(errmsg, "compute_motion failed.\n");
                goto fail_or_end;
            }
        }

        // copy to prev_buf
        memcpy(prev_blur_buf, blur_buf, data_sz);

#ifdef PRINT_PROGRESS
        printf("motion: %.3f, ", score);
#endif

        insert_array(motion_array, score);

        /* =========== vif ============== */

        if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores)))
        {
            sprintf(errmsg, "compute_vif failed.\n");
            goto fail_or_end;
        }

#ifdef PRINT_PROGRESS
        // printf("vif_num: %.3f, ", score_num);
        // printf("vif_den: %.3f, ", score_den);
        printf("vif_num_scale0: %.3f, ", scores[0]);
        printf("vif_den_scale0: %.3f, ", scores[1]);
        printf("vif_num_scale1: %.3f, ", scores[2]);
        printf("vif_den_scale1: %.3f, ", scores[3]);
        printf("vif_num_scale2: %.3f, ", scores[4]);
        printf("vif_den_scale2: %.3f, ", scores[5]);
        printf("vif_num_scale3: %.3f, ", scores[6]);
        printf("vif_den_scale3: %.3f, ", scores[7]);
        printf("vif: %.3f, ", score);
#endif

        insert_array(vif_num_scale0_array, scores[0]);
        insert_array(vif_den_scale0_array, scores[1]);
        insert_array(vif_num_scale1_array, scores[2]);
        insert_array(vif_den_scale1_array, scores[3]);
        insert_array(vif_num_scale2_array, scores[4]);
        insert_array(vif_den_scale2_array, scores[5]);
        insert_array(vif_num_scale3_array, scores[6]);
        insert_array(vif_den_scale3_array, scores[7]);
        insert_array(vif_array, score);

        // ref skip u and v
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            if (fread(temp_buf, 1, offset, ref_rfile) != (size_t)offset)
            {
                sprintf(errmsg, "ref fread u and v failed.\n");
                goto fail_or_end;
            }
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            if (fread(temp_buf, 2, offset, ref_rfile) != (size_t)offset)
            {
                sprintf(errmsg, "ref fread u and v failed.\n");
                goto fail_or_end;
            }
        }
        else
        {
            sprintf(errmsg, "unknown format %s.\n", fmt);
            goto fail_or_end;
        }

        // dis skip u and v
        if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
        {
            if (fread(temp_buf, 1, offset, dis_rfile) != (size_t)offset)
            {
                sprintf(errmsg, "dis fread u and v failed.\n");
                goto fail_or_end;
            }
        }
        else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
        {
            if (fread(temp_buf, 2, offset, dis_rfile) != (size_t)offset)
            {
                sprintf(errmsg, "dis fread u and v failed.\n");
                goto fail_or_end;
            }
        }
        else
        {
            sprintf(errmsg, "unknown format %s.\n", fmt);
            goto fail_or_end;
        }

#ifdef PRINT_PROGRESS
        printf("\n");
#endif

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

    aligned_free(prev_blur_buf);
    aligned_free(blur_buf);
    aligned_free(temp_buf);

    return ret;
}
