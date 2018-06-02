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
#include "combo.h"

#ifdef MULTI_THREADING
#include "common/blur_array.h"
#include "cpu_info.h"
#endif

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
int compute_ssim(const float *ref, const float *cmp, int w, int h, int ref_stride, int cmp_stride, double *score, double *l_score, double *c_score, double *s_score);
int compute_ms_ssim(const float *ref, const float *cmp, int w, int h, int ref_stride, int cmp_stride, double *score, double* l_scores, double* c_scores, double* s_scores);

void* combo_threadfunc(void* vmaf_thread_data)
{
    // this is our shared thread data
    VMAF_THREAD_STRUCT* thread_data = (VMAF_THREAD_STRUCT*)vmaf_thread_data;

    // set the local variables from the thread shared data
    size_t data_sz = thread_data->data_sz;
    int stride = thread_data->stride;
    double peak = thread_data->peak;
    double psnr_max = thread_data->psnr_max;
    int w = thread_data->w;
    int h = thread_data->h;
    char* errmsg = thread_data->errmsg;
    void* user_data = thread_data->user_data;
    const char* fmt = thread_data->fmt;

    double score = 0;
    double scores[4*2];
    double score_num = 0;
    double score_den = 0;
    double l_score = 0, c_score = 0, s_score = 0;
    double l_scores[SCALES], c_scores[SCALES], s_scores[SCALES];

#ifdef COMPUTE_ANSNR
    double score_psnr = 0;
#endif

    float *ref_buf = 0;
    float *dis_buf = 0;
    float *prev_blur_buf = 0;
    float *blur_buf = 0;
    float *temp_buf = 0;

    int ret = 1;

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

#ifndef MULTI_THREADING
    // prev_blur_buf, blur_buf for motion only
    if (!(prev_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
    {
        sprintf(errmsg, "aligned_malloc failed for prev_blur_buf.\n");
        goto fail_or_end;
    }
#endif

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

    int frm_idx = -1;
    while (1)
    {

#ifdef MULTI_THREADING
        pthread_mutex_lock(&thread_data->mutex_readframe);

        if (thread_data->stop_threads)
        {
            // this is the signal that another thread has reached the end of the input file, so we all quit
            pthread_mutex_unlock(&thread_data->mutex_readframe);
            break;
        }
#endif

        ret = read_frame(ref_buf, dis_buf, temp_buf, stride, user_data);

        if (ret == 1)
        {
#ifdef MULTI_THREADING
            thread_data->stop_threads = 1;
            pthread_mutex_unlock(&thread_data->mutex_readframe);
#endif
            goto fail_or_end;
        }
        if (ret == 2)
        {
#ifdef MULTI_THREADING
            thread_data->stop_threads = 1;
            pthread_mutex_unlock(&thread_data->mutex_readframe);
#endif
            break;
        }

#ifdef MULTI_THREADING
        // the next frame
        frm_idx = thread_data->frm_idx;
        thread_data->frm_idx++;

        pthread_mutex_unlock(&thread_data->mutex_readframe);
#else
        frm_idx++;
#endif

#ifdef PRINT_PROGRESS
        printf("frame: %d, ", frm_idx);
#endif

        // ===============================================================
        // for the PSNR, SSIM and MS-SSIM, offset are 0 - do them first
        // ===============================================================

        if (thread_data->psnr_array != NULL)
        {
            /* =========== psnr ============== */
            ret = compute_psnr(ref_buf, dis_buf, w, h, stride, stride, &score, peak, psnr_max);

            if (ret)
            {
                sprintf(errmsg, "compute_psnr failed.\n");
                goto fail_or_end;
            }

#ifdef PRINT_PROGRESS
            printf("psnr: %.3f, ", score);
#endif
            insert_array_at(thread_data->psnr_array, score, frm_idx);
        }

        if (thread_data->ssim_array != NULL)
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

            insert_array_at(thread_data->ssim_array, score, frm_idx);
        }

        if (thread_data->ms_ssim_array != NULL)
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

            insert_array_at(thread_data->ms_ssim_array, score, frm_idx);
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

        insert_array_at(thread_data->adm_num_array, score_num, frm_idx);
        insert_array_at(thread_data->adm_den_array, score_den, frm_idx);
        insert_array_at(thread_data->adm_num_scale0_array, scores[0], frm_idx);
        insert_array_at(thread_data->adm_den_scale0_array, scores[1], frm_idx);
        insert_array_at(thread_data->adm_num_scale1_array, scores[2], frm_idx);
        insert_array_at(thread_data->adm_den_scale1_array, scores[3], frm_idx);
        insert_array_at(thread_data->adm_num_scale2_array, scores[4], frm_idx);
        insert_array_at(thread_data->adm_den_scale2_array, scores[5], frm_idx);
        insert_array_at(thread_data->adm_num_scale3_array, scores[6], frm_idx);
        insert_array_at(thread_data->adm_den_scale3_array, scores[7], frm_idx);

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
        // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
        convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));

        // compute
        if (frm_idx == 0)
        {
            score = 0.0;
        }
        else
        {
#ifdef MULTI_THREADING
            prev_blur_buf = get_blur_buf(&thread_data->blur_array, frm_idx-1);
#endif
            if ((ret = compute_motion(prev_blur_buf, blur_buf, w, h, stride, stride, &score)))
            {
                sprintf(errmsg, "compute_motion failed.\n");
                goto fail_or_end;
            }
#ifdef MULTI_THREADING
            release_blur_buf(&thread_data->blur_array, frm_idx-1);
#endif
        }

#ifdef MULTI_THREADING
        put_blur_buf(&thread_data->blur_array, frm_idx, blur_buf);
#else
        // copy to prev_buf
        memcpy(prev_blur_buf, blur_buf, data_sz);
#endif

#ifdef PRINT_PROGRESS
        printf("motion: %.3f, ", score);
#endif

        insert_array_at(thread_data->motion_array, score, frm_idx);

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

        insert_array_at(thread_data->vif_num_scale0_array, scores[0], frm_idx);
        insert_array_at(thread_data->vif_den_scale0_array, scores[1], frm_idx);
        insert_array_at(thread_data->vif_num_scale1_array, scores[2], frm_idx);
        insert_array_at(thread_data->vif_den_scale1_array, scores[3], frm_idx);
        insert_array_at(thread_data->vif_num_scale2_array, scores[4], frm_idx);
        insert_array_at(thread_data->vif_den_scale2_array, scores[5], frm_idx);
        insert_array_at(thread_data->vif_num_scale3_array, scores[6], frm_idx);
        insert_array_at(thread_data->vif_den_scale3_array, scores[7], frm_idx);
        insert_array_at(thread_data->vif_array, score, frm_idx);

#ifdef PRINT_PROGRESS
        printf("\n");
#endif

    }

    ret = 0;

fail_or_end:
    aligned_free(ref_buf);
    aligned_free(dis_buf);

#ifndef MULTI_THREADING
    aligned_free(prev_blur_buf);
#endif
    aligned_free(blur_buf);
    aligned_free(temp_buf);

#ifdef MULTI_THREADING
    // when one thread ends we signal all other threads to also stop
    thread_data->stop_threads = 1;
    thread_data->ret = ret;
    pthread_exit(&ret);

#else
    thread_data->ret = ret;
#endif

}

#ifdef MULTI_THREADING

int combo(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt,
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
    // init shared thread data
    VMAF_THREAD_STRUCT combo_thread_data;
    combo_thread_data.read_frame = read_frame;
    combo_thread_data.user_data = user_data;
    combo_thread_data.w = w;
    combo_thread_data.h = h;
    combo_thread_data.fmt = fmt;
    combo_thread_data.adm_num_array = adm_num_array;
    combo_thread_data.adm_den_array = adm_den_array;
    combo_thread_data.adm_num_scale0_array = adm_num_scale0_array;
    combo_thread_data.adm_den_scale0_array = adm_den_scale0_array;
    combo_thread_data.adm_num_scale1_array = adm_num_scale1_array;
    combo_thread_data.adm_den_scale1_array = adm_den_scale1_array;
    combo_thread_data.adm_num_scale2_array = adm_num_scale2_array;
    combo_thread_data.adm_den_scale2_array = adm_den_scale2_array;
    combo_thread_data.adm_num_scale3_array = adm_num_scale3_array;
    combo_thread_data.adm_den_scale3_array = adm_den_scale3_array;
    combo_thread_data.motion_array = motion_array;
    combo_thread_data.vif_num_scale0_array = vif_num_scale0_array;
    combo_thread_data.vif_den_scale0_array = vif_den_scale0_array;
    combo_thread_data.vif_num_scale1_array = vif_num_scale1_array;
    combo_thread_data.vif_den_scale1_array = vif_den_scale1_array;
    combo_thread_data.vif_num_scale2_array = vif_num_scale2_array;
    combo_thread_data.vif_den_scale2_array = vif_den_scale2_array;
    combo_thread_data.vif_num_scale3_array = vif_num_scale3_array;
    combo_thread_data.vif_den_scale3_array = vif_den_scale3_array;
    combo_thread_data.vif_array = vif_array;
    combo_thread_data.psnr_array = psnr_array;
    combo_thread_data.ssim_array = ssim_array;
    combo_thread_data.ms_ssim_array = ms_ssim_array;
    combo_thread_data.errmsg = errmsg;
    combo_thread_data.frm_idx = 0;
    combo_thread_data.stop_threads = 0;

    // sanity check for width/height
    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        sprintf(errmsg, "wrong width %d or height %d.\n", w, h);
        return -1;
    }

    // calculate stride and data size
    combo_thread_data.stride = ALIGN_CEIL(w * sizeof(float));
    if ((size_t)h > SIZE_MAX / combo_thread_data.stride)
    {
        sprintf(errmsg, "height %d too large.\n", h);
        return -1;
    }

    if (psnr_constants(fmt, &combo_thread_data.peak, &combo_thread_data.psnr_max))
    {
        sprintf(errmsg, "unknown format %s.\n", fmt);
        return -1;
    }

    combo_thread_data.data_sz = (size_t)combo_thread_data.stride * h;

    combo_thread_data.thread_count = getNumCores();

    // for motion analysis we compare to previous buffer
    init_blur_array(&combo_thread_data.blur_array, combo_thread_data.thread_count + 1, combo_thread_data.data_sz, MAX_ALIGN);

    // initialize the mutex that protects the read_frame function
    pthread_mutex_init(&combo_thread_data.mutex_readframe, NULL);

    // create a joinable thread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // start threads
    int t;
    pthread_t thread[combo_thread_data.thread_count];
    for (t=0; t<combo_thread_data.thread_count; t++)
    {
        pthread_create(&thread[t], &attr, combo_threadfunc, &combo_thread_data);
    }

    pthread_attr_destroy(&attr);

    // wait for all threads to finish
    for (t=0; t<combo_thread_data.thread_count; t++)
    {
        void* thread_ret;
        int rc = pthread_join(thread[t], &thread_ret);

        if (rc)
        {
            printf("ERROR; return code from pthread_join() for thread[%d] is %d\n", t, (long)thread_ret);
            return -1;
        }
    }

    free_blur_buf(&combo_thread_data.blur_array);
    return 0;
}

#else // #ifdef MULTI_THREADING

int combo(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt,
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
    VMAF_THREAD_STRUCT combo_thread_data;
    combo_thread_data.read_frame = read_frame;
    combo_thread_data.user_data = user_data;
    combo_thread_data.w = w;
    combo_thread_data.h = h;
    combo_thread_data.fmt = fmt;
    combo_thread_data.adm_num_array = adm_num_array;
    combo_thread_data.adm_den_array = adm_den_array;
    combo_thread_data.adm_num_scale0_array = adm_num_scale0_array;
    combo_thread_data.adm_den_scale0_array = adm_den_scale0_array;
    combo_thread_data.adm_num_scale1_array = adm_num_scale1_array;
    combo_thread_data.adm_den_scale1_array = adm_den_scale1_array;
    combo_thread_data.adm_num_scale2_array = adm_num_scale2_array;
    combo_thread_data.adm_den_scale2_array = adm_den_scale2_array;
    combo_thread_data.adm_num_scale3_array = adm_num_scale3_array;
    combo_thread_data.adm_den_scale3_array = adm_den_scale3_array;
    combo_thread_data.motion_array = motion_array;
    combo_thread_data.vif_num_scale0_array = vif_num_scale0_array;
    combo_thread_data.vif_den_scale0_array = vif_den_scale0_array;
    combo_thread_data.vif_num_scale1_array = vif_num_scale1_array;
    combo_thread_data.vif_den_scale1_array = vif_den_scale1_array;
    combo_thread_data.vif_num_scale2_array = vif_num_scale2_array;
    combo_thread_data.vif_den_scale2_array = vif_den_scale2_array;
    combo_thread_data.vif_num_scale3_array = vif_num_scale3_array;
    combo_thread_data.vif_den_scale3_array = vif_den_scale3_array;
    combo_thread_data.vif_array = vif_array;
    combo_thread_data.psnr_array = psnr_array;
    combo_thread_data.ssim_array = ssim_array;
    combo_thread_data.ms_ssim_array = ms_ssim_array;
    combo_thread_data.errmsg = errmsg;
    combo_thread_data.frm_idx = 0;
    // combo_thread_data.stop_threads = 0;

    // sanity check for width/height
    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        sprintf(errmsg, "wrong width %d or height %d.\n", w, h);
        return -1;
    }

    // calculate stride and data size
    combo_thread_data.stride = ALIGN_CEIL(w * sizeof(float));
    if ((size_t)h > SIZE_MAX / combo_thread_data.stride)
    {
        sprintf(errmsg, "height %d too large.\n", h);
        return -1;
    }

    if (psnr_constants(fmt, &combo_thread_data.peak, &combo_thread_data.psnr_max))
    {
        sprintf(errmsg, "unknown format %s.\n", fmt);
        return -1;
    }

    combo_thread_data.data_sz = (size_t)combo_thread_data.stride * h;

    combo_threadfunc(&combo_thread_data);

    return combo_thread_data.ret;

}

#endif // #ifdef MULTI_THREADING
