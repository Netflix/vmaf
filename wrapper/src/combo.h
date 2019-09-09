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

#pragma once

#ifndef COMBO_H_
#define COMBO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "darray.h"
#include "common/blur_array.h"
#include "libvmaf.h"

#ifdef MULTI_THREADING
#include "pthread.h"
#endif

typedef struct
{
    int (*read_vmaf_picture)(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *user_data);
    void *user_data;
    int w;
    int h;
    enum VmafPixelFormat fmt;
    DArray *adm_num_array;
    DArray *adm_den_array;
    DArray *adm_num_scale0_array;
    DArray *adm_den_scale0_array;
    DArray *adm_num_scale1_array;
    DArray *adm_den_scale1_array;
    DArray *adm_num_scale2_array;
    DArray *adm_den_scale2_array;
    DArray *adm_num_scale3_array;
    DArray *adm_den_scale3_array;
    DArray *motion_array;
    DArray *motion2_array;
    DArray *vif_num_scale0_array;
    DArray *vif_den_scale0_array;
    DArray *vif_num_scale1_array;
    DArray *vif_den_scale1_array;
    DArray *vif_num_scale2_array;
    DArray *vif_den_scale2_array;
    DArray *vif_num_scale3_array;
    DArray *vif_den_scale3_array;
    DArray *vif_array;
    DArray *psnr_array;
    DArray *psnr_u_array;
    DArray *psnr_v_array;
    DArray *ssim_array;
    DArray *ms_ssim_array;
    char *errmsg;
    VmafFeatureCalculationSetting vmaf_feature_calculation_setting;
    int vmaf_feature_mode_setting;

    int frm_idx;
    int stride, stride_u, stride_v;
    double peak;
    double psnr_max;
    size_t data_sz, data_sz_u, data_sz_v;
#ifdef MULTI_THREADING
    int thread_count;
    int stop_threads;
    pthread_mutex_t mutex_readframe;
#endif
    BLUR_BUF_ARRAY blur_buf_array;
    BLUR_BUF_ARRAY ref_buf_array;
    BLUR_BUF_ARRAY dis_buf_array;
    BLUR_BUF_ARRAY ref_buf_u_array;
    BLUR_BUF_ARRAY dis_buf_u_array;
    BLUR_BUF_ARRAY ref_buf_v_array;
    BLUR_BUF_ARRAY dis_buf_v_array;
    DArray *motion_score_compute_flag_array;
    int ret;

} VMAF_THREAD_STRUCT;
void* combo_threadfunc(void* vmaf_thread_data);

int combo(int (*read_vmaf_picture)(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *user_data),
        void *user_data, int w, int h, enum VmafPixelFormat fmt,
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
        DArray *motion2_array,
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
        DArray *psnr_u_array,
        DArray *psnr_v_array,
        DArray *ssim_array,
        DArray *ms_ssim_array,
        char *errmsg,
        VmafFeatureCalculationSetting vmaf_feature_calculation_setting,
        int vmaf_feature_mode_setting
);

#ifdef __cplusplus
}
#endif

#endif /* COMBO_H_ */
