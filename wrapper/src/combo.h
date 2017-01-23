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

#pragma once

#ifndef COMBO_H_
#define COMBO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "darray.h"

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
);

#ifdef __cplusplus
}
#endif

#endif /* COMBO_H_ */
