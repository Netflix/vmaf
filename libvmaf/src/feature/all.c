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
#include "common/convolution_internal.h"
#include "psnr_tools.h"
#include "motion_tools.h"
#include "offset.h"
#include "vif_options.h"
#include "adm_options.h"

#define convolution_f32_c  convolution_f32_c_s
#define offset_image       offset_image_s
#define FILTER_5           FILTER_5_s
int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor, double adm_enhn_gain_limit, double adm_norm_view_dist, int adm_ref_display_height, int adm_csf_mode);
int compute_ansnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_psnr, double peak, double psnr_max);
int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double vif_enhn_gain_limit, double vif_kernelscale);
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
