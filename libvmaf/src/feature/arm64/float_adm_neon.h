/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#ifndef ARM64_NEON_FLOAT_ADM_H_
#define ARM64_NEON_FLOAT_ADM_H_

#include "feature/adm_tools.h"

void float_adm_dwt2_neon(const float *src, const adm_dwt_band_t_s *dst,
                          int **ind_y, int **ind_x,
                          int w, int h, int src_stride, int dst_stride);

void float_adm_csf_neon(const float *src, float *dst, float *flt,
                         int w, int h, int src_stride, int dst_stride,
                         float factor, float one_by_30);

float float_adm_csf_den_scale_neon(const float *src, int w, int h,
                                    int src_stride, int left, int top,
                                    int right, int bottom, float factor);

float float_adm_sum_cube_neon(const float *x, int w, int h, int stride,
                               int left, int top, int right, int bottom);

#endif /* ARM64_NEON_FLOAT_ADM_H_ */
