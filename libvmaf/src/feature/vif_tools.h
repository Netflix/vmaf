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

#pragma once

#ifndef VIF_TOOLS_H_
#define VIF_TOOLS_H_

enum vif_kernelscale_enum {
    vif_kernelscale_1 = 0,
    vif_kernelscale_1o2 = 1,
    vif_kernelscale_3o2 = 2,
    vif_kernelscale_2 = 3,
    vif_kernelscale_2o3 = 4,
    vif_kernelscale_24o10 = 5,
    vif_kernelscale_360o97 = 6,
    vif_kernelscale_4o3 = 7,
    vif_kernelscale_3d5o3 = 8,
    vif_kernelscale_3d75o3 = 9,
    vif_kernelscale_4d25o3 = 10,
};
extern const float vif_filter1d_table_s[11][4][65]; // 4 is scale. since this is separable filter, filtering is 1d repeat horizontally and vertically
extern const int vif_filter1d_width[11][4];

/* s single precision, d double precision */

void vif_dec2_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride); // stride >= width, multiple of 16 or 32 typically

float vif_sum_s(const float *x, int w, int h, int stride);

void vif_statistic_s(const float *mu1_sq, const float *mu2_sq, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
                     int w, int h, int mu1_sq_stride, int mu2_sq_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, double vif_enhn_gain_limit);

void vif_filter1d_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_sq_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_xy_s(const float *f, const float *src1, const float *src2, float *dst, float *tmpbuf, int w, int h, int src1_stride, int src2_stride, int dst_stride, int fwidth);

#endif /* VIF_TOOLS_H_ */
