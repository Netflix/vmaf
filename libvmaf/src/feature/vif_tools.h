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

#include <stdbool.h>

enum vif_scaling_method {
    vif_scale_nearest = 0,
    vif_scale_bicubic = 1,
    vif_scale_lanczos4 = 2,
    vif_scale_bilinear = 3,
};

# define NUM_KERNELSCALES 21

static const float valid_kernelscales[NUM_KERNELSCALES] = {
    1.0,
    1.0f / 2.0f,
    3.0f / 2.0f,
    2.0f,
    2.0f / 3.0f,
    24.0f / 10.0f,
    360.0f / 97.0f,
    4.0f / 3.0f,
    3.5f / 3.0f,
    3.75f / 3.0f,
    4.25f / 3.0f,
    5.0f / 3.0f,
    3.0f,
    1.0f / 2.25f,
    1.4746f,
    1.54f,
    1.6f,
    1.06667f,
    0.711111f,
    0.740740f,
    1.111111f,
};

/* s single precision, d double precision */

void vif_dec2_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride); // stride >= width, multiple of 16 or 32 typically

void vif_dec16_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride); // stride >= width, multiple of 16 or 32 typically

float vif_sum_s(const float *x, int w, int h, int stride);

void vif_statistic_s(const float *mu1_sq, const float *mu2_sq, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
                     int w, int h, int mu1_sq_stride, int mu2_sq_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, double vif_enhn_gain_limit, double vif_sigma_nsq);

void vif_filter1d_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_sq_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_xy_s(const float *f, const float *src1, const float *src2, float *dst, float *tmpbuf, int w, int h, int src1_stride, int src2_stride, int dst_stride, int fwidth);

int vif_get_scaling_method(char *scaling_method_str, enum vif_scaling_method *scale_method);

void vif_scale_frame_s(enum vif_scaling_method scale_method, const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_w, int dst_h, int dst_stride);

int vif_get_filter_size(int scale, float kernelscale);

void vif_get_filter(float *out, int scale, float kernelscale);

void speed_get_antialias_filter(float *out, int scale, float kernelscale);

bool vif_validate_kernelscale(float kernelscale);

#endif /* VIF_TOOLS_H_ */
