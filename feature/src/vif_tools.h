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

#ifndef VIF_TOOLS_H_
#define VIF_TOOLS_H_

extern const float vif_filter1d_table_s[4][17]; // 4 is scale. since this is separable filter, filtering is 1d repeat horizontally and vertically

extern const int vif_filter1d_width[4];

extern const float vif_filter2d_table_s[4][17*17];

extern const int vif_filter2d_width[4];

/* s single precision, d double precision */

void vif_dec2_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride); // stride >= width, multiple of 16 or 32 typically

float vif_sum_s(const float *x, int w, int h, int stride);

// calculate x**2, x*y, y**2, in one reading into memory
void vif_xx_yy_xy_s(const float *x, const float *y, float *xx, float *yy, float *xy, int w, int h, int xstride, int ystride, int xxstride, int yystride, int xystride);

void vif_statistic_s(const float *mu1_sq, const float *mu2_sq, const float *mu1_mu2, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
                     int w, int h, int mu1_sq_stride, int mu2_sq_stride, int mu1_mu2_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, int num_stride, int den_stride);

void vif_filter1d_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_sq_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth);

void vif_filter1d_xy_s(const float *f, const float *src1, const float *src2, float *dst, float *tmpbuf, int w, int h, int src1_stride, int src2_stride, int dst_stride, int fwidth);

void vif_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth);

#endif /* VIF_TOOLS_H_ */
