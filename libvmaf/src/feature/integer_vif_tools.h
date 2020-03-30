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

#ifndef INTEGER_VIF_TOOLS_H_
#define INTEGER_VIF_TOOLS_H_

extern const float vif_filter1d_table_s[4][17]; // 4 is scale. since this is separable filter, filtering is 1d repeat horizontally and vertically

extern const uint16_t integer_vif_filter1d_table_s[4][17];

extern const int vif_filter1d_width[4];

extern const float vif_filter2d_table_s[4][17 * 17];

extern const int vif_filter2d_width[4];

extern uint16_t log_values[65537];
/* s single precision, d double precision */

void integer_vif_dec2_s(const int16_t *src, int16_t *dst, int src_w, int src_h, int src_stride, int dst_stride);

// calculate x**2, x*y, y**2, in one reading into memory

void integer_vif_statistic_s(const int32_t *mu1, const int32_t *mu2, const int32_t *xx_filt, const int32_t *yy_filt, const int32_t *xy_filt, float *num, float *den, int w, int h, int mu1_stride, int scale);

void integer_vif_xx_yy_xy_s(const int16_t *x, const int16_t *y, int32_t *xx, int32_t *yy, int32_t *xy, int w, int h, int xstride, int ystride, int xxstride, int yystride, int xystride, int squared, int scale);

void integer_vif_filter1d_combined_s(const uint16_t *integer_filter, const int16_t *integer_curr_ref_scale, const int16_t *integer_curr_dis_scale, int32_t *integer_ref_sq_filt, int32_t *integer_dis_sq_filt, int32_t *integer_ref_dis_filt, int32_t *integer_mu1, int32_t *integer_mu2, int w, int h, int src_stride, int dst_stride, int fwidth, int scale,int32_t *tmp_mu1, int32_t *tmp_mu2, int32_t *tmp_ref, int32_t *tmp_dis, int32_t *tmp_ref_dis, int inp_size_bits);

void integer_vif_filter1d_rdCombine_s(const uint16_t *integer_filter, const int16_t *integer_curr_ref_scale, const int16_t *integer_curr_dis_scale, int16_t *integer_mu1, int16_t *integer_mu2,  int w, int h, int curr_ref_stride, int curr_dis_stride, int dst_stride, int fwidth, int scale, int32_t *tmp_ref_convol, int32_t *tmp_dis_convol, int inp_size_bits);

#endif /* INTEGER_VIF_TOOLS_H_ */
