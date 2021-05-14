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

#ifndef ANSNR_TOOLS_H_
#define ANSNR_TOOLS_H_

extern const float ansnr_filter1d_ref_s[3];

extern const float ansnr_filter1d_dis_s[5];

extern const int ansnr_filter1d_ref_width;
extern const int ansnr_filter1d_dis_width;

extern const float ansnr_filter2d_ref_s[3*3];

extern const float ansnr_filter2d_dis_s[5*5];

extern const int ansnr_filter2d_ref_width;
extern const int ansnr_filter2d_dis_width;

void ansnr_mse_s(const float *ref, const float *dis, float *sig, float *noise, int w, int h, int ref_stride, int dis_stride);

void ansnr_filter1d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth);

void ansnr_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth);

#endif /* ANSNR_TOOLS_H_ */
