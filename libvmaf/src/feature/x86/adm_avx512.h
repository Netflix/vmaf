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

#ifndef X86_AVX512_ADM_H_
#define X86_AVX512_ADM_H_

#include "feature/integer_adm.h"

void adm_dwt2_8_avx512(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride);

void adm_decouple_avx512(AdmBuffer *buf, int w, int h, int stride,
                     double adm_enhn_gain_limit, int32_t* adm_div_lookup);

void adm_decouple_s123_avx512(AdmBuffer *buf, int w, int h, int stride,
                              double adm_enhn_gain_limit, int32_t* adm_div_lookup);

float adm_cm_avx512(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                    double adm_norm_view_dist, int adm_ref_display_height);

float i4_adm_cm_avx512(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride, int scale,
                       double adm_norm_view_dist, int adm_ref_display_height);

void adm_dwt2_s123_combined_avx512(const int32_t *i4_ref_scale, const int32_t *i4_curr_dis,
                                   AdmBuffer *buf, int w, int h, int ref_stride,
                                   int dis_stride, int dst_stride, int scale);

void adm_dwt2_16_avx512(const uint16_t *src, const adm_dwt_band_t *dst, AdmBuffer *buf, int w, int h,
                        int src_stride, int dst_stride, int inp_size_bits);

void adm_csf_avx512(AdmBuffer *buf, int w, int h, int stride,
                    double adm_norm_view_dist, int adm_ref_display_height);

void i4_adm_csf_avx512(AdmBuffer *buf, int scale, int w, int h, int stride,
                       double adm_norm_view_dist, int adm_ref_display_height);

float adm_csf_den_scale_avx512(const adm_dwt_band_t *src, int w, int h,
                    int src_stride,
                    double adm_norm_view_dist, int adm_ref_display_height);

float adm_csf_den_s123_avx512(const i4_adm_dwt_band_t *src, int scale, int w, int h,
                              int src_stride,
                              double adm_norm_view_dist, int adm_ref_display_height);

#endif /* X86_AVX512_ADM_H_ */
