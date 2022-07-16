/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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
#include "integer_adm_cuda.h"
#include "feature_collector.h"

#include "common.h"
#ifdef __cplusplus

extern "C" {
#endif


// decouple function

void adm_decouple_s123_device(AdmBufferCuda *buf, int w, int h, int stride, AdmFixedParametersCuda* p,
                              CUstream c_stream);

void adm_decouple_device(AdmBufferCuda *buf, int w, int h, int stride, AdmFixedParametersCuda* p,
                         CUstream c_stream);

// csf_den_scale functions

void adm_csf_den_scale_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                              double adm_norm_view_dist,
                              int adm_ref_display_height, CUstream c_stream);

void adm_csf_den_s123_device(AdmBufferCuda *buf, int scale, int w, int h,
                             int src_stride, double adm_norm_view_dist,
                             int adm_ref_display_height, CUstream c_stream);

// csf functions

void adm_csf_device(AdmBufferCuda *buf, int w, int h, int stride,
                    AdmFixedParametersCuda *p, CUstream c_stream);

void i4_adm_csf_device(AdmBufferCuda *buf, int scale, int w, int h, int stride,
                       AdmFixedParametersCuda *p, CUstream c_stream);

// adm cm functions

void i4_adm_cm_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                      int csf_a_stride, int scale, AdmFixedParametersCuda* p, CUstream c_stream);

void adm_cm_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                   int csf_a_stride, AdmFixedParametersCuda* p, CUstream c_stream);

// adm dwt functions

void dwt2_8_device(const uint8_t *d_picture, cuda_adm_dwt_band_t *d_dst, cuda_i4_adm_dwt_band_t i4_dwt_dst,
                   short2 *tmp_buf, AdmBufferCuda *d_buf, int w, int h,
                   int src_stride, int dst_stride, AdmFixedParametersCuda* p, CUstream c_stream);

void adm_dwt2_16_device(const uint16_t *d_picture, cuda_adm_dwt_band_t *d_dst, cuda_i4_adm_dwt_band_t i4_dwt_dst,
                        short2 *tmp_buf, AdmBufferCuda *d_buf, int w, int h,
                        int src_stride, int dst_stride, int inp_size_bits,
                        AdmFixedParametersCuda* p, CUstream c_stream);

void adm_dwt2_s123_combined_device(const int32_t *d_i4_ref_scale,
                                   int32_t *tmp_buf,
                                   cuda_i4_adm_dwt_band_t i4_dwt,
                                   AdmBufferCuda *d_buf, int w, int h,
                                   int ref_stride, int dst_stride, int scale, AdmFixedParametersCuda* p,
                                   CUstream cu_stream);

#ifdef __cplusplus
}
#endif
