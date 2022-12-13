/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#ifndef FEATURE_ADM_CUDA_H_
#define FEATURE_ADM_CUDA_H_

#include "mem.h"
#include "stdio.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "integer_adm.h"
#include "common.h"

typedef struct cuda_adm_dwt_band_t {
    union {
        struct{
            int16_t *band_a; /* Low-pass V + low-pass H. */
            int16_t *band_h; /* High-pass V + low-pass H. */
            int16_t *band_v; /* Low-pass V + high-pass H. */
            int16_t *band_d; /* High-pass V + high-pass H. */
        };
        int16_t * bands[4];
    };
} cuda_adm_dwt_band_t;

typedef struct cuda_i4_adm_dwt_band_t {
    union {
        struct{
            int32_t *band_a; /* Low-pass V + low-pass H. */
            int32_t *band_h; /* High-pass V + low-pass H. */
            int32_t *band_v; /* Low-pass V + high-pass H. */
            int32_t *band_d; /* High-pass V + high-pass H. */
        };
        int32_t * bands[4];
    };
} cuda_i4_adm_dwt_band_t;

typedef struct AdmFixedParametersCuda {
    float rfactor[3*4];
    uint32_t i_rfactor[3*4];
    float factor1[4];
    float factor2[4];
    float log2_h;
    float log2_w;

    double adm_norm_view_dist;
    double adm_enhn_gain_limit;
    int32_t adm_ref_display_height;

    int32_t dwt2_db2_coeffs_lo[10];
    int32_t dwt2_db2_coeffs_hi[10];
    int32_t dwt2_db2_coeffs_lo_sum;
    int32_t dwt2_db2_coeffs_hi_sum;
} AdmFixedParametersCuda;

typedef struct AdmBufferCuda {
    size_t ind_size_x, ind_size_y; // strides size for intermidate buffers
    int *ind_y[4], *ind_x[4];

    struct VmafCudaBuffer* data_buf;
    struct VmafCudaBuffer* tmp_ref;
    struct VmafCudaBuffer* tmp_dis;
    struct VmafCudaBuffer* tmp_res;
    struct VmafCudaBuffer* tmp_accum;
    struct VmafCudaBuffer* tmp_accum_h;

    cuda_adm_dwt_band_t ref_dwt2;
    cuda_adm_dwt_band_t dis_dwt2;
    cuda_adm_dwt_band_t decouple_r;
    cuda_adm_dwt_band_t decouple_a;
    cuda_adm_dwt_band_t csf_a;
    cuda_adm_dwt_band_t csf_f;

    cuda_i4_adm_dwt_band_t i4_ref_dwt2;
    cuda_i4_adm_dwt_band_t i4_dis_dwt2;
    cuda_i4_adm_dwt_band_t i4_decouple_r;
    cuda_i4_adm_dwt_band_t i4_decouple_a;
    cuda_i4_adm_dwt_band_t i4_csf_a;
    cuda_i4_adm_dwt_band_t i4_csf_f;

    int64_t* adm_cm[4];
    uint64_t* adm_csf_den[4];
    void* results_host;
} AdmBufferCuda;

extern unsigned char src_adm_dwt2_ptx[];
extern unsigned char src_adm_csf_den_ptx[];
extern unsigned char src_adm_csf_ptx[];
extern unsigned char src_adm_decouple_ptx[];
extern unsigned char src_adm_cm_ptx[];

#endif /* _FEATURE_ADM_CUDA_H_ */
