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

/*
 * Parameters which are constant for all pipelines of a single frame.
 * All non-integer values are either exact host-computed constants
 * (cos_1deg_sq) or integer-valued options, so that the device never has to
 * redo floating point computations the CPU reference implementation does on
 * the host.
 */
typedef struct AdmFixedParametersCuda {
    float cos_1deg_sq;      /* cos(1degree)^2, computed in double on the host */

    int32_t dwt2_db2_coeffs_lo[10];
    int32_t dwt2_db2_coeffs_hi[10];
    int32_t dwt2_db2_coeffs_lo_sum;
    int32_t dwt2_db2_coeffs_hi_sum;
} AdmFixedParametersCuda;

/* per launch parameters of the csf kernels, indexed by band */
typedef struct AdmCsfKernelParamsCuda {
    uint32_t i_rfactor[3];
    /* dst = (rfactor * src + add_dst) >> shift_dst */
    uint32_t shift_dst[3];
    int64_t add_dst[3];
    /* flt = (FIX_ONE_BY_30 * abs(dst) + add_flt) >> shift_flt */
    uint32_t shift_flt[3];
    int64_t add_flt[3];
    int32_t measure_aim;
} AdmCsfKernelParamsCuda;

/* per launch parameters of the csf_den kernels */
typedef struct AdmCsfDenKernelParamsCuda {
    int32_t top, bottom, left, right;
    int32_t src_stride;
    uint32_t shift_accum;
    uint32_t add_shift_accum;
    uint32_t shift_sq;
    uint32_t add_shift_sq;
    uint32_t shift_cub;
    uint32_t add_shift_cub;
} AdmCsfDenKernelParamsCuda;

/* per launch parameters of the cm kernels, indexed by band */
typedef struct AdmCmKernelParamsCuda {
    int32_t w, h;
    int32_t top, bottom, left, right;
    int32_t start_row, end_row, start_col, end_col;
    int32_t src_stride;
    int32_t measure_aim;
    uint32_t i_rfactor[3];
    uint32_t shift_sub[3];
    uint32_t shift_sq[3];
    uint32_t add_shift_sq[3];
    uint32_t shift_cub[3];
    uint32_t add_shift_cub[3];
    uint32_t shift_inner_accum;
    uint32_t add_shift_inner_accum;
} AdmCmKernelParamsCuda;

typedef struct AdmBufferCuda {
    size_t ind_size_x, ind_size_y; // strides size for intermidate buffers
    int *ind_y[4], *ind_x[4];

    struct VmafCudaBuffer* data_buf;
    struct VmafCudaBuffer* tmp_ref;
    struct VmafCudaBuffer* tmp_dis;
    struct VmafCudaBuffer* tmp_res;

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
    int64_t* aim_num[4];
    void* results_host;
} AdmBufferCuda;

extern const unsigned char adm_dwt2_ptx[];
extern const unsigned char adm_csf_den_ptx[];
extern const unsigned char adm_csf_ptx[];
extern const unsigned char adm_decouple_ptx[];
extern const unsigned char adm_cm_ptx[];

#endif /* _FEATURE_ADM_CUDA_H_ */
