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

#ifndef FEATURE_VIF_CUDA_H_
#define FEATURE_VIF_CUDA_H_

#include <stdint.h>
#include "integer_vif.h"
#include "common.h"


/* Enhancement gain imposed on vif, must be >= 1.0, where 1.0 means the gain is completely disabled */
#ifndef DEFAULT_VIF_ENHN_GAIN_LIMIT
#define DEFAULT_VIF_ENHN_GAIN_LIMIT (100.0)
#endif // !DEFAULT_VIF_ENHN_GAIN_LIMIT

typedef struct VifBufferCuda {
    VmafCudaState cu_state;

    VmafCudaBuffer *data;
    VmafCudaBuffer *accum_data;

    CUdeviceptr ref;
    CUdeviceptr dis;
    uint16_t *mu1;
    uint16_t *mu2;
    uint32_t *mu1_32;
    uint32_t *mu2_32;
    uint32_t *ref_sq;
    uint32_t *dis_sq;
    uint32_t *ref_dis;
    int64_t *accum;
    void* accum_host;
    void* cpu_param_buf;
    struct {
        uint32_t *mu1;
        uint32_t *mu2;
        uint32_t *ref;
        uint32_t *dis;
        uint32_t *ref_dis;
        uint32_t *ref_convol;
        uint32_t *dis_convol;
        uint32_t *padding;
    } tmp;

    ptrdiff_t stride;
    ptrdiff_t stride_16;
    ptrdiff_t stride_32;
    ptrdiff_t stride_64;
    ptrdiff_t stride_tmp;
} VifBufferCuda;

typedef struct filter_table_stuct {
    uint16_t filter[4][18];
} filter_table_stuct;

typedef struct filter_width_struct {
    int w[4];
} filter_width_struct;

typedef struct vif_accums {
    int64_t x;
    int64_t x2;
    int64_t num_x;
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
} vif_accums;

extern unsigned char src_filter1d_ptx[];

#endif /* _FEATURE_VIF_CUDA_H_ */
