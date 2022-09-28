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

#ifndef __VMAF_SRC_CUDA_MOTION_KERNELS_H__
#define __VMAF_SRC_CUDA_MOTION_KERNELS_H__

#include "integer_motion_cuda.h"

#include "common.h"
#ifdef __cplusplus

extern "C" {
#endif

void calculate_motion_score(const VmafPicture *src, CudaVmafBuffer *src_blurred,
                            const CudaVmafBuffer *prev_blurred,
                            CudaVmafBuffer *sad, unsigned width,
                            unsigned height, ptrdiff_t src_stride,
                            ptrdiff_t blurred_stride, unsigned src_bpc,
                            CUfunction funcbpc8, CUfunction funcbpc16, CUstream stream);

#ifdef __cplusplus
}
#endif
#endif /* __VMAF_SRC_CUDA_MOTION_KERNELS_H__ */