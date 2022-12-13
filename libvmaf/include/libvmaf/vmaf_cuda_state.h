/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#ifndef __VMAF_CUDA_STATE_H__
#define __VMAF_CUDA_STATE_H__

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafCudaState {
    CUcontext ctx;
    CUstream str;
    CUdevice dev;
} VmafCudaState;

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_CUDA_STATE_H__ */
