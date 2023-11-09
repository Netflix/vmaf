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

#ifndef __VMAF_SRC_CUDA_PICTURE_CUDA_H__
#define __VMAF_SRC_CUDA_PICTURE_CUDA_H__

#include "common.h"
#include "cuda.h"
#include "libvmaf/picture.h"

typedef struct VmafCudaCookie {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc, w, h;
    VmafCudaState *state;
} VmafCudaCookie;

/**
 * Upload CPU VmafPicture to the VmafPicture on the GPU on the CUstream that
 * is passed. VmafPicture has a CUevent member that will be triggered as
 * soon as the upload is finished.
 *
 * @param cuda_pic   destination  image on the device/GPU
 *
 * @param pic        source image on the host/CPU
 *
 * @param bitmask    determines the channel to upload
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_picture_upload_async(VmafPicture *cuda_pic, VmafPicture *pic,
                                   uint8_t bitmask);

/**
 * Download a VmafPicture from the GPU to CPU on the CUstream passed.
 *
 * @param cuda_pic  source image on the device/GPU.
 *
 * @param pic       destination image on the host/CPU.
 *
 * @param bitmask   determines the channel to download
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_picture_download_async(VmafPicture *cuda_pic, VmafPicture *pic,
                                     uint8_t bitmask);

int vmaf_cuda_picture_alloc_pinned(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                                   unsigned bpc, unsigned w, unsigned h,
                                   VmafCudaState *cuda_state);

int vmaf_cuda_picture_alloc(VmafPicture *pic, void *cookie);

int vmaf_cuda_picture_free(VmafPicture *pic, void *cookie);

int vmaf_cuda_picture_synchronize(VmafPicture *pic, void *cookie);

CUstream vmaf_cuda_picture_get_stream(VmafPicture *pic);

CUevent vmaf_cuda_picture_get_finished_event(VmafPicture *pic);

CUevent vmaf_cuda_picture_get_ready_event(VmafPicture *pic);

#endif /* __VMAF_SRC_CUDA_PICTURE_CUDA_H__ */
