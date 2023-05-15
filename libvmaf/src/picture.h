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

#ifndef __VMAF_SRC_PICTURE_H__
#define __VMAF_SRC_PICTURE_H__

#ifdef HAVE_CUDA
#include <cuda.h>
#include "libvmaf/libvmaf_cuda.h"
#endif
#include "libvmaf/picture.h"

enum VmafPictureBufferType {
    VMAF_PICTURE_BUFFER_TYPE_HOST = 0,
    VMAF_PICTURE_BUFFER_TYPE_CUDA_HOST_PINNED,
    VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE,
};

typedef struct VmafPicturePrivate {
    void *cookie;
    int (*release_picture)(VmafPicture *pic, void *cookie);
#ifdef HAVE_CUDA
    struct {
        CUcontext ctx;
        CUevent ready, finished;
        CUstream str;
    } cuda;
#endif
    enum VmafPictureBufferType buf_type;
} VmafPicturePrivate;

int vmaf_picture_priv_init(VmafPicture *pic);

int vmaf_picture_ref(VmafPicture *dst, VmafPicture *src);

int vmaf_picture_set_release_callback(VmafPicture *pic, void *cookie,
                        int (*release_picture)(VmafPicture *pic, void *cookie));

#endif /* __VMAF_SRC_PICTURE_H__ */
