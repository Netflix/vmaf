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

#ifndef __VMAF_CUDA_H__
#define __VMAF_CUDA_H__

#include "libvmaf/libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafCudaState VmafCudaState;

typedef struct VmafCudaConfiguration {
    void *cu_ctx; ///< CUcontext
} VmafCudaConfiguration;

/**
 * Initialize VmafCudaState.
 * VmafCudaState can optionally be configured with VmafCudaConfiguration.
 *
 * @param cu_state The CUDA state to open.
 *
 * @param cfg      Optional configuration parameters.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_state_init(VmafCudaState **cu_state, VmafCudaConfiguration cfg);

/**
 * Import VmafCudaState for use during CUDA feature extraction.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()`.
 *
 * @param cu_state CUDA state allocated with `vmaf_cuda_state_init()`.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_import_state(VmafContext *vmaf, VmafCudaState *cu_state);

enum VmafCudaPicturePreallocationMethod {
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE = 0,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
};

typedef struct VmafCudaPictureConfiguration {
    struct {
        unsigned w, h;
        unsigned bpc;
        enum VmafPixelFormat pix_fmt;
    } pic_params;
    enum VmafCudaPicturePreallocationMethod pic_prealloc_method;
} VmafCudaPictureConfiguration;

/**
 * Config and preallocate VmafPictures for use during CUDA feature extraction.
 * The preallocated VmafPicture data buffers are set according to
 * cfg.pic_prealloc_method.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()`.
 *
 * @param cfg VmafPicture parameter configuration.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_preallocate_pictures(VmafContext *vmaf,
                                   VmafCudaPictureConfiguration cfg);

/**
 * Fetch a preallocated VmafPicture for use during CUDA feature extraction.
 * pictures are allocated during `vmaf_cuda_preallocate_pictures` and data
 * buffers are set according to cfg.pic_prealloc_method.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()` and
 *             initialized with `vmaf_cuda_preallocate_pictures()`.
 *
 * @param pic Preallocated picture.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_fetch_preallocated_picture(VmafContext *vmaf, VmafPicture* pic);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_CUDA_H__ */
