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

#ifndef __VMAF_SRC_CUDA_COMMON_H__
#define __VMAF_SRC_CUDA_COMMON_H__

#include <pthread.h>
#include <stdbool.h>

#include "config.h"
#include "picture.h"

#if HAVE_CUDA

#include <cuda.h>
#include <libvmaf/libvmaf_cuda.h>
#include "cuda_helper.cuh"

typedef struct VmafCudaBuffer {
    size_t size;
    CUdeviceptr data;
} VmafCudaBuffer;

typedef struct VmafCudaState {
    CUcontext ctx;
    CUstream str;
    CUdevice dev;
    int release_ctx;
} VmafCudaState;

#define VMAF_CUDA_THREADS_PER_WARP 32
#define VMAF_CUDA_CACHE_LINE_SIZE 128

/**
 * Synchronize a CUcontext from a VmafCudaState object.
 *
 * @param cu_state VmafCudaState to get its context and synchronize.
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_sync(VmafCudaState *cu_state);

/**
 * Destroys a VmafCudaState object by destroying all of its members.
 * If rel_ctx is true, it will release the GPU driver context and also
 * release the driver. CUDA cannot be used when the context has be released,
 * afterwards all VmafCudaState objects are invalid.
 *
 * @param cu_state  VmafCudaState to free.
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */

int vmaf_cuda_release(VmafCudaState *cu_state);

/**
 * Allocates a 1D buffer on the GPU.
 *
 * @param cu_state  Initialized VmafCudaState object.
 *
 * @param buf       VmafCudaBuffer to be allocated.
 *
 * @param size      bytes to allocate.
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_alloc(VmafCudaState *cu_state, VmafCudaBuffer **buf,
                           size_t size);

/**
 * Frees a VmafCudaBuffer from the GPU and sets the passed pointer to 0.
 *
 * @param cu_state  Initialized VmafCudaState object.
 *
 * @param buf       VmafCudaBuffer to be freed.
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_free(VmafCudaState *cu_state, VmafCudaBuffer *buf);

/**
 * Uploads data in the size of the VmafCudaBuffer from src pointer (Host/CPU)
 * to the Device/GPU asynchronously.
 *
 * @param cu_state  Initialized VmafCudaState object.
 *
 * @param buf       Destination buffer on the Device/GPU.
 *
 * @param src       Source Host/CPU buffer.
 *
 * @param c_stream  stream on which the upload will happen.
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_upload_async(VmafCudaState *cu_state, VmafCudaBuffer *buf,
                                  const void *src, CUstream c_stream);
/**
 * Downloads data in the size of the VmafCudaBuffer from the GPU asynchronously.
 *
 * @param cu_state  Initialized VmafCudaState object.
 *
 * @param buf       Destination buffer on the Device/GPU.
 *
 * @param src       Source Host/CPU buffer.
 *
 * @param c_stream  stream on which the upload will happen.
 *
 * @return CUDA_SUCCESS on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_download_async(VmafCudaState *cu_state, VmafCudaBuffer *buf,
                                    void *dst, CUstream c_stream);
/**
 * Device pointer getter for VmafCudaBuffer
 *
 * @param buf   Initialized VmafCudaBuffer.
 *
 * @param ptr   CUdeviceptr to be set.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_get_dptr(VmafCudaBuffer *buf, CUdeviceptr *ptr);

/**
 * Frees up pinned host (CPU) memory.
 *
 * @param cu_state  Initialized VmafCudaState.
 *
 * @param buf       pointer to buffer that will be freed
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_host_free(VmafCudaState *cu_state, void *buf);

/**
 * Allocate host (CPU) pinned memory.
 * Memory transfers to the device (GPU) are accelerated with pinned memory.
 *
 * @param cu_state  Initialized VmafCudaState.
 *
 * @param buf       pointer to a pointer for the allocated buffer.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_buffer_host_alloc(VmafCudaState *cu_state, void **p_buf,
                                size_t size);
#endif // !HAVE_CUDA

#endif /* __VMAF_SRC_CUDA_COMMON_H__ */
