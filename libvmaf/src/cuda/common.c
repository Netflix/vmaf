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

#include <errno.h>
#include <stdbool.h>
#include <string.h>

#include "common.h"
#include "log.h"

static int is_cudastate_empty(VmafCudaState *cu_state)
{
    if (!cu_state) return 1;
    if (!cu_state->ctx) return 1;

    return 0;
}

static int init_with_primary_context(VmafCudaState *cu_state)
{
    if (!cu_state) return -EINVAL;

    CUdevice cu_device = 0;
    CUcontext cu_context = 0;
    CUresult res = CUDA_SUCCESS;

    const int device_id = 0;
    int n_gpu;
    res |= cuDeviceGetCount(&n_gpu);
    if (device_id > n_gpu) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "Error: device_id %d is out of range\n", device_id);
        return -EINVAL;
    }

    res |= cuDeviceGet(&cu_device, device_id);
    res |= cuDevicePrimaryCtxRetain(&cu_context, cu_device);
    if (res != CUDA_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "Error: failed to initialize CUDA\n");
        return -EINVAL;
    }

    cu_state->ctx = cu_context;
    cu_state->release_ctx = 1;
    cu_state->dev = cu_device;

    CHECK_CUDA(cuCtxPushCurrent((cu_state->ctx)));

    int low, high;
    CHECK_CUDA(cuCtxGetStreamPriorityRange(&low, &high));
    const int prio = 0;
    const int prio2 = MAX(low, MIN(high, prio));
    CHECK_CUDA(cuStreamCreateWithPriority(&cu_state->str,
                                          CU_STREAM_NON_BLOCKING, prio2));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return 0;
}

static int init_with_provided_context(VmafCudaState *cu_state, CUcontext cu_context)
{
    if (!cu_state) return -EINVAL;
    if (!cu_context) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_context));

    CUdevice cu_device = 0;
    int err = cuCtxGetDevice(&cu_device);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "failed to get CUDA device\n");
        return -EINVAL;
    }

    cu_state->ctx = cu_context;
    cu_state->release_ctx = 0;
    cu_state->dev = cu_device;

    int low, high;
    CHECK_CUDA(cuCtxGetStreamPriorityRange(&low, &high));
    const int prio = 0;
    const int prio2 = MAX(low, MIN(high, prio));
    CHECK_CUDA(cuStreamCreateWithPriority(&cu_state->str,
                                          CU_STREAM_NON_BLOCKING, prio2));

    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return 0;
}

int vmaf_cuda_state_init(VmafCudaState **cu_state, VmafCudaConfiguration cfg)
{
    if (!cu_state) return -EINVAL;

    VmafCudaState *const c = *cu_state = malloc(sizeof(*c));
    if (!c) return -ENOMEM;
    memset(c, 0, sizeof(*c));

    int err = cuInit(0);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem during CUDA initialization\n");
        return -EINVAL;
    }

    if (cfg.cu_ctx)
        return init_with_provided_context(c, cfg.cu_ctx);
    else
        return init_with_primary_context(c);
}

int vmaf_cuda_sync(VmafCudaState *cu_state)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent((cu_state->ctx)));
    int err = cuCtxSynchronize();
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return err;
}

int vmaf_cuda_release(VmafCudaState *cu_state)
{
    if (is_cudastate_empty(cu_state)) return CUDA_SUCCESS;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuStreamDestroy(cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    if (cu_state->release_ctx)
        CHECK_CUDA(cuDevicePrimaryCtxRelease(cu_state->dev));

    memset((void *)cu_state, 0, sizeof(*cu_state));

    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_alloc(VmafCudaState *cu_state, VmafCudaBuffer **p_buf,
                           size_t size)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!p_buf) return -EINVAL;

    VmafCudaBuffer *buf = (VmafCudaBuffer *)calloc(1, sizeof(*buf));
    if (!buf) return -ENOMEM;

    *p_buf = buf;
    buf->size = size;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemAlloc(&buf->data, buf->size));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_free(VmafCudaState *cu_state, VmafCudaBuffer *buf)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!buf) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemFree(buf->data));
    memset(buf, 0, sizeof(*buf));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_host_alloc(VmafCudaState *cu_state, void **p_buf,
                           size_t size)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!p_buf) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemHostAlloc(p_buf, size, CU_MEMHOSTALLOC_PORTABLE));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_host_free(VmafCudaState *cu_state, void *buf)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!buf) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemFreeHost(buf));
    buf = NULL;
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_upload_async(VmafCudaState *cu_state, VmafCudaBuffer *buf,
                            const void *src, CUstream c_stream)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!buf) return -EINVAL;
    if (!src) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemcpyHtoDAsync(buf->data, src, buf->size,
                                 c_stream == 0 ? c_stream : cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_download_async(VmafCudaState *cu_state,
                                    VmafCudaBuffer *buf, void *dst,
                                    CUstream c_stream)
{
    if (is_cudastate_empty(cu_state)) return -EINVAL;
    if (!buf) return -EINVAL;
    if (!dst) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemcpyDtoHAsync(dst, buf->data, buf->size,
                                 c_stream == 0 ? c_stream : cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_get_dptr(VmafCudaBuffer *buf, CUdeviceptr *ptr)
{
    if (!buf) return -EINVAL;
    if (!ptr) return -EINVAL;

    *ptr = buf->data;
    return 0;
}
