/**
 *
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

int vmaf_cuda_state_init(VmafCudaState *cu_state, int prio, int device_id)
{
    if (!cu_state) return -EINVAL;
    int n_gpu;

    CUdevice cu_device = 0;
    CUcontext cu_context = 0;
    CUresult res = CUDA_SUCCESS;
    res |= cuInit(0);
    res |= cuDeviceGetCount(&n_gpu);
    if (device_id > n_gpu) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "Error: device_id %d is out of range\n", device_id);
    }

    res |= cuDeviceGet(&cu_device, device_id);
    res |= cuDevicePrimaryCtxRetain(&cu_context, cu_device);
    if (res != CUDA_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "Error: failed to initialize CUDA\n");
        return -1;
    }

    cu_state->ctx = cu_context;
    cu_state->dev = cu_device;

    CHECK_CUDA(cuCtxPushCurrent((cu_state->ctx)));

    int low, high;
    CHECK_CUDA(cuCtxGetStreamPriorityRange(&low, &high));
    int prio2 = MAX(low, MIN(high, prio));
    CHECK_CUDA(cuStreamCreateWithPriority(&cu_state->str, CU_STREAM_NON_BLOCKING, prio2));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_sync(VmafCudaState *cu_state) {

    if(!cu_state) return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent((cu_state->ctx)));
    int err = cuCtxSynchronize();
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return err;

}

int vmaf_cuda_release(VmafCudaState *cu_state, bool rel_ctx)
{
    if(!cu_state)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuStreamDestroy(cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    if(rel_ctx)
        CHECK_CUDA(cuDevicePrimaryCtxRelease(cu_state->dev));

    memset((void *)cu_state, 0, sizeof(*cu_state));


    return CUDA_SUCCESS;
}


int vmaf_cuda_buffer_alloc(VmafCudaState *cu_state, CudaVmafBuffer **p_buf,
                           size_t size)
{
    if (!cu_state || !p_buf)
        return -EINVAL;


    CudaVmafBuffer *buf = (CudaVmafBuffer *)calloc(1, sizeof(*buf));
    *p_buf = buf;
    buf->size = size;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemAlloc(&buf->data, buf->size));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_free(VmafCudaState *cu_state, CudaVmafBuffer *buf)
{
    if (!cu_state || !buf)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemFree(buf->data));
    memset(buf, 0, sizeof(*buf));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_host_alloc(VmafCudaState *cu_state, void **p_buf,
                           size_t size)
{
    if (!cu_state || !p_buf)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemHostAlloc(p_buf, size, CU_MEMHOSTALLOC_PORTABLE));

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_host_free(VmafCudaState *cu_state, void *buf)
{
    if (!cu_state || !buf)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemFreeHost(buf));
    buf = NULL;
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_upload_async(VmafCudaState *cu_state, CudaVmafBuffer *buf,
                            const void *src, CUstream c_stream)
{
    if (!cu_state || !buf || !src)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemcpyHtoDAsync(buf->data, src, buf->size, c_stream == 0 ? c_stream : cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_download_async(VmafCudaState *cu_state, CudaVmafBuffer *buf,
                              void *dst, CUstream c_stream)
{
    if (!cu_state || !buf || !dst)
        return -EINVAL;

    CHECK_CUDA(cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cuMemcpyDtoHAsync(dst, buf->data, buf->size, c_stream == 0 ? c_stream : cu_state->str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return CUDA_SUCCESS;
}

int vmaf_cuda_buffer_get_dptr(CudaVmafBuffer *buf, CUdeviceptr *ptr)
{
    if (!buf || !ptr)
        return -EINVAL;

    *ptr = buf->data;
    return 0;
}
