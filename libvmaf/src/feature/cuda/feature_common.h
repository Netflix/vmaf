/**
 *
 *  Copyright 2026 Bardie Høgh Joensen
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

#ifndef VMAF_CUDA_FEATURE_COMMON_H
#define VMAF_CUDA_FEATURE_COMMON_H

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "log.h"

/* All helpers below run with the extractor's CUDA context current. The
 * caller owns partial initialization and must close it on failure. */
typedef struct CudaFeatureResources {
    CUmodule module;
    CUstream str, host_stream;
    CUevent finished, consumed, slot_done[2];
} CudaFeatureResources;

static inline int cuda_fex_error(CudaFunctions *f, CUresult result)
{
    if (result == CUDA_SUCCESS) return 0;
    const char *name = "unknown CUDA error";
    f->cuGetErrorName(result, &name);
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "CUDA feature extractor: %s\n", name);
    return -EIO;
}

#define CUDA_FEX_CHECK(f, call) do { \
    const int cuda_err_ = cuda_fex_error(f, (f)->call); \
    if (cuda_err_) return cuda_err_; \
} while (0)

/* Initializers declare err and a fail label that unwinds their resources. */
#define CUDA_FEX_INIT(f, call) do { \
    err = cuda_fex_error(f, (f)->call); \
    if (err) goto fail; \
} while (0)

static inline int cuda_fex_resources_init(CudaFunctions *f,
        CudaFeatureResources *r, const void *ptx)
{
    /* Blocking streams also follow legacy NULL-stream device copies made
     * by producers such as FFmpeg. Picture-ready events order async uploads. */
    CUDA_FEX_CHECK(f, cuStreamCreateWithPriority(&r->str, CU_STREAM_DEFAULT, 0));
    CUDA_FEX_CHECK(f, cuStreamCreateWithPriority(&r->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CUDA_FEX_CHECK(f, cuEventCreate(&r->finished, CU_EVENT_DISABLE_TIMING));
    CUDA_FEX_CHECK(f, cuEventCreate(&r->consumed, CU_EVENT_DISABLE_TIMING));
    for (unsigned i = 0; i < 2; i++)
        CUDA_FEX_CHECK(f, cuEventCreate(&r->slot_done[i], CU_EVENT_DISABLE_TIMING));
    CUDA_FEX_CHECK(f, cuModuleLoadData(&r->module, ptx));
    return 0;
}

static inline int cuda_fex_resources_close(CudaFunctions *f, CudaFeatureResources *r)
{
    int err = 0;
    /* Complete callbacks before their parameter/readback memory is freed.
     * Keep releasing other resources if an individual release fails. */
    if (r->str) err |= cuda_fex_error(f, f->cuStreamSynchronize(r->str));
    if (r->host_stream) err |= cuda_fex_error(f, f->cuStreamSynchronize(r->host_stream));
    if (r->finished) err |= cuda_fex_error(f, f->cuEventDestroy(r->finished));
    if (r->consumed) err |= cuda_fex_error(f, f->cuEventDestroy(r->consumed));
    for (unsigned i = 0; i < 2; i++)
        if (r->slot_done[i]) err |= cuda_fex_error(f, f->cuEventDestroy(r->slot_done[i]));
    if (r->str) err |= cuda_fex_error(f, f->cuStreamDestroy(r->str));
    if (r->host_stream) err |= cuda_fex_error(f, f->cuStreamDestroy(r->host_stream));
    if (r->module) err |= cuda_fex_error(f, f->cuModuleUnload(r->module));
    memset(r, 0, sizeof(*r));
    return err;
}

static inline int cuda_fex_buffer_alloc(CudaFunctions *f, VmafCudaBuffer **out,
        size_t size)
{
    VmafCudaBuffer *buf = calloc(1, sizeof(*buf));
    if (!buf) return -ENOMEM;
    const int err = cuda_fex_error(f, f->cuMemAlloc(&buf->data, size));
    if (err) { free(buf); return err; }
    buf->size = size;
    *out = buf;
    return 0;
}

static inline int cuda_fex_buffer_free(CudaFunctions *f, VmafCudaBuffer **buf)
{
    if (!*buf) return 0;
    const int err = cuda_fex_error(f, f->cuMemFree((*buf)->data));
    free(*buf);
    *buf = NULL;
    return err;
}

#endif
