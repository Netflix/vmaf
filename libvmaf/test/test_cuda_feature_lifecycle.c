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

#include <stdint.h>
#include <stdlib.h>

#include "test.h"
#include "cuda/common.h"
#include "feature/feature_extractor.h"

/* Inject failures at each driver allocation/symbol lookup. No GPU is needed:
 * a failed extractor init must leave no resources or pushed contexts behind. */
static unsigned calls, fail_at, live, next_handle, context_depth;
static unsigned handles[128];

static int fail(void) { return ++calls == fail_at; }

static CUresult create_handle(uintptr_t *out)
{
    if (fail()) return (CUresult)2;
    *out = ++next_handle;
    handles[*out] = 1;
    live++;
    return CUDA_SUCCESS;
}

static CUresult destroy_handle(uintptr_t handle)
{
    if (!handle || handle >= 128 || !handles[handle]) abort();
    handles[handle] = 0;
    live--;
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI push(CUcontext ctx)
{ (void)ctx; context_depth++; return CUDA_SUCCESS; }
static CUresult CUDAAPI pop(CUcontext *ctx)
{ (void)ctx; if (!context_depth) abort(); context_depth--; return CUDA_SUCCESS; }
static CUresult CUDAAPI error_name(CUresult error, const char **name)
{ (void)error; *name = "injected initialization failure"; return CUDA_SUCCESS; }
static CUresult CUDAAPI stream_create(CUstream *stream, unsigned flags, int priority)
{
    (void)flags; (void)priority;
    uintptr_t h = 0; CUresult err = create_handle(&h);
    *stream = (CUstream)h; return err;
}
static CUresult CUDAAPI stream_sync(CUstream stream)
{ (void)stream; return CUDA_SUCCESS; }
static CUresult CUDAAPI stream_destroy(CUstream stream)
{ return destroy_handle((uintptr_t)stream); }
static CUresult CUDAAPI event_create(CUevent *event, unsigned flags)
{
    (void)flags; uintptr_t h = 0; CUresult err = create_handle(&h);
    *event = (CUevent)h; return err;
}
static CUresult CUDAAPI event_destroy(CUevent event)
{ return destroy_handle((uintptr_t)event); }
static CUresult CUDAAPI module_load(CUmodule *module, const void *data)
{
    (void)data; uintptr_t h = 0; CUresult err = create_handle(&h);
    *module = (CUmodule)h; return err;
}
static CUresult CUDAAPI module_unload(CUmodule module)
{ return destroy_handle((uintptr_t)module); }
static CUresult CUDAAPI get_function(CUfunction *function, CUmodule module, const char *name)
{
    (void)module; (void)name;
    if (fail()) return (CUresult)2;
    *function = (CUfunction)(uintptr_t)1;
    return CUDA_SUCCESS;
}
static CUresult CUDAAPI mem_alloc(CUdeviceptr *ptr, size_t size)
{
    (void)size; uintptr_t h = 0; CUresult err = create_handle(&h);
    *ptr = (CUdeviceptr)h; return err;
}
static CUresult CUDAAPI mem_free(CUdeviceptr ptr)
{ return destroy_handle((uintptr_t)ptr); }
static CUresult CUDAAPI host_alloc(void **ptr, size_t size, unsigned flags)
{
    (void)size; (void)flags; uintptr_t h = 0; CUresult err = create_handle(&h);
    *ptr = (void *)h; return err;
}
static CUresult CUDAAPI host_free(void *ptr)
{ return destroy_handle((uintptr_t)ptr); }

static char *lifecycle(const char *name, enum VmafPixelFormat format,
                       unsigned width, unsigned height, int valid)
{
    CudaFunctions functions = {
        .cuCtxPushCurrent = push, .cuCtxPopCurrent = pop,
        .cuGetErrorName = error_name,
        .cuStreamCreateWithPriority = stream_create,
        .cuStreamSynchronize = stream_sync, .cuStreamDestroy = stream_destroy,
        .cuEventCreate = event_create, .cuEventDestroy = event_destroy,
        .cuModuleLoadData = module_load, .cuModuleUnload = module_unload,
        .cuModuleGetFunction = get_function,
        .cuMemAlloc = mem_alloc, .cuMemFree = mem_free,
        .cuMemHostAlloc = host_alloc, .cuMemFreeHost = host_free,
    };
    VmafCudaState state = { .ctx = (CUcontext)(uintptr_t)1, .f = &functions };
    unsigned count = 0;
    for (fail_at = 0; fail_at <= count; fail_at++) {
        calls = live = next_handle = context_depth = 0;
        VmafFeatureExtractorContext *ctx;
        VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name(name);
        int err = vmaf_feature_extractor_context_create(&ctx, fex, NULL);
        mu_assert("extractor context create failed", !err);
        ctx->fex->cu_state = &state;
        err = vmaf_feature_extractor_context_init(ctx, format,
                                                 8, width, height);
        if (fail_at || !valid) {
            mu_assert("invalid or failed initialization was accepted", err < 0);
            mu_assert("failed initialization leaked CUDA resources", !live);
        } else {
            mu_assert("initialization failed without injection", !err);
            count = calls;
            err = vmaf_feature_extractor_context_close(ctx);
            mu_assert("extractor close failed", !err);
        }
        mu_assert("CUDA context push/pop is unbalanced", !context_depth);
        mu_assert("extractor close leaked CUDA resources", !live);
        if (!valid) mu_assert("invalid dimensions allocated CUDA resources", !calls);
        vmaf_feature_extractor_context_destroy(ctx);
    }
    return NULL;
}

static char *test_lifecycle(void)
{
    const char *names[] = { "psnr_cuda", "ssim_cuda", "ciede_cuda" };
    for (unsigned i = 0; i < 3; i++) {
        char *fail = lifecycle(names[i], VMAF_PIX_FMT_YUV420P, 64, 48, 1);
        if (fail) return fail;
    }
    /* Exercise both SSIM allocation layouts and its invalid-window path. */
    char *fail = lifecycle("ssim_cuda", VMAF_PIX_FMT_YUV420P, 768, 432, 1);
    if (fail) return fail;
    fail = lifecycle("ssim_cuda", VMAF_PIX_FMT_YUV420P, 8, 8, 0);
    if (fail) return fail;
    const struct {
        enum VmafPixelFormat format;
        unsigned width, height;
        int valid;
    } cases[] = {
        { VMAF_PIX_FMT_YUV420P, 64, 49, 0 },
        { VMAF_PIX_FMT_YUV420P, 65, 48, 0 },
        { VMAF_PIX_FMT_YUV422P, 65, 48, 0 },
        { VMAF_PIX_FMT_YUV422P, 64, 49, 1 },
        { VMAF_PIX_FMT_YUV444P, 65, 49, 1 },
        { VMAF_PIX_FMT_YUV400P, 64, 48, 0 },
        { VMAF_PIX_FMT_YUV444P, 0, 48, 0 },
        { VMAF_PIX_FMT_YUV444P, 64, 0, 0 },
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fail = lifecycle("ciede_cuda", cases[i].format,
                          cases[i].width, cases[i].height, cases[i].valid);
        if (fail) return fail;
    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_lifecycle);
    return NULL;
}
