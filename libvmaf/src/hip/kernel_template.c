/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP kernel-template helper bodies — runtime PR (T7-10b /
 *  ADR-0212 §"What lands next" step 1).
 *
 *  Replaces the audit-first `-ENOSYS` stubs (ADR-0241) with real
 *  ROCm HIP runtime calls. Mirrors `libvmaf/src/cuda/kernel_template.h`'s
 *  inline helpers but as out-of-line definitions so the runtime PR
 *  can flip implementations without forcing every consumer TU to
 *  recompile.
 */

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include "../kernel_lifecycle_common.h"
#include "common.h"
#include "kernel_template.h"

/*
 * Translate a HIP error code into a negative POSIX errno. Mapping is
 * intentionally coarse — feature kernels treat any negative rc as
 * "abort the per-frame submit" and do not branch on the specific
 * code. Mirrors the SYCL backend's `sycl_rc_to_errno` and the CUDA
 * `CHECK_CUDA_GOTO` aggregation pattern.
 */
static int hip_rc_to_errno(hipError_t rc)
{
    if (rc == hipSuccess) {
        return 0;
    }
    switch (rc) {
    case hipErrorInvalidValue:
    case hipErrorInvalidHandle:
        return -EINVAL;
    case hipErrorOutOfMemory:
        return -ENOMEM;
    case hipErrorNoDevice:
    case hipErrorInvalidDevice:
        return -ENODEV;
    case hipErrorNotSupported:
        return -ENOSYS;
    default:
        return -EIO;
    }
}

int vmaf_hip_kernel_lifecycle_init(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    /* NASA P10 r5: pin the precondition + the post-init invariants
     * the rest of the lifecycle depends on. */
    assert(lc != NULL);
    VMAF_LIFECYCLE_ZERO(lc);

    hipStream_t stream = NULL;
    hipError_t rc = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (rc != hipSuccess) {
        return hip_rc_to_errno(rc);
    }

    hipEvent_t submit_ev = NULL;
    rc = hipEventCreateWithFlags(&submit_ev, hipEventDisableTiming);
    if (rc != hipSuccess) {
        (void)hipStreamDestroy(stream);
        return hip_rc_to_errno(rc);
    }

    hipEvent_t finished_ev = NULL;
    rc = hipEventCreateWithFlags(&finished_ev, hipEventDisableTiming);
    if (rc != hipSuccess) {
        (void)hipEventDestroy(submit_ev);
        (void)hipStreamDestroy(stream);
        return hip_rc_to_errno(rc);
    }

    lc->str = (uintptr_t)stream;
    lc->submit = (uintptr_t)submit_ev;
    lc->finished = (uintptr_t)finished_ev;
    return 0;
}

int vmaf_hip_kernel_readback_alloc(VmafHipKernelReadback *rb, VmafHipContext *ctx, size_t bytes)
{
    (void)ctx;
    if (rb == NULL) {
        return -EINVAL;
    }
    VMAF_LIFECYCLE_ZERO(rb);
    rb->bytes = bytes;
    if (bytes == 0) {
        return -EINVAL;
    }
    /* NASA P10 r5: precondition + post-zero invariants. */
    assert(rb != NULL);
    assert(bytes > 0);

    /* Synchronous device alloc: `hipMallocAsync` requires a stream-
     * ordered memory pool which is feature-flagged on some ROCm
     * builds. The kernel-template's allocations are once-per-state
     * (not per-frame), so the synchronous variant is the safer
     * default and gives us identical observable lifetime to the
     * CUDA twin's `cuMemAlloc`. */
    void *device = NULL;
    hipError_t rc = hipMalloc(&device, bytes);
    if (rc != hipSuccess) {
        return hip_rc_to_errno(rc);
    }

    void *host_pinned = NULL;
    rc = hipHostMalloc(&host_pinned, bytes, hipHostMallocDefault);
    if (rc != hipSuccess) {
        (void)hipFree(device);
        return hip_rc_to_errno(rc);
    }

    rb->device = device;
    rb->host_pinned = host_pinned;
    return 0;
}

int vmaf_hip_kernel_submit_pre_launch(VmafHipKernelLifecycle *lc, VmafHipContext *ctx,
                                      VmafHipKernelReadback *rb, uintptr_t picture_stream,
                                      uintptr_t dist_ready_event)
{
    (void)ctx;
    if (lc == NULL || rb == NULL) {
        return -EINVAL;
    }
    if (lc->str == 0 || rb->device == NULL || rb->bytes == 0) {
        return -EINVAL;
    }

    hipError_t rc = hipMemsetAsync(rb->device, 0, rb->bytes, (hipStream_t)lc->str);
    if (rc != hipSuccess) {
        return hip_rc_to_errno(rc);
    }

    /* Picture stream waits for the dist-side ready event. A zero
     * handle short-circuits to "no cross-stream wait needed"; the
     * CUDA twin honours the same convention so first-pass / single-
     * frame consumers can pass 0. */
    if (picture_stream != 0 && dist_ready_event != 0) {
        rc = hipStreamWaitEvent((hipStream_t)picture_stream, (hipEvent_t)dist_ready_event, 0);
        if (rc != hipSuccess) {
            return hip_rc_to_errno(rc);
        }
    }
    return 0;
}

int vmaf_hip_kernel_collect_wait(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    if (lc->str == 0) {
        /* Lifecycle was never initialised — nothing to wait on. */
        return 0;
    }
    return hip_rc_to_errno(hipStreamSynchronize((hipStream_t)lc->str));
}

int vmaf_hip_kernel_lifecycle_close(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return 0;
    }
    /* NASA P10 r5: lc is non-NULL on entry past the early-return. */
    assert(lc != NULL);

    /* Aggregate the first non-zero rc across the teardown steps;
     * mirrors the CUDA twin's `lifecycle_close` policy. Every step
     * is best-effort — we always proceed past a failure so the
     * remaining handles still get released. Zero handles short-
     * circuit (the lifecycle may be partially-initialised after a
     * failed `_lifecycle_init`). */
    int first_err = 0;
    if (lc->str != 0) {
        hipError_t rc = hipStreamSynchronize((hipStream_t)lc->str);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
        rc = hipStreamDestroy((hipStream_t)lc->str);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
    }
    if (lc->submit != 0) {
        hipError_t rc = hipEventDestroy((hipEvent_t)lc->submit);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
    }
    if (lc->finished != 0) {
        hipError_t rc = hipEventDestroy((hipEvent_t)lc->finished);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
    }
    lc->str = 0;
    lc->submit = 0;
    lc->finished = 0;
    return first_err;
}

int vmaf_hip_kernel_readback_free(VmafHipKernelReadback *rb, VmafHipContext *ctx)
{
    (void)ctx;
    if (rb == NULL) {
        return 0;
    }
    /* NASA P10 r5: rb is non-NULL on entry past the early-return. */
    assert(rb != NULL);

    int first_err = 0;
    if (rb->device != NULL) {
        hipError_t rc = hipFree(rb->device);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
    }
    if (rb->host_pinned != NULL) {
        hipError_t rc = hipHostFree(rb->host_pinned);
        if (rc != hipSuccess && first_err == 0) {
            first_err = hip_rc_to_errno(rc);
        }
    }
    rb->device = NULL;
    rb->host_pinned = NULL;
    rb->bytes = 0;
    return first_err;
}

int vmaf_hip_kernel_submit_post_record(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    if (lc->str == 0 || lc->finished == 0) {
        return -EINVAL;
    }
    /* Record the `finished` event on the private readback stream
     * (`lc->str`) after the DtoH copy is enqueued. `collect_wait`
     * calls `hipStreamSynchronize(lc->str)` which waits for this
     * event to complete, ensuring the pinned host buffer is safe to
     * read. Mirrors the CUDA twin's `vmaf_cuda_kernel_submit_post_record`. */
    hipError_t rc = hipEventRecord((hipEvent_t)lc->finished, (hipStream_t)lc->str);
    return hip_rc_to_errno(rc);
}
