/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_HIP_COMMON_H_
#define LIBVMAF_HIP_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scaffolded by ADR-0212 / T7-10 (mirrors the Vulkan scaffold ADR-0175).
 * Replace the stubs in common.c, picture_hip.c, dispatch_strategy.c,
 * and feature/hip/<feature>_hip.c with real HIP runtime implementations.
 */

typedef struct VmafHipContext VmafHipContext;

/* HIP error type. Prefer the real runtime header if present (resolves
 * hipError_t to its native enum); otherwise fall back to a scaffold int
 * typedef so non-HIP TUs still compile. We probe via __has_include so the
 * decision is local to this TU and doesn't depend on macros set by the build
 * (HAVE_HIPCC is build-system level and can be inconsistent across targets,
 * e.g. test executables that include hip/hip_runtime_api.h directly). */
#if defined(__has_include)
#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#define VMAF_HIP_HAS_RUNTIME_HEADER 1
#endif
#endif
#ifndef VMAF_HIP_HAS_RUNTIME_HEADER
typedef int hipError_t;
#endif

int vmaf_hip_context_new(VmafHipContext **ctx, int device_index);
void vmaf_hip_context_destroy(VmafHipContext *ctx);
int vmaf_hip_device_count(void);

/*
 * Translate a HIP error code into a negative POSIX errno. Consolidates
 * 8 identical per-feature private helpers into a single shared implementation.
 * Feature extractors use this instead of defining their own static versions.
 */
int vmaf_hip_rc_to_errno(hipError_t rc);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_HIP_COMMON_H_ */
