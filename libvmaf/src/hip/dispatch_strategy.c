/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Dispatch strategy stub for the HIP backend (ADR-0212 / T7-10).
 *  Mirrors libvmaf/src/vulkan/dispatch_strategy.c. The runtime PR will
 *  replace this with a feature-name → kernel routing table.
 */

#include "dispatch_strategy.h"

int vmaf_hip_dispatch_supports(const VmafHipContext *ctx, const char *feature)
{
    (void)ctx;
    (void)feature;
    /* TODO: walk a feature-name → hip-kernel registry once kernels exist. */
    return 0;
}
