/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Vulkan dispatch_strategy stub. Today every extractor returns
 *  PRIMARY_CMDBUF; this TU exposes the registry-aware decision
 *  surface so future secondary-cmdbuf-reuse work for ADM (16
 *  dispatches/frame) can land without touching the registration
 *  sites. See ADR-0181.
 */
#include "dispatch_strategy.h"
#include "../gpu_dispatch_parse.h"

#include <stdlib.h>

/* Strategy name table — index matches VmafVulkanDispatchStrategy enum values:
 *   0 → VMAF_VULKAN_DISPATCH_PRIMARY_CMDBUF
 *   1 → VMAF_VULKAN_DISPATCH_SECONDARY_CMDBUF_REUSE
 * See ADR-0483. */
static const char *const k_vulkan_strategy_names[] = {
    "primary", /* VMAF_VULKAN_DISPATCH_PRIMARY_CMDBUF        = 0 */
    "reuse",   /* VMAF_VULKAN_DISPATCH_SECONDARY_CMDBUF_REUSE = 1 */
    NULL,
};

VmafVulkanDispatchStrategy vmaf_vulkan_select_strategy(const char *feature_name,
                                                       const VmafFeatureCharacteristics *chars,
                                                       unsigned frame_w, unsigned frame_h)
{
    (void)chars;
    (void)frame_w;
    (void)frame_h;

    const char *env_disp = getenv("VMAF_VULKAN_DISPATCH");
    int idx = (int)VMAF_VULKAN_DISPATCH_PRIMARY_CMDBUF;
    if (vmaf_gpu_dispatch_parse_env(env_disp, feature_name, k_vulkan_strategy_names, &idx))
        return (VmafVulkanDispatchStrategy)idx;

    /* Stub default — PRIMARY_CMDBUF for every feature. Secondary-
     * cmdbuf reuse is a follow-up PR (see ADR-0181 § Consequences). */
    return VMAF_VULKAN_DISPATCH_PRIMARY_CMDBUF;
}
