/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Dispatch support table for the HIP backend (ADR-0212 / T7-10).
 *  Lists every vmaf_fex_*_hip extractor name that is registered in
 *  feature_extractor.c under HAVE_HIP.  Update this table whenever a
 *  new HIP kernel is added (same PR as the kernel itself).
 *  Audit invariant: see libvmaf/src/hip/AGENTS.md.
 */

#include "dispatch_strategy.h"

#include <string.h>

/*
 * Every entry is the .name field of a registered vmaf_fex_*_hip
 * extractor.  Cross-reference against feature/hip/ before editing.
 */
static const char *const g_hip_features[] = {
    /* ADR-0241: integer_psnr_hip */
    "psnr_hip",
    /* ADR-0254: float_psnr_hip */
    "float_psnr_hip",
    /* ADR-0257 / ADR-0259: ciede_hip */
    "ciede_hip",
    /* ADR-0258 / ADR-0260: float_moment_hip */
    "float_moment_hip",
    /* ADR-0266: float_ansnr_hip */
    "float_ansnr_hip",
    /* ADR-0267: integer_motion_v2_hip */
    "motion_v2_hip",
    /* ADR-0273: float_motion_hip */
    "float_motion_hip",
    /* ADR-0274: float_ssim_hip */
    "float_ssim_hip",
    NULL,
};

int vmaf_hip_dispatch_supports(const VmafHipContext *ctx, const char *feature)
{
    if (!ctx || !feature)
        return 0;

    for (size_t i = 0; g_hip_features[i]; ++i) {
        if (strcmp(feature, g_hip_features[i]) == 0)
            return 1;
    }
    return 0;
}
