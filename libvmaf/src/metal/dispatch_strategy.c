/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Dispatch support table for the Metal backend (ADR-0421 / T8-1c-k).
 *  Mirrors the CUDA / Vulkan dispatch predicates: callers ask whether
 *  a feature can route to this backend before they bind GPU pictures.
 */

#include "dispatch_strategy.h"

#include <string.h>

static const char *const g_metal_features[] = {
    "motion_v2_metal",
    "VMAF_integer_feature_motion_v2_sad_score",
    "motion2_v2_score",
    "float_psnr_metal",
    "float_psnr",
    "float_moment_metal",
    "float_moment_ref1st",
    "float_moment_dis1st",
    "float_moment_ref2nd",
    "float_moment_dis2nd",
    "float_ansnr_metal",
    "float_ansnr",
    "integer_psnr_metal",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "float_motion_metal",
    "float_motion",
    "integer_motion_metal",
    "VMAF_integer_feature_motion_y_score",
    "motion2_score",
    "motion3_score",
    "float_ssim_metal",
    "float_ssim",
    "float_ms_ssim",
    NULL,
};

int vmaf_metal_dispatch_supports(const VmafMetalContext *ctx, const char *feature)
{
    if (!ctx || !feature)
        return 0;

    for (size_t i = 0; g_metal_features[i]; ++i) {
        if (strcmp(feature, g_metal_features[i]) == 0)
            return 1;
    }
    return 0;
}
