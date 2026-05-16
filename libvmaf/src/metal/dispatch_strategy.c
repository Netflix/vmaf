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

/*
 * Every entry is either the extractor .name string or one of the
 * feature keys listed in that extractor's provided_features[].
 * Cross-reference against feature/metal/ before editing.
 * Audit invariant: see libvmaf/src/metal/AGENTS.md.
 */
static const char *const g_metal_features[] = {
    /* integer_motion_v2_metal (.name = "motion_v2_metal") */
    "motion_v2_metal",
    "VMAF_integer_feature_motion_v2_sad_score",
    "VMAF_integer_feature_motion2_v2_score",
    /* float_psnr_metal (.name = "float_psnr_metal") */
    "float_psnr_metal",
    "float_psnr",
    /* float_moment_metal (.name = "float_moment_metal") */
    "float_moment_metal",
    "float_moment_ref1st",
    "float_moment_dis1st",
    "float_moment_ref2nd",
    "float_moment_dis2nd",
    /* float_ansnr_metal (.name = "float_ansnr_metal") */
    "float_ansnr_metal",
    "float_ansnr",
    "float_anpsnr",
    /* integer_psnr_metal (.name = "integer_psnr_metal") */
    "integer_psnr_metal",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    /* float_motion_metal (.name = "float_motion_metal") */
    "float_motion_metal",
    "VMAF_feature_motion_score",
    "VMAF_feature_motion2_score",
    /* integer_motion_metal (.name = "integer_motion_metal") */
    "integer_motion_metal",
    "VMAF_integer_feature_motion_y_score",
    "VMAF_integer_feature_motion2_score",
    /* float_ssim_metal (.name = "float_ssim_metal") */
    "float_ssim_metal",
    "float_ssim",
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
