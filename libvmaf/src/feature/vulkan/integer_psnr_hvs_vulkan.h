/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_psnr_hvs feature extractor on the Vulkan backend.
 *  Port of feature/cuda/integer_psnr_hvs_cuda.c to Vulkan.
 *  Public header — declares the feature extractor struct exported
 *  from integer_psnr_hvs_vulkan.c and consumed by feature_extractor.c.
 */
#ifndef FEATURE_INTEGER_PSNR_HVS_VULKAN_H_
#define FEATURE_INTEGER_PSNR_HVS_VULKAN_H_

#include "feature_extractor.h"

extern VmafFeatureExtractor vmaf_fex_integer_psnr_hvs_vulkan;

#endif /* FEATURE_INTEGER_PSNR_HVS_VULKAN_H_ */
