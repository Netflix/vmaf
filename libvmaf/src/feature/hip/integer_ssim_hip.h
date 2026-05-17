/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the integer_ssim feature extractor — eleventh
 *  kernel-template consumer.
 *
 *  Mirrors `libvmaf/src/feature/cuda/integer_ssim_cuda.h`. The HIP
 *  kernel artefact (`integer_ssim_score.hip`) is compiled by hipcc to
 *  a HSACO fat binary and embedded as a C byte array when
 *  `enable_hipcc=true`. The host code loads it via
 *  `hipModuleLoadData` + `hipModuleLaunchKernel`.
 *
 *  When `enable_hipcc=false` (default, e.g. on a non-ROCm CI agent),
 *  the symbol is not available and the feature init() returns -ENOSYS,
 *  identical to the pre-runtime scaffold posture.
 */
#ifndef FEATURE_INTEGER_SSIM_HIP_H_
#define FEATURE_INTEGER_SSIM_HIP_H_

#include <stdint.h>

#ifdef HAVE_HIPCC
/* HSACO fat binary embedded by xxd -i (analogous to `ssim_score_ptx`
 * in the CUDA twin). The array is defined in the generated
 * `integer_ssim_score_hsaco.c` custom_target output. */
extern const unsigned char integer_ssim_score_hsaco[];
extern const unsigned int integer_ssim_score_hsaco_len;
#endif /* HAVE_HIPCC */

#endif /* FEATURE_INTEGER_SSIM_HIP_H_ */
