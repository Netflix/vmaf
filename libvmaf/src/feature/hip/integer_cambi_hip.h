/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue header for the CAMBI banding-detection feature extractor.
 *  Direct port of `libvmaf/src/feature/cuda/integer_cambi_cuda.h`
 *  (T3-15 / ADR-0360) to the HIP backend.
 *
 *  The HSACO fat binary is compiled from `integer_cambi/cambi_score.hip`
 *  by hipcc and embedded as a C byte array when `enable_hipcc=true`.
 *  The host code loads it via `hipModuleLoadData` + `hipModuleLaunchKernel`
 *  — the direct HIP module-API analog of the CUDA path's
 *  `cuModuleLoadData` + `cuLaunchKernel`.
 *
 *  When `enable_hipcc=false` (CPU-only build or HIP without a ROCm
 *  toolchain), init() returns -ENOSYS — identical to the scaffold posture
 *  used by all other HIP kernel consumers (ADR-0254 et al.).
 */
#ifndef FEATURE_INTEGER_CAMBI_HIP_H_
#define FEATURE_INTEGER_CAMBI_HIP_H_

#include <stdint.h>

#ifdef HAVE_HIPCC
/* HSACO fat binary embedded by xxd -i during the meson hipcc pipeline.
 * Defined in the auto-generated `cambi_score_hsaco.c` custom_target output. */
extern const unsigned char cambi_score_hsaco[];
extern const unsigned int cambi_score_hsaco_len;
#endif /* HAVE_HIPCC */

#endif /* FEATURE_INTEGER_CAMBI_HIP_H_ */
