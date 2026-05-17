/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host header for the integer_motion feature extractor.
 *  Mirrors `libvmaf/src/feature/cuda/integer_motion_cuda.h`.
 *
 *  When `HAVE_HIPCC` is defined (enable_hipcc=true build), the
 *  HSACO fat binary embedded by xxd is exposed here so the host TU
 *  can load it via `hipModuleLoadData`.
 */
#ifndef FEATURE_INTEGER_MOTION_HIP_H_
#define FEATURE_INTEGER_MOTION_HIP_H_

#include <stdint.h>

#ifdef HAVE_HIPCC
/* HSACO fat binary embedded by `xxd -i` (analogous to `motion_score_ptx`
 * in the CUDA twin). Defined in the generated `motion_score_hsaco.c`
 * custom_target output. */
extern const unsigned char motion_score_hsaco[];
extern const unsigned int motion_score_hsaco_len;
#endif /* HAVE_HIPCC */

#endif /* FEATURE_INTEGER_MOTION_HIP_H_ */
