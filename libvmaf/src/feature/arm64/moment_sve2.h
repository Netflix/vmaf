/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  SVE2 dispatch for `float_moment`.  See ADR-0461.
 *
 *  Runtime gate: VMAF_ARM_CPU_FLAG_SVE2 (getauxval(AT_HWCAP2) & HWCAP2_SVE2).
 *  Build gate:   HAVE_SVE2 + -march=armv9-a+sve2.
 *  Darwin opt-out: ADR-0419 — the probe in arm/cpu.c is __linux__-gated
 *  so this flag is always 0 on Apple Silicon regardless of chip capability.
 */

#ifndef LIBVMAF_FEATURE_ARM64_MOMENT_SVE2_H_
#define LIBVMAF_FEATURE_ARM64_MOMENT_SVE2_H_

int compute_1st_moment_sve2(const float *pic, int w, int h, int stride, double *score);
int compute_2nd_moment_sve2(const float *pic, int w, int h, int stride, double *score);

#endif /* LIBVMAF_FEATURE_ARM64_MOMENT_SVE2_H_ */
