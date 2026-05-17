/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  aarch64 SVE2 port of compute_1st_moment / compute_2nd_moment for the
 *  float_moment feature extractor (ADR-0461).
 *
 *  Bit-exactness contract (ADR-0138 / ADR-0179):
 *  Per-row accumulation uses svfloat64_t so every f32 sample is widened to
 *  f64 before summation, bounding the cross-lane ULP error to the double
 *  precision range.  The pattern is vector-length-agnostic (VLA): the inner
 *  loop steps by `svcntd()` f32 elements per iteration (equal to the number
 *  of f64 lanes per register), using `svwhilelt_b64` to construct a predicate
 *  that covers exactly those elements and `svcvt_f64_f32_x` to widen them.
 *
 *  Why `svcntd()` step and `svwhilelt_b64`:
 *    - `svcvt_f64_f32_x(pg64, vf32)` widens the lower `svcntd()` f32 lanes of
 *      `vf32` into f64 under the b64 predicate.  The `_x` suffix leaves
 *      inactive-lane results undefined, so the same `pg64` must govern the
 *      subsequent `svadd_f64_x` to prevent undefined values from entering
 *      `dsum`.
 *    - `svwhilelt_b64(j, w)` creates a b64 predicate that is active for
 *      elements `j..min(j+svcntd()-1, w-1)` — exactly the range we convert.
 *    - The load uses `svwhilelt_b32(j, w)` so no f32 access goes out of
 *      bounds; we then only convert and accumulate the lower `svcntd()` of
 *      those f32 lanes via the b64 predicate.
 *    - Do NOT change step to `svcntw()`: that would skip the upper half of
 *      the f32 register on SVE2 hardware with registers wider than 128 bits.
 *
 *  Darwin opt-out: ADR-0419.  The runtime gate in arm/cpu.c is
 *  `__linux__`-gated so VMAF_ARM_CPU_FLAG_SVE2 is never set on Apple Silicon
 *  regardless of chip capability.  The meson build gate mirrors
 *  `is_sve2_supported` which is forced false on Darwin.
 */

#include <arm_sve.h>
#include <assert.h>
#include <stddef.h>

#include "moment_sve2.h"

#pragma STDC FP_CONTRACT OFF

int compute_1st_moment_sve2(const float *pic, int w, int h, int stride, double *score)
{
    assert(pic != NULL);
    assert(score != NULL);
    assert(w > 0);
    assert(h > 0);

    const int stride_f = stride / (int)sizeof(float);
    /* step: svcntd() == number of f64 lanes per register == half the f32 lanes.
     * svcvt_f64_f32_x widens the lower svcntd() f32 lanes only; stepping by
     * svcntd() ensures each iteration's f32 data occupies the lower position.
     * Do NOT change to svcntw() — that skips the upper half on wide hardware. */
    const int step = (int)svcntd();
    double cum = 0.0;

    for (int i = 0; i < h; ++i) {
        const float *row = pic + (size_t)i * (size_t)stride_f;
        svfloat64_t dsum = svdup_f64(0.0);
        int j = 0;

        while (j < w) {
            /* b64 predicate: active for elements j..min(j+svcntd()-1, w-1).
             * Governs both the conversion and the add so no undefined result
             * from svcvt_f64_f32_x (_x suffix) enters dsum. */
            const svbool_t pg64 = svwhilelt_b64((uint64_t)j, (uint64_t)w);
            /* b32 predicate for the load: same range but for 32-bit elements.
             * svwhilelt_b32(j, w) covers min(svcntw(), w-j) f32 elements;
             * we then only convert the lower svcntd() of those. */
            const svbool_t pg32 = svwhilelt_b32((uint32_t)j, (uint32_t)w);
            const svfloat32_t vf32 = svld1_f32(pg32, row + j);
            const svfloat64_t vf64 = svcvt_f64_f32_x(pg64, vf32);
            dsum = svadd_f64_x(pg64, dsum, vf64);
            j += step;
        }

        cum += svaddv_f64(svptrue_b64(), dsum);
    }

    cum /= (double)w * (double)h;
    *score = cum;
    return 0;
}

int compute_2nd_moment_sve2(const float *pic, int w, int h, int stride, double *score)
{
    assert(pic != NULL);
    assert(score != NULL);
    assert(w > 0);
    assert(h > 0);

    const int stride_f = stride / (int)sizeof(float);
    const int step = (int)svcntd();
    double cum = 0.0;

    for (int i = 0; i < h; ++i) {
        const float *row = pic + (size_t)i * (size_t)stride_f;
        svfloat64_t dsum = svdup_f64(0.0);
        int j = 0;

        while (j < w) {
            const svbool_t pg64 = svwhilelt_b64((uint64_t)j, (uint64_t)w);
            const svbool_t pg32 = svwhilelt_b32((uint32_t)j, (uint32_t)w);
            const svfloat32_t vf32 = svld1_f32(pg32, row + j);
            /* Square in f32 before widening: mirrors the NEON sibling which
             * squares in f32 then widens to f64 (ADR-0179).  Active-only
             * multiply under pg32; inactive lanes' f32 values are undefined
             * but svcvt_f64_f32_x is governed by pg64 which covers the same
             * or fewer elements. */
            const svfloat32_t vsq = svmul_f32_x(pg32, vf32, vf32);
            const svfloat64_t vf64 = svcvt_f64_f32_x(pg64, vsq);
            dsum = svadd_f64_x(pg64, dsum, vf64);
            j += step;
        }

        cum += svaddv_f64(svptrue_b64(), dsum);
    }

    cum /= (double)w * (double)h;
    *score = cum;
    return 0;
}
