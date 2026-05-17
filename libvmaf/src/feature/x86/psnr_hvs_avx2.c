/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/*
 * AVX2 port of calc_psnrhvs / od_bin_fdct8x8.
 *
 * Bit-exactness contract (ADR-0138 / ADR-0139 spirit):
 *
 *   Under FLT_EVAL_METHOD == 0 this TU produces byte-for-byte
 *   identical int32 DCT coefficients and byte-for-byte identical
 *   `double` return values as the scalar reference in
 *   libvmaf/src/feature/third_party/xiph/psnr_hvs.c.
 *
 *   The integer forward-DCT butterfly network is pure signed
 *   int32 arithmetic (adds, subs, `mullo`, unbiased arithmetic
 *   right-shifts). Applying the same 30 ops to 8 independent
 *   columns packed in a single __m256i lane-wise yields, by the
 *   commutativity of SIMD lanes, the same result as running the
 *   scalar butterfly on each column individually.
 *
 *   All float and double accumulations (means, variances, mask,
 *   error) remain scalar, so their left-to-right summation order
 *   and intermediate type promotions match the scalar reference
 *   trivially. That avoids the ADR-0139 per-lane-reduction
 *   gymnastics — simpler, cheaper, and equally bit-exact. The
 *   64-element reductions are not the hot path; the two
 *   `od_bin_fdct8x8` calls are.
 */

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "psnr_hvs_avx2.h"

/* Disable compiler-emitted FMA contraction. The scalar reference is
 * compiled without `-mfma`, so `a + b * c` always executes as two
 * IEEE-754-rounded ops. This TU is compiled with `-mfma` (meson
 * x86_avx2_sources cflags) to enable vector FMAs in the DCT, but
 * `calc_psnrhvs_avx2`'s scalar float accumulators must stay
 * uncontracted to preserve byte-for-byte parity with scalar. */
#pragma STDC FP_CONTRACT OFF

typedef int32_t od_coeff;

/* Mirror of scalar OD_UNBIASED_RSHIFT32 / OD_DCT_RSHIFT across 8 lanes.
 *
 *   Scalar: `(int32_t)(((uint32_t)a >> (32 - b)) + a) >> b` —
 *   unbiased (round-to-nearest-even-ish) arithmetic right shift by
 *   a small positive `b`. The high bit of `a` (the sign) ends up
 *   added as rounding bias, then arith-shift-right rounds toward
 *   -inf. Bit-identical to scalar by construction:
 *
 *     - `srli_epi32(v, 32 - b)` is a logical shift (treats the
 *       vector as uint32 lanes) → matches the `(uint32_t)` cast.
 *     - `add_epi32` wraps on overflow (undefined in C for signed,
 *       but scalar also casts through uint32 here) → matches.
 *     - `srai_epi32(_, b)` is the signed arith shift → matches.
 */
static inline __m256i od_dct_rshift_avx2(__m256i v, int b)
{
    const __m256i bias = _mm256_srli_epi32(v, 32 - b);
    return _mm256_srai_epi32(_mm256_add_epi32(v, bias), b);
}

/* Mirror of the scalar `(x * k + round) >> shift` pattern.
 *
 *   `mullo_epi32` returns the low 32 bits of the signed 32x32
 *   product — same truncation semantics as scalar multiplying two
 *   int32s whose product fits in int32 (which it does for every
 *   DCT butterfly multiplier with 12-bit input). `add_epi32` +
 *   `srai_epi32` match scalar's `+ round` then signed `>> shift`.
 */
static inline __m256i od_mulrshift_avx2(__m256i x, int32_t k, int32_t round, int shift)
{
    const __m256i kv = _mm256_set1_epi32(k);
    const __m256i rv = _mm256_set1_epi32(round);
    const __m256i prod = _mm256_mullo_epi32(x, kv);
    return _mm256_srai_epi32(_mm256_add_epi32(prod, rv), shift);
}

/*
 * Apply the scalar `od_bin_fdct8` butterfly network to 8 columns
 * in parallel. `in0..in7` are the 8 input rows (each an __m256i
 * with one column-element per lane after the pre-butterfly
 * transpose); the outputs are written back via `out0..out7` in
 * natural (row-major within the 8-point butterfly) order.
 *
 * The scalar reference applies a fixed 8-element input permutation
 * before the first butterfly (see `od_bin_fdct8`):
 *   t0=x[0], t4=x[1], t2=x[2], t6=x[3], t7=x[4], t3=x[5], t5=x[6], t1=x[7]
 * and writes `y[0..7] = t0..t7` unchanged. Mirroring that here
 * keeps the line-by-line diff against the scalar source trivially
 * auditable. The byte-for-byte mapping to the scalar ops is
 * one-to-one; line comments mirror the scalar source so that a
 * reviewer can diff them side-by-side.
 *
 * ADR-0141: the 30-butterfly network must be kept together — splitting
 * it would break the one-to-one scalar diff that the bit-exactness
 * audit depends on. The scalar `od_bin_fdct8` has the same structure.
 */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
static inline void od_bin_fdct8_simd(__m256i in0, __m256i in1, __m256i in2, __m256i in3,
                                     __m256i in4, __m256i in5, __m256i in6, __m256i in7,
                                     __m256i *out0, __m256i *out1, __m256i *out2, __m256i *out3,
                                     __m256i *out4, __m256i *out5, __m256i *out6, __m256i *out7)
{
    /* Initial permutation: mirror scalar `od_bin_fdct8` reads. */
    __m256i t0 = in0;
    __m256i t4 = in1;
    __m256i t2 = in2;
    __m256i t6 = in3;
    __m256i t7 = in4;
    __m256i t3 = in5;
    __m256i t5 = in6;
    __m256i t1 = in7;
    __m256i t1h;
    __m256i t4h;
    __m256i t6h;

    /* +1/-1 butterflies. */
    t1 = _mm256_sub_epi32(t0, t1);
    t1h = od_dct_rshift_avx2(t1, 1);
    t0 = _mm256_sub_epi32(t0, t1h);
    t4 = _mm256_add_epi32(t4, t5);
    t4h = od_dct_rshift_avx2(t4, 1);
    t5 = _mm256_sub_epi32(t5, t4h);
    t3 = _mm256_sub_epi32(t2, t3);
    t2 = _mm256_sub_epi32(t2, od_dct_rshift_avx2(t3, 1));
    t6 = _mm256_add_epi32(t6, t7);
    t6h = od_dct_rshift_avx2(t6, 1);
    t7 = _mm256_sub_epi32(t6h, t7);

    /* Embedded 4-point type-II DCT. */
    t0 = _mm256_add_epi32(t0, t6h);
    t6 = _mm256_sub_epi32(t0, t6);
    t2 = _mm256_sub_epi32(t4h, t2);
    t4 = _mm256_sub_epi32(t2, t4);

    /* Embedded 2-point type-II DCT. */
    /* 13573/32768 ~= sqrt(2) - 1 */
    t0 = _mm256_sub_epi32(t0, od_mulrshift_avx2(t4, 13573, 16384, 15));
    /* 11585/16384 ~= sqrt(1/2) */
    t4 = _mm256_add_epi32(t4, od_mulrshift_avx2(t0, 11585, 8192, 14));
    /* 13573/32768 */
    t0 = _mm256_sub_epi32(t0, od_mulrshift_avx2(t4, 13573, 16384, 15));

    /* Embedded 2-point type-IV DST. */
    /* 21895/32768 ~= (1 - cos(3pi/8))/sin(3pi/8) */
    t6 = _mm256_sub_epi32(t6, od_mulrshift_avx2(t2, 21895, 16384, 15));
    /* 15137/16384 ~= sin(3pi/8) */
    t2 = _mm256_add_epi32(t2, od_mulrshift_avx2(t6, 15137, 8192, 14));
    /* 21895/32768 */
    t6 = _mm256_sub_epi32(t6, od_mulrshift_avx2(t2, 21895, 16384, 15));

    /* Embedded 4-point type-IV DST. */
    /* 19195/32768 ~= 2 - sqrt(2) */
    t3 = _mm256_add_epi32(t3, od_mulrshift_avx2(t5, 19195, 16384, 15));
    /* 11585/16384 ~= sqrt(1/2) */
    t5 = _mm256_add_epi32(t5, od_mulrshift_avx2(t3, 11585, 8192, 14));
    /* 7489/8192 ~= sqrt(2) - 1/2 */
    t3 = _mm256_sub_epi32(t3, od_mulrshift_avx2(t5, 7489, 4096, 13));
    t7 = _mm256_sub_epi32(od_dct_rshift_avx2(t5, 1), t7);
    t5 = _mm256_sub_epi32(t5, t7);
    t3 = _mm256_sub_epi32(t1h, t3);
    t1 = _mm256_sub_epi32(t1, t3);
    /* 3227/32768 */
    t7 = _mm256_add_epi32(t7, od_mulrshift_avx2(t1, 3227, 16384, 15));
    /* 6393/32768 */
    t1 = _mm256_sub_epi32(t1, od_mulrshift_avx2(t7, 6393, 16384, 15));
    /* 3227/32768 */
    t7 = _mm256_add_epi32(t7, od_mulrshift_avx2(t1, 3227, 16384, 15));
    /* 2485/8192 */
    t5 = _mm256_add_epi32(t5, od_mulrshift_avx2(t3, 2485, 4096, 13));
    /* 18205/32768 ~= sin(3pi/16) */
    t3 = _mm256_sub_epi32(t3, od_mulrshift_avx2(t5, 18205, 16384, 15));
    /* 2485/8192 */
    t5 = _mm256_add_epi32(t5, od_mulrshift_avx2(t3, 2485, 4096, 13));

    *out0 = t0;
    *out1 = t1;
    *out2 = t2;
    *out3 = t3;
    *out4 = t4;
    *out5 = t5;
    *out6 = t6;
    *out7 = t7;
}

/*
 * Transpose an 8x8 int32 matrix held in 8 __m256i row vectors.
 * Standard 3-stage unpack/permute pattern. After the call, what
 * was row i lane j becomes row j lane i.
 */
static inline void transpose8x8_epi32(__m256i *r0, __m256i *r1, __m256i *r2, __m256i *r3,
                                      __m256i *r4, __m256i *r5, __m256i *r6, __m256i *r7)
{
    const __m256i a0 = _mm256_unpacklo_epi32(*r0, *r1);
    const __m256i a1 = _mm256_unpackhi_epi32(*r0, *r1);
    const __m256i a2 = _mm256_unpacklo_epi32(*r2, *r3);
    const __m256i a3 = _mm256_unpackhi_epi32(*r2, *r3);
    const __m256i a4 = _mm256_unpacklo_epi32(*r4, *r5);
    const __m256i a5 = _mm256_unpackhi_epi32(*r4, *r5);
    const __m256i a6 = _mm256_unpacklo_epi32(*r6, *r7);
    const __m256i a7 = _mm256_unpackhi_epi32(*r6, *r7);

    const __m256i b0 = _mm256_unpacklo_epi64(a0, a2);
    const __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
    const __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
    const __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
    const __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
    const __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
    const __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
    const __m256i b7 = _mm256_unpackhi_epi64(a5, a7);

    *r0 = _mm256_permute2x128_si256(b0, b4, 0x20);
    *r1 = _mm256_permute2x128_si256(b1, b5, 0x20);
    *r2 = _mm256_permute2x128_si256(b2, b6, 0x20);
    *r3 = _mm256_permute2x128_si256(b3, b7, 0x20);
    *r4 = _mm256_permute2x128_si256(b0, b4, 0x31);
    *r5 = _mm256_permute2x128_si256(b1, b5, 0x31);
    *r6 = _mm256_permute2x128_si256(b2, b6, 0x31);
    *r7 = _mm256_permute2x128_si256(b3, b7, 0x31);
}

void od_bin_fdct8x8_avx2(int32_t *y, int32_t ystride, const int32_t *x, int32_t xstride)
{
    assert(y != NULL);
    assert(x != NULL);
    assert(xstride >= 8);
    assert(ystride >= 8);
    const ptrdiff_t xs = (ptrdiff_t)xstride;
    const ptrdiff_t ys = (ptrdiff_t)ystride;
    /* Load 8 rows of x. Each row vector holds 8 int32s. */
    __m256i r0 = _mm256_loadu_si256((const __m256i *)(x + 0 * xs));
    __m256i r1 = _mm256_loadu_si256((const __m256i *)(x + 1 * xs));
    __m256i r2 = _mm256_loadu_si256((const __m256i *)(x + 2 * xs));
    __m256i r3 = _mm256_loadu_si256((const __m256i *)(x + 3 * xs));
    __m256i r4 = _mm256_loadu_si256((const __m256i *)(x + 4 * xs));
    __m256i r5 = _mm256_loadu_si256((const __m256i *)(x + 5 * xs));
    __m256i r6 = _mm256_loadu_si256((const __m256i *)(x + 6 * xs));
    __m256i r7 = _mm256_loadu_si256((const __m256i *)(x + 7 * xs));

    /* First pass: scalar does `z[i] = fdct(col_i_of_x)`. Using the
     * row-loaded layout directly (r_k[lane L] = x[k][L]), lane L of
     * in_k == x[k][L] matches the scalar butterfly's k-th input for
     * column L. After the butterfly, out_1[k][lane L] = z[L][k] —
     * which is "vector k of out_1 holds row k of z^T = column k of
     * z spread across lanes". */
    od_bin_fdct8_simd(r0, r1, r2, r3, r4, r5, r6, r7, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    /* Transpose so that each vector holds a row of z (r'_k[lane j] =
     * z[k][j]). The second butterfly then sees, per lane i, col_i_of_z
     * in the scalar reference order: in_k[lane i] = r'_k[lane i] =
     * z[k][i] = col_i_of_z's k-th element. */
    transpose8x8_epi32(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    /* Second pass: scalar does `y[i] = fdct(col_i_of_z)`. After the
     * butterfly, out_2[k][lane i] = y[i][k]. */
    od_bin_fdct8_simd(r0, r1, r2, r3, r4, r5, r6, r7, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    /* Final transpose: out_2[k][lane i] = y[i][k] holds y in
     * transposed form; transposing yields r_i[lane k] = y[i][k], i.e.
     * row-major y. */
    transpose8x8_epi32(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    _mm256_storeu_si256((__m256i *)(y + 0 * ys), r0);
    _mm256_storeu_si256((__m256i *)(y + 1 * ys), r1);
    _mm256_storeu_si256((__m256i *)(y + 2 * ys), r2);
    _mm256_storeu_si256((__m256i *)(y + 3 * ys), r3);
    _mm256_storeu_si256((__m256i *)(y + 4 * ys), r4);
    _mm256_storeu_si256((__m256i *)(y + 5 * ys), r5);
    _mm256_storeu_si256((__m256i *)(y + 6 * ys), r6);
    _mm256_storeu_si256((__m256i *)(y + 7 * ys), r7);
}

/*
 * Per-block scratch state threaded through the calc_psnrhvs_avx2
 * helpers. Grouping these into one struct keeps the helper
 * signatures readable and matches the scalar reference's locally
 * scoped variables one-to-one.
 */
typedef struct {
    od_coeff dct_s[8 * 8];
    od_coeff dct_d[8 * 8];
    float s_means[4];
    float d_means[4];
    float s_vars[4];
    float d_vars[4];
    float s_gmean;
    float d_gmean;
    float s_gvar;
    float d_gvar;
    float s_mask;
    float d_mask;
} psnr_hvs_block;

/* Load one 8x8 block, compute global + quadrant means in one pass. */
static void load_block_and_means(psnr_hvs_block *b, const unsigned char *src, int systride,
                                 const unsigned char *dst, int dystride, int depth, int x, int y)
{
    b->s_gmean = 0;
    b->d_gmean = 0;
    for (int i = 0; i < 4; i++) {
        b->s_means[i] = 0;
        b->d_means[i] = 0;
        b->s_vars[i] = 0;
        b->d_vars[i] = 0;
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
            if (depth > 8) {
                b->dct_s[i * 8 + j] = src[(y + i) * systride + (j + x) * 2] +
                                      (src[(y + i) * systride + (j + x) * 2 + 1] << 8);
                b->dct_d[i * 8 + j] = dst[(y + i) * dystride + (j + x) * 2] +
                                      (dst[(y + i) * dystride + (j + x) * 2 + 1] << 8);
            } else {
                b->dct_s[i * 8 + j] = src[(y + i) * systride + (j + x)];
                b->dct_d[i * 8 + j] = dst[(y + i) * dystride + (j + x)];
            }
            b->s_gmean += b->dct_s[i * 8 + j];
            b->d_gmean += b->dct_d[i * 8 + j];
            b->s_means[sub] += b->dct_s[i * 8 + j];
            b->d_means[sub] += b->dct_d[i * 8 + j];
        }
    }
    b->s_gmean /= 64.f;
    b->d_gmean /= 64.f;
    for (int i = 0; i < 4; i++) {
        b->s_means[i] /= 16.f;
    }
    for (int i = 0; i < 4; i++) {
        b->d_means[i] /= 16.f;
    }
}

/* Compute global + quadrant variances; fold quadrant-to-global ratio as the
 * scalar reference does at the end. */
static void compute_vars(psnr_hvs_block *b)
{
    b->s_gvar = 0;
    b->d_gvar = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
            b->s_gvar += (b->dct_s[i * 8 + j] - b->s_gmean) * (b->dct_s[i * 8 + j] - b->s_gmean);
            b->d_gvar += (b->dct_d[i * 8 + j] - b->d_gmean) * (b->dct_d[i * 8 + j] - b->d_gmean);
            b->s_vars[sub] +=
                (b->dct_s[i * 8 + j] - b->s_means[sub]) * (b->dct_s[i * 8 + j] - b->s_means[sub]);
            b->d_vars[sub] +=
                (b->dct_d[i * 8 + j] - b->d_means[sub]) * (b->dct_d[i * 8 + j] - b->d_means[sub]);
        }
    }
    b->s_gvar *= 1 / 63.f * 64;
    b->d_gvar *= 1 / 63.f * 64;
    for (int i = 0; i < 4; i++) {
        b->s_vars[i] *= 1 / 15.f * 16;
    }
    for (int i = 0; i < 4; i++) {
        b->d_vars[i] *= 1 / 15.f * 16;
    }
    if (b->s_gvar > 0) {
        b->s_gvar = (b->s_vars[0] + b->s_vars[1] + b->s_vars[2] + b->s_vars[3]) / b->s_gvar;
    }
    if (b->d_gvar > 0) {
        b->d_gvar = (b->d_vars[0] + b->d_vars[1] + b->d_vars[2] + b->d_vars[3]) / b->d_gvar;
    }
}

/* DCT + AC-only mask accumulation; `sqrt` is double-precision to match
 * scalar. d_mask > s_mask fold mirrors the scalar reference. */
static void compute_masks(psnr_hvs_block *b, const float mask[8][8])
{
    od_bin_fdct8x8_avx2(b->dct_s, 8, b->dct_s, 8);
    od_bin_fdct8x8_avx2(b->dct_d, 8, b->dct_d, 8);
    b->s_mask = 0;
    b->d_mask = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = (i == 0); j < 8; j++) {
            b->s_mask += b->dct_s[i * 8 + j] * b->dct_s[i * 8 + j] * mask[i][j];
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = (i == 0); j < 8; j++) {
            b->d_mask += b->dct_d[i * 8 + j] * b->dct_d[i * 8 + j] * mask[i][j];
        }
    }
    /* ADR-0138 bit-exactness fix: scalar reference at third_party/xiph/psnr_hvs.c:351
     * does `sqrt((double)s_mask * s_gvar) / 32.f` — the explicit cast makes the
     * multiplication double-precision before sqrt.  Without the cast, `s_mask *
     * s_gvar` is a float-precision multiply (both operands are float), and the
     * result differs by ~3.77e-7 on the Cb channel at frame 0.
     * Preserve the scalar's (double) cast to maintain byte-for-byte parity. */
    // NOLINTNEXTLINE(performance-type-promotion-in-math-fn) ADR-0138: matches scalar.
    b->s_mask = (float)(sqrt((double)b->s_mask * b->s_gvar) / 32.0);
    // NOLINTNEXTLINE(performance-type-promotion-in-math-fn) ADR-0138: matches scalar.
    b->d_mask = (float)(sqrt((double)b->d_mask * b->d_gvar) / 32.0);
    if (b->d_mask > b->s_mask) {
        b->s_mask = b->d_mask;
    }
}

/* Per-coefficient error accumulation; AC coefficients have the mask-
 * threshold subtraction applied, DC (i==j==0) is compared raw. Adds the
 * 64 per-coefficient contributions directly into `*ret`; increments
 * `*pixels` by 64.
 *
 * ADR-0159 bit-exactness: `*ret` is the cross-block running accumulator
 * (scalar's outer `ret`). Accumulating into a local float here and then
 * adding the per-block total to the caller would change the float
 * summation tree (IEEE-754 add is non-associative) and break byte-for-byte
 * parity with the scalar reference's inline accumulation at
 * third_party/xiph/psnr_hvs.c:355. */
static void accumulate_error(const psnr_hvs_block *b, const float mask[8][8], float csf[8][8],
                             float *ret, int *pixels)
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float err = abs(b->dct_s[i * 8 + j] - b->dct_d[i * 8 + j]);
            if (i != 0 || j != 0) {
                err = err < b->s_mask / mask[i][j] ? 0 : err - b->s_mask / mask[i][j];
            }
            *ret += (err * csf[i][j]) * (err * csf[i][j]);
            (*pixels)++;
        }
    }
}

double calc_psnrhvs_avx2(const unsigned char *src, int systride, const unsigned char *dst,
                         int dystride, double par, int depth, int w, int h, int step,
                         float csf[8][8])
{
    float mask[8][8];
    psnr_hvs_block b;
    float ret = 0;
    int pixels = 0;
    int32_t samplemax;
    (void)par;

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            mask[x][y] = (csf[x][y] * 0.3885746225901003) * (csf[x][y] * 0.3885746225901003);
        }
    }

    for (int y = 0; y < h - 7; y += step) {
        for (int x = 0; x < w - 7; x += step) {
            load_block_and_means(&b, src, systride, dst, dystride, depth, x, y);
            compute_vars(&b);
            compute_masks(&b, mask);
            accumulate_error(&b, mask, csf, &ret, &pixels);
        }
    }
    ret /= pixels;
    samplemax = (1 << depth) - 1;
    ret /= samplemax * samplemax;
    return ret;
}
