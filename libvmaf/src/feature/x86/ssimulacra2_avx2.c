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
 * AVX2 port of the SSIMULACRA 2 pointwise, reduction, and downsampling
 * kernels. The scalar reference is libvmaf/src/feature/ssimulacra2.c
 * — bit-exactness contract per ADR-0161 mirrors ADR-0159/ADR-0160:
 *
 *   - Pointwise float arithmetic is vectorised lane-wise (single-
 *     precision). Because IEEE-754 add/mul/sub are lane-commutative
 *     (same op applied independently to each lane = same result as
 *     scalar per-element), the vector result equals the scalar
 *     result byte-for-byte under FLT_EVAL_METHOD == 0.
 *
 *   - Transcendental calls (`cbrtf`, `powf`) are NOT vectorised.
 *     They are applied per-lane via scalar libm inside a SIMD loop,
 *     matching the scalar reference byte-for-byte. The surrounding
 *     float arithmetic (matmul, bias, scale) is vectorised.
 *
 *   - Reductions over large buffers (ssim_map, edge_diff_map) accumulate
 *     into `double` via per-lane scalar tails (ADR-0139 pattern) to
 *     preserve the scalar summation tree. Vectorising the per-pixel
 *     arithmetic and then reducing lane-parallel would change the
 *     summation order.
 *
 * `#pragma STDC FP_CONTRACT OFF` disables FMA contraction to match
 * the scalar reference compiled without FMA fusion.
 */

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "feature/ssimulacra2_math.h"
#include "ssimulacra2_avx2.h"

#pragma STDC FP_CONTRACT OFF

/* Scalar constants mirrored from ssimulacra2.c. Keeping them local
 * avoids dragging `Ssimu2State` into this TU. */
static const float kM00 = 0.30f;
static const float kM02 = 0.078f;
static const float kM10 = 0.23f;
static const float kM12 = 0.078f;
static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kOpsinBias = 0.0037930732552754493f;
static const float kC2 = 0.0009f;

/* Per-lane scalar cbrtf on an __m256. Applying libm `cbrtf` on each
 * lane preserves byte-for-byte parity with the scalar reference. */
static inline __m256 cbrtf_lane_avx2(__m256 v)
{
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (int k = 0; k < 8; k++) {
        tmp[k] = vmaf_ss2_cbrtf(tmp[k]);
    }
    return _mm256_load_ps(tmp);
}

void ssimulacra2_multiply_3plane_avx2(const float *a, const float *b, float *mul, unsigned w,
                                      unsigned h)
{
    const size_t n = 3u * (size_t)w * (size_t)h;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(mul + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) {
        mul[i] = a[i] * b[i];
    }
}

/* ADR-0141 carve-out: the main body keeps matmul, per-lane cbrtf, and
 * the XYB rescale together for line-for-line diff against the scalar
 * reference. Splitting it would break the bit-exactness audit story. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size) — bit-exactness invariant: splitting would perturb register allocation + reduction order vs scalar (ADR-0138/0139, ADR-0141)
void ssimulacra2_linear_rgb_to_xyb_avx2(const float *lin, float *xyb, unsigned w, unsigned h)
{
    assert(lin != NULL);
    assert(xyb != NULL);
    assert(w > 0 && h > 0);
    const size_t plane_sz = (size_t)w * (size_t)h;
    const float *rp = lin;
    const float *gp = lin + plane_sz;
    const float *bp = lin + 2 * plane_sz;
    float *xp = xyb;
    float *yp = xyb + plane_sz;
    float *bxp = xyb + 2 * plane_sz;

    const float m01 = 1.0f - kM00 - kM02;
    const float m11 = 1.0f - kM10 - kM12;
    const float m22 = 1.0f - kM20 - kM21;
    const float cbrt_bias = vmaf_ss2_cbrtf(kOpsinBias);

    const __m256 vm00 = _mm256_set1_ps(kM00);
    const __m256 vm01 = _mm256_set1_ps(m01);
    const __m256 vm02 = _mm256_set1_ps(kM02);
    const __m256 vm10 = _mm256_set1_ps(kM10);
    const __m256 vm11 = _mm256_set1_ps(m11);
    const __m256 vm12 = _mm256_set1_ps(kM12);
    const __m256 vm20 = _mm256_set1_ps(kM20);
    const __m256 vm21 = _mm256_set1_ps(kM21);
    const __m256 vm22 = _mm256_set1_ps(m22);
    const __m256 vbias = _mm256_set1_ps(kOpsinBias);
    const __m256 vzero = _mm256_setzero_ps();

    /* `cbrtf` applied per-lane via scalar libm to preserve bit-exactness;
     * see `cbrtf_lane_avx2`. */

    size_t i = 0;
    for (; i + 8 <= plane_sz; i += 8) {
        const __m256 r = _mm256_loadu_ps(rp + i);
        const __m256 g = _mm256_loadu_ps(gp + i);
        const __m256 b = _mm256_loadu_ps(bp + i);
        /* LMS mixing matrix + opsin bias. Addition order MUST match
         * scalar left-to-right: ((a + b) + c) + d — IEEE-754 add is
         * non-associative and test_xyb catches the drift. */
        __m256 l = _mm256_add_ps(_mm256_mul_ps(vm00, r), _mm256_mul_ps(vm01, g));
        l = _mm256_add_ps(l, _mm256_mul_ps(vm02, b));
        l = _mm256_add_ps(l, vbias);
        __m256 m = _mm256_add_ps(_mm256_mul_ps(vm10, r), _mm256_mul_ps(vm11, g));
        m = _mm256_add_ps(m, _mm256_mul_ps(vm12, b));
        m = _mm256_add_ps(m, vbias);
        __m256 sv = _mm256_add_ps(_mm256_mul_ps(vm20, r), _mm256_mul_ps(vm21, g));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(vm22, b));
        sv = _mm256_add_ps(sv, vbias);
        /* Clamp to zero below — matches scalar `if (l < 0.0f) l = 0.0f`. */
        l = _mm256_max_ps(l, vzero);
        m = _mm256_max_ps(m, vzero);
        sv = _mm256_max_ps(sv, vzero);

        const __m256 vcbrt_bias = _mm256_set1_ps(cbrt_bias);
        const __m256 L = _mm256_sub_ps(cbrtf_lane_avx2(l), vcbrt_bias);
        const __m256 M = _mm256_sub_ps(cbrtf_lane_avx2(m), vcbrt_bias);
        const __m256 S = _mm256_sub_ps(cbrtf_lane_avx2(sv), vcbrt_bias);

        /* X = 0.5 * (L - M); Y = 0.5 * (L + M); B = S. */
        /* X-channel folds 0.5*(L-M)*14 to (L-M)*7 to mirror scalar -O2's
         * compiler-folded constant; previous mul-by-0.5 then *14 path was
         * not bit-exact with scalar's emitted (L-M)*7. ADR-0138/0139. */
        const __m256 Y = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(L, M));
        const __m256 B = S;

        /* MakePositiveXYB rescale in libjxl order (B uses Y before Y offset). */
        const __m256 Bfinal = _mm256_add_ps(_mm256_sub_ps(B, Y), _mm256_set1_ps(0.55f));
        const __m256 Xfinal = _mm256_add_ps(
            _mm256_mul_ps(_mm256_sub_ps(L, M), _mm256_set1_ps(7.0f)),
            _mm256_set1_ps(0.42f));
        const __m256 Yfinal = _mm256_add_ps(Y, _mm256_set1_ps(0.01f));

        _mm256_storeu_ps(xp + i, Xfinal);
        _mm256_storeu_ps(yp + i, Yfinal);
        _mm256_storeu_ps(bxp + i, Bfinal);
    }

    /* Scalar tail — identical to the scalar reference body. */
    for (; i < plane_sz; i++) {
        float r = rp[i];
        float g = gp[i];
        float bb = bp[i];
        float l = kM00 * r + m01 * g + kM02 * bb + kOpsinBias;
        float m = kM10 * r + m11 * g + kM12 * bb + kOpsinBias;
        float s = kM20 * r + kM21 * g + m22 * bb + kOpsinBias;
        if (l < 0.0f)
            l = 0.0f;
        if (m < 0.0f)
            m = 0.0f;
        if (s < 0.0f)
            s = 0.0f;
        float L = vmaf_ss2_cbrtf(l) - cbrt_bias;
        float M = vmaf_ss2_cbrtf(m) - cbrt_bias;
        float S = vmaf_ss2_cbrtf(s) - cbrt_bias;
        float X = 0.5f * (L - M);
        float Y = 0.5f * (L + M);
        float B = S;
        B = (B - Y) + 0.55f;
        X = X * 14.0f + 0.42f;
        Y = Y + 0.01f;
        xp[i] = X;
        yp[i] = Y;
        bxp[i] = B;
    }
}

void ssimulacra2_downsample_2x2_avx2(const float *in, unsigned iw, unsigned ih, float *out,
                                     unsigned *ow_out, unsigned *oh_out)
{
    const unsigned ow = (iw + 1) / 2;
    const unsigned oh = (ih + 1) / 2;
    *ow_out = ow;
    *oh_out = oh;

    const size_t in_plane = (size_t)iw * (size_t)ih;
    const size_t out_plane = (size_t)ow * (size_t)oh;
    /* The scalar reference clamps ix/iy to (iw-1)/(ih-1) when they
     * exceed the input bounds. Vectorising this requires handling
     * the last column / last row carefully; we stay scalar on those
     * edges and SIMD on the interior. */
    for (int c = 0; c < 3; c++) {
        const float *ip = in + (size_t)c * in_plane;
        float *op = out + (size_t)c * out_plane;
        for (unsigned oy = 0; oy < oh; oy++) {
            const unsigned iy0 = oy * 2;
            const unsigned iy1 = (iy0 + 1 < ih) ? iy0 + 1 : ih - 1;
            const float *row0 = ip + (size_t)iy0 * iw;
            const float *row1 = ip + (size_t)iy1 * iw;
            float *orow = op + (size_t)oy * ow;
            unsigned ox = 0;
            /* SIMD interior: 8 output lanes per iter. To preserve scalar's
             * left-to-right summation order `((r0e + r0o) + r1e) + r1o`,
             * deinterleave even/odd pairs then sum sequentially. */
            const unsigned interior_end = (ow > 0 && iw >= 2) ? ((ow - 1) / 8) * 8 : 0;
            const __m256 vquarter = _mm256_set1_ps(0.25f);
            for (; ox < interior_end; ox += 8) {
                const size_t base = (size_t)ox * 2u;
                const __m256 r00 = _mm256_loadu_ps(row0 + base);
                const __m256 r01 = _mm256_loadu_ps(row0 + base + 8);
                const __m256 r10 = _mm256_loadu_ps(row1 + base);
                const __m256 r11 = _mm256_loadu_ps(row1 + base + 8);
                const __m256 r0e_raw = _mm256_shuffle_ps(r00, r01, 0x88);
                const __m256 r0o_raw = _mm256_shuffle_ps(r00, r01, 0xDD);
                const __m256 r1e_raw = _mm256_shuffle_ps(r10, r11, 0x88);
                const __m256 r1o_raw = _mm256_shuffle_ps(r10, r11, 0xDD);
                const __m256 r0e =
                    _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r0e_raw), 0xD8));
                const __m256 r0o =
                    _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r0o_raw), 0xD8));
                const __m256 r1e =
                    _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r1e_raw), 0xD8));
                const __m256 r1o =
                    _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r1o_raw), 0xD8));
                __m256 acc = _mm256_add_ps(r0e, r0o);
                acc = _mm256_add_ps(acc, r1e);
                acc = _mm256_add_ps(acc, r1o);
                _mm256_storeu_ps(orow + ox, _mm256_mul_ps(acc, vquarter));
            }
            /* Scalar tail for the last-column wraparound + leftover lanes. */
            for (; ox < ow; ox++) {
                unsigned ix0 = ox * 2;
                unsigned ix1 = (ix0 + 1 < iw) ? ix0 + 1 : iw - 1;
                float sum = row0[ix0] + row0[ix1] + row1[ix0] + row1[ix1];
                orow[ox] = sum * 0.25f;
            }
        }
    }
}

static inline double quartic_d(double x)
{
    x *= x;
    return x * x;
}

/* ADR-0141 carve-out: the body interleaves SIMD pointwise arithmetic
 * with a per-lane double-accumulator tail for the two reductions; the
 * two parts are semantically coupled and splitting them would force a
 * second SIMD spill/reload per 8-pixel block. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size) — bit-exactness invariant: splitting would perturb register allocation + reduction order vs scalar (ADR-0138/0139, ADR-0141)
void ssimulacra2_ssim_map_avx2(const float *m1, const float *m2, const float *s11, const float *s22,
                               const float *s12, unsigned w, unsigned h, double plane_averages[6])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    const __m256 vc2 = _mm256_set1_ps(kC2);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vtwo = _mm256_set1_ps(2.0f);

    for (int c = 0; c < 3; c++) {
        double sum_l1 = 0.0;
        double sum_l4 = 0.0;
        const float *rm1 = m1 + (size_t)c * plane;
        const float *rm2 = m2 + (size_t)c * plane;
        const float *rs11 = s11 + (size_t)c * plane;
        const float *rs22 = s22 + (size_t)c * plane;
        const float *rs12 = s12 + (size_t)c * plane;

        size_t i = 0;
        for (; i + 8 <= plane; i += 8) {
            const __m256 mu1 = _mm256_loadu_ps(rm1 + i);
            const __m256 mu2 = _mm256_loadu_ps(rm2 + i);
            const __m256 mu11 = _mm256_mul_ps(mu1, mu1);
            const __m256 mu22 = _mm256_mul_ps(mu2, mu2);
            const __m256 mu12 = _mm256_mul_ps(mu1, mu2);
            const __m256 diff = _mm256_sub_ps(mu1, mu2);
            const __m256 num_m = _mm256_sub_ps(vone, _mm256_mul_ps(diff, diff));
            const __m256 num_s = _mm256_add_ps(
                _mm256_mul_ps(vtwo, _mm256_sub_ps(_mm256_loadu_ps(rs12 + i), mu12)), vc2);
            const __m256 denom_s =
                _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(_mm256_loadu_ps(rs11 + i), mu11),
                                            _mm256_sub_ps(_mm256_loadu_ps(rs22 + i), mu22)),
                              vc2);
            /* Compute d = 1.0 - (num_m * num_s / denom_s) per-lane in double
             * to match scalar's (double)num_m * (double)num_s / (double)denom_s.
             * Spill + per-lane double accumulate for bit-exactness (ADR-0139). */
            alignas(32) float num_m_f[8];
            alignas(32) float num_s_f[8];
            alignas(32) float denom_s_f[8];
            _mm256_store_ps(num_m_f, num_m);
            _mm256_store_ps(num_s_f, num_s);
            _mm256_store_ps(denom_s_f, denom_s);
            for (int k = 0; k < 8; k++) {
                double d = 1.0 - ((double)num_m_f[k] * (double)num_s_f[k] / (double)denom_s_f[k]);
                if (d < 0.0)
                    d = 0.0;
                sum_l1 += d;
                sum_l4 += quartic_d(d);
            }
        }
        /* Scalar tail — identical to scalar reference. */
        for (; i < plane; i++) {
            float mu1 = rm1[i];
            float mu2 = rm2[i];
            float mu11 = mu1 * mu1;
            float mu22 = mu2 * mu2;
            float mu12 = mu1 * mu2;
            float num_m = 1.0f - (mu1 - mu2) * (mu1 - mu2);
            float num_s = 2.0f * (rs12[i] - mu12) + kC2;
            float denom_s = (rs11[i] - mu11) + (rs22[i] - mu22) + kC2;
            double d = 1.0 - ((double)num_m * (double)num_s / (double)denom_s);
            if (d < 0.0)
                d = 0.0;
            sum_l1 += d;
            sum_l4 += quartic_d(d);
        }
        plane_averages[c * 2 + 0] = one_per_pixels * sum_l1;
        plane_averages[c * 2 + 1] = sqrt(sqrt(one_per_pixels * sum_l4));
    }
}

void ssimulacra2_edge_diff_map_avx2(const float *img1, const float *mu1, const float *img2,
                                    const float *mu2, unsigned w, unsigned h,
                                    double plane_averages[12])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    const __m256 vsignmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int c = 0; c < 3; c++) {
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        const float *r1 = img1 + (size_t)c * plane;
        const float *rm1 = mu1 + (size_t)c * plane;
        const float *r2 = img2 + (size_t)c * plane;
        const float *rm2 = mu2 + (size_t)c * plane;

        size_t i = 0;
        for (; i + 8 <= plane; i += 8) {
            const __m256 a1 = _mm256_loadu_ps(r1 + i);
            const __m256 a2 = _mm256_loadu_ps(r2 + i);
            const __m256 am1 = _mm256_loadu_ps(rm1 + i);
            const __m256 am2 = _mm256_loadu_ps(rm2 + i);
            const __m256 d1 = _mm256_and_ps(vsignmask, _mm256_sub_ps(a1, am1));
            const __m256 d2 = _mm256_and_ps(vsignmask, _mm256_sub_ps(a2, am2));
            /* Promote to double per-lane for the divide, matching scalar. */
            alignas(32) float d1f[8];
            alignas(32) float d2f[8];
            _mm256_store_ps(d1f, d1);
            _mm256_store_ps(d2f, d2);
            for (int k = 0; k < 8; k++) {
                double ed1 = (double)d1f[k];
                double ed2 = (double)d2f[k];
                double d = (1.0 + ed2) / (1.0 + ed1) - 1.0;
                double art = d > 0.0 ? d : 0.0;
                double det = d < 0.0 ? -d : 0.0;
                s0 += art;
                s1 += quartic_d(art);
                s2 += det;
                s3 += quartic_d(det);
            }
        }
        /* Scalar tail. */
        for (; i < plane; i++) {
            double ed1 = fabs((double)r1[i] - (double)rm1[i]);
            double ed2 = fabs((double)r2[i] - (double)rm2[i]);
            double d = (1.0 + ed2) / (1.0 + ed1) - 1.0;
            double art = d > 0.0 ? d : 0.0;
            double det = d < 0.0 ? -d : 0.0;
            s0 += art;
            s1 += quartic_d(art);
            s2 += det;
            s3 += quartic_d(det);
        }
        plane_averages[c * 4 + 0] = one_per_pixels * s0;
        plane_averages[c * 4 + 1] = sqrt(sqrt(one_per_pixels * s1));
        plane_averages[c * 4 + 2] = one_per_pixels * s2;
        plane_averages[c * 4 + 3] = sqrt(sqrt(one_per_pixels * s3));
    }
}

/*
 * FastGaussian separable IIR — horizontal pass.
 *
 * Scalar `fast_gaussian_1d` is a 3-pole serial recurrence per row;
 * the recurrence can't vectorise within a single row. We batch 8
 * rows in parallel: each AVX2 vector lane holds a column of the 8
 * rows being processed together.
 *
 * Data layout: scalar input is row-major (`in[y * w + x]`). To
 * feed 8 rows into SIMD lanes, we use `vpgatherdd`-style strided
 * loads — lane i reads `in[(y_batch + i) * w + x]` via a stride-w
 * index vector. Output is written lane-by-lane (AVX2 has no
 * scatter) via a spill + 8 scalar stores per iteration. The
 * per-lane IIR state vectors hold prev1/prev2 for 8 independent
 * rows in lock-step.
 *
 * Bit-exact to scalar: each lane computes the same recurrence on
 * the same row of input, in the same left-to-right order. IEEE-754
 * lane-commutative arithmetic yields byte-identical results.
 */
/* ADR-0141 carve-out: gather loads + 3-pole IIR recurrence + scatter-via-
 * scalar-store kept together for line-for-line scalar diff. Splitting would
 * spill the IIR state across call boundaries and break the bit-exact audit
 * vs the scalar reference in `fast_gaussian_1d`. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size) — bit-exactness invariant: splitting would perturb register allocation + reduction order vs scalar (ADR-0138/0139, ADR-0141)
static void hblur_8rows_avx2(const float rg_n2[3], const float rg_d1[3], int rg_radius,
                             const float *in, float *out, unsigned w, unsigned y_base,
                             unsigned row_count)
{
    const ptrdiff_t N = (ptrdiff_t)rg_radius;
    const ptrdiff_t W = (ptrdiff_t)w;

    /* Per-lane gather index vector: lane i reads at offset i * w. */
    int32_t idx_tmp[8];
    for (int i = 0; i < 8; i++) {
        idx_tmp[i] = (i < (int)row_count) ? (int32_t)((y_base + (unsigned)i) * w) : 0;
    }
    const __m256i vindices = _mm256_loadu_si256((const __m256i *)idx_tmp);

    /* Mask: lanes beyond `row_count` should not contribute to output. */
    const int32_t lane_valid = (row_count >= 8) ? 0 : (int32_t)row_count;
    (void)lane_valid;

    const __m256 vn2_0 = _mm256_set1_ps(rg_n2[0]);
    const __m256 vn2_1 = _mm256_set1_ps(rg_n2[1]);
    const __m256 vn2_2 = _mm256_set1_ps(rg_n2[2]);
    const __m256 vd1_0 = _mm256_set1_ps(rg_d1[0]);
    const __m256 vd1_1 = _mm256_set1_ps(rg_d1[1]);
    const __m256 vd1_2 = _mm256_set1_ps(rg_d1[2]);

    __m256 prev1_0 = _mm256_setzero_ps();
    __m256 prev1_1 = _mm256_setzero_ps();
    __m256 prev1_2 = _mm256_setzero_ps();
    __m256 prev2_0 = _mm256_setzero_ps();
    __m256 prev2_1 = _mm256_setzero_ps();
    __m256 prev2_2 = _mm256_setzero_ps();

    alignas(32) float store_tmp[8];

    for (ptrdiff_t n = -N + 1; n < W; n++) {
        const ptrdiff_t left = n - N - 1;
        const ptrdiff_t right = n + N - 1;
        const __m256 lv = (left >= 0) ? _mm256_i32gather_ps(in + left, vindices, sizeof(float)) :
                                        _mm256_setzero_ps();
        const __m256 rv = (right < W) ? _mm256_i32gather_ps(in + right, vindices, sizeof(float)) :
                                        _mm256_setzero_ps();
        const __m256 sum = _mm256_add_ps(lv, rv);

        /* Scalar order: n2_k * sum - d1_k * prev1_k - prev2_k. */
        __m256 o0 = _mm256_sub_ps(_mm256_mul_ps(vn2_0, sum), _mm256_mul_ps(vd1_0, prev1_0));
        o0 = _mm256_sub_ps(o0, prev2_0);
        __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(vn2_1, sum), _mm256_mul_ps(vd1_1, prev1_1));
        o1 = _mm256_sub_ps(o1, prev2_1);
        __m256 o2 = _mm256_sub_ps(_mm256_mul_ps(vn2_2, sum), _mm256_mul_ps(vd1_2, prev1_2));
        o2 = _mm256_sub_ps(o2, prev2_2);

        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            /* Scalar order: (o0 + o1) + o2. */
            const __m256 res = _mm256_add_ps(_mm256_add_ps(o0, o1), o2);
            _mm256_store_ps(store_tmp, res);
            for (unsigned i = 0; i < row_count; i++) {
                out[((size_t)y_base + i) * w + (size_t)n] = store_tmp[i];
            }
        }
    }
}

/*
 * FastGaussian separable IIR — vertical pass.
 *
 * The per-column IIR state (`prev1_*`, `prev2_*`) already lives in
 * contiguous 6 × w float arrays (`col_state`), so 8 consecutive
 * columns' state load/store naturally with aligned vectors. No
 * gather required.
 *
 * Bit-exact to scalar: same summation order, lane-commutative.
 */
/* ADR-0141 carve-out: SIMD main loop + scalar tail share the same IIR
 * state arrays and must stay in one function to preserve the scalar
 * per-column state-update ordering. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size) — bit-exactness invariant: splitting would perturb register allocation + reduction order vs scalar (ADR-0138/0139, ADR-0141)
static void vblur_simd_8cols_avx2(const float rg_n2[3], const float rg_d1[3], int rg_radius,
                                  float *col_state, const float *in, float *out, unsigned w,
                                  unsigned h)
{
    const size_t xsize = (size_t)w;
    float *prev1_0 = col_state + 0u * xsize;
    float *prev1_1 = col_state + 1u * xsize;
    float *prev1_2 = col_state + 2u * xsize;
    float *prev2_0 = col_state + 3u * xsize;
    float *prev2_1 = col_state + 4u * xsize;
    float *prev2_2 = col_state + 5u * xsize;
    memset(col_state, 0, 6u * xsize * sizeof(float));

    const __m256 vn2_0 = _mm256_set1_ps(rg_n2[0]);
    const __m256 vn2_1 = _mm256_set1_ps(rg_n2[1]);
    const __m256 vn2_2 = _mm256_set1_ps(rg_n2[2]);
    const __m256 vd1_0 = _mm256_set1_ps(rg_d1[0]);
    const __m256 vd1_1 = _mm256_set1_ps(rg_d1[1]);
    const __m256 vd1_2 = _mm256_set1_ps(rg_d1[2]);

    const ptrdiff_t N = (ptrdiff_t)rg_radius;
    const ptrdiff_t ysize = (ptrdiff_t)h;

    for (ptrdiff_t n = -N + 1; n < ysize; n++) {
        const ptrdiff_t left = n - N - 1;
        const ptrdiff_t right = n + N - 1;
        const float *lrow = (left >= 0) ? (in + (size_t)left * xsize) : NULL;
        const float *rrow = (right < ysize) ? (in + (size_t)right * xsize) : NULL;
        float *orow = (n >= 0) ? (out + (size_t)n * xsize) : NULL;

        size_t x = 0;
        for (; x + 8 <= xsize; x += 8) {
            const __m256 lv = lrow ? _mm256_loadu_ps(lrow + x) : _mm256_setzero_ps();
            const __m256 rv = rrow ? _mm256_loadu_ps(rrow + x) : _mm256_setzero_ps();
            const __m256 sum = _mm256_add_ps(lv, rv);
            const __m256 p1_0 = _mm256_loadu_ps(prev1_0 + x);
            const __m256 p1_1 = _mm256_loadu_ps(prev1_1 + x);
            const __m256 p1_2 = _mm256_loadu_ps(prev1_2 + x);
            const __m256 p2_0 = _mm256_loadu_ps(prev2_0 + x);
            const __m256 p2_1 = _mm256_loadu_ps(prev2_1 + x);
            const __m256 p2_2 = _mm256_loadu_ps(prev2_2 + x);
            __m256 o0 = _mm256_sub_ps(_mm256_mul_ps(vn2_0, sum), _mm256_mul_ps(vd1_0, p1_0));
            o0 = _mm256_sub_ps(o0, p2_0);
            __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(vn2_1, sum), _mm256_mul_ps(vd1_1, p1_1));
            o1 = _mm256_sub_ps(o1, p2_1);
            __m256 o2 = _mm256_sub_ps(_mm256_mul_ps(vn2_2, sum), _mm256_mul_ps(vd1_2, p1_2));
            o2 = _mm256_sub_ps(o2, p2_2);
            _mm256_storeu_ps(prev2_0 + x, p1_0);
            _mm256_storeu_ps(prev2_1 + x, p1_1);
            _mm256_storeu_ps(prev2_2 + x, p1_2);
            _mm256_storeu_ps(prev1_0 + x, o0);
            _mm256_storeu_ps(prev1_1 + x, o1);
            _mm256_storeu_ps(prev1_2 + x, o2);
            if (orow) {
                const __m256 res = _mm256_add_ps(_mm256_add_ps(o0, o1), o2);
                _mm256_storeu_ps(orow + x, res);
            }
        }
        /* Scalar tail — identical to scalar reference body. */
        for (; x < xsize; x++) {
            const float lv = lrow ? lrow[x] : 0.f;
            const float rv = rrow ? rrow[x] : 0.f;
            const float sum = lv + rv;
            const float o0 = rg_n2[0] * sum - rg_d1[0] * prev1_0[x] - prev2_0[x];
            const float o1 = rg_n2[1] * sum - rg_d1[1] * prev1_1[x] - prev2_1[x];
            const float o2 = rg_n2[2] * sum - rg_d1[2] * prev1_2[x] - prev2_2[x];
            prev2_0[x] = prev1_0[x];
            prev2_1[x] = prev1_1[x];
            prev2_2[x] = prev1_2[x];
            prev1_0[x] = o0;
            prev1_1[x] = o1;
            prev1_2[x] = o2;
            if (orow) {
                orow[x] = o0 + o1 + o2;
            }
        }
    }
}

void ssimulacra2_blur_plane_avx2(const float rg_n2[3], const float rg_d1[3], int rg_radius,
                                 float *col_state, const float *in, float *out, float *scratch,
                                 unsigned w, unsigned h)
{
    assert(col_state != NULL);
    assert(in != NULL);
    assert(out != NULL);
    assert(scratch != NULL);
    assert(w > 0 && h > 0);

    /* Horizontal pass: 8-row batches through `scratch`. */
    unsigned y = 0;
    for (; y + 8 <= h; y += 8) {
        hblur_8rows_avx2(rg_n2, rg_d1, rg_radius, in, scratch, w, y, 8);
    }
    if (y < h) {
        /* Tail: fewer than 8 rows. Still use the 8-wide path with a
         * masked-store row_count — bit-exact because lanes beyond
         * `row_count` don't write to output. */
        hblur_8rows_avx2(rg_n2, rg_d1, rg_radius, in, scratch, w, y, h - y);
    }

    /* Vertical pass: 8-column SIMD with scalar tail. */
    vblur_simd_8cols_avx2(rg_n2, rg_d1, rg_radius, col_state, scratch, out, w, h);
}

/*
 * YUV → linear RGB SIMD port (ADR-0163).
 *
 * Strategy: per-lane scalar `read_plane`-equivalent reads (handles
 * all chroma subsampling ratios + 8/16-bit uniformly), SIMD matmul
 * + normalise + clamp, per-lane scalar `srgb_to_linear` (branch +
 * libm `powf` — bit-exact to scalar reference).
 *
 * The scalar-read-per-lane pattern mirrors the ADR-0161 `cbrtf_lane`
 * approach: SIMD arithmetic is the win; the libm transcendental
 * and the pixel-format dispatch stay scalar to preserve
 * bit-exactness and handle arbitrary chroma ratios.
 */

/* Scalar pixel read with bounds clamping — mirror of scalar read_plane. */
static inline float read_plane_scalar_s2(const simd_plane_t *p, unsigned lw, unsigned lh, int x,
                                         int y, unsigned bpc)
{
    const unsigned pw = p->w;
    const unsigned ph = p->h;
    int sx;
    int sy;
    if (pw == lw) {
        sx = x;
    } else if (pw * 2 == lw) {
        sx = x >> 1;
    } else {
        sx = (int)((int64_t)x * (int64_t)pw / (int64_t)lw);
    }
    if (ph == lh) {
        sy = y;
    } else if (ph * 2 == lh) {
        sy = y >> 1;
    } else {
        sy = (int)((int64_t)y * (int64_t)ph / (int64_t)lh);
    }
    if (sx < 0)
        sx = 0;
    if (sy < 0)
        sy = 0;
    if ((unsigned)sx >= pw)
        sx = (int)pw - 1;
    if ((unsigned)sy >= ph)
        sy = (int)ph - 1;

    if (bpc > 8) {
        const uint16_t *row = (const uint16_t *)((const uint8_t *)p->data + (size_t)sy * p->stride);
        return (float)row[sx];
    }
    const uint8_t *row = (const uint8_t *)p->data + (size_t)sy * p->stride;
    return (float)row[sx];
}

/* Per-lane scalar sRGB EOTF. Branch + `powf` per lane; bit-exact to
 * the scalar reference's `srgb_to_linear`. */
static inline __m256 srgb_to_linear_lane_avx2(__m256 v)
{
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (int k = 0; k < 8; k++) {
        const float x = tmp[k];
        tmp[k] = vmaf_ss2_srgb_eotf(x);
    }
    return _mm256_load_ps(tmp);
}

/* Extract YUV→RGB matrix coefficients for a given enum value. The
 * scalar reference uses the same formula; we materialise the
 * derived coefficients here so SIMD callers avoid the scalar
 * switch inside the hot loop. */
static inline void compute_matrix_coefs(int yuv_matrix, float *kr_out, float *kg_out, float *kb_out,
                                        int *limited_out)
{
    switch (yuv_matrix) {
    case 2: /* BT709_FULL */
        *limited_out = 0;
        *kr_out = 0.2126f;
        *kg_out = 0.7152f;
        *kb_out = 0.0722f;
        break;
    case 0: /* BT709_LIMITED */
        *limited_out = 1;
        *kr_out = 0.2126f;
        *kg_out = 0.7152f;
        *kb_out = 0.0722f;
        break;
    case 3: /* BT601_FULL */
        *limited_out = 0;
        *kr_out = 0.299f;
        *kg_out = 0.587f;
        *kb_out = 0.114f;
        break;
    case 1: /* BT601_LIMITED */
    default:
        *limited_out = 1;
        *kr_out = 0.299f;
        *kg_out = 0.587f;
        *kb_out = 0.114f;
        break;
    }
}

/* ADR-0141 carve-out: the main loop ties scalar pixel reads + SIMD
 * matmul + per-lane scalar sRGB EOTF together for the line-for-line
 * scalar diff. Splitting would force per-iteration spills. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size) — bit-exactness invariant: splitting would perturb register allocation + reduction order vs scalar (ADR-0138/0139, ADR-0141)
void ssimulacra2_picture_to_linear_rgb_avx2(int yuv_matrix, unsigned bpc, unsigned w, unsigned h,
                                            const simd_plane_t planes[3], float *out)
{
    assert(planes != NULL);
    assert(out != NULL);
    assert(w > 0 && h > 0);

    const size_t plane_sz = (size_t)w * (size_t)h;
    float *rp = out;
    float *gp = out + plane_sz;
    float *bp = out + 2 * plane_sz;

    const float peak = (float)((1u << bpc) - 1u);
    const float inv_peak = 1.0f / peak;

    float kr;
    float kg;
    float kb;
    int limited;
    compute_matrix_coefs(yuv_matrix, &kr, &kg, &kb, &limited);

    /* Derived coefficients mirror the scalar reference. */
    const float cr_r = 2.0f * (1.0f - kr);
    const float cb_b = 2.0f * (1.0f - kb);
    const float cb_g = -(2.0f * kb * (1.0f - kb)) / kg;
    const float cr_g = -(2.0f * kr * (1.0f - kr)) / kg;

    const float y_scale = limited ? (255.0f / 219.0f) : 1.0f;
    const float c_scale = limited ? (255.0f / 224.0f) : 1.0f;
    const float y_off = limited ? (16.0f / 255.0f) : 0.0f;
    const float c_off = 0.5f;

    const __m256 vinv_peak = _mm256_set1_ps(inv_peak);
    const __m256 vy_scale = _mm256_set1_ps(y_scale);
    const __m256 vc_scale = _mm256_set1_ps(c_scale);
    const __m256 vy_off = _mm256_set1_ps(y_off);
    const __m256 vc_off = _mm256_set1_ps(c_off);
    const __m256 vcr_r = _mm256_set1_ps(cr_r);
    const __m256 vcb_b = _mm256_set1_ps(cb_b);
    const __m256 vcb_g = _mm256_set1_ps(cb_g);
    const __m256 vcr_g = _mm256_set1_ps(cr_g);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);

    alignas(32) float y_tmp[8];
    alignas(32) float u_tmp[8];
    alignas(32) float v_tmp[8];

    for (unsigned y = 0; y < h; y++) {
        unsigned x = 0;
        for (; x + 8 <= w; x += 8) {
            for (int i = 0; i < 8; i++) {
                y_tmp[i] =
                    read_plane_scalar_s2(&planes[0], w, h, (int)(x + (unsigned)i), (int)y, bpc);
                u_tmp[i] =
                    read_plane_scalar_s2(&planes[1], w, h, (int)(x + (unsigned)i), (int)y, bpc);
                v_tmp[i] =
                    read_plane_scalar_s2(&planes[2], w, h, (int)(x + (unsigned)i), (int)y, bpc);
            }
            const __m256 Y = _mm256_mul_ps(_mm256_load_ps(y_tmp), vinv_peak);
            const __m256 U = _mm256_mul_ps(_mm256_load_ps(u_tmp), vinv_peak);
            const __m256 V = _mm256_mul_ps(_mm256_load_ps(v_tmp), vinv_peak);

            const __m256 Yn = _mm256_mul_ps(_mm256_sub_ps(Y, vy_off), vy_scale);
            const __m256 Un = _mm256_mul_ps(_mm256_sub_ps(U, vc_off), vc_scale);
            const __m256 Vn = _mm256_mul_ps(_mm256_sub_ps(V, vc_off), vc_scale);

            /* Scalar-order matmul: R = Yn + cr_r*Vn; G = Yn + cb_g*Un + cr_g*Vn;
             * B = Yn + cb_b*Un. Preserve left-to-right associativity. */
            __m256 R = _mm256_add_ps(Yn, _mm256_mul_ps(vcr_r, Vn));
            __m256 G = _mm256_add_ps(Yn, _mm256_mul_ps(vcb_g, Un));
            G = _mm256_add_ps(G, _mm256_mul_ps(vcr_g, Vn));
            __m256 B = _mm256_add_ps(Yn, _mm256_mul_ps(vcb_b, Un));

            R = _mm256_max_ps(_mm256_min_ps(R, vone), vzero);
            G = _mm256_max_ps(_mm256_min_ps(G, vone), vzero);
            B = _mm256_max_ps(_mm256_min_ps(B, vone), vzero);

            R = srgb_to_linear_lane_avx2(R);
            G = srgb_to_linear_lane_avx2(G);
            B = srgb_to_linear_lane_avx2(B);

            _mm256_storeu_ps(rp + (size_t)y * w + x, R);
            _mm256_storeu_ps(gp + (size_t)y * w + x, G);
            _mm256_storeu_ps(bp + (size_t)y * w + x, B);
        }
        /* Scalar tail — identical to scalar reference body. */
        for (; x < w; x++) {
            const float Ys = read_plane_scalar_s2(&planes[0], w, h, (int)x, (int)y, bpc) * inv_peak;
            const float Us = read_plane_scalar_s2(&planes[1], w, h, (int)x, (int)y, bpc) * inv_peak;
            const float Vs = read_plane_scalar_s2(&planes[2], w, h, (int)x, (int)y, bpc) * inv_peak;
            const float Yn = (Ys - y_off) * y_scale;
            const float Un = (Us - c_off) * c_scale;
            const float Vn = (Vs - c_off) * c_scale;
            float R = Yn + cr_r * Vn;
            float G = Yn + cb_g * Un + cr_g * Vn;
            float B = Yn + cb_b * Un;
            if (R < 0.0f)
                R = 0.0f;
            if (R > 1.0f)
                R = 1.0f;
            if (G < 0.0f)
                G = 0.0f;
            if (G > 1.0f)
                G = 1.0f;
            if (B < 0.0f)
                B = 0.0f;
            if (B > 1.0f)
                B = 1.0f;
            const float Rl = vmaf_ss2_srgb_eotf(R);
            const float Gl = vmaf_ss2_srgb_eotf(G);
            const float Bl = vmaf_ss2_srgb_eotf(B);
            const size_t idx = (size_t)y * w + x;
            rp[idx] = Rl;
            gp[idx] = Gl;
            bp[idx] = Bl;
        }
    }
}
