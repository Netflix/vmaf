/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "cpu.h"
#include "mem.h"
#include "common/convolution.h"
#include "vif_options.h"
#include "vif_tools.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef VIF_OPT_FAST_LOG2 // option to replace log2 calculation with faster speed

static const float log2_poly_s[9] = { -0.012671635276421, 0.064841182402670, -0.157048836463065, 0.257167726303123, -0.353800560300520, 0.480131410397451, -0.721314327952201, 1.442694803896991, 0 };

static float horner_s(const float *poly, float x, int n)
{
    float var = 0;
    int i;

    for (i = 0; i < n; ++i) {
        var = var * x + poly[i];
    }
    return var;
}

static float log2f_approx(float x)
{
    const uint32_t exp_zero_const = 0x3F800000UL;

    const uint32_t exp_expo_mask = 0x7F800000UL;
    const uint32_t exp_mant_mask = 0x007FFFFFUL;

    float remain;
    float log_base, log_remain;
    uint32_t u32, u32remain;
    uint32_t exponent, mant;

    if (x == 0)
        return -INFINITY;
    if (x < 0)
        return NAN;

    memcpy(&u32, &x, sizeof(float));

    exponent  = (u32 & exp_expo_mask) >> 23;
    mant      = (u32 & exp_mant_mask) >> 0;
    u32remain = mant | exp_zero_const;

    memcpy(&remain, &u32remain, sizeof(float));

    log_base = (int32_t)exponent - 127;
    log_remain = horner_s(log2_poly_s, (remain - 1.0f), sizeof(log2_poly_s) / sizeof(float));

    return log_base + log_remain;
}

#define log2f log2f_approx

#endif /* VIF_FAST_LOG2 */

#if 1 //defined(_MSC_VER)
const float vif_filter1d_table_s[3][4][17] = {
        { // kernelscale = 1.0
                {0.00745626912, 0.0142655009, 0.0250313189, 0.0402820669, 0.0594526194, 0.0804751068, 0.0999041125, 0.113746084, 0.118773937, 0.113746084, 0.0999041125, 0.0804751068, 0.0594526194, 0.0402820669, 0.0250313189, 0.0142655009, 0.00745626912},
                {0.0189780835, 0.0558981746, 0.120920904, 0.192116052, 0.224173605, 0.192116052, 0.120920904, 0.0558981746, 0.0189780835},
                {0.054488685, 0.244201347, 0.402619958, 0.244201347, 0.054488685},
                {0.166378498, 0.667243004, 0.166378498}
        },
        { // kernelscale = 0.5
                {0.01483945381483146, 0.04981728920101665, 0.11832250618647212, 0.19882899654808195, 0.2363835084991954, 0.19882899654808195, 0.11832250618647212, 0.04981728920101665, 0.01483945381483146},
                {0.03765705331865383, 0.23993597758503457, 0.4448139381926232, 0.23993597758503457, 0.03765705331865383},
                {0.10650697891920077, 0.7869860421615985, 0.10650697891920077},
                {0.00383625879916893, 0.9923274824016621, 0.00383625879916893}
        },
        { // kernelscale = 1.5
                {0.003061411879755733, 0.004950361473107714, 0.007702910451476288, 0.011533883865671363, 0.01661877803810386, 0.02304227676257057, 0.030743582802885815, 0.039471747635457924, 0.04876643642871142, 0.057977365173867444, 0.06632827707341536, 0.07301998612791924, 0.0773548527900372, 0.07885625899404015, 0.0773548527900372, 0.07301998612791924, 0.06632827707341536, 0.057977365173867444, 0.04876643642871142, 0.039471747635457924, 0.030743582802885815, 0.02304227676257057, 0.01661877803810386, 0.011533883865671363, 0.007702910451476288, 0.004950361473107714, 0.003061411879755733},
                {0.005155279152239917, 0.012574282300781493, 0.026738695950137843, 0.049570492793231676, 0.08011839616706516, 0.11289306247174812, 0.13868460659463525, 0.14853036914032117, 0.13868460659463525, 0.11289306247174812, 0.08011839616706516, 0.049570492793231676, 0.026738695950137843, 0.012574282300781493, 0.005155279152239917},
                {0.007614419169296345, 0.03607496968918391, 0.10958608179781394, 0.2134445419434044, 0.26655997480060273, 0.2134445419434044, 0.10958608179781394, 0.03607496968918391, 0.007614419169296345},
                {0.03765705331865383, 0.23993597758503457, 0.4448139381926232, 0.23993597758503457, 0.03765705331865383}
        }
};
#else
const float vif_filter1d_table_s[3][4][17] = {
        { // kernelscale = 1.0
                { 0x1.e8a77p-8,  0x1.d373b2p-7, 0x1.9a1cf6p-6, 0x1.49fd9ep-5, 0x1.e7092ep-5, 0x1.49a044p-4, 0x1.99350ep-4, 0x1.d1e76ap-4, 0x1.e67f8p-4, 0x1.d1e76ap-4, 0x1.99350ep-4, 0x1.49a044p-4, 0x1.e7092ep-5, 0x1.49fd9ep-5, 0x1.9a1cf6p-6, 0x1.d373b2p-7, 0x1.e8a77p-8 },
                { 0x1.36efdap-6, 0x1.c9eaf8p-5, 0x1.ef4ac2p-4, 0x1.897424p-3, 0x1.cb1b88p-3, 0x1.897424p-3, 0x1.ef4ac2p-4, 0x1.c9eaf8p-5, 0x1.36efdap-6 },
                { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 },
                { 0x1.54be4p-3,  0x1.55a0ep-1,  0x1.54be4p-3 }
        },
        { // kernelscale = 0.5
                { 1.483945e-02, 4.981729e-02, 1.183225e-01, 1.988290e-01, 2.363835e-01, 1.988290e-01, 1.183225e-01, 4.981729e-02, 1.483945e-02 },
                { 3.765705e-02, 2.399360e-01, 4.448139e-01, 2.399360e-01, 3.765705e-02 },
                { 1.065070e-01, 7.869860e-01, 1.065070e-01 },
                { 3.836259e-03, 9.923275e-01, 3.836259e-03 }
        },
        { // kernelscale = 1.5
                { 3.061412e-03, 4.950361e-03, 7.702910e-03, 1.153388e-02, 1.661878e-02, 2.304228e-02, 3.074358e-02, 3.947175e-02, 4.876644e-02, 5.797737e-02, 6.632828e-02, 7.301999e-02, 7.735485e-02, 7.885626e-02, 7.735485e-02, 7.301999e-02, 6.632828e-02, 5.797737e-02, 4.876644e-02, 3.947175e-02, 3.074358e-02, 2.304228e-02, 1.661878e-02, 1.153388e-02, 7.702910e-03, 4.950361e-03, 3.061412e-03 },
                { 5.155279e-03, 1.257428e-02, 2.673870e-02, 4.957049e-02, 8.011840e-02, 1.128931e-01, 1.386846e-01, 1.485304e-01, 1.386846e-01, 1.128931e-01, 8.011840e-02, 4.957049e-02, 2.673870e-02, 1.257428e-02, 5.155279e-03 },
                { 7.614419e-03, 3.607497e-02, 1.095861e-01, 2.134445e-01, 2.665600e-01, 2.134445e-01, 1.095861e-01, 3.607497e-02, 7.614419e-03 },
                { 3.765705e-02, 2.399360e-01, 4.448139e-01, 2.399360e-01, 3.765705e-02 }
        }
};
#endif

const int vif_filter1d_width[3][4] = {
        { 17, 9, 5, 3 }, // kernelscale = 1.0
        { 9, 5, 3, 3 }, // kernelscale = 0.5
        { 27, 15, 9, 5 }  // kernelscale = 1.5
};

void vif_dec2_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride)
{
    int src_px_stride = src_stride / sizeof(float); // src_stride is in bytes
    int dst_px_stride = dst_stride / sizeof(float);

    int i, j;

    // decimation by 2 in each direction (after gaussian blur? check)
    for (i = 0; i < src_h / 2; ++i) {
        for (j = 0; j < src_w / 2; ++j) {
            dst[i * dst_px_stride + j] = src[(i * 2) * src_px_stride + (j * 2)];
        }
    }
}

float vif_sum_s(const float *x, int w, int h, int stride)
{
    int px_stride = stride / sizeof(float);
    int i, j;

    float accum = 0;

    for (i = 0; i < h; ++i) {
        float accum_inner = 0;

        for (j = 0; j < w; ++j) {
            accum_inner += x[i * px_stride + j];
        } // having an inner accumulator help reduce numerical error (no accumulation of near-0 terms)

        accum += accum_inner;
    }

    return accum;
}

void vif_statistic_s(const float *mu1, const float *mu2, const float *mu1_mu2, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
	int w, int h, int mu1_stride, int mu2_stride, int mu1_mu2_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, int num_stride, int den_stride,
	double vif_enhn_gain_limit)
{
	static const float sigma_nsq = 2;
	static const float sigma_max_inv = 4.0 / (255.0*255.0);

	int mu1_px_stride = mu1_stride / sizeof(float);
	int mu2_px_stride = mu2_stride / sizeof(float);
	int xx_filt_px_stride = xx_filt_stride / sizeof(float);
	int yy_filt_px_stride = yy_filt_stride / sizeof(float);
	int xy_filt_px_stride = xy_filt_stride / sizeof(float);

	float mu1_sq_val, mu2_sq_val, mu1_mu2_val, xx_filt_val, yy_filt_val, xy_filt_val;
	float sigma1_sq, sigma2_sq, sigma12;
	float num_val, den_val;
	int i, j;

    /* ==== vif_stat_mode = 'matching_c' ==== */
    // float num_log_den, num_log_num;
    /* ==== vif_stat_mode = 'matching_matlab' ==== */
	float g, sv_sq, eps = 1.0e-10f;
	float vif_enhn_gain_limit_f = (float) vif_enhn_gain_limit;
    /* == end of vif_stat_mode = 'matching_matlab' == */

	float accum_num = 0.0f;
	float accum_den = 0.0f;

	for (i = 0; i < h; ++i) {
		float accum_inner_num = 0;
		float accum_inner_den = 0;
		for (j = 0; j < w; ++j) {
			float mu1_val = mu1[i * mu1_px_stride + j];
			float mu2_val = mu2[i * mu2_px_stride + j];
			mu1_sq_val = mu1_val * mu1_val; // same name as the Matlab code vifp_mscale.m
			mu2_sq_val = mu2_val * mu2_val;
			mu1_mu2_val = mu1_val * mu2_val; //mu1_mu2[i * mu1_mu2_px_stride + j];
			xx_filt_val = xx_filt[i * xx_filt_px_stride + j];
			yy_filt_val = yy_filt[i * yy_filt_px_stride + j];
			xy_filt_val = xy_filt[i * xy_filt_px_stride + j];

			sigma1_sq = xx_filt_val - mu1_sq_val;
			sigma2_sq = yy_filt_val - mu2_sq_val;
			sigma12 = xy_filt_val - mu1_mu2_val;

			/* ==== vif_stat_mode = 'matching_c' ==== */

            /* if (sigma1_sq < sigma_nsq) {
                num_val = 1.0 - sigma2_sq * sigma_max_inv;
                den_val = 1.0;
            }
            else {
                num_log_num = (sigma2_sq + sigma_nsq) * sigma1_sq;
                if (sigma12 < 0)
                {
                    num_val = 0.0;
                }
                else
                {
                    num_log_den = num_log_num - sigma12 * sigma12;
                    num_val = log2f(num_log_num / num_log_den);
                }
                den_val = log2f(1.0f + sigma1_sq / sigma_nsq);
            } */

            /* ==== vif_stat_mode = 'matching_matlab' ==== */

            sigma1_sq = MAX(sigma1_sq, 0.0f);
            sigma2_sq = MAX(sigma2_sq, 0.0f);

			g = sigma12 / (sigma1_sq + eps);
			sv_sq = sigma2_sq - g * sigma12;

			if (sigma1_sq < eps) {
			    g = 0.0f;
                sv_sq = sigma2_sq;
                sigma1_sq = 0.0f;
			}

			if (sigma2_sq < eps) {
			    g = 0.0f;
			    sv_sq = 0.0f;
			}

			if (g < 0.0f) {
			    sv_sq = sigma2_sq;
			    g = 0.0f;
			}
			sv_sq = MAX(sv_sq, eps);

            g = MIN(g, vif_enhn_gain_limit_f);

            num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
            den_val = log2f(1.0f + (sigma1_sq) / (sigma_nsq));

            if (sigma12 < 0.0f) {
                num_val = 0.0f;
            }

            if (sigma1_sq < sigma_nsq) {
                num_val = 1.0f - sigma2_sq * sigma_max_inv;
                den_val = 1.0f;
            }

            /* == end of vif_stat_mode = 'matching_matlab' == */

            accum_inner_num += num_val;
			accum_inner_den += den_val;
		}

		accum_num += accum_inner_num;
		accum_den += accum_inner_den;
	}
	num[0] = accum_num;
	den[0] = accum_den;
}

void vif_filter1d_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth)
{

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    /* if support avx */

#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        convolution_f32_avx_s(f, fwidth, src, dst, tmpbuf, w, h,
                              src_px_stride, dst_px_stride);
        return;
    }
#endif

    /* fall back */

    float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = f[fi];

                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = f[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

                imgcoeff = tmp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);
}

// Code optimized by adding intrinsic code for the functions,
// vif_filter1d_sq and vif_filter1d_sq

void vif_filter1d_sq_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth)
{

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	/* if support avx */
	
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        convolution_f32_avx_sq_s(f, fwidth, src, dst, tmpbuf, w, h,
                                 src_px_stride, dst_px_stride);
        return;
    }
#endif

	/* fall back */

	float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
	float fcoeff, imgcoeff;

	int i, j, fi, fj, ii, jj;

	for (i = 0; i < h; ++i) {
		/* Vertical pass. */
		for (j = 0; j < w; ++j) {
			float accum = 0;

			for (fi = 0; fi < fwidth; ++fi) {
				fcoeff = f[fi];

				ii = i - fwidth / 2 + fi;
				ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

				imgcoeff = src[ii * src_px_stride + j];

				accum += fcoeff * (imgcoeff * imgcoeff);
			}

			tmp[j] = accum;
		}

		/* Horizontal pass. */
		for (j = 0; j < w; ++j) {
			float accum = 0;

			for (fj = 0; fj < fwidth; ++fj) {
				fcoeff = f[fj];

				jj = j - fwidth / 2 + fj;
				jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

				imgcoeff = tmp[jj];

				accum += fcoeff * imgcoeff;
			}

			dst[i * dst_px_stride + j] = accum;
		}
	}

	aligned_free(tmp);
}

void vif_filter1d_xy_s(const float *f, const float *src1, const float *src2, float *dst, float *tmpbuf, int w, int h, int src1_stride, int src2_stride, int dst_stride, int fwidth)
{

	int src1_px_stride = src1_stride / sizeof(float);
	int src2_px_stride = src1_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	/* if support avx */

#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        convolution_f32_avx_xy_s(f, fwidth, src1, src2, dst, tmpbuf, w, h,
                                 src1_px_stride, src2_px_stride, dst_px_stride);
        return;
    }
#endif

	/* fall back */

	float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
	float fcoeff, imgcoeff, imgcoeff1, imgcoeff2;

	int i, j, fi, fj, ii, jj;

	for (i = 0; i < h; ++i) {
		/* Vertical pass. */
		for (j = 0; j < w; ++j) {
			float accum = 0;

			for (fi = 0; fi < fwidth; ++fi) {
				fcoeff = f[fi];

				ii = i - fwidth / 2 + fi;
				ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

				imgcoeff1 = src1[ii * src1_px_stride + j];
				imgcoeff2 = src2[ii * src2_px_stride + j];

				accum += fcoeff * (imgcoeff1 * imgcoeff2);
			}

			tmp[j] = accum;
		}

		/* Horizontal pass. */
		for (j = 0; j < w; ++j) {
			float accum = 0;

			for (fj = 0; fj < fwidth; ++fj) {
				fcoeff = f[fj];

				jj = j - fwidth / 2 + fj;
				jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

				imgcoeff = tmp[jj];

				accum += fcoeff * imgcoeff;
			}

			dst[i * dst_px_stride + j] = accum;
		}
	}

	aligned_free(tmp);
}
