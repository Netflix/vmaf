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
#include "mem.h"
#include "common/convolution.h"
#include "vif_options.h"
#include "vif_tools.h"
#include "common/cpu.h"
#include "common/macros.h"

extern enum vmaf_cpu cpu;

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

//log value generation for integer point implementation
void log_generate()
    {
    int i;
	for (i = 32767; i < 65536; i++)

    {
		log_values[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
    }

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

void vif_xx_yy_xy_s(const float *x, const float *y, float *xx, float *yy, float *xy, int w, int h, int xstride, int ystride, int xxstride, int yystride, int xystride)
{
    int x_px_stride = xstride / sizeof(float);
    int y_px_stride = ystride / sizeof(float);
    int xx_px_stride = xxstride / sizeof(float);
    int yy_px_stride = yystride / sizeof(float);
    int xy_px_stride = xystride / sizeof(float);

    int i, j;

    // L1 is 32 - 64 KB
    // L2 is 2-4 MB
    // w, h = 1920 x 1080 at floating point is 8 MB
    // not going to fit into L2

    float xval, yval, xxval, yyval, xyval;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            xval = x[i * x_px_stride + j];
            yval = y[i * y_px_stride + j];

            xxval = xval * xval;
            yyval = yval * yval;
            xyval = xval * yval;

            xx[i * xx_px_stride + j] = xxval;
            yy[i * yy_px_stride + j] = yyval;
            xy[i * xy_px_stride + j] = xyval;
        }
    }
}

void vif_statistic_s(const float *mu1, const float *mu2, const float *mu1_mu2, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
	int w, int h, int mu1_stride, int mu2_stride, int mu1_mu2_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, int num_stride, int den_stride)
{
	static const float sigma_nsq = 2;
	static const float sigma_max_inv = 4.0 / (255.0*255.0);

	int mu1_px_stride = mu1_stride / sizeof(float);
	int mu2_px_stride = mu2_stride / sizeof(float);
	int xx_filt_px_stride = xx_filt_stride / sizeof(float);
	int yy_filt_px_stride = yy_filt_stride / sizeof(float);
	int xy_filt_px_stride = xy_filt_stride / sizeof(float);

	float mu1_sq_val, mu2_sq_val, mu1_mu2_val, xx_filt_val, yy_filt_val, xy_filt_val;
	float sigma1_sq, sigma2_sq, sigma12, g, sv_sq;
	float num_val, den_val;
	int i, j;

	float accum_num = 0.0;
	float accum_den = 0.0;

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

			if (sigma1_sq < sigma_nsq) {
				num_val = 1.0 - sigma2_sq * sigma_max_inv;
				den_val = 1.0;
			}
			else {
				sv_sq = (sigma2_sq + sigma_nsq) * sigma1_sq;
				if (sigma12 < 0)
				{
					num_val = 0.0;
				}
				else
				{
					g = sv_sq - sigma12 * sigma12;
					num_val = log2f(sv_sq / g);
				}
				den_val = log2f(1.0f + sigma1_sq / sigma_nsq);
			}

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

    if (cpu >= VMAF_CPU_AVX)
    {
        convolution_f32_avx_s(f, fwidth, src, dst, tmpbuf, w, h, src_px_stride, dst_px_stride);
        return;
    }

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
	
	if (cpu >= VMAF_CPU_AVX)
	{
		convolution_f32_avx_sq_s(f, fwidth, src, dst, tmpbuf, w, h, src_px_stride, dst_px_stride);
		return;
	}

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

	if (cpu >= VMAF_CPU_AVX)
	{
		convolution_f32_avx_xy_s(f, fwidth, src1, src2, dst, tmpbuf, w, h, src1_px_stride, src2_px_stride, dst_px_stride);
		return;
	}

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

void vif_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float fcoeff, imgcoeff;
    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                float accum_inner = 0;

                for (fj = 0; fj < fwidth; ++fj) {
                    fcoeff = f[fi * fwidth + fj];

                    ii = i - fwidth / 2 + fi;
                    jj = j - fwidth / 2 + fj;

                    ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);
                    jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

                    imgcoeff = src[ii * src_px_stride + jj];

                    accum_inner += fcoeff * imgcoeff;
                }

                accum += accum_inner;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}
