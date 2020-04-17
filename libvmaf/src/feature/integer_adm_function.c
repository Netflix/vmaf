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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mem.h"
#include "common/macros.h"
#include "integer_adm_function.h"
#include "adm_options.h"

#define M_PI 3.14159265358979323846264338327

#ifdef ADM_OPT_RECIP_DIVISION

#include <emmintrin.h>

static float rcp_s(float x)
{
    float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

#define DIVS(n, d) ((n) * rcp_s(d))

#else

#define DIVS(n, d) ((n) / (d))

#endif /* ADM_OPT_RECIP_DIVISION */

static const int16_t integer_dwt2_db2_coeffs_lo_s[4] = { 15826, 27411, 7345, -4240 };
static const int16_t integer_dwt2_db2_coeffs_hi_s[4] = { -4240, -7345, 27411, -15826 };

static const int32_t integer_dwt2_db2_coeffs_lo_sum = 46342;
static const int32_t integer_dwt2_db2_coeffs_hi_sum = 0;

#ifndef INTEGER_ONE_BY_15
#define INTEGER_ONE_BY_15 8738
#endif

#ifndef INTEGER_I4_ONE_BY_15
#define INTEGER_I4_ONE_BY_15 286331153
#endif

/* ================= */
/* Noise floor model */
/* ================= */

#define VIEW_DIST 3.0f

#define REF_DISPLAY_HEIGHT 1080

/*
 * The following dwt visibility threshold parameters are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1170, formula (7) and corresponding Table IV
 * Table IV has 2 entries for Cb and Cr thresholds
 * Chose those corresponding to subject "sfl" since they are lower
 * These thresholds were obtained and modeled for the 7-9 biorthogonal wavelet basis
 */

 /*
  * The following dwt visibility threshold parameters are taken from
  * "Visibility of Wavelet Quantization Noise"
  * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
  * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
  * Page 1170, formula (7) and corresponding Table IV
  * Table IV has 2 entries for Cb and Cr thresholds
  * Chose those corresponding to subject "sfl" since they are lower
  * These thresholds were obtained and modeled for the 7-9 biorthogonal wavelet basis
  */
static struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

// 0 -> Y, 1 -> Cb, 2 -> Cr
static const struct dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    {.a = 0.495,.k = 0.466,.f0 = 0.401,.g = { 1.501, 1.0, 0.534, 1.0} },
    {.a = 1.633,.k = 0.353,.f0 = 0.209,.g = { 1.520, 1.0, 0.502, 1.0} },
    {.a = 0.944,.k = 0.521,.f0 = 0.404,.g = { 1.868, 1.0, 0.516, 1.0} }
};

/*
 * The following dwt basis function amplitudes, A(lambda,theta), are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1172, Table V
 * The table has been transposed, i.e. it can be used directly to obtain A[lambda][theta]
 * These amplitudes were calculated for the 7-9 biorthogonal wavelet basis
 */
static const float dwt_7_9_basis_function_amplitudes[6][4] = {
    { 0.62171,  0.67234,  0.72709,  0.67234  },
    { 0.34537,  0.41317,  0.49428,  0.41317  },
    { 0.18004,  0.22727,  0.28688,  0.22727  },
    { 0.091401, 0.11792,  0.15214,  0.11792  },
    { 0.045943, 0.059758, 0.077727, 0.059758 },
    { 0.023013, 0.030018, 0.039156, 0.030018 }
};

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
FORCE_INLINE static inline float dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = VIEW_DIST * REF_DISPLAY_HEIGHT * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0, lambda + 1)*params->f0*params->g[theta] / r);
    float Q = 2.0*params->a*pow(10.0, params->k*temp*temp) / dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

typedef struct integer_adm_dwt_band_t_s {
    int16_t *band_a; /* Low-pass V + low-pass H. */
    int16_t *band_v; /* Low-pass V + high-pass H. */
    int16_t *band_h; /* High-pass V + low-pass H. */
    int16_t *band_d; /* High-pass V + high-pass H. */
} integer_adm_dwt_band_t_s;

typedef struct integer_i4_adm_dwt_band_t_s {
    int32_t *band_a; /* Low-pass V + low-pass H. */
    int32_t *band_v; /* Low-pass V + high-pass H. */
    int32_t *band_h; /* High-pass V + low-pass H. */
    int32_t *band_d; /* High-pass V + high-pass H. */
} integer_i4_adm_dwt_band_t_s;

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define INTEGER_ADM_CM_THRESH_S_0_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
			sum += flt_ptr[1]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define INTEGER_ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define INTEGER_ADM_CM_THRESH_S_0_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1];  \
			*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define INTEGER_ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
			sum += flt_ptr[1]; \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define INTEGER_ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define INTEGER_ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[j - 1];\
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define INTEGER_ADM_CM_THRESH_S_I_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
			int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define INTEGER_ADM_CM_THRESH_S_I_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define INTEGER_ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	int32_t sum = 0; \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
			int16_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (i-1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += (int16_t)(((INTEGER_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define INTEGER_I4_ADM_CM_THRESH_S_0_0(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs( src_ptr[0]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[1]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define INTEGER_I4_ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define INTEGER_I4_ADM_CM_THRESH_S_0_J(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1];  \
			*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define INTEGER_I4_ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[1]; \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define INTEGER_I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define INTEGER_I4_ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[j - 1];\
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define INTEGER_I4_ADM_CM_THRESH_S_I_J(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t sum = 0; \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define INTEGER_I4_ADM_CM_THRESH_S_I_0(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define INTEGER_I4_ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	int32_t sum = 0; \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			int32_t *src_ptr = angles[theta]; \
			int32_t *flt_ptr = flt_angles[theta]; \
			int32_t sum = 0; \
			src_ptr += (src_px_stride * (i-1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += (int32_t)((((int64_t)INTEGER_I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

static char *integer_init_dwt_band(integer_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_h = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_v = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_d = (int16_t *)data_top; data_top += buf_sz_one;
    return data_top;
}

static char *integer_i4_init_dwt_band(integer_i4_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_h = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_v = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_d = (int32_t *)data_top;
	data_top += buf_sz_one;
	return data_top;
}

static char *integer_init_dwt_band_hvd(integer_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = NULL;
	band->band_h = (int16_t *)data_top; data_top += buf_sz_one;
	band->band_v = (int16_t *)data_top; data_top += buf_sz_one;
	band->band_d = (int16_t *)data_top; data_top += buf_sz_one;
	return data_top;
}

static char *integer_i4_init_dwt_band_hvd(integer_i4_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = NULL;
	band->band_h = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_v = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_d = (int32_t *)data_top;
	data_top += buf_sz_one;
	return data_top;
}

/**
 * This function stores the imgcoeff values used for integer_adm_dwt2_16_s and integer_adm_dwt2_scale123_combined_s in buffers, which reduces the control code cycles.
 */
static void integer_dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h)
{
    int i, j;
    int ind0, ind1, ind2, ind3;
    /* Vertical pass */
    for (i = 0; i < (h + 1) / 2; ++i) { /* Index = 2 * i - 1 + fi */
        ind0 = 2 * i - 1;
        ind0 = (ind0 < 0) ? -ind0 : ((ind0 >= h) ? (2 * h - ind0 - 1) : ind0);
        ind1 = 2 * i;
        if (ind1 >= h) {
            ind1 = (2 * h - ind1 - 1);
        }
        ind2 = 2 * i + 1;
        if (ind2 >= h) {
            ind2 = (2 * h - ind2 - 1);
        }
        ind3 = 2 * i + 2;
        if (ind3 >= h) {
            ind3 = (2 * h - ind3 - 1);
        }
        src_ind_y[0][i] = ind0;
        src_ind_y[1][i] = ind1;
        src_ind_y[2][i] = ind2;
        src_ind_y[3][i] = ind3;
    }
    /* Horizontal pass */
    for (j = 0; j < (w + 1) / 2; ++j) { /* Index = 2 * j - 1 + fj */
        ind0 = 2 * j - 1;
        ind0 = (ind0 < 0) ? -ind0 : ((ind0 >= w) ? (2 * w - ind0 - 1) : ind0);
        ind1 = 2 * j;
        if (ind1 >= w) {
            ind1 = (2 * w - ind1 - 1);
        }
        ind2 = 2 * j + 1;
        if (ind2 >= w) {
            ind2 = (2 * w - ind2 - 1);
        }
        ind3 = 2 * j + 2;
        if (ind3 >= w) {
            ind3 = (2 * w - ind3 - 1);
        }
        src_ind_x[0][j] = ind0;
        src_ind_x[1][j] = ind1;
        src_ind_x[2][j] = ind2;
        src_ind_x[3][j] = ind3;
    }
}

/**
 * Works similar to floating-point function adm_decouple_s
 */
static void integer_adm_decouple_s(const integer_adm_dwt_band_t_s *ref, const integer_adm_dwt_band_t_s *dis, const integer_adm_dwt_band_t_s *r, const integer_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor)
{
#ifdef ADM_OPT_AVOID_ATAN
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
    const float eps = 1e-30;

    int ref_px_stride = ref_stride >> 2;    //divide by sizeof(int32_t)
    int dis_px_stride = dis_stride >> 2;    //divide by sizeof(int32_t)
    int r_px_stride = r_stride >> 2;        //divide by sizeof(int32_t)
    int a_px_stride = a_stride >> 2;        //divide by sizeof(int32_t)

    /* The computation of the score is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5 - 1; // -1 for filter tap
    int top = h * border_factor - 0.5 - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0) {
        left = 0;
    }
    if (right > w) {
        right = w;
    }
    if (top < 0) {
        top = 0;
    }
    if (bottom > h) {
        bottom = h;
    }

    int16_t oh, ov, od, th, tv, td;
    int32_t kh, kv, kd;
    int16_t tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
    int64_t ot_dp, o_mag_sq, t_mag_sq;
#else
    float oa, ta, diff;
#endif
    int angle_flag;
    int i, j;

    for (i = top; i < bottom; ++i) {
        for (j = left; j < right; ++j) {
            oh = ref->band_h[i * ref_px_stride + j];
            ov = ref->band_v[i * ref_px_stride + j];
            od = ref->band_d[i * ref_px_stride + j];
            th = dis->band_h[i * dis_px_stride + j];
            tv = dis->band_v[i * dis_px_stride + j];
            td = dis->band_d[i * dis_px_stride + j];

            /**
             * This shift is done so that precision is maintained after division
             * division is Q31/Q16
             */
            int32_t tmp_th = ((int32_t)th) << 15;
            int32_t tmp_tv = ((int32_t)tv) << 15;
            int32_t tmp_td = ((int32_t)td) << 15;

            int32_t tmp_kh = (oh == 0) ? 32768 : (tmp_th / oh);
            int32_t tmp_kv = (ov == 0) ? 32768 : (tmp_tv / ov);
            int32_t tmp_kd = (od == 0) ? 32768 : (tmp_td / od);

            kh = tmp_kh < 0 ? 0 : (tmp_kh > (int32_t) 32768 ? (int32_t)32768 : (int32_t)tmp_kh);
            kv = tmp_kv < 0 ? 0 : (tmp_kv > (int32_t) 32768 ? (int32_t)32768 : (int32_t)tmp_kv);
            kd = tmp_kd < 0 ? 0 : (tmp_kd > (int32_t) 32768 ? (int32_t)32768 : (int32_t)tmp_kd);

            /**
             * kh,kv,kd are in Q15 type and oh,ov,od are in Q16 type hence shifted by 15 to make result Q16
             */
            tmph = ((kh * oh) + 16384) >> 15;
            tmpv = ((kv * ov) + 16384) >> 15;
            tmpd = ((kd * od) + 16384) >> 15;
#ifdef ADM_OPT_AVOID_ATAN
            /* Determine if angle between (oh,ov) and (th,tv) is less than 1 degree.
             * Given that u is the angle (oh,ov) and v is the angle (th,tv), this can
             * be done by testing the inequvality.
             *
             * { (u.v.) >= 0 } AND { (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2 }
             *
             * Proof:
             *
             * cos(theta) = (u.v) / (||u|| * ||v||)
             *
             * IF u.v >= 0 THEN
             *   cos(theta)^2 = (u.v)^2 / (||u||^2 * ||v||^2)
             *   (u.v)^2 = cos(theta)^2 * ||u||^2 * ||v||^2
             *
             *   IF |theta| < 1deg THEN
             *     (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2
             *   END
             * ELSE
             *   |theta| > 90deg
             * END
             */
            ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
            o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
            t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

            /**
             * angle_flag is calculated in floating-point by converting fixed-point variables back to floating-point
             */
            angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) && (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >= cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
#else
            oa = atanf(DIVS(ov, oh + eps));
            ta = atanf(DIVS(tv, th + eps));

            if (oh < 0.0f)
                oa += (float)M_PI;
            if (th < 0.0f)
                ta += (float)M_PI;

            diff = fabsf(oa - ta) * 180.0f / M_PI;
            angle_flag = diff < 1.0f;
#endif
            if (angle_flag) {
                tmph = th;
                tmpv = tv;
                tmpd = td;
            }

            r->band_h[i * r_px_stride + j] = tmph;
            r->band_v[i * r_px_stride + j] = tmpv;
            r->band_d[i * r_px_stride + j] = tmpd;

            a->band_h[i * a_px_stride + j] = th - tmph;
            a->band_v[i * a_px_stride + j] = tv - tmpv;
            a->band_d[i * a_px_stride + j] = td - tmpd;
        }
    }
}

/**
 * Works similar to integer_adm_decouple_s function for scale 1,2,3
 */
static void integer_adm_decouple_scale123_s(const integer_i4_adm_dwt_band_t_s *ref, const integer_i4_adm_dwt_band_t_s *dis, const integer_i4_adm_dwt_band_t_s *r, const integer_i4_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor, int scale)
{
#ifdef ADM_OPT_AVOID_ATAN
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
    const float eps = 1e-30;

    int ref_px_stride = ref_stride >> 2;    //divide by sizeof(int32_t)
    int dis_px_stride = dis_stride >> 2;    //divide by sizeof(int32_t)
    int r_px_stride = r_stride >> 2;        //divide by sizeof(int32_t)
    int a_px_stride = a_stride >> 2;        //divide by sizeof(int32_t)
    float f_shift[3] = { pow(2, 21), pow(2, 19), pow(2, 18) };
    /* The computation of the score is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5 - 1; // -1 for filter tap
    int top = h * border_factor - 0.5 - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0)
    {
        left = 0;
    }
    if (right > w)
    {
        right = w;
    }
    if (top < 0)
    {
        top = 0;
    }
    if (bottom > h)
    {
        bottom = h;
    }

    int32_t oh, ov, od, th, tv, td;
    int64_t kh, kv, kd;
    int32_t tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
    int64_t ot_dp, o_mag_sq, t_mag_sq;
#else
    float oa, ta, diff;
#endif
    int angle_flag;
    int i, j;

    for (i = top; i < bottom; ++i)
    {
        for (j = left; j < right; ++j)
        {
            oh = ref->band_h[i * ref_px_stride + j];
            ov = ref->band_v[i * ref_px_stride + j];
            od = ref->band_d[i * ref_px_stride + j];
            th = dis->band_h[i * dis_px_stride + j];
            tv = dis->band_v[i * dis_px_stride + j];
            td = dis->band_d[i * dis_px_stride + j];

            int64_t tmp_th = ((int64_t)th) << 15;
            int64_t tmp_tv = ((int64_t)tv) << 15;
            int64_t tmp_td = ((int64_t)td) << 15;

            int64_t tmp_kh = (oh == 0) ? 32768 : (tmp_th / oh);
            int64_t tmp_kv = (ov == 0) ? 32768 : (tmp_tv / ov);
            int64_t tmp_kd = (od == 0) ? 32768 : (tmp_td / od);

            kh = tmp_kh < 0 ? 0 : (tmp_kh > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kh);
            kv = tmp_kv < 0 ? 0 : (tmp_kv > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kv);
            kd = tmp_kd < 0 ? 0 : (tmp_kd > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kd);

            tmph = ((kh * oh) + 16384) >> 15;
            tmpv = ((kv * ov) + 16384) >> 15;
            tmpd = ((kd * od) + 16384) >> 15;
#ifdef ADM_OPT_AVOID_ATAN
            /* Determine if angle between (oh,ov) and (th,tv) is less than 1 degree.
             * Given that u is the angle (oh,ov) and v is the angle (th,tv), this can
             * be done by testing the inequvality.
             *
             * { (u.v.) >= 0 } AND { (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2 }
             *
             * Proof:
             *
             * cos(theta) = (u.v) / (||u|| * ||v||)
             *
             * IF u.v >= 0 THEN
             *   cos(theta)^2 = (u.v)^2 / (||u||^2 * ||v||^2)
             *   (u.v)^2 = cos(theta)^2 * ||u||^2 * ||v||^2
             *
             *   IF |theta| < 1deg THEN
             *     (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2
             *   END
             * ELSE
             *   |theta| > 90deg
             * END
             */
            ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
            o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
            t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

            angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) && (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >= cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
#else
            oa = atanf(DIVS(ov, oh + eps));
            ta = atanf(DIVS(tv, th + eps));

            if (oh < 0.0f)
                oa += (float)M_PI;
            if (th < 0.0f)
                ta += (float)M_PI;

            diff = fabsf(oa - ta) * 180.0f / M_PI;
            angle_flag = diff < 1.0f;
#endif
            if (angle_flag)
            {
                tmph = th;
                tmpv = tv;
                tmpd = td;
            }

            r->band_h[i * r_px_stride + j] = tmph;
            r->band_v[i * r_px_stride + j] = tmpv;
            r->band_d[i * r_px_stride + j] = tmpd;

            a->band_h[i * a_px_stride + j] = th - tmph;
            a->band_v[i * a_px_stride + j] = tv - tmpv;
            a->band_d[i * a_px_stride + j] = td - tmpd;
        }
    }
}

/**
 * Works similar to floating-point function adm_csf_s
 */
static void integer_adm_csf_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *dst, const integer_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor)
{
    const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    const int16_t *src_ptr;
    int16_t *dst_ptr;
    int16_t *flt_ptr;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    /**
     * rfactor is converted to fixed-point for scale0 and stored in i_rfactor multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2]
     */
    uint16_t i_rfactor[3] = { 36453,36453,49417 };

    /**
     * Shifts pending from previous stage is 6
     * hence variables multiplied by i_rfactor[0,1] has to be shifted by 21+6=27 to convert into floating-point. But shifted by 15 to make it Q16
     * and variables multiplied by i_factor[2] has to be shifted by 23+6=29 to convert into floating-point. But shifted by 17 to make it Q16
     * Hence remaining shifts after shifting by i_shifts is 12 to make it equivalent to floating-point
     */
    uint8_t i_shifts[3] = { 15,15,17 };
    uint16_t i_shiftsadd[3] = { 16384, 16384, 65535 };
    uint16_t FIX_ONE_BY_30 = 4369; //(1/30)*2^17
    /* The computation of the csf values is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5 - 1; // -1 for filter tap
    int top = h * border_factor - 0.5 - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0) {
        left = 0;
    }
    if (right > w) {
        right = w;
    }
    if (top < 0) {
        top = 0;
    }
    if (bottom > h) {
        bottom = h;
    }

    int i, j, theta, src_offset, dst_offset;
    int32_t dst_val;

    for (theta = 0; theta < 3; ++theta) {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];
        flt_ptr = flt_angles[theta];

        for (i = top; i < bottom; ++i) {
            src_offset = i * src_px_stride;
            dst_offset = i * dst_px_stride;

            for (j = left; j < right; ++j) {
                dst_val = i_rfactor[theta] * (int32_t)src_ptr[src_offset + j];
                int16_t i16_dst_val = ((int16_t)((dst_val + i_shiftsadd[theta]) >> i_shifts[theta]));
                dst_ptr[dst_offset + j] = i16_dst_val;
                flt_ptr[dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val)) + 2048) >> 12));//shifted by 12 to make it Q16. Remaining shifts to make it equivalent to floating point is 12+17-12=12
            }
        }
    }
}

/**
 * Works similar to fixed-point function integer_adm_csf_s for scale 1,2,3
 */
static void integer_i4_adm_csf_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *dst, const integer_i4_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor)
{
    const int32_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int32_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int32_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    const int32_t *src_ptr;
    int32_t *dst_ptr;
    int32_t *flt_ptr;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    //rfactor in fixed-point
    uint32_t i_rfactor[3] = { (uint32_t)(rfactor1[0] * pow(2, 32)), (uint32_t)(rfactor1[1] * pow(2, 32)), (uint32_t)(rfactor1[2] * pow(2, 32)) };

    uint32_t FIX_ONE_BY_30 = 143165577;

    /**
     * The shifts are done such that overflow doesn't happen i.e results are less than 2^31 i.e 1Q31
     */
    int32_t shift_dst[3] = { 28, 28, 28 };
    int32_t shift_flt[3] = { 32, 32, 32 };

    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    int idx;
    for (idx = 0; idx < 3; idx++)
    {
        add_bef_shift_dst[idx] = (int32_t)pow(2, (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (int32_t)pow(2, (shift_flt[idx] - 1));
    }

    /* The computation of the csf values is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5 - 1; // -1 for filter tap
    int top = h * border_factor - 0.5 - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0)
    {
        left = 0;
    }
    if (right > w)
    {
        right = w;
    }
    if (top < 0)
    {
        top = 0;
    }
    if (bottom > h)
    {
        bottom = h;
    }

    int i, j, theta, src_offset, dst_offset;
    int32_t dst_val;

    for (theta = 0; theta < 3; ++theta)
    {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];
        flt_ptr = flt_angles[theta];

        for (i = top; i < bottom; ++i)
        {
            src_offset = i * src_px_stride;
            dst_offset = i * dst_px_stride;

            for (j = left; j < right; ++j)
            {
                dst_val = (int32_t)(((i_rfactor[theta] * (int64_t)src_ptr[src_offset + j]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                dst_ptr[dst_offset + j] = dst_val;
                flt_ptr[dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) + add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);
            }
        }
    }
}

/**
 * Combination of adm_csf_s and adm_sum_cube_s for csf_o based den_scale
 */
static float integer_adm_csf_den_scale_s(const integer_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor)
{
    int16_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    uint64_t accum_inner_h, accum_inner_v, accum_inner_d;
    float den_scale_h, den_scale_v, den_scale_d;

    uint64_t val;

    /* The computation of the denominator scales is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5;
    int top = h * border_factor - 0.5;
    int right = w - left;
    int bottom = h - top;

    int32_t shift_inner_accum = (int32_t)ceil(log2((bottom - top)*(right - left)) - 20);
    shift_inner_accum = shift_inner_accum > 0 ? shift_inner_accum : 0;
    int32_t add_before_shift_inner_accum = (int32_t)pow(2, (shift_inner_accum - 1));
    add_before_shift_inner_accum = add_before_shift_inner_accum > 0 ? add_before_shift_inner_accum : 0;
    uint8_t shift_accum = 18 - shift_inner_accum;

    int i, j;
    /**
     * The rfactor is multiplied at the end after cubing
     * Because d+ = (a[i]^3)*(r^3)
     * is equivalent to d+=a[i]^3 and d=d*(r^3)
     */
    for (i = top; i < bottom; ++i) {
        accum_inner_h = 0;
        accum_inner_v = 0;
        accum_inner_d = 0;
        src_h = src->band_h + i * src_px_stride;
        src_v = src->band_v + i * src_px_stride;
        src_d = src->band_d + i * src_px_stride;
        for (j = left; j < right; ++j) {

            uint16_t h = (uint16_t)abs(src_h[j]);
            uint16_t v = (uint16_t)abs(src_v[j]);
            uint16_t d = (uint16_t)abs(src_d[j]);

            val = ((uint64_t)h * h) * h;

            accum_inner_h += val;
            val = ((uint64_t)v * v) * v;
            accum_inner_v += val;
            val = ((uint64_t)d * d) * d;
            accum_inner_d += val;
        }
        /**
         * max_value of h^3, v^3, d^3 is 1.205624776×10^13
         * accum_h can hold till 1.844674407×10^19
         * accum_h's maximum is reached when it is 2^20 * max(h^3)
         * Therefore the accum_h,v,d is shifted based on width and height subtracted by 20
         */
        accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
        accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
        accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
    }
    double csf_h, csf_v, csf_d;

    /**
     * rfactor is multiplied after cubing
     * accum_h,v,d is converted to floating-point for score calculation
     * 6bits are yet to be shifted from previous stage that is after dwt hence after cubing 18bits are to shifted
     * Hence final shift is 18-shift_inner_accum
     */
    csf_h = (double)(accum_h / pow(2, shift_accum)) * pow(rfactor[0], 3);
    csf_v = (double)(accum_v / pow(2, shift_accum)) * pow(rfactor[1], 3);
    csf_d = (double)(accum_d / pow(2, shift_accum)) * pow(rfactor[2], 3);

    den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return(den_scale_h + den_scale_v + den_scale_d);

}

/**
 * Combination of adm_csf_s and adm_sum_cube_s for csf_o based den_scale for scale 1,2,3
 */
static float integer_adm_csf_den_scale123_s(const integer_i4_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor)
{
    int32_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    uint64_t accum_inner_h, accum_inner_v, accum_inner_d;
    float den_scale_h, den_scale_v, den_scale_d;

    uint32_t shift_sq[3] = { 31, 30, 31 };
    uint32_t add_before_shift_sq[3];
    int idx;
    for (idx = 0; idx < 3; idx++)
    {
        add_before_shift_sq[idx] = (uint32_t)pow(2, shift_sq[idx]);
    }

    uint64_t val;

    /* The computation of the denominator scales is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5;
    int top = h * border_factor - 0.5;
    int right = w - left;
    int bottom = h - top;

    uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    uint32_t add_before_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

    uint32_t shift_inner_accum = (uint32_t)ceil(log2(bottom - top));
    uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    int accum_convert_float[3] = { 32, 27, 23 };
    int i, j;
    /**
     * The shifts are done such that overflow doesn't happen i.e results are less than 2^64
     */
    for (i = top; i < bottom; ++i)
    {
        accum_inner_h = 0;
        accum_inner_v = 0;
        accum_inner_d = 0;
        src_h = src->band_h + i * src_px_stride;
        src_v = src->band_v + i * src_px_stride;
        src_d = src->band_d + i * src_px_stride;
        for (j = left; j < right; ++j)
        {

            uint32_t h = (uint32_t)abs(src_h[j]);
            uint32_t v = (uint32_t)abs(src_v[j]);
            uint32_t d = (uint32_t)abs(src_d[j]);

            val = ((((((uint64_t)h * h) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * h) + add_before_shift_cub) >> shift_cub;

            accum_inner_h += val;

            val = ((((((uint64_t)v * v) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * v) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;

            val = ((((((uint64_t)d * d) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * d) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;
        }

        accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
        accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
        accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
    }
    double csf_h, csf_v, csf_d;

    /**
     * All the results are converted to floating-point to calculate the scores
     * For all scales the final shift is 3*shifts from dwt - total shifts done here
     */
    csf_h = (double)(accum_h / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[0], 3);
    csf_v = (double)(accum_v / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[1], 3);
    csf_d = (double)(accum_d / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[2], 3);

    den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return (den_scale_h + den_scale_v + den_scale_d);
}

/**
 * Works similar to fixed-point function adm_cm_s
 */
static float integer_adm_cm_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *csf_f, const integer_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int flt_stride, int csf_a_stride, double border_factor, int scale)
{
    /* Take decouple_r as src and do dsf_s on decouple_r here to get csf_r */
    int16_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

    //rfactor is left shifted by 21 for rfactor[0,1] and by 23 for rfactor[2]
    uint16_t rfactor[3] = { 36453,36453,49417 };

    int32_t shift_xhsq = 29;
    int32_t shift_xvsq = 29;
    int32_t shift_xdsq = 30;

    int32_t add_before_shift_xhsq = 268435456;
    int32_t add_before_shift_xvsq = 268435456;
    int32_t add_before_shift_xdsq = 536870912;

    uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
    uint32_t add_before_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));

    uint32_t shift_xvcub = (uint32_t)ceil(log2(w) - 4);
    uint32_t add_before_shift_xvcub = (uint32_t)pow(2, (shift_xvcub - 1));

    uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
    uint32_t add_before_shift_xdcub = (uint32_t)pow(2, (shift_xdcub - 1));

    uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    int32_t shift_xhsub = 10;
    int32_t shift_xvsub = 10;
    int32_t shift_xdsub = 12;

    const int16_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    const int16_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    int src_px_stride = src_stride >> 2;    //divide by sizeof(int32_t)
    int flt_px_stride = flt_stride >> 2;    //divide by sizeof(int32_t)
    int csf_px_stride = csf_a_stride >> 2;  //divide by sizeof(int32_t)

    int32_t xh, xv, xd;
    int32_t thr;
    int32_t xh_sq, xv_sq, xd_sq;

    int64_t val;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h, accum_inner_v, accum_inner_d;
    float num_scale_h, num_scale_v, num_scale_d;

    /* The computation of the scales is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5;
    int top = h * border_factor - 0.5;
    int right = w - left;
    int bottom = h - top;

    int start_col = (left > 1) ? left : 1;
    int end_col = (right < (w - 1)) ? right : (w - 1);
    int start_row = (top > 1) ? top : 1;
    int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int i, j;

    /* i=0,j=0 */
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)src->band_h[0] * rfactor[0];
        xv = (int32_t)src->band_v[0] * rfactor[1];
        xd = (int32_t)src->band_d[0] * rfactor[2];
        INTEGER_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_px_stride, &thr, w, h, 0, 0);

        //thr is shifted to make it's Q format equivalent to xh,xv,xd
        xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
        xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
        xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        //Shifted after squaring to make it Q32. xh,xv,xd max value is 22930*rfactor[i]
        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);

        /**
         * max value of xh_sq and xv_sq is 1301381973 and that of xd_sq is 1195806729
         *
         * max(val before shift for h and v) is 9.995357299×10^17.
         * 9.995357299×10^17 * 2^4 is close to 2^64.
         * Hence shift is done based on width subtracting 4
         *
         * max(val before shift for d) is 1.355006643×10^18
         * 1.355006643×10^18 * 2^3 is close to 2^64
         * Hence shift is done based on width subtracting 3
         */
        val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub; //max(val before shift) is 9.995357299×10^17. 9.995357299×10^17 * 2^4 is close to 2^64. Hence shift is done based on width subtracting 4
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
        accum_inner_d += val;
    }

    /* i=0, j */
    if (top <= 0) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[j] * rfactor[0];
            xv = src->band_v[j] * rfactor[1];
            xd = src->band_d[j] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_px_stride, &thr, w, h, 0, j);

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = src->band_h[w - 1] * rfactor[0];
        xv = src->band_v[w - 1] * rfactor[1];
        xd = src->band_d[w - 1] * rfactor[2];
        INTEGER_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, 0, (w - 1));

        xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
        xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
        xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
        accum_inner_d += val;

    }
    //Shift is done based on height
    accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_px_stride + j] * rfactor[0];
                xv = src->band_v[i * src_px_stride + j] * rfactor[1];
                xd = src->band_d[i * src_px_stride + j] * rfactor[2];
                INTEGER_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

                xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
                xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
                xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
                accum_inner_d += val;

            }
            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_px_stride] * rfactor[0];
            xv = src->band_v[i * src_px_stride] * rfactor[1];
            xd = src->band_d[i * src_px_stride] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_px_stride + j] * rfactor[0];
                xv = src->band_v[i * src_px_stride + j] * rfactor[1];
                xd = src->band_d[i * src_px_stride + j] * rfactor[2];
                INTEGER_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

                xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
                xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
                xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
                accum_inner_d += val;

            }
            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else if ((left > 0) && (right > (w - 1))) /* Left border within frame, right outside */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_px_stride + j] * rfactor[0];
                xv = src->band_v[i * src_px_stride + j] * rfactor[1];
                xd = src->band_d[i * src_px_stride + j] * rfactor[2];
                INTEGER_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

                xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
                xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
                xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
                accum_inner_d += val;

            }
            /* j = w-1 */
            xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
            xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
            xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else /* Both borders outside frame */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_px_stride] * rfactor[0];
            xv = src->band_v[i * src_px_stride] * rfactor[1];
            xd = src->band_d[i * src_px_stride] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_px_stride + j] * rfactor[0];
                xv = src->band_v[i * src_px_stride + j] * rfactor[1];
                xd = src->band_d[i * src_px_stride + j] * rfactor[2];
                INTEGER_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

                xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
                xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
                xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
                accum_inner_d += val;

            }
            /* j = w-1 */
            xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
            xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
            xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = src->band_h[(h - 1) * src_px_stride] * rfactor[0];
        xv = src->band_v[(h - 1) * src_px_stride] * rfactor[1];
        xd = src->band_d[(h - 1) * src_px_stride] * rfactor[2];
        INTEGER_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), 0);

        xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
        xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
        xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
        accum_inner_d += val;

    }

    /* i=h-1,j */
    if (bottom > (h - 1)) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[(h - 1) * src_px_stride + j] * rfactor[0];
            xv = src->band_v[(h - 1) * src_px_stride + j] * rfactor[1];
            xd = src->band_d[(h - 1) * src_px_stride + j] * rfactor[2];
            INTEGER_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), j);

            xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
            xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
            xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
            accum_inner_d += val;

        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = src->band_h[(h - 1) * src_px_stride + w - 1] * rfactor[0];
        xv = src->band_v[(h - 1) * src_px_stride + w - 1] * rfactor[1];
        xd = src->band_d[(h - 1) * src_px_stride + w - 1] * rfactor[2];
        INTEGER_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), (w - 1));

        xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
        xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
        xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
        accum_inner_d += val;

    }
    accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

    /**
     * For h and v total shifts pending from last stage is 6 rfactor[0,1] has 21 shifts
     * => after cubing (6+21)*3=81 after squaring shifted by 29
     * hence pending is 52-shift's done based on width and height
     *
     * For d total shifts pending from last stage is 6 rfactor[2] has 23 shifts
     * => after cubing (6+23)*3=87 after squaring shifted by 30
     * hence pending is 57-shift's done based on width and height
     */
    float f_accum_h = (float)(accum_h / pow(2, (52 - shift_xhcub - shift_inner_accum)));
    float f_accum_v = (float)(accum_v / pow(2, (52 - shift_xvcub - shift_inner_accum)));
    float f_accum_d = (float)(accum_d / pow(2, (57 - shift_xdcub - shift_inner_accum)));

    num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

/**
 * Works similar to fixed-point function integer_adm_cm_s for scale 1,2,3
 */
static float integer_i4_adm_cm_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *csf_f, const integer_i4_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int flt_stride, int csf_a_stride, double border_factor, int scale)
{
    /* Take decouple_r as src and do dsf_s on decouple_r here to get csf_r */
    int32_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint32_t rfactor[3] = { (uint32_t)(rfactor1[0] * pow(2, 32)), (uint32_t)(rfactor1[1] * pow(2, 32)), (uint32_t)(rfactor1[2] * pow(2, 32)) };

    int32_t shift_dst[3] = { 28, 28, 28 };
    int32_t shift_flt[3] = { 32, 32, 32 };

    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    int idx;
    for (idx = 0; idx < 3; idx++)
    {
        add_bef_shift_dst[idx] = (int32_t)pow(2, (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (int32_t)pow(2, (shift_flt[idx] - 1));

    }

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    uint32_t add_before_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

    uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    float final_shift[3] = { pow(2,(45 - shift_cub - shift_inner_accum)),pow(2,(39 - shift_cub - shift_inner_accum)), pow(2,(36 - shift_cub - shift_inner_accum)) };

    int32_t shift_sq = 30;

    int32_t add_before_shift_sq = 536870912; //2^29

    int32_t shift_sub = 0;

    const int32_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    const int32_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    int src_px_stride = src_stride >> 2;    //divide by sizeof(int32_t)
    int flt_px_stride = flt_stride >> 2;    //divide by sizeof(int32_t)
    int csf_px_stride = csf_a_stride >> 2;  //divide by sizeof(int32_t)

    int32_t xh, xv, xd;
    int32_t thr;
    int32_t xh_sq, xv_sq, xd_sq;

    int64_t val;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h, accum_inner_v, accum_inner_d;
    float num_scale_h, num_scale_v, num_scale_d;

    /* The computation of the scales is not required for the regions which lie outside the frame borders */
    int left = w * border_factor - 0.5;
    int top = h * border_factor - 0.5;
    int right = w - left;
    int bottom = h - top;

    // printf("left %d right %d top %d bottom %d\n",left,right,top,bottom);
    // exit(0);
    int start_col = (left > 1) ? left : 1;
    int end_col = (right < (w - 1)) ? right : (w - 1);
    int start_row = (top > 1) ? top : 1;
    int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int i, j;

    /* i=0,j=0 */
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[0] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[0] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[0] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        INTEGER_I4_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_px_stride, &thr, w, h, 0, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        xh = abs(xh) - (thr >> shift_sub); //Shifted to make it equivalent to xh
        xv = abs(xv) - (thr >> shift_sub);
        xd = abs(xd) - (thr >> shift_sub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
        accum_inner_d += val;

    }

    /* i=0, j */
    if (top <= 0)
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_px_stride, &thr, w, h, 0, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;
        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[w - 1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[w - 1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[w - 1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        INTEGER_I4_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, 0, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        xh = abs(xh) - (thr >> shift_sub);
        xv = abs(xv) - (thr >> shift_sub);
        xd = abs(xd) - (thr >> shift_sub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
        accum_inner_d += val;
    }

    accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            for (j = start_col; j < end_col; ++j)
            {

                xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                INTEGER_I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                xh = abs(xh) - (thr >> shift_sub);
                xv = abs(xv) - (thr >> shift_sub);
                xd = abs(xd) - (thr >> shift_sub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
                accum_inner_d += val;
            }
            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                INTEGER_I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                xh = abs(xh) - (thr >> shift_sub);
                xv = abs(xv) - (thr >> shift_sub);
                xd = abs(xd) - (thr >> shift_sub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
                accum_inner_d += val;
            }
            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else if ((left > 0) && (right > (w - 1))) /* Left border within frame, right outside */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                INTEGER_I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                xh = abs(xh) - (thr >> shift_sub);
                xv = abs(xv) - (thr >> shift_sub);
                xd = abs(xd) - (thr >> shift_sub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
                accum_inner_d += val;
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;

            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else /* Both borders outside frame */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                INTEGER_I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                xh = abs(xh) - (thr >> shift_sub);
                xv = abs(xv) - (thr >> shift_sub);
                xd = abs(xd) - (thr >> shift_sub);

                xh = xh < 0 ? 0 : xh;
                xv = xv < 0 ? 0 : xv;
                xd = xd < 0 ? 0 : xd;

                xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
                xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
                xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
                val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
                accum_inner_h += val;
                val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
                accum_inner_v += val;
                val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
                accum_inner_d += val;
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + w - 1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + w - 1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + w - 1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;

            accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        INTEGER_I4_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        xh = abs(xh) - (thr >> shift_sub);
        xv = abs(xv) - (thr >> shift_sub);
        xd = abs(xd) - (thr >> shift_sub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
        accum_inner_d += val;
    }

    /* i=h-1,j */
    if (bottom > (h - 1))
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            INTEGER_I4_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            xh = abs(xh) - (thr >> shift_sub);
            xv = abs(xv) - (thr >> shift_sub);
            xd = abs(xd) - (thr >> shift_sub);

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
            xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
            xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
            val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
            accum_inner_d += val;
        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride + w - 1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride + w - 1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride + w - 1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        INTEGER_I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        xh = abs(xh) - (thr >> shift_sub);
        xv = abs(xv) - (thr >> shift_sub);
        xd = abs(xd) - (thr >> shift_sub);

        xh = xh < 0 ? 0 : xh;
        xv = xv < 0 ? 0 : xv;
        xd = xd < 0 ? 0 : xd;

        xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
        xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
        xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
        val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
        accum_inner_h += val;
        val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
        accum_inner_v += val;
        val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
        accum_inner_d += val;
    }
    accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

    /**
     * Converted to floating-point for calculating the final scores
     * Final shifts is calculated from 3*(shifts_from_previous_stage(i.e src comes from dwt)+32)-total_shifts_done_in_this_function
     */
    float f_accum_h = (float)(accum_h / final_shift[scale - 1]);
    float f_accum_v = (float)(accum_v / final_shift[scale - 1]);
    float f_accum_d = (float)(accum_d / final_shift[scale - 1]);

    num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

/*
 * This function just type casts the results of scale 0 such that next stage input is 32bits
 */
static void i16_to_i32(const integer_adm_dwt_band_t_s *src, integer_i4_adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride)
{
    int i, j;
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)

    for (i = 0; i < (h + 1) / 2; i++)
    {
        int32_t *dst_band_a_addr = &dst->band_a[i * dst_px_stride + 0];
        int16_t *src_band_a_addr = &src->band_a[i * dst_px_stride + 0];
        for (j = 0; j < (w + 1) / 2; ++j)
        {
            *(dst_band_a_addr++) = (int32_t)(*(src_band_a_addr++));
        }
    }
}

/**
 * Works similar to floating-point function adm_dwt2_s for 8 bit input
 */
static void integer_adm_dwt2_8_s(const uint8_t *src, const integer_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, const int32_t *tmp_ref, int inp_size_bits)
{
    const int16_t *filter_lo = integer_dwt2_db2_coeffs_lo_s;
    const int16_t *filter_hi = integer_dwt2_db2_coeffs_hi_s;

    int32_t add_bef_shift_round_VP = (int32_t)pow(2, (inp_size_bits - 1));
    int32_t add_bef_shift_round_HP = 32768;
    int16_t shift_VerticalPass = inp_size_bits;
    int16_t shift_HorizontalPass = 16;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)
    int16_t *tmplo = tmp_ref;
    int16_t *tmphi = tmplo + w;
    int16_t s0, s1, s2, s3;
    uint16_t u_s0, u_s1, u_s2, u_s3;
    int32_t accum;

    int i, j;
    int j0, j1, j2, j3;

    for (i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            u_s0 = src[ind_y[0][i] * src_px_stride + j];
            u_s1 = src[ind_y[1][i] * src_px_stride + j];
            u_s2 = src[ind_y[2][i] * src_px_stride + j];
            u_s3 = src[ind_y[3][i] * src_px_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)integer_dwt2_db2_coeffs_lo_sum * add_bef_shift_round_VP;

            /**
            *  Accum is Q24 as imgcoeff1 is Q8 fcoeff sums to -54822 to 54822. hence accum varies between -7017216 to 7017216 shifted by 8 to make tmp Q16(max(tmp is -27411 to 27411))
            */
            tmplo[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)integer_dwt2_db2_coeffs_hi_sum * add_bef_shift_round_VP;

            tmphi[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;
        }

        /* Horizontal pass (lo and hi). */
        for (j = 0; j < (w + 1) / 2; ++j) {

            j0 = ind_x[0][j];
            j1 = ind_x[1][j];
            j2 = ind_x[2][j];
            j3 = ind_x[3][j];

            s0 = tmplo[j0];
            s1 = tmplo[j1];
            s2 = tmplo[j2];
            s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            /**
             *  Accum is Q32 as imgcoeff1 is Q16(-27411 to 27411) fcoeff sums to -54822 to 54822. hence accum varies between -1502725842 to 1502725842 shifted by 16 to make tmp Q16(max(tmp is -22930 to 22930))
             */
            dst->band_a[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

        }
    }
}

/**
 * Works similar to floating-point function adm_dwt2_s for 16 bit input
 */
static void integer_adm_dwt2_16_s(const uint16_t *src, const integer_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, const int32_t *tmp_ref, int inp_size_bits)
{
    const int16_t *filter_lo = integer_dwt2_db2_coeffs_lo_s;
    const int16_t *filter_hi = integer_dwt2_db2_coeffs_hi_s;

    int32_t add_bef_shift_round_VP = (int32_t)pow(2, (inp_size_bits - 1));
    int32_t add_bef_shift_round_HP = 32768;
    int16_t shift_VerticalPass = inp_size_bits;
    int16_t shift_HorizontalPass = 16;

    int src_px_stride = src_stride >> 2;  //divide by sizeof(int32_t)
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)

    int16_t *tmplo = tmp_ref;
    int16_t *tmphi = tmplo + w;
    int16_t s0, s1, s2, s3;
    uint16_t u_s0, u_s1, u_s2, u_s3;
    int32_t accum;

    int i, j;
    int j0, j1, j2, j3;

    for (i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            u_s0 = src[ind_y[0][i] * src_px_stride + j];
            u_s1 = src[ind_y[1][i] * src_px_stride + j];
            u_s2 = src[ind_y[2][i] * src_px_stride + j];
            u_s3 = src[ind_y[3][i] * src_px_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)integer_dwt2_db2_coeffs_lo_sum * add_bef_shift_round_VP;
            
            /**
            *  Accum is Q24 as imgcoeff1 is Q10 fcoeff sums to -54822 to 54822. hence accum varies between -7017216 to 7017216 shifted by 8 to make tmp Q16(max(tmp is -27411 to 27411))
            */
            tmplo[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)integer_dwt2_db2_coeffs_hi_sum * add_bef_shift_round_VP;

            tmphi[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;
        }

        /* Horizontal pass (lo and hi). */
        for (j = 0; j < (w + 1) / 2; ++j) {

            j0 = ind_x[0][j];
            j1 = ind_x[1][j];
            j2 = ind_x[2][j];
            j3 = ind_x[3][j];

            s0 = tmplo[j0];
            s1 = tmplo[j1];
            s2 = tmplo[j2];
            s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            /**
             *  Accum is Q32 as imgcoeff1 is Q16(-27411 to 27411) fcoeff sums to -54822 to 54822. hence accum varies between -1502725842 to 1502725842 shifted by 16 to make tmp Q16(max(tmp is -22930 to 22930))
             */
            dst->band_a[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;
            
            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

        }
    }
}

/**
 * Works similar to floating-point function adm_dwt2_s for scale 1, 2, 3
 */
static void integer_adm_dwt2_scale123_combined_s(const int32_t *integer_i4_curr_ref_scale, const integer_i4_adm_dwt_band_t_s *integer_i4_ref_dwt2, const int32_t *integer_i4_curr_dis_scale, const integer_i4_adm_dwt_band_t_s *integer_i4_dis_dwt2, int **ind_y, int **ind_x, int w, int h, int curr_ref_stride, int curr_dis_stride, int dst_stride, int scale, const int32_t *tmp_ref)
{
    const int16_t *filter_lo = integer_dwt2_db2_coeffs_lo_s;
    const int16_t *filter_hi = integer_dwt2_db2_coeffs_hi_s;

    int32_t add_bef_shift_round_VP[3] = { 0, 32768, 32768 };
    int32_t add_bef_shift_round_HP[3] = { 16384, 32768, 16384 };
    int16_t shift_VerticalPass[3] = { 0, 16, 16 };
    int16_t shift_HorizontalPass[3] = { 15, 16, 15 };

    //float f_shift[3] = { pow(2, 21), pow(2, 19), pow(2, 18) };

    int src1_px_stride = curr_ref_stride >> 2;  //divide by sizeof(int32_t)
    int src2_px_stride = curr_dis_stride >> 2;  //divide by sizeof(int32_t)
    int dst_px_stride = dst_stride >> 2;  //divide by sizeof(int32_t)

    int32_t *tmplo_ref = tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;
    int32_t s10, s11, s12, s13;

    int64_t accum_ref;

    int i, j;
    int j0, j1, j2, j3;

    for (i = 0; i < (h + 1) / 2; ++i)
    {
        /* Vertical pass. */
        for (j = 0; j < w; ++j)
        {
            s10 = integer_i4_curr_ref_scale[ind_y[0][i] * src1_px_stride + j];
            s11 = integer_i4_curr_ref_scale[ind_y[1][i] * src1_px_stride + j];
            s12 = integer_i4_curr_ref_scale[ind_y[2][i] * src1_px_stride + j];
            s13 = integer_i4_curr_ref_scale[ind_y[3][i] * src1_px_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);

            s10 = integer_i4_curr_dis_scale[ind_y[0][i] * src2_px_stride + j];
            s11 = integer_i4_curr_dis_scale[ind_y[1][i] * src2_px_stride + j];
            s12 = integer_i4_curr_dis_scale[ind_y[2][i] * src2_px_stride + j];
            s13 = integer_i4_curr_dis_scale[ind_y[3][i] * src2_px_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
        }
        /* Horizontal pass (lo and hi). */
        for (j = 0; j < (w + 1) / 2; ++j)
        {
            j0 = ind_x[0][j];
            j1 = ind_x[1][j];
            j2 = ind_x[2][j];
            j3 = ind_x[3][j];

            s10 = tmplo_ref[j0];
            s11 = tmplo_ref[j1];
            s12 = tmplo_ref[j2];
            s13 = tmplo_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            integer_i4_ref_dwt2->band_a[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            integer_i4_ref_dwt2->band_v[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            s10 = tmphi_ref[j0];
            s11 = tmphi_ref[j1];
            s12 = tmphi_ref[j2];
            s13 = tmphi_ref[j3];
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            integer_i4_ref_dwt2->band_h[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            integer_i4_ref_dwt2->band_d[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            s10 = tmplo_dis[j0];
            s11 = tmplo_dis[j1];
            s12 = tmplo_dis[j2];
            s13 = tmplo_dis[j3];
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            integer_i4_dis_dwt2->band_a[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            integer_i4_dis_dwt2->band_v[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            s10 = tmphi_dis[j0];
            s11 = tmphi_dis[j1];
            s12 = tmphi_dis[j2];
            s13 = tmphi_dis[j3];
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            integer_i4_dis_dwt2->band_h[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            integer_i4_dis_dwt2->band_d[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
        }
    }
}

int integer_compute_adm(const pixel *ref, const pixel *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor, int inp_size_bits, Integer_AdmState *s)
{
	double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);

	char *data_top;
    int32_t *tmp_ref;

    char *ind_buf_y = NULL;
    char *ind_buf_x = NULL;
	int *ind_y[4], *ind_x[4];

	integer_adm_dwt_band_t_s integer_ref_dwt2;
	integer_adm_dwt_band_t_s integer_dis_dwt2;

	integer_adm_dwt_band_t_s integer_decouple_r;
	integer_adm_dwt_band_t_s integer_decouple_a;

	integer_adm_dwt_band_t_s integer_csf_a;
	integer_adm_dwt_band_t_s integer_csf_f;

	integer_i4_adm_dwt_band_t_s integer_i4_ref_dwt2;
	integer_i4_adm_dwt_band_t_s integer_i4_dis_dwt2;

	integer_i4_adm_dwt_band_t_s integer_i4_decouple_r;
	integer_i4_adm_dwt_band_t_s integer_i4_decouple_a;

	integer_i4_adm_dwt_band_t_s integer_i4_csf_a;
	integer_i4_adm_dwt_band_t_s integer_i4_csf_f;

    uint8_t *integer_curr_ref_scale_8 = NULL;
    uint8_t *integer_curr_dis_scale_8 = NULL;

    uint16_t *integer_curr_ref_scale_16 = NULL;
    uint16_t *integer_curr_dis_scale_16 = NULL;

    int32_t *integer_i4_curr_ref_scale = NULL;
    int32_t *integer_i4_curr_dis_scale = NULL;

    if (inp_size_bits == 8)
    {
        integer_curr_ref_scale_8 = (uint8_t *)ref;
        integer_curr_dis_scale_8 = (uint8_t *)dis;
    }
    else
    {
        integer_curr_ref_scale_16 = (uint16_t *)ref;
        integer_curr_dis_scale_16 = (uint16_t *)dis;
    }

	int curr_ref_stride = ref_stride;
	int curr_dis_stride = dis_stride;

	int orig_h = h;
    int buf_stride = s->ind_size_x;

	double num = 0;
	double den = 0;

	int scale;
	int ret = 1;

    tmp_ref = (int32_t*)s->tmp_ref;

	data_top = (char *)s->data_buf;
    data_top = integer_init_dwt_band(&integer_ref_dwt2, data_top, s->buf_sz_one / 2);
	data_top = integer_init_dwt_band(&integer_dis_dwt2, data_top, s->buf_sz_one / 2);
	data_top = integer_init_dwt_band_hvd(&integer_decouple_r, data_top, s->buf_sz_one / 2);
	data_top = integer_init_dwt_band_hvd(&integer_decouple_a, data_top, s->buf_sz_one / 2);
	data_top = integer_init_dwt_band_hvd(&integer_csf_a, data_top, s->buf_sz_one / 2);
	data_top = integer_init_dwt_band_hvd(&integer_csf_f, data_top, s->buf_sz_one / 2);
	
	data_top = integer_i4_init_dwt_band(&integer_i4_ref_dwt2, data_top, s->buf_sz_one);
	data_top = integer_i4_init_dwt_band(&integer_i4_dis_dwt2, data_top, s->buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_decouple_r, data_top, s->buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_decouple_a, data_top, s->buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_csf_a, data_top, s->buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_csf_f, data_top, s->buf_sz_one);

    ind_buf_y = (char *)s->buf_y_orig;
	ind_y[0] = (int*)ind_buf_y; ind_buf_y += s->ind_size_y;
	ind_y[1] = (int*)ind_buf_y; ind_buf_y += s->ind_size_y;
	ind_y[2] = (int*)ind_buf_y; ind_buf_y += s->ind_size_y;
	ind_y[3] = (int*)ind_buf_y; ind_buf_y += s->ind_size_y;

    ind_buf_x = (char *)s->buf_x_orig;
    ind_x[0] = (int*)ind_buf_x; ind_buf_x += s->ind_size_x;
	ind_x[1] = (int*)ind_buf_x; ind_buf_x += s->ind_size_x;
	ind_x[2] = (int*)ind_buf_x; ind_buf_x += s->ind_size_x;
	ind_x[3] = (int*)ind_buf_x; ind_buf_x += s->ind_size_x;

	for (scale = 0; scale < 4; ++scale) {
		float num_scale = 0.0;
		float den_scale = 0.0;

        integer_dwt2_src_indices_filt_s(ind_y, ind_x, w, h);
		if(scale==0)
		{
            if (inp_size_bits == 8)
            {
                integer_adm_dwt2_8_s(integer_curr_ref_scale_8, &integer_ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride, buf_stride, tmp_ref, inp_size_bits);
                integer_adm_dwt2_8_s(integer_curr_dis_scale_8, &integer_dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride, buf_stride, tmp_ref, inp_size_bits);
            }
            else
            {
                integer_adm_dwt2_16_s(integer_curr_ref_scale_16, &integer_ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride, buf_stride, tmp_ref, inp_size_bits);
                integer_adm_dwt2_16_s(integer_curr_dis_scale_16, &integer_dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride, buf_stride, tmp_ref, inp_size_bits);
            }

			i16_to_i32(&integer_ref_dwt2, &integer_i4_ref_dwt2, w, h, buf_stride, buf_stride);
			i16_to_i32(&integer_dis_dwt2, &integer_i4_dis_dwt2, w, h, buf_stride, buf_stride);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			integer_adm_decouple_s(&integer_ref_dwt2, &integer_dis_dwt2, &integer_decouple_r, &integer_decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride, border_factor);

			den_scale = integer_adm_csf_den_scale_s(&integer_ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor);

			integer_adm_csf_s(&integer_decouple_a, &integer_csf_a, &integer_csf_f, orig_h, scale, w, h, buf_stride, buf_stride, border_factor);

			num_scale = integer_adm_cm_s(&integer_decouple_r, &integer_csf_f, &integer_csf_a, w, h, buf_stride, buf_stride, buf_stride, border_factor, scale);

		}
		else
		{
            integer_adm_dwt2_scale123_combined_s(integer_i4_curr_ref_scale, &integer_i4_ref_dwt2, integer_i4_curr_dis_scale, &integer_i4_dis_dwt2, ind_y, ind_x, w, h, curr_ref_stride, curr_dis_stride, buf_stride, scale, tmp_ref);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			integer_adm_decouple_scale123_s(&integer_i4_ref_dwt2, &integer_i4_dis_dwt2, &integer_i4_decouple_r, &integer_i4_decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride, border_factor, scale);

			den_scale = integer_adm_csf_den_scale123_s(&integer_i4_ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor);

			integer_i4_adm_csf_s(&integer_i4_decouple_a, &integer_i4_csf_a, &integer_i4_csf_f, orig_h, scale, w, h, buf_stride, buf_stride, border_factor);

			num_scale = integer_i4_adm_cm_s(&integer_i4_decouple_r, &integer_i4_csf_f, &integer_i4_csf_a, w, h, buf_stride, buf_stride, buf_stride, border_factor, scale);

		}

		num += num_scale;
		den += den_scale;

		integer_i4_curr_ref_scale = integer_i4_ref_dwt2.band_a;
		integer_i4_curr_dis_scale = integer_i4_dis_dwt2.band_a;

		curr_ref_stride = buf_stride;
		curr_dis_stride = buf_stride;

		scores[2 * scale + 0] = num_scale;
		scores[2 * scale + 1] = den_scale;
	}

	num = num < numden_limit ? 0 : num;
	den = den < numden_limit ? 0 : den;

	if (den == 0.0)
	{
		*score = 1.0f;
	}
	else
	{
		*score = num / den;
	}
	*score_num = num;
	*score_den = den;

	ret = 0;

	return ret;
}
