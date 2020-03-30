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

#define M_PI 3.14159265358979323846264338327
#include <math.h>
#include <stdint.h>
#include "common/macros.h"

#pragma once

#ifndef INTEGER_ADM_TOOLS_H_
#define INTEGER_ADM_TOOLS_H_

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
#define INTEGER_ADM_CM_THRESH_FIXED_S_I_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
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
#define INTEGER_ADM_CM_THRESH_FIXED_S_I_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
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
#define INTEGER_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
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
#define INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_J(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
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
#define INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_0(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
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
#define INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j,add_bef_shift,shift) \
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

void integer_adm_decouple_s(const integer_adm_dwt_band_t_s *ref, const integer_adm_dwt_band_t_s *dis, const integer_adm_dwt_band_t_s *r, const integer_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor);

void integer_adm_decouple_scale123_s(const integer_i4_adm_dwt_band_t_s *ref, const integer_i4_adm_dwt_band_t_s *dis, const integer_i4_adm_dwt_band_t_s *r, const integer_i4_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor, int scale);

void integer_adm_csf_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *dst, const integer_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor);

void integer_i4_adm_csf_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *dst, const integer_i4_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor);

float integer_adm_csf_den_scale_s(const integer_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor);

float integer_adm_csf_den_scale123_s(const integer_i4_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor);

float integer_adm_cm_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *dst, const integer_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int dst_stride, int csf_a_stride, double border_factor, int scale);

float integer_i4_adm_cm_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *dst, const integer_i4_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int dst_stride, int csf_a_stride, double border_factor, int scale);

void integer_adm_dwt2_s(const int16_t *src, const integer_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, const int32_t *tmp_ref, int inp_size_bits);

void integer_adm_dwt2_scale123_s(const int32_t *src, const integer_i4_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, int scale);

void i16_to_i32(const integer_adm_dwt_band_t_s *src, integer_i4_adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride);

void integer_adm_dwt2_scale123_combined_s(const int32_t *i4_curr_ref_scale_fixed, const integer_i4_adm_dwt_band_t_s *i4_ref_dwt2_fixed, const int32_t *i4_curr_dis_scale_fixed, const integer_i4_adm_dwt_band_t_s *i4_dis_dwt2_fixed, int **ind_y, int **ind_x, int w, int h, int curr_ref_stride, int curr_dis_stride, int dst_stride, int scale, const int32_t *tmp_ref);

#endif /* INTEGER_ADM_TOOLS_H_ */
