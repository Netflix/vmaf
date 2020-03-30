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

#ifndef ADM_TOOLS_H_
#define ADM_TOOLS_H_

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1
#define ADM_CM_THRESH_S_0_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			float sum = 0; \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[0]); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[src_px_stride + 1]; \
			sum += flt_ptr[src_px_stride]; \
			sum += flt_ptr[src_px_stride + 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1 
#define ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			float sum = 0; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[w - 1]); \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[src_px_stride + w - 2]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			sum += flt_ptr[src_px_stride + w - 1]; \
			*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1 
#define ADM_CM_THRESH_S_0_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			float sum = 0; \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[j]); \
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[src_px_stride + j - 1]; \
			sum += flt_ptr[src_px_stride + j]; \
			sum += flt_ptr[src_px_stride + j + 1];  \
			*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1 
#define ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			float sum = 0; \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[0]); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1 
#define ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			float sum = 0; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[w - 1]); \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1 
#define ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			float sum = 0; \
		src_ptr += (src_px_stride * (h - 2)); \
			flt_ptr += (src_px_stride * (h - 2)); \
			sum += flt_ptr[j - 1];\
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[j]); \
			sum += flt_ptr[j + 1]; \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1 
#define ADM_CM_THRESH_S_I_J(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			float sum = 0; \
			float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[j - 1]; \
			sum += flt_ptr[j]; \
			sum += flt_ptr[j + 1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[j - 1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[j]); \
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
#define ADM_CM_THRESH_S_I_0(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
			float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			float sum = 0; \
			src_ptr += (src_px_stride * (i - 1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[1]; \
			sum += flt_ptr[0]; \
			sum += flt_ptr[1]; \
			src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[1]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[0]); \
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
#define ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_px_stride,accum,w,h,i,j) \
{ \
	float sum = 0; \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		float *src_ptr = angles[theta]; \
			float *flt_ptr = flt_angles[theta]; \
			float sum = 0; \
		src_ptr += (src_px_stride * (i-1)); \
			flt_ptr += (src_px_stride * (i - 1)); \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
		src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += FLOAT_ONE_BY_15 * fabsf(src_ptr[w - 1]); \
			sum += flt_ptr[w - 1]; \
		src_ptr += src_px_stride; \
			flt_ptr += src_px_stride; \
			sum += flt_ptr[w - 2]; \
			sum += flt_ptr[w - 1]; \
			sum += flt_ptr[w - 1]; \
			*accum += sum; \
	} \
}

float adm_sum_cube_s(const float *x, int w, int h, int stride, double border_factor);

void adm_decouple_s(const adm_dwt_band_t_s *ref, const adm_dwt_band_t_s *dis, const adm_dwt_band_t_s *r, const adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor);

void adm_csf_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor);

void adm_cm_thresh_s(const adm_dwt_band_t_s *src, float *dst, int w, int h, int src_stride, int dst_stride);

float adm_csf_den_scale_s(const adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor);

float adm_cm_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int dst_stride, int csf_a_stride, double border_factor, int scale);

void dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h);

void adm_dwt2_s(const float *src, const adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, int scale);

#endif /* ADM_TOOLS_H_ */
