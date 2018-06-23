/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#define M_PI 3.14159265358979323846264338327
#include "common/macros.h"

#pragma once

#ifndef ADM_TOOLS_H_
#define ADM_TOOLS_H_

typedef struct adm_dwt_band_t_s {
    float *band_a; /* Low-pass V + low-pass H. */
    float *band_v; /* Low-pass V + high-pass H. */
    float *band_h; /* High-pass V + low-pass H. */
    float *band_d; /* High-pass V + high-pass H. */
} adm_dwt_band_t_s;

float adm_sum_cube_s(const float *x, int w, int h, int stride, double border_factor);

void adm_decouple_s(const adm_dwt_band_t_s *ref, const adm_dwt_band_t_s *dis, const adm_dwt_band_t_s *r, const adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride);

void adm_csf_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride);

void adm_cm_thresh_s(const adm_dwt_band_t_s *src, float *dst, int w, int h, int src_stride, int dst_stride);

void adm_cm_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const float *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride);

void adm_dwt2_s(const float *src, const adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride);

void adm_buffer_copy(const void *src, void *dst, int linewidth, int h, int src_stride, int dst_stride);

/* ================= */
/* Noise floor model */
/* ================= */

#define VIEW_DIST 3.0f
//#define VIEW_DIST 4.0f
//#define VIEW_DIST 5.0f
//#define VIEW_DIST 6.0f

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
struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

// 0 -> Y, 1 -> Cb, 2 -> Cr
static const struct dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    { .a = 0.495, .k = 0.466, .f0 = 0.401, .g = { 1.501, 1.0, 0.534, 1.0} },
    { .a = 1.633, .k = 0.353, .f0 = 0.209, .g = { 1.520, 1.0, 0.502, 1.0} },
    { .a = 0.944, .k = 0.521, .f0 = 0.404, .g = { 1.868, 1.0, 0.516, 1.0} }
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
FORCE_INLINE inline float dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = VIEW_DIST * REF_DISPLAY_HEIGHT * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0,lambda+1)*params->f0*params->g[theta]/r);
    float Q = 2.0*params->a*pow(10.0,params->k*temp*temp)/dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

#endif /* ADM_TOOLS_H_ */
