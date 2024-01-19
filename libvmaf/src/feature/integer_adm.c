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

#include "cpu.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_adm.h"
#include "log.h"

#if ARCH_X86
#include "x86/adm_avx2.h"
#elif ARCH_AARCH64
#include "arm64/adm_neon.h"
#include <arm_neon.h>
#endif

typedef struct AdmState {
    size_t integer_stride;
    AdmBuffer buf;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    void (*dwt2_8)(const uint8_t *src, const adm_dwt_band_t *dst,
                   AdmBuffer *buf, int w, int h, int src_stride,
                   int dst_stride);
    VmafDictionary *feature_name_dict;
} AdmState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdh",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static inline float
dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
        double adm_norm_view_dist, int adm_ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree
    // of visual angle. This should be 56.55
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0, lambda + 1)*params->f0*params->g[theta] / r);
    float Q = 2.0*params->a*pow(10.0, params->k*temp*temp) /
        dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define ADM_CM_THRESH_S_0_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_0_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1];  \
		*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[j - 1];\
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_I_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define ADM_CM_THRESH_S_I_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i-1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define I4_ADM_CM_THRESH_S_0_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs( src_ptr[0]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define I4_ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_0_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1];  \
		*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define I4_ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
	    int32_t sum = 0; \
	    int32_t *src_ptr = angles[theta]; \
	    int32_t *flt_ptr = flt_angles[theta]; \
	    src_ptr += (src_stride * (h - 2)); \
	    flt_ptr += (src_stride * (h - 2)); \
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[0]; \
	    sum += flt_ptr[1]; \
	    src_ptr += src_stride; \
	    flt_ptr += src_stride; \
	    sum += flt_ptr[1]; \
	    sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[0]; \
	    sum += flt_ptr[1]; \
	    *accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[j - 1];\
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_I_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define I4_ADM_CM_THRESH_S_I_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define I4_ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i-1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

#define ADM_CM_ACCUM_ROUND(x, thr, shift_xsub, x_sq, add_shift_xsq, shift_xsq, val, \
                           add_shift_xcub, shift_xcub, accum_inner) \
{ \
    x = abs(x) - ((int32_t)(thr) << shift_xsub); \
    x = x < 0 ? 0 : x; \
    x_sq = (int32_t)((((int64_t)x * x) + add_shift_xsq) >> shift_xsq); \
    val = (((int64_t)x_sq * x) + add_shift_xcub) >> shift_xcub; \
    accum_inner += val; \
}

#define I4_ADM_CM_ACCUM_ROUND(x, thr, shift_sub, x_sq, add_shift_sq, shift_sq, val, \
                              add_shift_cub, shift_cub, accum_inner)    \
{ \
    x = abs(x) - (thr >> shift_sub); \
    x = x < 0 ? 0 : x; \
    x_sq = (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq); \
    val = (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub; \
    accum_inner += val; \
}

static void dwt2_src_indices_filt(int **src_ind_y, int **src_ind_x, int w, int h)
{
    int ind0, ind1, ind2, ind3;
    const unsigned h_half = (h + 1) / 2;
    const unsigned w_half = (w + 1) / 2;
    unsigned i, j;
    /* Vertical pass */
    {   /* i : 0 */
        src_ind_y[0][0] = 1;
        src_ind_y[1][0] = 0;
        src_ind_y[2][0] = 1;
        src_ind_y[3][0] = 2;
    }
    for (i = 1; i < h_half - 2; ++i) { /* i : 1 to  h_half - 3*/
        ind1 = 2 * i;
        ind0 = ind1 - 1;
        ind2 = ind1 + 1;
        ind3 = ind1 + 2;
        src_ind_y[0][i] = ind0;
        src_ind_y[1][i] = ind1;
        src_ind_y[2][i] = ind2;
        src_ind_y[3][i] = ind3;
    }
    for (i = h_half - 2; i < h_half; ++i) { /* i : h_half - 3 to  h_half */
        ind1 = 2 * i;
        ind0 = ind1 - 1;
        ind2 = ind1 + 1;
        ind3 = ind1 + 2;
        if (ind0 >= h) {
            ind0 = (2 * h - ind0 - 1);
        }
        if (ind1 >= h) {
            ind1 = (2 * h - ind1 - 1);
        }
        if (ind2 >= h) {
            ind2 = (2 * h - ind2 - 1);
        }
        if (ind3 >= h) {
            ind3 = (2 * h - ind3 - 1);
        }
        src_ind_y[0][i] = ind0;
        src_ind_y[1][i] = ind1;
        src_ind_y[2][i] = ind2;
        src_ind_y[3][i] = ind3;
    }

    /* Horizontal pass */
    {   /* j : 0 */
        src_ind_x[0][0] = 1;
        src_ind_x[1][0] = 0;
        src_ind_x[2][0] = 1;
        src_ind_x[3][0] = 2;
    }
    for (j = 1; j < w_half - 2; ++j) { /* j : 1 to  w_half - 3 */
        ind1 = 2 * j;
        ind0 = ind1 - 1;
        ind2 = ind1 + 1;
        ind3 = ind1 + 2;
        src_ind_x[0][j] = ind0;
        src_ind_x[1][j] = ind1;
        src_ind_x[2][j] = ind2;
        src_ind_x[3][j] = ind3;
    }
    for (j = w_half - 2; j < w_half; ++j) { /* j : w_half - 3 to  w_half */
        ind1 = 2 * j;
        ind0 = ind1 - 1;
        ind2 = ind1 + 1;
        ind3 = ind1 + 2;
        if (ind0 >= w) {
            ind0 = (2 * w - ind0 - 1);
        }
        if (ind1 >= w) {
            ind1 = (2 * w - ind1 - 1);
        }
        if (ind2 >= w) {
            ind2 = (2 * w - ind2 - 1);
        }
        if (ind3 >= w) {
            ind3 = (2 * w - ind3 - 1);
        }
        src_ind_x[0][j] = ind0;
        src_ind_x[1][j] = ind1;
        src_ind_x[2][j] = ind2;
        src_ind_x[3][j] = ind3;
    }
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static void adm_decouple(AdmBuffer *buf, int w, int h, int stride,
                         double adm_enhn_gain_limit)
{
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);

    const adm_dwt_band_t *ref = &buf->ref_dwt2;
    const adm_dwt_band_t *dis = &buf->dis_dwt2;
    const adm_dwt_band_t *r = &buf->decouple_r;
    const adm_dwt_band_t *a = &buf->decouple_a;

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int64_t ot_dp, o_mag_sq, t_mag_sq;

    for (int i = top; i < bottom; ++i) {
        for (int j = left; j < right; ++j) {
            int16_t oh = ref->band_h[i * stride + j];
            int16_t ov = ref->band_v[i * stride + j];
            int16_t od = ref->band_d[i * stride + j];
            int16_t th = dis->band_h[i * stride + j];
            int16_t tv = dis->band_v[i * stride + j];
            int16_t td = dis->band_d[i * stride + j];
            int16_t rst_h, rst_v, rst_d;

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
             * angle_flag is calculated in floating-point by converting fixed-point variables back to
             * floating-point
             */
            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

            /**
             * Division th/oh is carried using lookup table and converted to multiplication
             */

            int32_t tmp_kh = (oh == 0) ?
                32768 : (((int64_t)div_lookup[oh + 32768] * th) + 16384) >> 15;
            int32_t tmp_kv = (ov == 0) ?
                32768 : (((int64_t)div_lookup[ov + 32768] * tv) + 16384) >> 15;
            int32_t tmp_kd = (od == 0) ?
                32768 : (((int64_t)div_lookup[od + 32768] * td) + 16384) >> 15;

            int32_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
            int32_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
            int32_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

            /**
             * kh,kv,kd are in Q15 type and oh,ov,od are in Q16 type hence shifted by
             * 15 to make result Q16
             */
            rst_h = ((kh * oh) + 16384) >> 15;
            rst_v = ((kv * ov) + 16384) >> 15;
            rst_d = ((kd * od) + 16384) >> 15;

            const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
            const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
            const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

            if (angle_flag && (rst_h_f > 0.)) rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
            if (angle_flag && (rst_h_f < 0.)) rst_h = MAX((rst_h * adm_enhn_gain_limit), th);

            if (angle_flag && (rst_v_f > 0.)) rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v_f < 0.)) rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d_f > 0.)) rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d_f < 0.)) rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            r->band_h[i * stride + j] = rst_h;
            r->band_v[i * stride + j] = rst_v;
            r->band_d[i * stride + j] = rst_d;

            a->band_h[i * stride + j] = th - rst_h;
            a->band_v[i * stride + j] = tv - rst_v;
            a->band_d[i * stride + j] = td - rst_d;
        }
    }
}

static inline uint16_t get_best15_from32(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);    //built in for intel
    k = 17 - k;
    temp = (temp + (1 << (k - 1))) >> k;
    *x = k;
    return temp;
}

static void adm_decouple_s123(AdmBuffer *buf, int w, int h, int stride,
                              double adm_enhn_gain_limit)
{
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);

    const i4_adm_dwt_band_t *ref = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *dis = &buf->i4_dis_dwt2;
    const i4_adm_dwt_band_t *r = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *a = &buf->i4_decouple_a;
    /* The computation of the score is not required for the regions
    which lie outside the frame borders */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int64_t ot_dp, o_mag_sq, t_mag_sq;

    for (int i = top; i < bottom; ++i)
    {
        for (int j = left; j < right; ++j)
        {
            int32_t oh = ref->band_h[i * stride + j];
            int32_t ov = ref->band_v[i * stride + j];
            int32_t od = ref->band_d[i * stride + j];
            int32_t th = dis->band_h[i * stride + j];
            int32_t tv = dis->band_v[i * stride + j];
            int32_t td = dis->band_d[i * stride + j];
            int32_t rst_h, rst_v, rst_d;

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

            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

            /**
             * Division th/oh is carried using lookup table and converted to multiplication
             * int64 / int32 is converted to multiplication using following method
             * num /den :
             * DenAbs = Abs(den)
             * MSBDen = MSB(DenAbs)     (gives position of first 1 bit form msb side)
             * If (DenAbs < (1 << 15))
             *      Round = (1<<14)
             *      Score = (num *  div_lookup[den] + Round ) >> 15
             * else
             *      RoundD  = (1<< (16 - MSBDen))
             *      Round   = (1<< (14 + (17 - MSBDen))
             *      Score   = (num * div_lookup[(DenAbs + RoundD )>>(17 - MSBDen)]*sign(Denominator) + Round)
             *                  >> ((15 + (17 - MSBDen))
             */

            int32_t kh_shift = 0;
            int32_t kv_shift = 0;
            int32_t kd_shift = 0;

            uint32_t abs_oh = abs(oh);
            uint32_t abs_ov = abs(ov);
            uint32_t abs_od = abs(od);

            int8_t kh_sign = (oh < 0 ? -1 : 1);
            int8_t kv_sign = (ov < 0 ? -1 : 1);
            int8_t kd_sign = (od < 0 ? -1 : 1);

            uint16_t kh_msb = (abs_oh < (32768) ? abs_oh : get_best15_from32(abs_oh, &kh_shift));
            uint16_t kv_msb = (abs_ov < (32768) ? abs_ov : get_best15_from32(abs_ov, &kv_shift));
            uint16_t kd_msb = (abs_od < (32768) ? abs_od : get_best15_from32(abs_od, &kd_shift));

            int64_t tmp_kh = (oh == 0) ? 32768 : (((int64_t)div_lookup[kh_msb + 32768] * th)*(kh_sign) +
                (1 << (14 + kh_shift))) >> (15 + kh_shift);
            int64_t tmp_kv = (ov == 0) ? 32768 : (((int64_t)div_lookup[kv_msb + 32768] * tv)*(kv_sign) +
                (1 << (14 + kv_shift))) >> (15 + kv_shift);
            int64_t tmp_kd = (od == 0) ? 32768 : (((int64_t)div_lookup[kd_msb + 32768] * td)*(kd_sign) +
                (1 << (14 + kd_shift))) >> (15 + kd_shift);

            int64_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
            int64_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
            int64_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

            rst_h = ((kh * oh) + 16384) >> 15;
            rst_v = ((kv * ov) + 16384) >> 15;
            rst_d = ((kd * od) + 16384) >> 15;

            const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
            const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
            const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

            if (angle_flag && (rst_h_f > 0.)) rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
            if (angle_flag && (rst_h_f < 0.)) rst_h = MAX((rst_h * adm_enhn_gain_limit), th);

            if (angle_flag && (rst_v_f > 0.)) rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v_f < 0.)) rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d_f > 0.)) rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d_f < 0.)) rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            r->band_h[i * stride + j] = rst_h;
            r->band_v[i * stride + j] = rst_v;
            r->band_d[i * stride + j] = rst_d;

            a->band_h[i * stride + j] = th - rst_h;
            a->band_v[i * stride + j] = tv - rst_v;
            a->band_d[i * stride + j] = td - rst_d;
        }
    }
}

static void adm_csf(AdmBuffer *buf, int w, int h, int stride,
                    double adm_norm_view_dist, int adm_ref_display_height)
{
    const adm_dwt_band_t *src = &buf->decouple_a;
    const adm_dwt_band_t *dst = &buf->csf_a;
    const adm_dwt_band_t *flt = &buf->csf_f;

    const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    // 0 is scale zero passed to dwt_quant_step

    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    /**
     * rfactor is converted to fixed-point for scale0 and stored in i_rfactor
     * multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2].
     * For adm_norm_view_dist 3.0 and adm_ref_display_height 1080,
     * i_rfactor is around { 36453,36453,49417 }
     */
    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    }
    else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t) (rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t) (rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t) (rfactor1[2] * pow2_23);
    }

    /**
     * Shifts pending from previous stage is 6
     * hence variables multiplied by i_rfactor[0,1] has to be shifted by 21+6=27 to convert
     * into floating-point. But shifted by 15 to make it Q16
     * and variables multiplied by i_factor[2] has to be shifted by 23+6=29 to convert into
     * floating-point. But shifted by 17 to make it Q16
     * Hence remaining shifts after shifting by i_shifts is 12 to make it equivalent to
     * floating-point
     */
    uint8_t i_shifts[3] = { 15,15,17 };
    uint16_t i_shiftsadd[3] = { 16384, 16384, 65535 };
    uint16_t FIX_ONE_BY_30 = 4369; //(1/30)*2^17
    /* The computation of the csf values is not required for the regions which
     *lie outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    for (int theta = 0; theta < 3; ++theta) {
        const int16_t *src_ptr = src_angles[theta];
        int16_t *dst_ptr = dst_angles[theta];
        int16_t *flt_ptr = flt_angles[theta];

        for (int i = top; i < bottom; ++i) {
            int src_offset = i * stride;
            int dst_offset = i * stride;

            for (int j = left; j < right; ++j) {
                int32_t dst_val = i_rfactor[theta] * (int32_t)src_ptr[src_offset + j];
                int16_t i16_dst_val = ((int16_t)((dst_val + i_shiftsadd[theta]) >> i_shifts[theta]));
                dst_ptr[dst_offset + j] = i16_dst_val;
                flt_ptr[dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val))
                    + 2048) >> 12));
            }
        }
    }
}

static void i4_adm_csf(AdmBuffer *buf, int scale, int w, int h, int stride,
                       double adm_norm_view_dist, int adm_ref_display_height)
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_a;
    const i4_adm_dwt_band_t *dst = &buf->i4_csf_a;
    const i4_adm_dwt_band_t *flt = &buf->i4_csf_f;

    const int32_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int32_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int32_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    //i_rfactor in fixed-point
    const double pow2_32 = pow(2, 32);
    const uint32_t i_rfactor[3] = { (uint32_t)(rfactor1[0] * pow2_32),
                                    (uint32_t)(rfactor1[1] * pow2_32),
                                    (uint32_t)(rfactor1[2] * pow2_32) };

    const uint32_t FIX_ONE_BY_30 = 143165577;
    const uint32_t shift_dst[3] = { 28, 28, 28 };
    const uint32_t shift_flt[3] = { 32, 32, 32 };
    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    for (unsigned idx = 0; idx < 3; ++idx) {
        add_bef_shift_dst[idx] = (1u << (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1));
    }

    /* The computation of the csf values is not required for the regions
     * which lie outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    for (int theta = 0; theta < 3; ++theta)
    {
        const int32_t *src_ptr = src_angles[theta];
        int32_t *dst_ptr = dst_angles[theta];
        int32_t *flt_ptr = flt_angles[theta];

        for (int i = top; i < bottom; ++i)
        {
            int src_offset = i * stride;
            int dst_offset = i * stride;

            for (int j = left; j < right; ++j)
            {
                int32_t dst_val = (int32_t)(((i_rfactor[theta] * (int64_t)src_ptr[src_offset + j]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                dst_ptr[dst_offset + j] = dst_val;
                flt_ptr[dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) +
                    add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);
            }
        }
    }
}

static float adm_csf_den_scale(const adm_dwt_band_t *src, int w, int h,
                               int src_stride,
                               double adm_norm_view_dist, int adm_ref_display_height)
{
    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;

    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    int32_t shift_accum = (int32_t)ceil(log2((bottom - top)*(right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    int32_t add_shift_accum =
        shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;

    /**
     * The rfactor is multiplied at the end after cubing
     * Because d+ = (a[i]^3)*(r^3)
     * is equivalent to d+=a[i]^3 and d=d*(r^3)
     */
    int16_t *src_h = src->band_h + top * src_stride;
    int16_t *src_v = src->band_v + top * src_stride;
    int16_t *src_d = src->band_d + top * src_stride;
    for (int i = top; i < bottom; ++i) {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        for (int j = left; j < right; ++j) {
            uint16_t h = (uint16_t)abs(src_h[j]);
            uint16_t v = (uint16_t)abs(src_v[j]);
            uint16_t d = (uint16_t)abs(src_d[j]);

            uint64_t val = ((uint64_t)h * h) * h;
            accum_inner_h += val;
            val = ((uint64_t)v * v) * v;
            accum_inner_v += val;
            val = ((uint64_t)d * d) * d;
            accum_inner_d += val;
        }
        /**
         * max_value of h^3, v^3, d^3 is 1.205624776 * —10^13
         * accum_h can hold till 1.844674407 * —10^19
         * accum_h's maximum is reached when it is 2^20 * max(h^3)
         * Therefore the accum_h,v,d is shifted based on width and height subtracted by 20
         */
        accum_h += (accum_inner_h + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + add_shift_accum) >> shift_accum;
        src_h += src_stride;
        src_v += src_stride;
        src_d += src_stride;
    }
    /**
     * rfactor is multiplied after cubing
     * accum_h,v,d is converted to floating-point for score calculation
     * 6bits are yet to be shifted from previous stage that is after dwt hence
     * after cubing 18bits are to shifted
     * Hence final shift is 18-shift_accum
     */
    double shift_csf = pow(2, (18 - shift_accum));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return(den_scale_h + den_scale_v + den_scale_d);

}

static float adm_csf_den_s123(const i4_adm_dwt_band_t *src, int scale, int w, int h,
                              int src_stride,
                              double adm_norm_view_dist, int adm_ref_display_height)
{
    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    const uint32_t shift_sq[3] = { 31, 30, 31 };
    const uint32_t accum_convert_float[3] = { 32, 27, 23 };
    const uint32_t add_shift_sq[3] =
        { 1u << shift_sq[0], 1u << shift_sq[1], 1u << shift_sq[2] };

    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));
    uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
    uint32_t add_shift_accum = (uint32_t)pow(2, (shift_accum - 1));

    int32_t *src_h = src->band_h + top * src_stride;
    int32_t *src_v = src->band_v + top * src_stride;
    int32_t *src_d = src->band_d + top * src_stride;
    for (int i = top; i < bottom; ++i)
    {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        for (int j = left; j < right; ++j)
        {
            uint32_t h = (uint32_t)abs(src_h[j]);
            uint32_t v = (uint32_t)abs(src_v[j]);
            uint32_t d = (uint32_t)abs(src_d[j]);

            uint64_t val = ((((((uint64_t)h * h) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * h) + add_shift_cub) >> shift_cub;

            accum_inner_h += val;

            val = ((((((uint64_t)v * v) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * v) + add_shift_cub) >> shift_cub;
            accum_inner_v += val;

            val = ((((((uint64_t)d * d) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * d) + add_shift_cub) >> shift_cub;
            accum_inner_d += val;
        }

        accum_h += (accum_inner_h + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + add_shift_accum) >> shift_accum;

        src_h += src_stride;
        src_v += src_stride;
        src_d += src_stride;
    }
    /**
     * All the results are converted to floating-point to calculate the scores
     * For all scales the final shift is 3*shifts from dwt - total shifts done here
     */
    double shift_csf = pow(2, (accum_convert_float[scale - 1] - shift_accum - shift_cub));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return (den_scale_h + den_scale_v + den_scale_d);
}

static float adm_cm(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                    double adm_norm_view_dist, int adm_ref_display_height)
{
    const adm_dwt_band_t *src   = &buf->decouple_r;
    const adm_dwt_band_t *csf_f = &buf->csf_f;
    const adm_dwt_band_t *csf_a = &buf->csf_a;

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    // 0 is scale zero passed to dwt_quant_step

    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    /**
     * rfactor is converted to fixed-point for scale0 and stored in i_rfactor
     * multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2].
     * For adm_norm_view_dist 3.0 and adm_ref_display_height 1080,
     * i_rfactor is around { 36453,36453,49417 }
     */
    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    }
    else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t) (rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t) (rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t) (rfactor1[2] * pow2_23);
    }

    const int32_t shift_xhsq = 29;
    const int32_t shift_xvsq = 29;
    const int32_t shift_xdsq = 30;
    const int32_t add_shift_xhsq = 268435456;
    const int32_t add_shift_xvsq = 268435456;
    const int32_t add_shift_xdsq = 536870912;

    const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));

    const uint32_t shift_xvcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xvcub = (uint32_t)pow(2, (shift_xvcub - 1));

    const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
    const uint32_t add_shift_xdcub = (uint32_t)pow(2, (shift_xdcub - 1));

    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    const uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    const int32_t shift_xhsub = 10;
    const int32_t shift_xvsub = 10;
    const int32_t shift_xdsub = 12;

    int16_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int16_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    /* The computation of the scales is not required for the regions which lie
     * outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int i, j;
    int64_t val;
    int32_t xh, xv, xd, thr;
    int32_t xh_sq, xv_sq, xd_sq;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;

    /* i=0,j=0 */
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)src->band_h[0] * i_rfactor[0];
        xv = (int32_t)src->band_v[0] * i_rfactor[1];
        xd = (int32_t)src->band_d[0] * i_rfactor[2];
        ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_a_stride, &thr, w, h, 0, 0);

        //thr is shifted to make it's Q format equivalent to xh,xv,xd

        /**
         * max value of xh_sq and xv_sq is 1301381973 and that of xd_sq is 1195806729
         *
         * max(val before shift for h and v) is 9.995357299 * —10^17.
         * 9.995357299 * —10^17 * 2^4 is close to 2^64.
         * Hence shift is done based on width subtracting 4
         *
         * max(val before shift for d) is 1.355006643 * —10^18
         * 1.355006643 * —10^18 * 2^3 is close to 2^64
         * Hence shift is done based on width subtracting 3
         */
        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }

    /* i=0, j */
    if (top <= 0) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[j] * i_rfactor[0];
            xv = src->band_v[j] * i_rfactor[1];
            xd = src->band_d[j] * i_rfactor[2];
            ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_a_stride, &thr, w, h, 0, j);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);
        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = src->band_h[w - 1] * i_rfactor[0];
        xv = src->band_v[w - 1] * i_rfactor[1];
        xd = src->band_d[w - 1] * i_rfactor[2];
        ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, 0, (w - 1));

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }
    //Shift is done based on height
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
        }
    }
    else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_stride] * i_rfactor[0];
            xv = src->band_v[i * src_stride] * i_rfactor[1];
            xd = src->band_d[i * src_stride] * i_rfactor[2];
            ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            /* j = w-1 */
            xh = src->band_h[i * src_stride + w - 1] * i_rfactor[0];
            xv = src->band_v[i * src_stride + w - 1] * i_rfactor[1];
            xd = src->band_d[i * src_stride + w - 1] * i_rfactor[2];
            ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1));

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

        }
    }
    else /* Both borders outside frame */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_stride] * i_rfactor[0];
            xv = src->band_v[i * src_stride] * i_rfactor[1];
            xd = src->band_d[i * src_stride] * i_rfactor[2];
            ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            /* j = w-1 */
            xh = src->band_h[i * src_stride + w - 1] * i_rfactor[0];
            xv = src->band_v[i * src_stride + w - 1] * i_rfactor[1];
            xd = src->band_d[i * src_stride + w - 1] * i_rfactor[2];
            ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1));

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = src->band_h[(h - 1) * src_stride] * i_rfactor[0];
        xv = src->band_v[(h - 1) * src_stride] * i_rfactor[1];
        xd = src->band_d[(h - 1) * src_stride] * i_rfactor[2];
        ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), 0);

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }

    /* i=h-1,j */
    if (bottom > (h - 1)) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[(h - 1) * src_stride + j] * i_rfactor[0];
            xv = src->band_v[(h - 1) * src_stride + j] * i_rfactor[1];
            xd = src->band_d[(h - 1) * src_stride + j] * i_rfactor[2];
            ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), j);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);
        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = src->band_h[(h - 1) * src_stride + w - 1] * i_rfactor[0];
        xv = src->band_v[(h - 1) * src_stride + w - 1] * i_rfactor[1];
        xd = src->band_d[(h - 1) * src_stride + w - 1] * i_rfactor[2];
        ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h,
            (h - 1), (w - 1));

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

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

    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

static float i4_adm_cm(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride, int scale,
                       double adm_norm_view_dist, int adm_ref_display_height)
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *csf_f = &buf->i4_csf_f;
    const i4_adm_dwt_band_t *csf_a = &buf->i4_csf_a;

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    const uint32_t rfactor[3] = { (uint32_t)(rfactor1[0] * pow(2, 32)),
                                  (uint32_t)(rfactor1[1] * pow(2, 32)),
                                  (uint32_t)(rfactor1[2] * pow(2, 32)) };

    const uint32_t shift_dst[3] = { 28, 28, 28 };
    const uint32_t shift_flt[3] = { 32, 32, 32 };
    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    for (unsigned idx = 0; idx < 3; ++idx) {
        add_bef_shift_dst[idx] = (1u << (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1));

    }

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

    uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    float final_shift[3] = { pow(2,(45 - shift_cub - shift_inner_accum)),
                             pow(2,(39 - shift_cub - shift_inner_accum)),
                             pow(2,(36 - shift_cub - shift_inner_accum)) };

    const int32_t shift_sq = 30;
    const int32_t add_shift_sq = 536870912; //2^29
    const int32_t shift_sub = 0;
    int32_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int32_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    /* The computation of the scales is not required for the regions which lie
     * outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int i, j;
    int32_t xh, xv, xd, thr;
    int32_t xh_sq, xv_sq, xd_sq;
    int64_t val;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;
    /* i=0,j=0 */
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[0] * rfactor[0]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[0] * rfactor[1]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[0] * rfactor[2]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_a_stride, &thr, w, h, 0, 0,
                                       add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    /* i=0, j */
    if (top <= 0)
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[j] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[j] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[j] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_a_stride, &thr, w, h,
                                           0, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);
        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[w - 1] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[w - 1] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[w - 1] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, 0, (w - 1),
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            for (j = start_col; j < end_col; ++j)
            {

                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1),
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride + w - 1] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride + w - 1] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride + w - 1] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1),
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    /* i=h-1,j */
    if (bottom > (h - 1))
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride + j] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride + j] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride + j] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);
        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride + w - 1] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride + w - 1] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride + w - 1] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1),
                                               (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    /**
     * Converted to floating-point for calculating the final scores
     * Final shifts is calculated from 3*(shifts_from_previous_stage(i.e src comes from dwt)+32)-total_shifts_done_in_this_function
     */
    float f_accum_h = (float)(accum_h / final_shift[scale - 1]);
    float f_accum_v = (float)(accum_v / final_shift[scale - 1]);
    float f_accum_d = (float)(accum_d / final_shift[scale - 1]);

    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

static void i16_to_i32(adm_dwt_band_t *src, i4_adm_dwt_band_t *dst,
                       int w, int h, int stride)
{
    for (int i = 0; i < (h + 1) / 2; ++i) {
        int16_t *src_band_a_addr = &src->band_a[i * stride];
        int32_t *dst_band_a_addr = &dst->band_a[i * stride];
        for (int j = 0; j < (w + 1) / 2; ++j) {
            *(dst_band_a_addr++) = (int32_t)(*(src_band_a_addr++));
        }
    }
}

static void adm_dwt2_8(const uint8_t *src, const adm_dwt_band_t *dst,
                       AdmBuffer *buf, int w, int h, int src_stride,
                       int dst_stride)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_VP = 8;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 128;
    const int32_t add_shift_HP = 32768;

    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (int j = 0; j < w; ++j) {
            uint16_t u_s0 = src[ind_y[0][i] * src_stride + j];
            uint16_t u_s1 = src[ind_y[1][i] * src_stride + j];
            uint16_t u_s2 = src[ind_y[2][i] * src_stride + j];
            uint16_t u_s3 = src[ind_y[3][i] * src_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP;

            tmplo[j] = (accum + add_shift_VP) >> shift_VP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP;

            tmphi[j] = (accum + add_shift_VP) >> shift_VP;
        }

        /* Horizontal pass (lo and hi). */
        for (int j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }
    }
}

static void adm_dwt2_16(const uint16_t *src, const adm_dwt_band_t *dst, AdmBuffer *buf, int w, int h,
                        int src_stride, int dst_stride, int inp_size_bits)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_VP = inp_size_bits;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 1 << (inp_size_bits - 1);
    const int32_t add_shift_HP = 32768;

    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (int j = 0; j < w; ++j) {
            uint16_t u_s0 = src[ind_y[0][i] * src_stride + j];
            uint16_t u_s1 = src[ind_y[1][i] * src_stride + j];
            uint16_t u_s2 = src[ind_y[2][i] * src_stride + j];
            uint16_t u_s3 = src[ind_y[3][i] * src_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP;

            tmplo[j] = (accum + add_shift_VP) >> shift_VP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP;

            tmphi[j] = (accum + add_shift_VP) >> shift_VP;
        }

        /* Horizontal pass (lo and hi). */
        for (int j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }
    }
}

static void adm_dwt2_s123_combined(const int32_t *i4_ref_scale, const int32_t *i4_curr_dis,
                                   AdmBuffer *buf, int w, int h, int ref_stride,
                                   int dis_stride, int dst_stride, int scale)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int32_t add_bef_shift_round_VP[3] = { 0, 32768, 32768 };
    const int32_t add_bef_shift_round_HP[3] = { 16384, 32768, 16384 };
    const int16_t shift_VerticalPass[3] = { 0, 16, 16 };
    const int16_t shift_HorizontalPass[3] = { 15, 16, 15 };

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;
    int32_t s10, s11, s12, s13;

    int64_t accum_ref;

    for (int i = 0; i < (h + 1) / 2; ++i)
    {
        /* Vertical pass. */
        for (int j = 0; j < w; ++j)
        {
            s10 = i4_ref_scale[ind_y[0][i] * ref_stride + j];
            s11 = i4_ref_scale[ind_y[1][i] * ref_stride + j];
            s12 = i4_ref_scale[ind_y[2][i] * ref_stride + j];
            s13 = i4_ref_scale[ind_y[3][i] * ref_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1])
                >> shift_VerticalPass[scale - 1]);
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1])
                >> shift_VerticalPass[scale - 1]);

            s10 = i4_curr_dis[ind_y[0][i] * dis_stride + j];
            s11 = i4_curr_dis[ind_y[1][i] * dis_stride + j];
            s12 = i4_curr_dis[ind_y[2][i] * dis_stride + j];
            s13 = i4_curr_dis[ind_y[3][i] * dis_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1])
                >> shift_VerticalPass[scale - 1]);
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1])
                >> shift_VerticalPass[scale - 1]);
        }
        /* Horizontal pass (lo and hi). */
        for (int j = 0; j < (w + 1) / 2; ++j)
        {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            s10 = tmplo_ref[j0];
            s11 = tmplo_ref[j1];
            s12 = tmplo_ref[j2];
            s13 = tmplo_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            s10 = tmphi_ref[j0];
            s11 = tmphi_ref[j1];
            s12 = tmphi_ref[j2];
            s13 = tmphi_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            s10 = tmplo_dis[j0];
            s11 = tmplo_dis[j1];
            s12 = tmplo_dis[j2];
            s13 = tmplo_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            s10 = tmphi_dis[j0];
            s11 = tmphi_dis[j1];
            s12 = tmphi_dis[j2];
            s13 = tmphi_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref +
                add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
        }
    }
}

void integer_compute_adm(AdmState *s, VmafPicture *ref_pic, VmafPicture *dis_pic,
                         double *score, double *score_num, double *score_den, double *scores, AdmBuffer *buf,
                         double adm_enhn_gain_limit,
                         double adm_norm_view_dist, int adm_ref_display_height)
{
    int w = ref_pic->w[0];
    int h = ref_pic->h[0];

    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);

    size_t curr_ref_stride;
    size_t curr_dis_stride;
    size_t buf_stride = buf->ind_size_x >> 2;

    int32_t *i4_curr_ref_scale = NULL;
    int32_t *i4_curr_dis_scale = NULL;

    if (ref_pic->bpc == 8) {
        curr_ref_stride = ref_pic->stride[0];
        curr_dis_stride = dis_pic->stride[0];
    }
    else {
        curr_ref_stride = ref_pic->stride[0] >> 1;
        curr_dis_stride = dis_pic->stride[0] >> 1;
    }

    double num = 0;
    double den = 0;
	for (unsigned scale = 0; scale < 4; ++scale) {
		float num_scale = 0.0;
		float den_scale = 0.0;

        dwt2_src_indices_filt(buf->ind_y, buf->ind_x, w, h);
		if(scale==0) {
            if (ref_pic->bpc == 8) {
                s->dwt2_8(ref_pic->data[0], &buf->ref_dwt2, buf, w, h,
                          curr_ref_stride, buf_stride);
                s->dwt2_8(dis_pic->data[0], &buf->dis_dwt2, buf, w, h,
                          curr_dis_stride, buf_stride);
            }
            else {
                adm_dwt2_16(ref_pic->data[0], &buf->ref_dwt2, buf, w, h,
                            curr_ref_stride, buf_stride, ref_pic->bpc);
                adm_dwt2_16(dis_pic->data[0], &buf->dis_dwt2, buf, w, h,
                            curr_dis_stride, buf_stride, dis_pic->bpc);
            }

			i16_to_i32(&buf->ref_dwt2, &buf->i4_ref_dwt2, w, h, buf_stride);
			i16_to_i32(&buf->dis_dwt2, &buf->i4_dis_dwt2, w, h, buf_stride);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			adm_decouple(buf, w, h, buf_stride, adm_enhn_gain_limit);

			den_scale = adm_csf_den_scale(&buf->ref_dwt2, w, h, buf_stride,
                                 adm_norm_view_dist, adm_ref_display_height);

			adm_csf(buf, w, h, buf_stride, adm_norm_view_dist, adm_ref_display_height);

			num_scale = adm_cm(buf, w, h, buf_stride, buf_stride,
                               adm_norm_view_dist, adm_ref_display_height);
		}
		else {
            adm_dwt2_s123_combined(i4_curr_ref_scale, i4_curr_dis_scale, buf, w, h, curr_ref_stride,
                                   curr_dis_stride, buf_stride, scale);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			adm_decouple_s123(buf, w, h, buf_stride, adm_enhn_gain_limit);

			den_scale = adm_csf_den_s123(
			        &buf->i4_ref_dwt2, scale, w, h, buf_stride,
			        adm_norm_view_dist, adm_ref_display_height);

			i4_adm_csf(buf, scale, w, h, buf_stride,
              adm_norm_view_dist, adm_ref_display_height);

			num_scale = i4_adm_cm(buf, w, h, buf_stride, buf_stride, scale,
                         adm_norm_view_dist, adm_ref_display_height);
		}

		num += num_scale;
		den += den_scale;

		i4_curr_ref_scale = buf->i4_ref_dwt2.band_a;
		i4_curr_dis_scale = buf->i4_dis_dwt2.band_a;

		curr_ref_stride = buf_stride;
		curr_dis_stride = buf_stride;

		scores[2 * scale + 0] = num_scale;
		scores[2 * scale + 1] = den_scale;
	}

	num = num < numden_limit ? 0 : num;
	den = den < numden_limit ? 0 : den;

	if (den == 0.0) {
		*score = 1.0f;
	}
	else {
		*score = num / den;
	}
    *score_num = num;
    *score_den = den;

}

static inline void *init_dwt_band(adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = (int16_t *)data_top; data_top += stride;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *init_index(int32_t **index, char *data_top, size_t stride)
{
    index[0] = (int32_t *)data_top; data_top += stride;
    index[1] = (int32_t *)data_top; data_top += stride;
    index[2] = (int32_t *)data_top; data_top += stride;
    index[3] = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *i4_init_dwt_band(i4_adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = (int32_t *)data_top; data_top += stride;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *init_dwt_band_hvd(adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *i4_init_dwt_band_hvd(i4_adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmState *s = fex->priv;
    (void) pix_fmt;
    (void) bpc;

    if (w <= 32 || h <= 32) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "%s: invalid size (%dx%d), "
                 "width/height must be greater than 32\n",
                 fex->name, w, h);
        return -EINVAL;
    }

    s->dwt2_8 = adm_dwt2_8;

#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        if (!(w % 8)) s->dwt2_8 = adm_dwt2_8_avx2;
    }
#elif ARCH_AARCH64
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_ARM_CPU_FLAG_NEON) {
        if (!(w % 8)) s->dwt2_8 = adm_dwt2_8_neon;
    }
#endif

    s->integer_stride   = ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x   = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y   = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    size_t buf_sz_one   = s->buf.ind_size_x * ((h + 1) / 2);

    s->buf.data_buf     = aligned_malloc(buf_sz_one * NUM_BUFS_ADM, MAX_ALIGN);
    if (!s->buf.data_buf) goto fail;
    s->buf.tmp_ref      = aligned_malloc(s->integer_stride * 4, MAX_ALIGN);
    if (!s->buf.tmp_ref) goto fail;
    s->buf.buf_x_orig   = aligned_malloc(s->buf.ind_size_x * 4, MAX_ALIGN);
    if (!s->buf.buf_x_orig) goto fail;
    s->buf.buf_y_orig   = aligned_malloc(s->buf.ind_size_y * 4, MAX_ALIGN);
    if (!s->buf.buf_y_orig) goto fail;

    void *data_top = s->buf.data_buf;
    data_top = init_dwt_band(&s->buf.ref_dwt2, data_top, buf_sz_one / 2);
    data_top = init_dwt_band(&s->buf.dis_dwt2, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.decouple_r, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.decouple_a, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.csf_a, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.csf_f, data_top, buf_sz_one / 2);

    data_top = i4_init_dwt_band(&s->buf.i4_ref_dwt2, data_top, buf_sz_one);
    data_top = i4_init_dwt_band(&s->buf.i4_dis_dwt2, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_decouple_r, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_decouple_a, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_csf_a, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_csf_f, data_top, buf_sz_one);

    void *ind_buf_y = s->buf.buf_y_orig;
    init_index(s->buf.ind_y, ind_buf_y, s->buf.ind_size_y);
    void *ind_buf_x = s->buf.buf_x_orig;
    init_index(s->buf.ind_x, ind_buf_x, s->buf.ind_size_x);

    div_lookup_generator();

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (s->buf.data_buf)    aligned_free(s->buf.data_buf);
    if (s->buf.tmp_ref)     aligned_free(s->buf.tmp_ref);
    if (s->buf.buf_x_orig)  aligned_free(s->buf.buf_x_orig);
    if (s->buf.buf_y_orig)  aligned_free(s->buf.buf_y_orig);
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    AdmState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    double score, score_num, score_den;
    double scores[8];

    // current implementation is limited by the 16-bit data pipeline, thus
    // cannot handle an angular frequency smaller than 1080p * 3H
    if (s->adm_norm_view_dist * s->adm_ref_display_height <
        DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) {
        return -EINVAL;
    }

    integer_compute_adm(s, ref_pic, dist_pic, &score, &score_num, &score_den,
                        scores, &s->buf,
                        s->adm_enhn_gain_limit,
                        s->adm_norm_view_dist, s->adm_ref_display_height);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_adm2_score", score,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale0", scores[0] / scores[1],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale1", scores[2] / scores[3],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale2", scores[4] / scores[5],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale3", scores[6] / scores[7],
            index);

    if (!s->debug) return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale0", scores[0], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale0", scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale1", scores[2], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale1", scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale2", scores[4], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale2", scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale3", scores[6], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale3", scores[7], index);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;

    if (s->buf.data_buf)    aligned_free(s->buf.data_buf);
    if (s->buf.tmp_ref)     aligned_free(s->buf.tmp_ref);
    if (s->buf.buf_x_orig)  aligned_free(s->buf.buf_x_orig);
    if (s->buf.buf_y_orig)  aligned_free(s->buf.buf_y_orig);
    vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_adm2_score", "integer_adm_scale0",
    "integer_adm_scale1", "integer_adm_scale2", "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0", "integer_adm_num_scale1",
    "integer_adm_den_scale1", "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_adm = {
    .name = "adm",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
