/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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

#include "common.h"
#include <stdio.h>

#include "cuda_helper.cuh"
#include "mem.h"
#include "libvmaf/vmaf_cuda_state.h"

#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cpu.h"
#include "integer_adm_cuda.h"
#include "picture_cuda.h"
#include "integer_adm_kernels.h"
#include <unistd.h>

#define RES_BUFFER_SIZE 4 * 3 * 2


typedef struct AdmStateCuda {
    size_t integer_stride;
    AdmBufferCuda buf;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    void (*dwt2_8)(const uint8_t *src, const cuda_adm_dwt_band_t *dst, short2* tmp_buf,
                   AdmBufferCuda *buf, int w, int h, int src_stride,
                   int dst_stride, CUstream c_stream);
    CUstream str, host_stream;
    void* write_score_parameters;
    CUevent ref_event, dis_event, finished;
    VmafDictionary *feature_name_dict;
} AdmStateCuda;

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static inline float dwt_quant_step(const struct dwt_model_params *params,
        int lambda, int theta, double adm_norm_view_dist, int adm_ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0,lambda+1)*params->f0*params->g[theta]/r);
    float Q = 2.0*params->a*pow(10.0,params->k*temp*temp)/dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}


static void conclude_adm_cm(int64_t *accum, int h,
                            int w, int scale, float *result) {
  int left = w * ADM_BORDER_FACTOR - 0.5;
  int top = h * ADM_BORDER_FACTOR - 0.5;
  int right = w - left;
  int bottom = h - top;
  const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));

  // scale 0
  const uint32_t shift_xcub[3] = {(uint32_t)ceil(log2(w) - 4),
                                  (uint32_t)ceil(log2(w) - 4),
                                  (uint32_t)ceil(log2(w) - 3)};
  int constant_offset[3] = {52, 52, 57};

  // scale 123
  uint32_t shift_cub = (uint32_t)ceil(log2(w));
  float final_shift[3] = {powf(2, (45 - shift_cub - shift_inner_accum)),
                          powf(2, (39 - shift_cub - shift_inner_accum)),
                          powf(2, (36 - shift_cub - shift_inner_accum))};
  float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

  float f_accum;
  *result = 0;
  for (int i = 0; i < 3; ++i) {
    if (scale == 0) {
      f_accum = (float)(accum[i] / pow(2, (constant_offset[i] - shift_xcub[i] -
                                           shift_inner_accum)));
    } else {
      f_accum = (float)(accum[i] / final_shift[scale - 1]);
    }
    *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
  }
}

static void conclude_adm_csf_den(uint64_t *accum, int h, int w, int scale, float *result,
                           float adm_norm_view_dist,
                           float adm_ref_display_height) {
  const int left = w * ADM_BORDER_FACTOR - 0.5;
  const int top = h * ADM_BORDER_FACTOR - 0.5;
  const int right = w - left;
  const int bottom = h - top;
  const float factor1 =
      dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                     adm_ref_display_height);
  const float factor2 =
      dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                     adm_ref_display_height);
  const float rfactor[3] = {1.0f / factor1, 1.0f / factor1, 1.0f / factor2};
  const uint32_t accum_convert_float[4] = {18, 32, 27, 23};

  int32_t shift_accum;
  double shift_csf;
  if (scale == 0) {
    shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    shift_csf = pow(2, (accum_convert_float[scale] - shift_accum));
  } else {
    shift_accum = (int32_t)ceil(log2(bottom - top));
    const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    shift_csf = pow(2, (accum_convert_float[scale] - shift_accum - shift_cub));
  }
  const float powf_add =
      powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

  *result = 0;
  for (int i = 0; i < 3; ++i) {
    const double csf = (double)(accum[i] / shift_csf) * pow(rfactor[i], 3);
    *result += powf(csf, 1.0f / 3.0f) + powf_add;
  }
}

static const VmafOption options_cuda[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_enhn_gain_limit",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmStateCuda, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
    },
    {
        .name = "adm_norm_view_dist",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmStateCuda, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
    },
    {
        .name = "adm_ref_display_height",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateCuda, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
    },
    { 0 }
};

typedef struct write_score_parameters_adm {
    VmafFeatureCollector *feature_collector;
    AdmStateCuda *s;
    unsigned index, h, w;
} write_score_parameters_adm;

static int write_scores(write_score_parameters_adm* params)
{

    VmafFeatureCollector *feature_collector = params->feature_collector;
    AdmStateCuda *s = params->s;
    unsigned index = params->index;

    double scores[8];
    double score, score_num, score_den;

    double num = 0;
    double den = 0;

    unsigned w = params->w;
    unsigned h = params->h;

    int64_t* adm_cm = (int64_t*)s->buf.results_host;
    uint64_t* adm_csf = &((uint64_t*)s->buf.results_host)[RES_BUFFER_SIZE / 2];
	float num_scale;
    float den_scale;
    for (unsigned scale = 0; scale < 4; ++scale) {

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        conclude_adm_cm(&adm_cm[scale * 3], h, w, scale, &num_scale);
        conclude_adm_csf_den(&adm_csf[scale * 3], h, w, scale, &den_scale, s->adm_norm_view_dist, s->adm_ref_display_height);

    	num += num_scale;
		den += den_scale;

		scores[2 * scale + 0] = num_scale;
		scores[2 * scale + 1] = den_scale;
    }    
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);


	num = num < numden_limit ? 0 : num;
	den = den < numden_limit ? 0 : den;

	if (den == 0.0) {
		score = 1.0f;
	}
	else {
		score = num / den;
	}
    score_num = num;
    score_den = den;

    int err = 0;
    char *key =
        s->adm_enhn_gain_limit != DEFAULT_ADM_ENHN_GAIN_LIMIT ?
        "adm_enhn_gain_limit" : NULL;
    double val = s->adm_enhn_gain_limit;

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

static void integer_compute_adm_cuda(VmafFeatureExtractor *fex, AdmStateCuda *s,
                                     VmafPicture *ref_pic, VmafPicture *dis_pic, AdmBufferCuda *buf,
                                     double adm_enhn_gain_limit,
                                     double adm_norm_view_dist,
                                     int adm_ref_display_height)
{
    int w = ref_pic->w[0];
    int h = ref_pic->h[0];

    AdmFixedParametersCuda p = {
        .dwt2_db2_coeffs_lo = {15826, 27411, 7345, -4240},
        .dwt2_db2_coeffs_hi = {-4240, -7345, 27411, -15826},
        .dwt2_db2_coeffs_lo_sum = 46342,
        .dwt2_db2_coeffs_hi_sum = 0,
        .log2_w = log2(w),
        .log2_h = log2(h),
        .adm_ref_display_height = adm_ref_display_height,
        .adm_norm_view_dist = adm_norm_view_dist,
        .adm_enhn_gain_limit = adm_enhn_gain_limit,
    };

    const double pow2_32 = pow(2, 32);
    const double pow2_21 = pow(2, 21);
    const double pow2_23 = pow(2, 23);
    for (unsigned scale = 0; scale < 4; ++scale) {
        float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1,
                                        adm_norm_view_dist, adm_ref_display_height);
        float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2,
                                        adm_norm_view_dist, adm_ref_display_height);
        p.factor1[scale] = factor1;
        p.factor2[scale] = factor2;
        p.rfactor[scale*3] = 1.0f / factor1;
        p.rfactor[scale*3+1] = 1.0f / factor1;
        p.rfactor[scale*3+2] = 1.0f / factor2;
        if (scale == 0) {
            if (fabs(p.adm_norm_view_dist * p.adm_ref_display_height -
                    DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) <
                1.0e-8) {
                p.i_rfactor[scale * 3] = 36453;
                p.i_rfactor[scale * 3 + 1] = 36453;
                p.i_rfactor[scale * 3 + 2] = 49417;
            } else {
                p.i_rfactor[scale * 3] = (uint32_t)(p.rfactor[scale * 3] * pow2_21);
                p.i_rfactor[scale * 3 + 1] =
                    (uint32_t)(p.rfactor[scale * 3 + 1] * pow2_21);
                p.i_rfactor[scale * 3 + 2] =
                    (uint32_t)(p.rfactor[scale * 3 + 2] * pow2_23);
            }
        } else {
        p.i_rfactor[scale * 3] = (uint32_t)(p.rfactor[scale * 3] * pow2_32);
        p.i_rfactor[scale * 3 + 1] =
            (uint32_t)(p.rfactor[scale * 3 + 1] * pow2_32);
        p.i_rfactor[scale * 3 + 2] =
            (uint32_t)(p.rfactor[scale * 3 + 2] * pow2_32);
        }
        uint32_t *i_rfactor = &p.i_rfactor[scale*3];
    }
    CHECK_CUDA(cuMemsetD8Async(buf->tmp_res->data, 0, sizeof(int64_t) * RES_BUFFER_SIZE, s->str));

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
        curr_ref_stride = dis_pic->stride[0] >> 1;
        curr_dis_stride = ref_pic->stride[0] >> 1;
    }

	for (unsigned scale = 0; scale < 4; ++scale) {
		float num_scale = 0.0;
		float den_scale = 0.0;


		if(scale==0) {
            // run these first dwt kernels on the input iamge stream to make sure it is consumed afterwards continue
            // consumes reference picture
            // produces buf->ref_dwt2, buf->dis_dwt2
            if (ref_pic->bpc == 8) {    
                dwt2_8_device((const uint8_t*)ref_pic->data[0], &buf->ref_dwt2, buf->i4_ref_dwt2, (int16_t*)buf->tmp_ref->data, buf, w, h, curr_ref_stride, buf_stride, &p, vmaf_cuda_picture_get_stream(ref_pic));

                dwt2_8_device((const uint8_t*)dis_pic->data[0], &buf->dis_dwt2, buf->i4_dis_dwt2, (int16_t*)buf->tmp_dis->data, buf, w, h, curr_dis_stride, buf_stride, &p,  vmaf_cuda_picture_get_stream(dis_pic));
            }
            else {
                adm_dwt2_16_device((uint16_t*)ref_pic->data, &buf->ref_dwt2, buf->i4_dis_dwt2, (int16_t*)buf->tmp_ref->data, buf, w, h, curr_ref_stride, buf_stride, ref_pic->bpc, &p,  vmaf_cuda_picture_get_stream(ref_pic));
                
                adm_dwt2_16_device((uint16_t*)dis_pic->data, &buf->dis_dwt2, buf->i4_dis_dwt2, (int16_t*)buf->tmp_dis->data, buf, w, h, curr_dis_stride, buf_stride, dis_pic->bpc, &p,  vmaf_cuda_picture_get_stream(dis_pic));

            }
            CHECK_CUDA(cuEventRecord(s->ref_event,  vmaf_cuda_picture_get_stream(ref_pic)));
            CHECK_CUDA(cuEventRecord(s->dis_event,  vmaf_cuda_picture_get_stream(dis_pic)));

			w = (w + 1) / 2;
			h = (h + 1) / 2;

            // This event ensures the input buffer is consumed
            CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
            
            CHECK_CUDA(cuStreamWaitEvent(s->str, s->dis_event, CU_EVENT_WAIT_DEFAULT));
            CHECK_CUDA(cuEventDestroy(s->dis_event));
            CHECK_CUDA(cuEventCreate(&s->dis_event, CU_EVENT_DEFAULT));

            CHECK_CUDA(cuStreamWaitEvent(s->str, s->ref_event, CU_EVENT_WAIT_DEFAULT));
            CHECK_CUDA(cuEventDestroy(s->ref_event));
            CHECK_CUDA(cuEventCreate(&s->ref_event, CU_EVENT_DEFAULT));
            
            CHECK_CUDA(cuCtxPopCurrent(NULL));
            // consumes buf->ref_dwt2 , buf->dis_dwt2
            // produces buf->decouple_r , buf->decouple_a
			adm_decouple_device(buf, w, h, buf_stride, &p, s->str);

            // consumes buf->ref_dwt2
            // produces buf->adm_csf_den[0]
			adm_csf_den_scale_device(buf, w, h, buf_stride,
                                 adm_norm_view_dist, adm_ref_display_height, s->str);

            // consumes buf->decouple_a
            // produces buf->csf_a , buf->csf_f
			adm_csf_device(buf, w, h, buf_stride, &p, s->str);

            // consumes buf->decouple_r, buf->csf_a, buf->csf_a
            // produces buf->adm_cm[0]
            adm_cm_device(buf, w, h, buf_stride, buf_stride, &p, s->str);
		}
		else {
            // consumes buf->i4_ref_dwt2.band_a , buf->i4_dis_dwt2.band_a
            // produces buf->i4_ref_dwt2.band_[ahvd] , buf->i4_dis_dwt2.band_[ahvd]
            // uses buf->tmp_ref
            adm_dwt2_s123_combined_device(i4_curr_ref_scale, (int32_t*)buf->tmp_ref->data, buf->i4_ref_dwt2, buf, w, h,
                            curr_ref_stride, buf_stride, scale, &p, s->str);
            adm_dwt2_s123_combined_device(i4_curr_dis_scale, (int32_t*)buf->tmp_dis->data, buf->i4_dis_dwt2, buf, w, h,
                            curr_dis_stride, buf_stride, scale, &p, s->str);

			w = (w + 1) / 2;
			h = (h + 1) / 2;
            
            // consumes buf->i4_ref_dwt2 , buf->i4_dis_dwt2
            // produces buf->i4_decouple_r , buf->i4_decouple_a
			adm_decouple_s123_device(buf, w, h, buf_stride, &p, s->str);

            // consumes buf->i4_ref_dwt2
            // produces buf->adm_csf_den[1,2,3]
			adm_csf_den_s123_device(
			        buf, scale, w, h, buf_stride,
			        adm_norm_view_dist, adm_ref_display_height, s->str);

            // consumes buf->i4_decouple_a
            // produces buf->i4_csf_a , buf->i4_csf_f
			i4_adm_csf_device(buf, scale, w, h, buf_stride, &p, s->str);

            // consumes buf->i4_decouple_r, buf->i4_csf_a, buf->i4_csf_a
            // produces buf->adm_cm[1,2,3]
			i4_adm_cm_device(buf, w, h, buf_stride, buf_stride, scale, &p, s->str);
		}

		i4_curr_ref_scale = buf->i4_ref_dwt2.band_a;
		i4_curr_dis_scale = buf->i4_dis_dwt2.band_a;

		curr_ref_stride = buf_stride;
		curr_dis_stride = buf_stride;
    }
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuMemcpyDtoHAsync(buf->results_host, buf->tmp_res->data, sizeof(int64_t) * RES_BUFFER_SIZE, s->str));
    CHECK_CUDA(cuEventRecord(s->finished, s->str));
}

static CUdeviceptr init_dwt_band_cuda(struct VmafCudaState *cu_state,
                                      struct cuda_adm_dwt_band_t *band,
                                      CUdeviceptr data_top, size_t stride)
{
    band->band_a = (int16_t *)data_top; data_top += stride;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
                                          struct cuda_adm_dwt_band_t *band,
                                          CUdeviceptr data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr i4_init_dwt_band_cuda(struct VmafCudaState *cu_state,
                                         struct cuda_i4_adm_dwt_band_t *band,
                                         CUdeviceptr data_top, size_t stride)
{
    band->band_a = (int32_t *)data_top; data_top += stride;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}
static CUdeviceptr i4_init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
                                             struct cuda_i4_adm_dwt_band_t *band,
                                             CUdeviceptr data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr init_res_cm_cuda(struct VmafCudaState *cu_state,
                                    int64_t *scale_pointer[],
                                    CUdeviceptr data_top)
{
    const int stride = 3 * sizeof(int64_t);
    scale_pointer[0] = (int64_t *)data_top; data_top += stride;
    scale_pointer[1] = (int64_t *)data_top; data_top += stride;
    scale_pointer[2] = (int64_t *)data_top; data_top += stride;
    scale_pointer[3] = (int64_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr init_res_csf_cuda(struct VmafCudaState *cu_state,
                                     uint64_t *scale_pointer[],
                                     CUdeviceptr data_top)
{
    const int stride = 3 * sizeof(uint64_t);
    scale_pointer[0] = (uint64_t *)data_top; data_top += stride;
    scale_pointer[1] = (uint64_t *)data_top; data_top += stride;
    scale_pointer[2] = (uint64_t *)data_top; data_top += stride;
    scale_pointer[3] = (uint64_t *)data_top; data_top += stride;
    return data_top;
}


static inline CUdeviceptr init_index_cuda(int32_t **index, CUdeviceptr data_top,
                                     size_t stride)
{
    index[0] = (int32_t *)data_top; data_top += stride;
    index[1] = (int32_t *)data_top; data_top += stride;
    index[2] = (int32_t *)data_top; data_top += stride;
    index[3] = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmStateCuda *s = fex->priv;
    
    (void) pix_fmt;
    (void) bpc;
    int ret = 0;

    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->ref_event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->dis_event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    s->dwt2_8 = dwt2_8_device;

    s->integer_stride   = ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x   = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y   = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    size_t buf_sz_one   = s->buf.ind_size_x * ((h + 1) / 2);

    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.data_buf, buf_sz_one * NUM_BUFS_ADM);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_ref, (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_dis, (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_accum, sizeof(uint64_t) * 3 * w * h);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_accum_h, sizeof(uint64_t) * 3 * h);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_res, sizeof(uint64_t) * RES_BUFFER_SIZE);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->buf.results_host, sizeof(uint64_t) * RES_BUFFER_SIZE);
    if (ret) goto free_ref;

    CUdeviceptr cu_res_top;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.tmp_res, &cu_res_top);
    if (ret) goto free_ref;

    cu_res_top = init_res_cm_cuda(fex->cu_state, s->buf.adm_cm, cu_res_top);
    cu_res_top = init_res_csf_cuda(fex->cu_state, s->buf.adm_csf_den, cu_res_top);

    CUdeviceptr cu_data_top;
    vmaf_cuda_buffer_get_dptr(s->buf.data_buf, &cu_data_top);

    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.ref_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.dis_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.decouple_r, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.decouple_a, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.csf_a, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.csf_f, cu_data_top, buf_sz_one / 2);

    cu_data_top = i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_ref_dwt2, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_dis_dwt2, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_decouple_r, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_decouple_a, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_csf_a, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_csf_f, cu_data_top, buf_sz_one);

    s->write_score_parameters = malloc(sizeof(write_score_parameters_adm));
    ((write_score_parameters_adm*)s->write_score_parameters)->s = s;


    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto free_ref;

    return 0;

free_ref:
    if (s->buf.data_buf) {    
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data_buf);
        free(s->buf.data_buf);
    }
    if (s->buf.tmp_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_ref);
        free(s->buf.tmp_ref);
    }
    if (s->buf.tmp_accum) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum);
        free(s->buf.tmp_accum);
    }
    if (s->buf.tmp_accum_h) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum_h);
        free(s->buf.tmp_accum_h);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }
    vmaf_dictionary_free(&s->feature_name_dict);

    return -ENOMEM;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{

    AdmStateCuda *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    
    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cuStreamSynchronize(s->str));
    // CHECK_CUDA(cuEventSynchronize(s->finished));
    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuEventDestroy(s->finished));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    
    // current implementation is limited by the 16-bit data pipeline, thus
    // cannot handle an angular frequency smaller than 1080p * 3H
    if (s->adm_norm_view_dist * s->adm_ref_display_height <
        DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) {
        return -EINVAL;
    }

    integer_compute_adm_cuda(fex, s, ref_pic, dist_pic, &s->buf,
                        s->adm_enhn_gain_limit,
                        s->adm_norm_view_dist, s->adm_ref_display_height);
    
    write_score_parameters_adm *data = s->write_score_parameters;
    data->feature_collector = feature_collector;
    data->index = index;
    data->h = ref_pic->h[0];
    data->w = ref_pic->w[0];
    CHECK_CUDA(cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cuLaunchHostFunc(s->host_stream, (CUhostFn)write_scores, data));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    AdmStateCuda *s = fex->priv;
    CHECK_CUDA(cuStreamSynchronize(s->str));

    int ret = 0;

    if (s->buf.data_buf) {    
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data_buf);
        free(s->buf.data_buf);
    }
    if (s->buf.tmp_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_ref);
        free(s->buf.tmp_ref);
    }
    if (s->buf.tmp_accum) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum);
        free(s->buf.tmp_accum);
    }
    if (s->buf.tmp_accum_h) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum_h);
        free(s->buf.tmp_accum_h);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }
    if (s->write_score_parameters)  free(s->write_score_parameters);

    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector)
{
    AdmStateCuda *s = fex->priv;
    CHECK_CUDA(cuStreamSynchronize(s->str));
    return 1;
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

VmafFeatureExtractor vmaf_fex_integer_adm_cuda = {
    .name = "adm_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .options = options_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(AdmStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA
};
