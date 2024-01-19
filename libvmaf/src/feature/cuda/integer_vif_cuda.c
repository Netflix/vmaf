/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <errno.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "cpu.h"
#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "picture.h"
#include "cuda/integer_vif_cuda.h"
#include "picture_cuda.h"

#if ARCH_X86
#include "x86/vif_avx2.h"
#if HAVE_AVX512
#include "x86/vif_avx512.h"
#endif
#endif

typedef struct VifStateCuda {
    VifBufferCuda buf;
    CUevent event, finished;
    CUstream str, host_stream;
    bool debug;
    double vif_enhn_gain_limit;
    void (*filter1d_8)(VifBufferCuda *buf, uint8_t* ref_in, uint8_t* dis_in, unsigned w, unsigned h, double vif_enhn_gain_limit, CUstream stream);
    void (*filter1d_16)(VifBufferCuda *buf,  uint16_t* ref_in, uint16_t* dis_in, unsigned w, unsigned h, int scale,
            int bpc, double vif_enhn_gain_limit, CUstream stream);
    VmafDictionary *feature_name_dict;
    CUfunction func_filter1d_8_vertical_kernel_uint32_t_17_9,
               func_filter1d_8_horizontal_kernel_2_17_9,
               func_filter1d_16_vertical_kernel_uint2_17_9_0,
               func_filter1d_16_vertical_kernel_uint2_9_5_1,
               func_filter1d_16_vertical_kernel_uint2_5_3_2,
               func_filter1d_16_vertical_kernel_uint2_3_0_3,
               func_filter1d_16_horizontal_kernel_2_17_9_0,
               func_filter1d_16_horizontal_kernel_2_9_5_1,
               func_filter1d_16_horizontal_kernel_2_5_3_2,
               func_filter1d_16_horizontal_kernel_2_3_0_3
} VifStateCuda;

typedef struct write_score_parameters_vif {
    VmafFeatureCollector *feature_collector;
    VifStateCuda *s;
    unsigned index;
} write_score_parameters_vif;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
            "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifStateCuda, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
        unsigned bpc, unsigned w, unsigned h)
{
    VifStateCuda *s = fex->priv;

    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));

    // make this static
    CUmodule filter1d_module;
    CHECK_CUDA(cuModuleLoadData(&filter1d_module, src_filter1d_ptx));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_8_vertical_kernel_uint32_t_17_9,
                filter1d_module, "filter1d_8_vertical_kernel_uint32_t_17_9"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_8_horizontal_kernel_2_17_9,
                filter1d_module, "filter1d_8_horizontal_kernel_2_17_9"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_17_9_0,
                filter1d_module, "filter1d_16_vertical_kernel_uint2_17_9_0"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_9_5_1,
                filter1d_module, "filter1d_16_vertical_kernel_uint2_9_5_1"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_5_3_2,
                filter1d_module, "filter1d_16_vertical_kernel_uint2_5_3_2"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_3_0_3,
                filter1d_module, "filter1d_16_vertical_kernel_uint2_3_0_3"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_17_9_0,
                filter1d_module, "filter1d_16_horizontal_kernel_2_17_9_0"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_9_5_1,
                filter1d_module, "filter1d_16_horizontal_kernel_2_9_5_1"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_5_3_2,
                filter1d_module, "filter1d_16_horizontal_kernel_2_5_3_2"));
    CHECK_CUDA(cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_3_0_3,
                filter1d_module, "filter1d_16_horizontal_kernel_2_3_0_3"));

    CHECK_CUDA(cuCtxPopCurrent(NULL));

    (void) pix_fmt;
    const bool hbd = bpc > 8;

    int tex_alignment;
    CHECK_CUDA(cuDeviceGetAttribute(&tex_alignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                fex->cu_state->dev));
    s->buf.stride = tex_alignment * (((w * (1 << (int)hbd) + tex_alignment - 1) / tex_alignment));
    s->buf.stride_16 = ALIGN_CEIL(w * sizeof(uint16_t));
    s->buf.stride_32 = ALIGN_CEIL(w * sizeof(uint32_t));
    s->buf.stride_64 = ALIGN_CEIL(w * sizeof(uint64_t));
    s->buf.stride_tmp =
        ALIGN_CEIL(w * sizeof(uint32_t));
    const size_t frame_size = s->buf.stride * h;
    const size_t pad_size = s->buf.stride * 8;
    const size_t data_sz = 2 * frame_size +
        2 * (h * s->buf.stride_16) + 5 * (h * s->buf.stride_32) + 8 * (s->buf.stride_tmp * h); // intermediater buffers
    int ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.data, data_sz);
    if (ret) goto free_ref;

    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.accum_data, sizeof(vif_accums) * 4);
    if (ret) goto free_ref;

    ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, (void**)&s->buf.accum_host, sizeof(vif_accums) * 4);
    if (ret) goto free_ref;

    CUdeviceptr data;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.data, &data);
    if (ret) goto free_ref;

    s->buf.ref = data; data += frame_size;
    s->buf.dis = data; data += frame_size;
    s->buf.mu1 = data; data += h * s->buf.stride_16;
    s->buf.mu2 = data; data += h * s->buf.stride_16;
    s->buf.mu1_32 = data; data += h * s->buf.stride_32;
    s->buf.mu2_32 = data; data += h * s->buf.stride_32;
    s->buf.ref_sq = data; data += h * s->buf.stride_32;
    s->buf.dis_sq = data; data += h * s->buf.stride_32;
    s->buf.ref_dis = data; data += h * s->buf.stride_32;
    s->buf.tmp.mu1 = data; data += s->buf.stride_tmp * h;
    s->buf.tmp.mu2 = data; data += s->buf.stride_tmp * h;
    s->buf.tmp.ref = data; data += s->buf.stride_tmp * h;
    s->buf.tmp.dis = data; data += s->buf.stride_tmp * h;
    s->buf.tmp.ref_dis = data; data += s->buf.stride_tmp * h;
    s->buf.tmp.ref_convol = data; data += s->buf.stride_tmp  * h;
    s->buf.tmp.dis_convol = data; data += s->buf.stride_tmp  * h;
    s->buf.tmp.padding = data; data += s->buf.stride_tmp  * h;

    CUdeviceptr data_accum;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.accum_data, &data_accum);
    if (ret) goto free_ref;

    s->buf.accum = data_accum;

    s->buf.cpu_param_buf = malloc(sizeof(write_score_parameters_vif));
    write_score_parameters_vif *data_p = s->buf.cpu_param_buf;
    data_p->s = s;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto free_ref;
    return 0;

free_ref:
    if (s->buf.data) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data);
        free(s->buf.data);
    }
    if (s->buf.accum_data) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.accum_data);
        free(s->buf.accum_data);
    }
    if (s->buf.accum_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.accum_host);
    }

    return -ENOMEM;
}

void filter1d_8(VifStateCuda *s, VifBufferCuda *buf, uint8_t* ref_in, uint8_t* dis_in, int w, int h, double vif_enhn_gain_limit,
        CUstream stream) {
    {

        const int size_of_alignment_type = sizeof(uint32_t),
        BLOCKX = 128 / size_of_alignment_type,
        BLOCKY = 128 / (VMAF_CUDA_CACHE_LINE_SIZE / size_of_alignment_type);
        void* args_vert[] = {
            &*buf, &ref_in, &dis_in, &w, &h, &vif_filter1d_table
        };

        CHECK_CUDA(cuLaunchKernel(s->func_filter1d_8_vertical_kernel_uint32_t_17_9,
                    DIV_ROUND_UP(w, BLOCKX * size_of_alignment_type), DIV_ROUND_UP(h, BLOCKY), 1,
                    BLOCKX, BLOCKY, 1,
                    0, stream, args_vert, NULL));
    }
    {
        const int BLOCKX = 128, BLOCKY = 1, val_per_thread = 2;

        void* args_hori[] = {
            &*buf, &w, &h, &vif_filter1d_table,
            &vif_enhn_gain_limit, &buf->accum
        };

        CHECK_CUDA(cuLaunchKernel(s->func_filter1d_8_horizontal_kernel_2_17_9,
                    DIV_ROUND_UP(w, BLOCKX * val_per_thread), DIV_ROUND_UP(h, BLOCKY), 1,
                    BLOCKX, BLOCKY, 1,
                    0, stream, args_hori, NULL));
    }
}

void filter1d_16(VifStateCuda *s, VifBufferCuda *buf, uint16_t* ref_in, uint16_t* dis_in, int w, int h, int scale, int bpc,
        double vif_enhn_gain_limit, CUstream stream) {

    int32_t add_shift_round_HP, shift_HP;
    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    if (scale == 0) {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    } else {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }
    
    struct uint2 {
        unsigned x ,y;
    } uint2;

    const int size_of_alginment = sizeof(uint2),
          val_per_thread = size_of_alginment / sizeof(uint16_t),
          BLOCKX = 128,
          BLOCK_VERT_X = VMAF_CUDA_CACHE_LINE_SIZE / val_per_thread, BLOCK_VERT_Y = 128 / (VMAF_CUDA_CACHE_LINE_SIZE / val_per_thread);
    const int GRID_VERT_X = DIV_ROUND_UP(w, BLOCK_VERT_X * val_per_thread), GRID_VERT_Y = DIV_ROUND_UP(h, BLOCK_VERT_Y),
          GRID_HORI_X = DIV_ROUND_UP(w, BLOCKX), GRID_HORI_Y = h;

    void * args_vert[] = {
        &*buf, &ref_in, &dis_in, &w, &h, &add_shift_round_VP, &shift_VP, &add_shift_round_VP_sq,
        &shift_VP_sq, &(*(filter_table_stuct *)vif_filter1d_table)
    };

    vif_accums * ptr = &((vif_accums *)buf->accum)[scale];

    void * args_hori[] = {
        &*buf, &w, &h, &add_shift_round_HP, &shift_HP,
        &vif_filter1d_table, &vif_enhn_gain_limit,
        &(ptr)
    };

    switch (scale) {
    case 0: {
                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_17_9_0,
                            GRID_VERT_X, GRID_VERT_Y, 1,
                            BLOCK_VERT_X, BLOCK_VERT_Y, 1,
                            0, stream, args_vert, NULL));

                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_17_9_0,
                            GRID_HORI_X, GRID_HORI_Y, 1,
                            BLOCKX, 1, 1,
                            0, stream, args_hori, NULL));
                break;
            }
    case 1: {
                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_9_5_1,
                            GRID_VERT_X, GRID_VERT_Y, 1,
                            BLOCK_VERT_X, BLOCK_VERT_Y, 1,
                            0, stream, args_vert, NULL));

                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_9_5_1,
                            GRID_HORI_X, GRID_HORI_Y, 1,
                            BLOCKX, 1, 1,
                            0, stream, args_hori, NULL));
                break;
            }
    case 2: {
                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_5_3_2,
                            GRID_VERT_X, GRID_VERT_Y, 1,
                            BLOCK_VERT_X, BLOCK_VERT_Y, 1,
                            0, stream, args_vert, NULL));

                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_5_3_2,
                            GRID_HORI_X, GRID_HORI_Y, 1,
                            BLOCKX, 1, 1,
                            0, stream, args_hori, NULL));
                break;
            }
    case 3: {
                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_3_0_3,
                            GRID_VERT_X, GRID_VERT_Y, 1,
                            BLOCK_VERT_X, BLOCK_VERT_Y, 1,
                            0, stream, args_vert, NULL));

                CHECK_CUDA(cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_3_0_3,
                            GRID_HORI_X, GRID_HORI_Y, 1,
                            BLOCKX, 1, 1,
                            0, stream, args_hori, NULL));
                break;
            }
    }
}


typedef struct VifScore {
    struct {
        float num;
        float den;
    } scale[4];
} VifScore;

static int write_scores(write_score_parameters_vif* data)
{
    VmafFeatureCollector *feature_collector = data->feature_collector;
    VifStateCuda *s = data->s;
    unsigned index = data->index;

    VifScore vif;
    vif_accums *accum = (vif_accums*)data->s->buf.accum_host;
    for (unsigned scale = 0; scale < 4; ++scale) {
        vif.scale[scale].num =
            accum[scale].num_log / 2048.0 + accum[scale].x2 +
            (accum[scale].den_non_log -
             ((accum[scale].num_non_log) / 16384.0) / (65025.0));
        vif.scale[scale].den =
            accum[scale].den_log / 2048.0 -
            (accum[scale].x + (accum[scale].num_x * 17)) +
            accum[scale].den_non_log;
    }
    int err = 0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale0_score",
            vif.scale[0].num / vif.scale[0].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale1_score",
            vif.scale[1].num / vif.scale[1].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale2_score",
            vif.scale[2].num / vif.scale[2].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale3_score",
            vif.scale[3].num / vif.scale[3].den, index);

    if (!s->debug) return err;

    const double score_num =
        (double)vif.scale[0].num + (double)vif.scale[1].num +
        (double)vif.scale[2].num + (double)vif.scale[3].num;

    const double score_den =
        (double)vif.scale[0].den + (double)vif.scale[1].den +
        (double)vif.scale[2].den + (double)vif.scale[3].den;

    const double score =
        score_den == 0.0 ? 1.0f : score_num / score_den;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale0", vif.scale[0].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale0", vif.scale[0].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale1", vif.scale[1].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale1", vif.scale[1].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale2", vif.scale[2].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale2", vif.scale[2].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale3", vif.scale[3].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale3", vif.scale[3].den,
            index);

    return err;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
        VmafPicture *ref_pic, VmafPicture *ref_pic_90,
        VmafPicture *dist_pic, VmafPicture *dist_pic_90,
        unsigned index, VmafFeatureCollector *feature_collector)
{
    VifStateCuda *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;

    int w = ref_pic->w[0];
    int h = dist_pic->h[0];

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    // before the GPU has finished writing to it.
    CHECK_CUDA(cuStreamSynchronize(s->str));
    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuEventDestroy(s->finished));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    CHECK_CUDA(cuMemsetD8Async(s->buf.accum_data->data, 0, sizeof(vif_accums) * 4, s->str));
    CHECK_CUDA(cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic), vmaf_cuda_picture_get_ready_event(dist_pic), CU_EVENT_WAIT_DEFAULT));
    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale > 0) {
            w /= 2; h /= 2;
        }

        if (ref_pic->bpc == 8 && scale == 0) {
            filter1d_8(s, &s->buf, (uint8_t*)ref_pic->data[0], (uint8_t*)dist_pic->data[0], w, h, s->vif_enhn_gain_limit, vmaf_cuda_picture_get_stream(ref_pic));
        } else if (scale == 0) {
            filter1d_16(s, &s->buf, (uint16_t*)ref_pic->data[0], (uint16_t*)dist_pic->data[0], w, h, scale, ref_pic->bpc, s->vif_enhn_gain_limit, vmaf_cuda_picture_get_stream(ref_pic));
        } else {
            filter1d_16(s, &s->buf, (uint16_t*)s->buf.ref, (uint16_t*)s->buf.dis, w, h, scale, ref_pic->bpc, s->vif_enhn_gain_limit, s->str);
        }
        if(scale == 0)
        {
            // This event ensures the input buffer is consumed
            CHECK_CUDA(cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
            CHECK_CUDA(cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
            CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
            CHECK_CUDA(cuEventDestroy(s->event));
            CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
            CHECK_CUDA(cuCtxPopCurrent(NULL));
        }
    }

    // log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    // num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log /
    // 65536.0) / (255.0*255.0)); den[0] = accum_den_log / 2048.0 +
    vif_accums *accum = (vif_accums*)s->buf.accum_host;
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuMemcpyDtoHAsync(accum, s->buf.accum_data->data,
                sizeof(*accum) * 4, s->str));
    CHECK_CUDA(cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));

    write_score_parameters_vif *data = s->buf.cpu_param_buf;
    data->feature_collector = feature_collector;
    data->index = index;
    CHECK_CUDA(cuLaunchHostFunc(s->str, write_scores, data));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    VifStateCuda *s = fex->priv;
    CHECK_CUDA(cuStreamSynchronize(s->str));

    int ret = 0;
    if (s->buf.data) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data);
        free(s->buf.data);
    }
    if (s->buf.accum_data) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.accum_data);
        free(s->buf.accum_data);
    }
    if (s->buf.accum_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.accum_host);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    VifStateCuda *s = fex->priv;

    CHECK_CUDA(cuStreamSynchronize(s->str));
    return 1;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale3_score",
    "integer_vif", "integer_vif_num", "integer_vif_den", "integer_vif_num_scale0",
    "integer_vif_den_scale0", "integer_vif_num_scale1", "integer_vif_den_scale1",
    "integer_vif_num_scale2", "integer_vif_den_scale2", "integer_vif_num_scale3",
    "integer_vif_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif_cuda = {
    .name = "vif_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .options = options,
    .close = close_fex_cuda,
    .priv_size = sizeof(VifStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA
};
