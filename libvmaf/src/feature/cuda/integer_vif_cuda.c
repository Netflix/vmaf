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
#include "drain_batch.h"
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
    CUstream str;
    /* Engine-scope fence batching opt-in flag (T-GPU-OPT-1, ADR-0242). */
    bool drained;
    bool debug;
    double vif_enhn_gain_limit;
    void (*filter1d_8)(VifBufferCuda *buf, uint8_t *ref_in, uint8_t *dis_in, unsigned w, unsigned h,
                       double vif_enhn_gain_limit, CUstream stream);
    void (*filter1d_16)(VifBufferCuda *buf, uint16_t *ref_in, uint16_t *dis_in, unsigned w,
                        unsigned h, int scale, int bpc, double vif_enhn_gain_limit,
                        CUstream stream);
    VmafDictionary *feature_name_dict;
    CUfunction func_filter1d_8_vertical_kernel_uint32_t_17_9,
        func_filter1d_8_horizontal_kernel_2_17_9, func_filter1d_16_vertical_kernel_uint2_17_9_0,
        func_filter1d_16_vertical_kernel_uint2_9_5_1, func_filter1d_16_vertical_kernel_uint2_5_3_2,
        func_filter1d_16_vertical_kernel_uint2_3_0_3, func_filter1d_16_horizontal_kernel_2_17_9_0,
        func_filter1d_16_horizontal_kernel_2_9_5_1, func_filter1d_16_horizontal_kernel_2_5_3_2,
        func_filter1d_16_horizontal_kernel_2_3_0_3;
} VifStateCuda;

typedef struct write_score_parameters_vif {
    VmafFeatureCollector *feature_collector;
    VifStateCuda *s;
    unsigned index;
} write_score_parameters_vif;

static const VmafOption options[] = {{
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
                                     {0}};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    VifStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);
    // make this static
    CUmodule filter1d_module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&filter1d_module, filter1d_ptx), fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_8_vertical_kernel_uint32_t_17_9,
                                        filter1d_module,
                                        "filter1d_8_vertical_kernel_uint32_t_17_9"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_8_horizontal_kernel_2_17_9,
                                        filter1d_module, "filter1d_8_horizontal_kernel_2_17_9"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_17_9_0,
                                        filter1d_module,
                                        "filter1d_16_vertical_kernel_uint2_17_9_0"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_9_5_1,
                                        filter1d_module, "filter1d_16_vertical_kernel_uint2_9_5_1"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_5_3_2,
                                        filter1d_module, "filter1d_16_vertical_kernel_uint2_5_3_2"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_vertical_kernel_uint2_3_0_3,
                                        filter1d_module, "filter1d_16_vertical_kernel_uint2_3_0_3"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_17_9_0,
                                        filter1d_module, "filter1d_16_horizontal_kernel_2_17_9_0"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_9_5_1,
                                        filter1d_module, "filter1d_16_horizontal_kernel_2_9_5_1"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_5_3_2,
                                        filter1d_module, "filter1d_16_horizontal_kernel_2_5_3_2"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter1d_16_horizontal_kernel_2_3_0_3,
                                        filter1d_module, "filter1d_16_horizontal_kernel_2_3_0_3"),
                    fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    (void)pix_fmt;
    const bool hbd = bpc > 8;

    int tex_alignment;
    CHECK_CUDA_GOTO(cu_f,
                    cuDeviceGetAttribute(&tex_alignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                         fex->cu_state->dev),
                    fail_after_pop);
    s->buf.stride = tex_alignment * (((w * (1 << (int)hbd) + tex_alignment - 1) / tex_alignment));
    {
        const int rd_w_bytes = ((w + 1) / 2) * sizeof(uint16_t);
        s->buf.rd_stride = tex_alignment * ((rd_w_bytes + tex_alignment - 1) / tex_alignment);
    }
    s->buf.stride_16 = ALIGN_CEIL(w * sizeof(uint16_t));
    s->buf.stride_32 = ALIGN_CEIL(w * sizeof(uint32_t));
    s->buf.stride_64 = ALIGN_CEIL(w * sizeof(uint64_t));
    s->buf.stride_tmp = ALIGN_CEIL(w * sizeof(uint32_t));
    const size_t rd_size = s->buf.rd_stride * ((h + 1) / 2);
    const size_t data_sz = 2 * rd_size + 2 * (h * s->buf.stride_16) + 5 * (h * s->buf.stride_32) +
                           8 * (s->buf.stride_tmp * h); // intermediater buffers
    int ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.data, data_sz);
    if (ret)
        goto free_ref;

    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.accum_data, sizeof(vif_accums) * 4);
    if (ret)
        goto free_ref;

    ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->buf.accum_host,
                                      sizeof(vif_accums) * 4);
    if (ret)
        goto free_ref;

    CUdeviceptr data;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.data, &data);
    if (ret)
        goto free_ref;

    s->buf.ref = data;
    data += rd_size;
    s->buf.dis = data;
    data += rd_size;
    // CUDA device pointers arrive as CUdeviceptr (unsigned long long) and must
    // be cast to host-visible pointer types to carve the buffer into typed
    // subregions the launch-time kernel-args struct references. The cast is
    // inherent to the CUDA Driver API and cannot be refactored away without
    // changing the public libvmaf-CUDA contract. Per ADR-0141 touched-file
    // rule, upstream-parity exception.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    s->buf.mu1 = (uint16_t *)data;
    data += h * s->buf.stride_16;
    s->buf.mu2 = (uint16_t *)data;
    data += h * s->buf.stride_16;
    s->buf.mu1_32 = (uint32_t *)data;
    data += h * s->buf.stride_32;
    s->buf.mu2_32 = (uint32_t *)data;
    data += h * s->buf.stride_32;
    s->buf.ref_sq = (uint32_t *)data;
    data += h * s->buf.stride_32;
    s->buf.dis_sq = (uint32_t *)data;
    data += h * s->buf.stride_32;
    s->buf.ref_dis = (uint32_t *)data;
    data += h * s->buf.stride_32;
    s->buf.tmp.mu1 = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.mu2 = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.ref = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.dis = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.ref_dis = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.ref_convol = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.dis_convol = (uint32_t *)data;
    data += s->buf.stride_tmp * h;
    s->buf.tmp.padding = (uint32_t *)data;
    data += s->buf.stride_tmp * h;

    CUdeviceptr data_accum;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.accum_data, &data_accum);
    if (ret)
        goto free_ref;

    s->buf.accum = (int64_t *)data_accum;
    // NOLINTEND(performance-no-int-to-ptr)

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;
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
    (void)ret; // accumulated cleanup status intentionally discarded on error path

    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

int filter1d_8(VifStateCuda *s, VifBufferCuda *buf, uint8_t *ref_in, uint8_t *dis_in, int w, int h,
               double vif_enhn_gain_limit, CudaFunctions *cu_f, CUstream stream)
{
    {

        const int size_of_alignment_type = sizeof(uint32_t), BLOCKX = 128 / size_of_alignment_type,
                  BLOCKY = 128 / (VMAF_CUDA_CACHE_LINE_SIZE / size_of_alignment_type);
        void *args_vert[] = {&*buf, &ref_in, &dis_in, &w, &h, (uint16_t *)&vif_filter1d_table};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_8_vertical_kernel_uint32_t_17_9,
                                               DIV_ROUND_UP(w, BLOCKX * size_of_alignment_type),
                                               DIV_ROUND_UP(h, BLOCKY), 1, BLOCKX, BLOCKY, 1, 0,
                                               stream, args_vert, NULL));
    }
    {
        const int BLOCKX = 128, BLOCKY = 1, val_per_thread = 2;

        void *args_hori[] = {
            &*buf, &w, &h, (uint16_t *)&vif_filter1d_table, &vif_enhn_gain_limit, &buf->accum};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_8_horizontal_kernel_2_17_9,
                                               DIV_ROUND_UP(w, BLOCKX * val_per_thread),
                                               DIV_ROUND_UP(h, BLOCKY), 1, BLOCKX, BLOCKY, 1, 0,
                                               stream, args_hori, NULL));
    }
    return 0;
}

int filter1d_16(VifStateCuda *s, VifBufferCuda *buf, uint16_t *ref_in, uint16_t *dis_in, int w,
                int h, int scale, int bpc, double vif_enhn_gain_limit, CudaFunctions *cu_f,
                CUstream stream)
{

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
        unsigned x, y;
    } uint2;

    const int size_of_alginment = sizeof(uint2),
              val_per_thread = size_of_alginment / sizeof(uint16_t), BLOCKX = 128,
              BLOCK_VERT_X = VMAF_CUDA_CACHE_LINE_SIZE / val_per_thread,
              BLOCK_VERT_Y = 128 / (VMAF_CUDA_CACHE_LINE_SIZE / val_per_thread);
    const int GRID_VERT_X = DIV_ROUND_UP(w, BLOCK_VERT_X * val_per_thread),
              GRID_VERT_Y = DIV_ROUND_UP(h, BLOCK_VERT_Y), GRID_HORI_X = DIV_ROUND_UP(w, BLOCKX),
              GRID_HORI_Y = h;

    void *args_vert[] = {&*buf,        &ref_in,
                         &dis_in,      &w,
                         &h,           &add_shift_round_VP,
                         &shift_VP,    &add_shift_round_VP_sq,
                         &shift_VP_sq, &(*(filter_table_stuct *)vif_filter1d_table)};

    vif_accums *ptr = &((vif_accums *)buf->accum)[scale];

    void *args_hori[] = {&*buf,
                         &w,
                         &h,
                         &add_shift_round_HP,
                         &shift_HP,
                         (uint16_t *)&vif_filter1d_table,
                         &vif_enhn_gain_limit,
                         &(ptr)};

    switch (scale) {
    case 0: {
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_17_9_0,
                                               GRID_VERT_X, GRID_VERT_Y, 1, BLOCK_VERT_X,
                                               BLOCK_VERT_Y, 1, 0, stream, args_vert, NULL));

        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_17_9_0,
                                               GRID_HORI_X, GRID_HORI_Y, 1, BLOCKX, 1, 1, 0, stream,
                                               args_hori, NULL));
        break;
    }
    case 1: {
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_9_5_1,
                                               GRID_VERT_X, GRID_VERT_Y, 1, BLOCK_VERT_X,
                                               BLOCK_VERT_Y, 1, 0, stream, args_vert, NULL));

        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_9_5_1, GRID_HORI_X,
                                         GRID_HORI_Y, 1, BLOCKX, 1, 1, 0, stream, args_hori, NULL));
        break;
    }
    case 2: {
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_5_3_2,
                                               GRID_VERT_X, GRID_VERT_Y, 1, BLOCK_VERT_X,
                                               BLOCK_VERT_Y, 1, 0, stream, args_vert, NULL));

        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_5_3_2, GRID_HORI_X,
                                         GRID_HORI_Y, 1, BLOCKX, 1, 1, 0, stream, args_hori, NULL));
        break;
    }
    case 3: {
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_filter1d_16_vertical_kernel_uint2_3_0_3,
                                               GRID_VERT_X, GRID_VERT_Y, 1, BLOCK_VERT_X,
                                               BLOCK_VERT_Y, 1, 0, stream, args_vert, NULL));

        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_filter1d_16_horizontal_kernel_2_3_0_3, GRID_HORI_X,
                                         GRID_HORI_Y, 1, BLOCKX, 1, 1, 0, stream, args_hori, NULL));
        break;
    }
    default:
        break; /* VIF has exactly 4 scales (0–3); unreachable with valid input */
    }
    return 0;
}

typedef struct VifScore {
    struct {
        float num;
        float den;
    } scale[4];
} VifScore;

static int write_scores(write_score_parameters_vif *data)
{
    VmafFeatureCollector *feature_collector = data->feature_collector;
    VifStateCuda *s = data->s;
    unsigned index = data->index;

    VifScore vif;
    vif_accums *accum = (vif_accums *)data->s->buf.accum_host;
    for (unsigned scale = 0; scale < 4; ++scale) {
        vif.scale[scale].num =
            accum[scale].num_log / 2048.0 + accum[scale].x2 +
            (accum[scale].den_non_log - ((accum[scale].num_non_log) / 16384.0) / (65025.0));
        vif.scale[scale].den = accum[scale].den_log / 2048.0 -
                               (accum[scale].x + (accum[scale].num_x * 17)) +
                               accum[scale].den_non_log;
    }
    int err = 0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_vif_scale0_score",
                                                   vif.scale[0].num / vif.scale[0].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_vif_scale1_score",
                                                   vif.scale[1].num / vif.scale[1].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_vif_scale2_score",
                                                   vif.scale[2].num / vif.scale[2].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_vif_scale3_score",
                                                   vif.scale[3].num / vif.scale[3].den, index);

    if (!s->debug)
        return err;

    const double score_num = (double)vif.scale[0].num + (double)vif.scale[1].num +
                             (double)vif.scale[2].num + (double)vif.scale[3].num;

    const double score_den = (double)vif.scale[0].den + (double)vif.scale[1].den +
                             (double)vif.scale[2].den + (double)vif.scale[3].den;

    const double score = score_den == 0.0 ? 1.0f : score_num / score_den;

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_vif", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_vif_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_vif_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_num_scale0", vif.scale[0].num, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_den_scale0", vif.scale[0].den, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_num_scale1", vif.scale[1].num, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_den_scale1", vif.scale[1].den, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_num_scale2", vif.scale[2].num, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_den_scale2", vif.scale[2].den, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_num_scale3", vif.scale[3].num, index);

    err |= vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "integer_vif_den_scale3", vif.scale[3].den, index);

    return err;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    VifStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    (void)ref_pic_90;
    (void)dist_pic_90;

    int w = ref_pic->w[0];
    int h = dist_pic->h[0];

    CHECK_CUDA_RETURN(cu_f,
                      cuMemsetD8Async(s->buf.accum_data->data, 0, sizeof(vif_accums) * 4, s->str));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                                              vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));
    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale > 0) {
            w /= 2;
            h /= 2;
        }

        int err = 0;
        if (ref_pic->bpc == 8 && scale == 0) {
            err =
                filter1d_8(s, &s->buf, (uint8_t *)ref_pic->data[0], (uint8_t *)dist_pic->data[0], w,
                           h, s->vif_enhn_gain_limit, cu_f, vmaf_cuda_picture_get_stream(ref_pic));
        } else if (scale == 0) {
            err = filter1d_16(s, &s->buf, (uint16_t *)ref_pic->data[0],
                              (uint16_t *)dist_pic->data[0], w, h, scale, ref_pic->bpc,
                              s->vif_enhn_gain_limit, cu_f, vmaf_cuda_picture_get_stream(ref_pic));
        } else {
            // s->buf.ref / s->buf.dis are CUdeviceptr (unsigned long long) carved
            // from the master buffer in init; the filter1d_16 contract takes a
            // typed uint16_t* view. Cast is inherent to the CUDA Driver API.
            // Per ADR-0141 touched-file rule, upstream-parity exception.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            err = filter1d_16(s, &s->buf, (uint16_t *)s->buf.ref, (uint16_t *)s->buf.dis, w, h,
                              scale, ref_pic->bpc, s->vif_enhn_gain_limit, cu_f, s->str);
        }
        if (err)
            return err;
        if (scale == 0) {
            // This event ensures the input buffer is consumed
            CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
            CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
        }
    }

    // Queue async download of accumulators
    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->buf.accum_host, s->buf.accum_data->data,
                                              sizeof(vif_accums) * 4, s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    /* Engine-scope fence batching opt-in (T-GPU-OPT-1, ADR-0242). */
    (void)vmaf_cuda_drain_batch_register_event(s->finished, &s->drained);
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    VifStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (s->drained) {
        s->drained = false;
    } else {
        CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));
    }

    // Results are now in accum_host — compute and write scores
    write_score_parameters_vif data = {
        .feature_collector = feature_collector,
        .s = s,
        .index = index,
    };
    return write_scores(&data);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    VifStateCuda *s = fex->priv;
    /* Close path continues unwinding on CUDA error. */
    int _cuda_err = 0;
    CHECK_CUDA_GOTO(fex->cu_state->f, cuStreamSynchronize(s->str), after_sync);
after_sync:
    CHECK_CUDA_GOTO(fex->cu_state->f, cuStreamDestroy(s->str), after_stream);
after_stream:
    CHECK_CUDA_GOTO(fex->cu_state->f, cuEventDestroy(s->event), after_ev1);
after_ev1:
    CHECK_CUDA_GOTO(fex->cu_state->f, cuEventDestroy(s->finished), after_ev2);
after_ev2:;

    int ret = _cuda_err;
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

static int flush_fex_cuda(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    VifStateCuda *s = fex->priv;

    CHECK_CUDA_RETURN(fex->cu_state->f, cuStreamSynchronize(s->str));
    return 1;
}

static const char *provided_features[] = {"VMAF_integer_feature_vif_scale0_score",
                                          "VMAF_integer_feature_vif_scale1_score",
                                          "VMAF_integer_feature_vif_scale2_score",
                                          "VMAF_integer_feature_vif_scale3_score",
                                          "integer_vif",
                                          "integer_vif_num",
                                          "integer_vif_den",
                                          "integer_vif_num_scale0",
                                          "integer_vif_den_scale0",
                                          "integer_vif_num_scale1",
                                          "integer_vif_den_scale1",
                                          "integer_vif_num_scale2",
                                          "integer_vif_den_scale2",
                                          "integer_vif_num_scale3",
                                          "integer_vif_den_scale3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_integer_vif_cuda = {.name = "vif_cuda",
                                                  .init = init_fex_cuda,
                                                  .submit = submit_fex_cuda,
                                                  .collect = collect_fex_cuda,
                                                  .flush = flush_fex_cuda,
                                                  .options = options,
                                                  .close = close_fex_cuda,
                                                  .priv_size = sizeof(VifStateCuda),
                                                  .provided_features = provided_features,
                                                  .flags = VMAF_FEATURE_EXTRACTOR_CUDA};
