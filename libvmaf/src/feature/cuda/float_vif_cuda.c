/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature kernel on the CUDA backend (T7-23 / batch 3
 *  part 5b — ADR-0192 / ADR-0197). CUDA twin of float_vif_vulkan.
 *
 *  v1: kernelscale=1.0 only. CPU's VIF_OPT_HANDLE_BORDERS branch:
 *  per-scale dims = prev/2 (no border crop); decimate samples at
 *  (2*gx, 2*gy) with mirror padding on the input filter taps.
 *
 *  Per-frame flow: 4 compute + 3 decimate launches. Submit/collect
 *  async stream pattern matches motion_cuda.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cuda/float_vif_cuda.h"
#include "cuda/kernel_template.h"
#include "cuda_helper.cuh"
#include "picture.h"
#include "picture_cuda.h"

#define FVIF_BX 16
#define FVIF_BY 16

typedef struct FloatVifStateCuda {
    bool debug;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;

    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0246). Multi-scale 4-pyramid state stays outside the
     * template's single-pair readback bundle. */
    VmafCudaKernelLifecycle lc;
    CUfunction func_compute;
    CUfunction func_decimate;

    VmafCudaBuffer *ref_raw;
    VmafCudaBuffer *dis_raw;
    VmafCudaBuffer *ref_buf[2];
    VmafCudaBuffer *dis_buf[2];

    VmafCudaBuffer *num_partials[4];
    VmafCudaBuffer *den_partials[4];
    float *num_host[4];
    float *den_host[4];
    unsigned wg_count[4];

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[4];
    unsigned scale_h[4];

    VmafDictionary *feature_name_dict;
} FloatVifStateCuda;

static const VmafOption options[] = {{.name = "debug",
                                      .help = "debug mode: enable additional output",
                                      .offset = offsetof(FloatVifStateCuda, debug),
                                      .type = VMAF_OPT_TYPE_BOOL,
                                      .default_val.b = false},
                                     {.name = "vif_enhn_gain_limit",
                                      .alias = "egl",
                                      .help = "enhancement gain imposed on vif (>= 1.0)",
                                      .offset = offsetof(FloatVifStateCuda, vif_enhn_gain_limit),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 100.0,
                                      .min = 1.0,
                                      .max = 100.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_kernelscale",
                                      .help = "scaling factor for the gaussian kernel",
                                      .offset = offsetof(FloatVifStateCuda, vif_kernelscale),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 1.0,
                                      .min = 0.1,
                                      .max = 4.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_sigma_nsq",
                                      .alias = "snsq",
                                      .help = "neural noise variance",
                                      .offset = offsetof(FloatVifStateCuda, vif_sigma_nsq),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 2.0,
                                      .min = 0.0,
                                      .max = 5.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {0}};

static void compute_per_scale_dims(FloatVifStateCuda *s)
{
    s->scale_w[0] = s->width;
    s->scale_h[0] = s->height;
    for (int i = 1; i < 4; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatVifStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (s->vif_kernelscale != 1.0)
        return -EINVAL;

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    compute_per_scale_dims(s);

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, float_vif_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_compute, module, "float_vif_compute"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_decimate, module, "float_vif_decimate"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t raw_bytes = (size_t)w * h * bpp;
    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->ref_raw, raw_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dis_raw, raw_bytes);
    const size_t fbytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->ref_buf[0], fbytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dis_buf[0], fbytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->ref_buf[1], fbytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dis_buf[1], fbytes);
    if (ret)
        goto free_buffers;

    for (int i = 0; i < 4; i++) {
        const unsigned gx = (s->scale_w[i] + FVIF_BX - 1u) / FVIF_BX;
        const unsigned gy = (s->scale_h[i] + FVIF_BY - 1u) / FVIF_BY;
        s->wg_count[i] = gx * gy;
        const size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->num_partials[i], pbytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->den_partials[i], pbytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->num_host[i], pbytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->den_host[i], pbytes);
    }
    if (ret)
        goto free_buffers;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_buffers;
    return 0;

free_buffers:
    if (s->ref_raw) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->ref_raw);
        free(s->ref_raw);
    }
    if (s->dis_raw) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->dis_raw);
        free(s->dis_raw);
    }
    for (int i = 0; i < 2; i++) {
        if (s->ref_buf[i]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->ref_buf[i]);
            free(s->ref_buf[i]);
        }
        if (s->dis_buf[i]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->dis_buf[i]);
            free(s->dis_buf[i]);
        }
    }
    for (int i = 0; i < 4; i++) {
        if (s->num_partials[i]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->num_partials[i]);
            free(s->num_partials[i]);
        }
        if (s->den_partials[i]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->den_partials[i]);
            free(s->den_partials[i]);
        }
    }
    (void)vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return _cuda_err;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)index;
    FloatVifStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    const ptrdiff_t raw_stride = (ptrdiff_t)(s->width * (s->bpc <= 8u ? 1u : 2u));

    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA_RETURN(cu_f,
                      cuStreamWaitEvent(pic_stream, vmaf_cuda_picture_get_ready_event(dist_pic),
                                        CU_EVENT_WAIT_DEFAULT));

    CUDA_MEMCPY2D cpy = {0};
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = (CUdeviceptr)ref_pic->data[0];
    cpy.srcPitch = ref_pic->stride[0];
    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstDevice = (CUdeviceptr)s->ref_raw->data;
    cpy.dstPitch = raw_stride;
    cpy.WidthInBytes = raw_stride;
    cpy.Height = s->height;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy, pic_stream));

    CUDA_MEMCPY2D cpy_d = cpy;
    cpy_d.srcDevice = (CUdeviceptr)dist_pic->data[0];
    cpy_d.srcPitch = dist_pic->stride[0];
    cpy_d.dstDevice = (CUdeviceptr)s->dis_raw->data;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy_d, pic_stream));

    /* Reset partials. */
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA_RETURN(cu_f,
                          cuMemsetD8Async(s->num_partials[i]->data, 0,
                                          (size_t)s->wg_count[i] * sizeof(float), pic_stream));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemsetD8Async(s->den_partials[i]->data, 0,
                                          (size_t)s->wg_count[i] * sizeof(float), pic_stream));
    }

    /* Launch sequence: 4 compute + 3 decimate, one stream so launches
     * serialise naturally. */
    CUdeviceptr ref_raw_d = (CUdeviceptr)s->ref_raw->data;
    CUdeviceptr dis_raw_d = (CUdeviceptr)s->dis_raw->data;
    CUdeviceptr ref_buf0 = (CUdeviceptr)s->ref_buf[0]->data;
    CUdeviceptr dis_buf0 = (CUdeviceptr)s->dis_buf[0]->data;
    CUdeviceptr ref_buf1 = (CUdeviceptr)s->ref_buf[1]->data;
    CUdeviceptr dis_buf1 = (CUdeviceptr)s->dis_buf[1]->data;
    CUdeviceptr null_dptr = 0;

    /* Helper for compute: pass scale, raw or float input via the
     * appropriate set of args; the kernel selects via `is_raw =
     * (scale == 0)`. */
    {
        int scale = 0;
        CUdeviceptr num_d = (CUdeviceptr)s->num_partials[0]->data;
        CUdeviceptr den_d = (CUdeviceptr)s->den_partials[0]->data;
        ptrdiff_t f_stride = (ptrdiff_t)s->scale_w[0];
        unsigned w = s->scale_w[0];
        unsigned h = s->scale_h[0];
        unsigned grid_x = (w + FVIF_BX - 1u) / FVIF_BX;
        unsigned grid_y = (h + FVIF_BY - 1u) / FVIF_BY;
        void *args[] = {&scale,         &ref_raw_d, &dis_raw_d,        (void *)&raw_stride,
                        &null_dptr,     &null_dptr, (void *)&f_stride, &num_d,
                        &den_d,         (void *)&w, (void *)&h,        (void *)&s->bpc,
                        (void *)&grid_x};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_compute, grid_x, grid_y, 1, FVIF_BX, FVIF_BY,
                                               1, 0, pic_stream, args, NULL));
    }

    for (int next_scale = 1; next_scale < 4; next_scale++) {
        /* Decimate prev → ref_buf[(next-1)%2], dis_buf[(next-1)%2]. */
        const int dst_idx = (next_scale - 1) % 2;
        const bool prev_is_raw = (next_scale == 1);
        CUdeviceptr ref_in = prev_is_raw ? ref_raw_d : (dst_idx == 0 ? ref_buf1 : ref_buf0);
        CUdeviceptr dis_in = prev_is_raw ? dis_raw_d : (dst_idx == 0 ? dis_buf1 : dis_buf0);
        const ptrdiff_t in_f_stride = prev_is_raw ? 0 : (ptrdiff_t)s->scale_w[next_scale - 1];
        CUdeviceptr ref_out = (dst_idx == 0) ? ref_buf0 : ref_buf1;
        CUdeviceptr dis_out = (dst_idx == 0) ? dis_buf0 : dis_buf1;
        const ptrdiff_t out_f_stride = (ptrdiff_t)s->scale_w[next_scale];
        unsigned out_w = s->scale_w[next_scale];
        unsigned out_h = s->scale_h[next_scale];
        unsigned in_w = s->scale_w[next_scale - 1];
        unsigned in_h = s->scale_h[next_scale - 1];
        unsigned dec_grid_x = (out_w + FVIF_BX - 1u) / FVIF_BX;
        unsigned dec_grid_y = (out_h + FVIF_BY - 1u) / FVIF_BY;
        int scale_arg = next_scale;
        void *args[] = {
            &scale_arg,
            &ref_in,
            &dis_in,
            (void *)&raw_stride,
            &ref_in,
            &dis_in,
            (void *)&in_f_stride,
            &ref_out,
            &dis_out,
            (void *)&out_f_stride,
            (void *)&out_w,
            (void *)&out_h,
            (void *)&in_w,
            (void *)&in_h,
            (void *)&s->bpc,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_decimate, dec_grid_x, dec_grid_y, 1, FVIF_BX,
                                               FVIF_BY, 1, 0, pic_stream, args, NULL));

        /* Compute at this scale on the just-written buffer. */
        CUdeviceptr num_d = (CUdeviceptr)s->num_partials[next_scale]->data;
        CUdeviceptr den_d = (CUdeviceptr)s->den_partials[next_scale]->data;
        const ptrdiff_t comp_f_stride = (ptrdiff_t)s->scale_w[next_scale];
        unsigned w = s->scale_w[next_scale];
        unsigned h = s->scale_h[next_scale];
        unsigned grid_x = (w + FVIF_BX - 1u) / FVIF_BX;
        unsigned grid_y = (h + FVIF_BY - 1u) / FVIF_BY;
        int scale_arg2 = next_scale;
        void *cargs[] = {&scale_arg2,
                         &ref_raw_d,
                         &dis_raw_d,
                         (void *)&raw_stride,
                         &ref_out,
                         &dis_out,
                         (void *)&comp_f_stride,
                         &num_d,
                         &den_d,
                         (void *)&w,
                         (void *)&h,
                         (void *)&s->bpc,
                         (void *)&grid_x};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_compute, grid_x, grid_y, 1, FVIF_BX, FVIF_BY,
                                               1, 0, pic_stream, cargs, NULL));
    }

    /* Sync over to our event-driven stream + D2H copy partials. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, pic_stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->num_host[i], (CUdeviceptr)s->num_partials[i]->data,
                                            (size_t)s->wg_count[i] * sizeof(float), s->lc.str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->den_host[i], (CUdeviceptr)s->den_partials[i]->data,
                                            (size_t)s->wg_count[i] * sizeof(float), s->lc.str));
    }
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    FloatVifStateCuda *s = fex->priv;
    /* Drain via the template helper so engine-scope fence batching
     * (T-GPU-OPT-1, ADR-0242) can short-circuit the per-stream
     * cuStreamSynchronize when the engine has already waited on
     * lc.finished as part of a batched drain. */
    int sync_err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (sync_err) {
        return sync_err;
    }

    double scores[8];
    for (int i = 0; i < 4; i++) {
        double n = 0.0, d = 0.0;
        for (unsigned j = 0; j < s->wg_count[i]; j++) {
            n += (double)s->num_host[i][j];
            d += (double)s->den_host[i][j];
        }
        scores[2 * i + 0] = n;
        scores[2 * i + 1] = d;
    }

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale0_score",
                                                   scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale1_score",
                                                   scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale2_score",
                                                   scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale3_score",
                                                   scores[6] / scores[7], index);

    if (s->debug && !err) {
        double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        double score = score_den == 0.0 ? 1.0 : score_num / score_den;
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif", score, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_den", score_den, index);
        const char *names[8] = {"vif_num_scale0", "vif_den_scale0", "vif_num_scale1",
                                "vif_den_scale1", "vif_num_scale2", "vif_den_scale2",
                                "vif_num_scale3", "vif_den_scale3"};
        for (int i = 0; i < 8; i++) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                           names[i], scores[i], index);
        }
    }

    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    FloatVifStateCuda *s = fex->priv;
    int ret = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    if (s->ref_raw) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->ref_raw);
        free(s->ref_raw);
    }
    if (s->dis_raw) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->dis_raw);
        free(s->dis_raw);
    }
    for (int i = 0; i < 2; i++) {
        if (s->ref_buf[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->ref_buf[i]);
            free(s->ref_buf[i]);
        }
        if (s->dis_buf[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->dis_buf[i]);
            free(s->dis_buf[i]);
        }
    }
    for (int i = 0; i < 4; i++) {
        if (s->num_partials[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->num_partials[i]);
            free(s->num_partials[i]);
        }
        if (s->den_partials[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->den_partials[i]);
            free(s->den_partials[i]);
        }
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"VMAF_feature_vif_scale0_score",
                                          "VMAF_feature_vif_scale1_score",
                                          "VMAF_feature_vif_scale2_score",
                                          "VMAF_feature_vif_scale3_score",
                                          "vif",
                                          "vif_num",
                                          "vif_den",
                                          "vif_num_scale0",
                                          "vif_den_scale0",
                                          "vif_num_scale1",
                                          "vif_den_scale1",
                                          "vif_num_scale2",
                                          "vif_den_scale2",
                                          "vif_num_scale3",
                                          "vif_den_scale3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_float_vif_cuda = {
    .name = "float_vif_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(FloatVifStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
