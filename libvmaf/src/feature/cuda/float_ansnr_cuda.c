/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ansnr feature kernel on the CUDA backend (T7-23 / batch 3
 *  part 2b — ADR-0192 / ADR-0194). CUDA twin of float_ansnr_vulkan.
 *
 *  Single-dispatch kernel produces per-block (sig, noise) float
 *  partials; host accumulates in `double` and applies the CPU
 *  formulas:
 *    float_ansnr  = 10 * log10(sig / noise)   (or psnr_max if noise == 0)
 *    float_anpsnr = MIN(10*log10(peak² · w · h / max(noise, 1e-10)), psnr_max)
 *
 *  Pattern reference: integer_ciede_cuda.c (per-block float partials
 *  + double host reduction). Uses submit/collect async stream pattern.
 */

#include <errno.h>
#include <math.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cuda/float_ansnr_cuda.h"
#include "cuda/kernel_template.h"
#include "cuda_helper.cuh"
#include "picture.h"
#include "picture_cuda.h"

typedef struct FloatAnsnrStateCuda {
    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0246). */
    VmafCudaKernelLifecycle lc;
    /* Per-block (sig, noise) interleaved float partials: device +
     * pinned host. Owned by the template's readback bundle. */
    VmafCudaKernelReadback rb;

    CUfunction funcbpc8;
    CUfunction funcbpc16;

    /* Per-frame upload of ref + dis raw pixels. */
    VmafCudaBuffer *ref_in;
    VmafCudaBuffer *dis_in;
    unsigned wg_count;

    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    double peak;
    double psnr_max;

    VmafDictionary *feature_name_dict;
} FloatAnsnrStateCuda;

#define ANSNR_BX 16
#define ANSNR_BY 16

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatAnsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;

    if (bpc == 8u) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10u) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12u) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16u) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else {
        return -EINVAL;
    }

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, float_ansnr_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "float_ansnr_kernel_8bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "float_ansnr_kernel_16bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    const size_t bytes_per_pixel = (bpc <= 8u) ? 1u : 2u;
    const size_t plane_bytes = (size_t)w * h * bytes_per_pixel;
    const unsigned gx = (w + ANSNR_BX - 1u) / ANSNR_BX;
    const unsigned gy = (h + ANSNR_BY - 1u) / ANSNR_BY;
    s->wg_count = gx * gy;
    const size_t partials_bytes = (size_t)s->wg_count * 2u * sizeof(float);

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->ref_in, plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dis_in, plane_bytes);
    if (ret)
        goto free_buffers;
    ret = vmaf_cuda_kernel_readback_alloc(&s->rb, fex->cu_state, partials_bytes);
    if (ret)
        goto free_buffers;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        ret = -ENOMEM;
        goto free_buffers;
    }

    return 0;

free_buffers:
    if (s->ref_in) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->ref_in);
        free(s->ref_in);
    }
    if (s->dis_in) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->dis_in);
        free(s->dis_in);
    }
    (void)vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
    (void)vmaf_dictionary_free(&s->feature_name_dict);
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return ret;

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
    FloatAnsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * (s->bpc <= 8u ? 1u : 2u));

    /* Pack ref and dis Y planes from VmafPicture device pointers
     * (which may have arbitrary stride) into our tightly-packed
     * staging buffers. */
    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA_RETURN(cu_f,
                      cuStreamWaitEvent(pic_stream, vmaf_cuda_picture_get_ready_event(dist_pic),
                                        CU_EVENT_WAIT_DEFAULT));

    CUDA_MEMCPY2D cpy_ref = {0};
    cpy_ref.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy_ref.srcDevice = (CUdeviceptr)ref_pic->data[0];
    cpy_ref.srcPitch = ref_pic->stride[0];
    cpy_ref.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy_ref.dstDevice = (CUdeviceptr)s->ref_in->data;
    cpy_ref.dstPitch = plane_pitch;
    cpy_ref.WidthInBytes = plane_pitch;
    cpy_ref.Height = s->frame_h;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy_ref, pic_stream));

    CUDA_MEMCPY2D cpy_dis = {0};
    cpy_dis.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy_dis.srcDevice = (CUdeviceptr)dist_pic->data[0];
    cpy_dis.srcPitch = dist_pic->stride[0];
    cpy_dis.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy_dis.dstDevice = (CUdeviceptr)s->dis_in->data;
    cpy_dis.dstPitch = plane_pitch;
    cpy_dis.WidthInBytes = plane_pitch;
    cpy_dis.Height = s->frame_h;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy_dis, pic_stream));

    /* Reset partials buffer. */
    CHECK_CUDA_RETURN(cu_f, cuMemsetD8Async(s->rb.device->data, 0,
                                            (size_t)s->wg_count * 2u * sizeof(float), pic_stream));

    const unsigned grid_x = (s->frame_w + ANSNR_BX - 1u) / ANSNR_BX;
    const unsigned grid_y = (s->frame_h + ANSNR_BY - 1u) / ANSNR_BY;

    if (s->bpc == 8u) {
        void *args[] = {
            &s->ref_in->data,     &s->dis_in->data,    (void *)&plane_pitch, (void *)&plane_pitch,
            (void *)s->rb.device, (void *)&s->frame_w, (void *)&s->frame_h,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->funcbpc8, grid_x, grid_y, 1, ANSNR_BX, ANSNR_BY,
                                               1, 0, pic_stream, args, NULL));
    } else {
        void *args[] = {
            &s->ref_in->data,     &s->dis_in->data,    (void *)&plane_pitch, (void *)&plane_pitch,
            (void *)s->rb.device, (void *)&s->frame_w, (void *)&s->frame_h,  (void *)&s->bpc,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->funcbpc16, grid_x, grid_y, 1, ANSNR_BX, ANSNR_BY,
                                               1, 0, pic_stream, args, NULL));
    }

    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, pic_stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));

    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(((float *)s->rb.host_pinned),
                                              (CUdeviceptr)s->rb.device->data,
                                              (size_t)s->wg_count * 2u * sizeof(float), s->lc.str));
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    FloatAnsnrStateCuda *s = fex->priv;

    int sync_err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (sync_err)
        return sync_err;

    const float *partials_host = s->rb.host_pinned;
    double sig = 0.0;
    double noise = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++) {
        sig += (double)partials_host[2 * i + 0];
        noise += (double)partials_host[2 * i + 1];
    }

    const double score = (noise == 0.0) ? s->psnr_max : 10.0 * log10(sig / noise);
    const double eps = 1e-10;
    const double n_pix = (double)s->frame_w * (double)s->frame_h;
    const double max_noise = noise > eps ? noise : eps;
    double score_psnr = 10.0 * log10(s->peak * s->peak * n_pix / max_noise);
    if (score_psnr > s->psnr_max)
        score_psnr = s->psnr_max;

    int err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_ansnr", score, index);
    if (err)
        return err;
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_anpsnr", score_psnr, index);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    FloatAnsnrStateCuda *s = fex->priv;
    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);

    if (s->ref_in) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->ref_in);
        free(s->ref_in);
        if (rc == 0)
            rc = e;
    }
    if (s->dis_in) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->dis_in);
        free(s->dis_in);
        if (rc == 0)
            rc = e;
    }
    const int rb_rc = vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
    if (rc == 0)
        rc = rb_rc;
    const int dict_rc = vmaf_dictionary_free(&s->feature_name_dict);
    if (rc == 0)
        rc = dict_rc;
    return rc;
}

static const char *provided_features[] = {"float_ansnr", "float_anpsnr", NULL};

VmafFeatureExtractor vmaf_fex_float_ansnr_cuda = {
    .name = "float_ansnr_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(FloatAnsnrStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
