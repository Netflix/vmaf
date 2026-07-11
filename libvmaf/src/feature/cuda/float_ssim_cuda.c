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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "cuda/float_ssim_cuda.h"
#include "opt.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

#define GAUSSIAN_LEN 11
#define REDUCE_BLOCK 256

typedef struct SsimStateCuda {
    CUevent event, finished;
    CUfunction f_norm8, f_norm16, f_decimate, f_products;
    CUfunction f_conv_h, f_conv_v, f_map_reduce;
    CUstream str, host_stream;
    VmafCudaBuffer *ref_f, *cmp_f;
    VmafCudaBuffer *refd, *cmpd;
    VmafCudaBuffer *ref2, *cmp2, *both;
    VmafCudaBuffer *cache;
    VmafCudaBuffer *mu1, *mu2, *cref2, *ccmp2, *cboth;
    VmafCudaBuffer *partials;
    double *partials_host;
    void *write_score_parameters;
    unsigned w, h, sw, sh, cw, ch;
    unsigned n_blocks;
    unsigned bpc;
    int factor;
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    double max_db;
    int scale;
} SsimStateCuda;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(SsimStateCuda, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(SsimStateCuda, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(SsimStateCuda, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling, 2-10=explicit)",
        .offset = offsetof(SsimStateCuda, scale),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    { 0 }
};

typedef struct write_score_parameters_ssim {
    VmafFeatureCollector *feature_collector;
    SsimStateCuda *s;
    unsigned index;
} write_score_parameters_ssim;

// iqa/math_utils.c _round: round half away from zero
static int iqa_round(float a)
{
    int sign_a = a > 0.0f ? 1 : -1;
    return a - (int)a >= 0.5 ? (int)a + sign_a : (int)a;
}

static int alloc_buf(VmafFeatureExtractor *fex, VmafCudaBuffer **buf,
                     size_t size)
{
    return vmaf_cuda_buffer_alloc(fex->cu_state, buf, size);
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) pix_fmt;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, ssim_ptx));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_norm8, module, "ssim_normalize_8bpc"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_norm16, module, "ssim_normalize_16bpc"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_decimate, module, "ssim_decimate"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_products, module, "ssim_products"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_conv_h, module, "ssim_conv_h"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_conv_v, module, "ssim_conv_v"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->f_map_reduce, module, "ssim_map_reduce"));

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    s->w = w;
    s->h = h;
    s->bpc = bpc;

    // compute_ssim: scale = max(1, round(min(w,h) / 256.0)), or the override
    const unsigned min_wh = w < h ? w : h;
    s->factor = s->scale > 0 ?
        s->scale : (iqa_round((float)min_wh / 256.0f) < 1 ?
                    1 : iqa_round((float)min_wh / 256.0f));

    if (s->factor > 1) {
        // _iqa_decimate: sw = w/factor + (w&1)
        s->sw = w / s->factor + (w & 1);
        s->sh = h / s->factor + (h & 1);
    } else {
        s->sw = w;
        s->sh = h;
    }
    if (s->sw < GAUSSIAN_LEN || s->sh < GAUSSIAN_LEN)
        return -EINVAL;
    s->cw = s->sw - GAUSSIAN_LEN + 1;
    s->ch = s->sh - GAUSSIAN_LEN + 1;
    s->n_blocks = DIV_ROUND_UP(s->cw * s->ch, REDUCE_BLOCK);

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    int ret = 0;

    s->write_score_parameters = malloc(sizeof(write_score_parameters_ssim));
    if (!s->write_score_parameters) return -ENOMEM;
    ((write_score_parameters_ssim*)s->write_score_parameters)->s = s;

    const size_t full = sizeof(float) * w * h;
    const size_t dec = sizeof(float) * s->sw * s->sh;
    const size_t conv = sizeof(float) * s->cw * s->ch;

    ret |= alloc_buf(fex, &s->ref_f, full);
    ret |= alloc_buf(fex, &s->cmp_f, full);
    if (s->factor > 1) {
        ret |= alloc_buf(fex, &s->refd, dec);
        ret |= alloc_buf(fex, &s->cmpd, dec);
    }
    ret |= alloc_buf(fex, &s->ref2, dec);
    ret |= alloc_buf(fex, &s->cmp2, dec);
    ret |= alloc_buf(fex, &s->both, dec);
    ret |= alloc_buf(fex, &s->cache, dec);
    ret |= alloc_buf(fex, &s->mu1, conv);
    ret |= alloc_buf(fex, &s->mu2, conv);
    ret |= alloc_buf(fex, &s->cref2, conv);
    ret |= alloc_buf(fex, &s->ccmp2, conv);
    ret |= alloc_buf(fex, &s->cboth, conv);
    ret |= alloc_buf(fex, &s->partials, sizeof(double) * 4 * s->n_blocks);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void**)&s->partials_host,
                                       sizeof(double) * 4 * s->n_blocks);
    if (ret) return -ENOMEM;

    return 0;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static int write_scores(write_score_parameters_ssim *params)
{
    SsimStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    // sequential sum over block partials keeps the result deterministic
    double ssim_sum = 0., l_sum = 0., c_sum = 0., s_sum = 0.;
    for (unsigned b = 0; b < s->n_blocks; b++) {
        ssim_sum += s->partials_host[b * 4 + 0];
        l_sum += s->partials_host[b * 4 + 1];
        c_sum += s->partials_host[b * 4 + 2];
        s_sum += s->partials_host[b * 4 + 3];
    }

    // _iqa_ssim returns float means; compute_ssim widens them to double
    const double n = (double)(s->cw * s->ch);
    double score = (double)(float)(ssim_sum / n);
    const double l_score = (double)(float)(l_sum / n);
    const double c_score = (double)(float)(c_sum / n);
    const double s_score = (double)(float)(s_sum / n);

    if (s->enable_db)
        score = convert_to_db(score, s->max_db);

    int err = vmaf_feature_collector_append(feature_collector, "float_ssim",
                                            score, params->index);
    if (s->enable_lcs) {
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_l",
                                             l_score, params->index);
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_c",
                                             c_score, params->index);
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_s",
                                             s_score, params->index);
    }

    return err;
}

static void launch_conv(SsimStateCuda *s, CudaFunctions *cu_f, CUstream stream,
                        VmafCudaBuffer *in, VmafCudaBuffer *out)
{
    int w = s->sw, h = s->sh, dst_w = s->cw, dst_h = s->ch;
    {
        void *args[] = { (void*)in, (void*)s->cache, &w, &h, &dst_w };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_conv_h,
                    DIV_ROUND_UP(dst_w, 16), DIV_ROUND_UP(h, 16), 1,
                    16, 16, 1, 0, stream, args, NULL));
    }
    {
        void *args[] = { (void*)s->cache, (void*)out, &w, &dst_w, &dst_h };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_conv_v,
                    DIV_ROUND_UP(dst_w, 16), DIV_ROUND_UP(dst_h, 16), 1,
                    16, 16, 1, 0, stream, args, NULL));
    }
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) ref_pic_90;
    (void) dist_pic_90;

    // this is done to ensure that the CPU does not overwrite the buffer
    // params for write_scores
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));

    const CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA(cu_f, cuStreamWaitEvent(pic_stream,
                vmaf_cuda_picture_get_ready_event(dist_pic),
                CU_EVENT_WAIT_DEFAULT));

    unsigned w = s->w, h = s->h;
    float scaler = 4.0f;
    if (s->bpc == 12) scaler = 16.0f;
    if (s->bpc == 16) scaler = 256.0f;

    if (s->bpc == 8) {
        void *a1[] = { (void*)ref_pic, (void*)s->ref_f, &w, &h };
        void *a2[] = { (void*)dist_pic, (void*)s->cmp_f, &w, &h };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_norm8, DIV_ROUND_UP(w, 16),
                    DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, pic_stream, a1, NULL));
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_norm8, DIV_ROUND_UP(w, 16),
                    DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, pic_stream, a2, NULL));
    } else {
        void *a1[] = { (void*)ref_pic, (void*)s->ref_f, &w, &h, &scaler };
        void *a2[] = { (void*)dist_pic, (void*)s->cmp_f, &w, &h, &scaler };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_norm16, DIV_ROUND_UP(w, 16),
                    DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, pic_stream, a1, NULL));
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_norm16, DIV_ROUND_UP(w, 16),
                    DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, pic_stream, a2, NULL));
    }

    VmafCudaBuffer *ref_in = s->ref_f;
    VmafCudaBuffer *cmp_in = s->cmp_f;
    if (s->factor > 1) {
        int iw = w, ih = h, sw = s->sw, sh = s->sh, factor = s->factor;
        void *a1[] = { (void*)s->ref_f, (void*)s->refd, &iw, &ih, &sw, &sh, &factor };
        void *a2[] = { (void*)s->cmp_f, (void*)s->cmpd, &iw, &ih, &sw, &sh, &factor };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_decimate, DIV_ROUND_UP(sw, 16),
                    DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, pic_stream, a1, NULL));
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_decimate, DIV_ROUND_UP(sw, 16),
                    DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, pic_stream, a2, NULL));
        ref_in = s->refd;
        cmp_in = s->cmpd;
    }

    {
        int n = s->sw * s->sh;
        void *args[] = { (void*)ref_in, (void*)cmp_in, (void*)s->ref2,
                         (void*)s->cmp2, (void*)s->both, &n };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_products,
                    DIV_ROUND_UP(n, REDUCE_BLOCK), 1, 1,
                    REDUCE_BLOCK, 1, 1, 0, pic_stream, args, NULL));
    }

    launch_conv(s, cu_f, pic_stream, ref_in, s->mu1);
    launch_conv(s, cu_f, pic_stream, cmp_in, s->mu2);
    launch_conv(s, cu_f, pic_stream, s->ref2, s->cref2);
    launch_conv(s, cu_f, pic_stream, s->cmp2, s->ccmp2);
    launch_conv(s, cu_f, pic_stream, s->both, s->cboth);

    {
        // _iqa_ssim: C1 = (K1*L)^2, C2 = (K2*L)^2, C3 = C2/2, L = 255
        float c1 = (0.01f * 255) * (0.01f * 255);
        float c2 = (0.03f * 255) * (0.03f * 255);
        float c3 = c2 / 2.0f;
        int n = s->cw * s->ch;
        void *args[] = { (void*)s->mu1, (void*)s->mu2, (void*)s->cref2,
                         (void*)s->ccmp2, (void*)s->cboth, (void*)s->partials,
                         &n, &c1, &c2, &c3 };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->f_map_reduce,
                    s->n_blocks, 1, 1, REDUCE_BLOCK, 1, 1, 0,
                    pic_stream, args, NULL));
    }

    CHECK_CUDA(cu_f, cuEventRecord(s->event, pic_stream));
    // This event ensures the input buffer is consumed
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));

    // Download block partials
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(s->partials_host, s->partials->data,
                sizeof(double) * 4 * s->n_blocks, s->str));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->host_stream, s->finished,
                CU_EVENT_WAIT_DEFAULT));

    write_score_parameters_ssim *params = s->write_score_parameters;
    params->feature_collector = feature_collector;
    params->index = index;
    CHECK_CUDA(cu_f, cuLaunchHostFunc(s->host_stream, (CUhostFn*)write_scores,
                s->write_score_parameters));

    return 0;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    // drain the pending write_scores host callback so the final frame's
    // score is in the collector before anything reads it
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    return 1;
}

static int free_buf(VmafFeatureExtractor *fex, VmafCudaBuffer *buf)
{
    int ret = 0;
    if (buf) {
        ret = vmaf_cuda_buffer_free(fex->cu_state, buf);
        free(buf);
    }
    return ret;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cu_f, cuEventDestroy(s->event));
    CHECK_CUDA(cu_f, cuEventDestroy(s->finished));
    CHECK_CUDA(cu_f, cuStreamDestroy(s->str));
    CHECK_CUDA(cu_f, cuStreamDestroy(s->host_stream));

    int ret = 0;
    ret |= free_buf(fex, s->ref_f);
    ret |= free_buf(fex, s->cmp_f);
    ret |= free_buf(fex, s->refd);
    ret |= free_buf(fex, s->cmpd);
    ret |= free_buf(fex, s->ref2);
    ret |= free_buf(fex, s->cmp2);
    ret |= free_buf(fex, s->both);
    ret |= free_buf(fex, s->cache);
    ret |= free_buf(fex, s->mu1);
    ret |= free_buf(fex, s->mu2);
    ret |= free_buf(fex, s->cref2);
    ret |= free_buf(fex, s->ccmp2);
    ret |= free_buf(fex, s->cboth);
    ret |= free_buf(fex, s->partials);
    if (s->partials_host)
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->partials_host);
    if (s->write_score_parameters)
        free(s->write_score_parameters);

    return ret;
}

static const char *provided_features[] = {
    "float_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ssim_cuda = {
    .name = "ssim_cuda",
    .options = options,
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(SsimStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
