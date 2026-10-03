/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  Copyright 2026 Bardie Høgh Joensen
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
#include "cuda/feature_common.h"

#define GAUSSIAN_LEN 11
#define REDUCE_BLOCK 256

typedef struct SsimStateCuda {
    CudaFeatureResources cuda;
    CUfunction f_norm8, f_norm16, f_dec8, f_dec16, f_products;
    CUfunction f_conv_h, f_conv_v, f_map_reduce;
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
    const double *partials;
    unsigned index;
    int err; /* Read after callback completion. */
} write_score_parameters_ssim;

// Match IQA's rounding for positive dimensions.
static int iqa_round(float a)
{
    int sign_a = a > 0.0f ? 1 : -1;
    return a - (int)a >= 0.5 ? (int)a + sign_a : (int)a;
}

static int close_fex_cuda(VmafFeatureExtractor *fex);

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) pix_fmt;

    if (!w || !h || (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16))
        return -EINVAL;
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

    int err;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    err = cuda_fex_resources_init(cu_f, &s->cuda, ssim_ptx);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_norm8, s->cuda.module, "ssim_normalize_8bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_norm16, s->cuda.module, "ssim_normalize_16bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_dec8, s->cuda.module, "ssim_decimate_8bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_dec16, s->cuda.module, "ssim_decimate_16bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_products, s->cuda.module, "ssim_products"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_conv_h, s->cuda.module, "ssim_conv_h"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_conv_v, s->cuda.module, "ssim_conv_v"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->f_map_reduce, s->cuda.module, "ssim_map_reduce"));
    s->write_score_parameters = calloc(2, sizeof(write_score_parameters_ssim));
    if (!s->write_score_parameters) { err = -ENOMEM; goto fail; }
    for (unsigned i = 0; i < 2; i++)
        ((write_score_parameters_ssim *)s->write_score_parameters)[i].s = s;

    const size_t full = sizeof(float) * w * h;
    const size_t dec = sizeof(float) * s->sw * s->sh;
    const size_t conv = sizeof(float) * s->cw * s->ch;
    if (s->factor > 1) {
        err = cuda_fex_buffer_alloc(cu_f, &s->refd, dec);
        if (err) goto fail;
        err = cuda_fex_buffer_alloc(cu_f, &s->cmpd, dec);
        if (err) goto fail;
    } else {
        err = cuda_fex_buffer_alloc(cu_f, &s->ref_f, full);
        if (err) goto fail;
        err = cuda_fex_buffer_alloc(cu_f, &s->cmp_f, full);
        if (err) goto fail;
    }
    err = cuda_fex_buffer_alloc(cu_f, &s->ref2, dec);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->cmp2, dec);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->both, dec);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->cache, dec);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->mu1, conv);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->mu2, conv);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->cref2, conv);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->ccmp2, conv);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->cboth, conv);
    if (err) goto fail;
    err = cuda_fex_buffer_alloc(cu_f, &s->partials, sizeof(double) * 4 * s->n_blocks);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuMemHostAlloc((void **)&s->partials_host,
            sizeof(double) * 4 * s->n_blocks * 2, 0x01));
    CUDA_FEX_INIT(cu_f, cuCtxPopCurrent(NULL));
    return 0;
fail:
    close_fex_cuda(fex);
    cu_f->cuCtxPopCurrent(NULL);
    return err;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static void CUDAAPI write_scores(void *opaque)
{
    write_score_parameters_ssim *params = opaque;
    SsimStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    // sequential sum over block partials keeps the result deterministic
    double ssim_sum = 0., l_sum = 0., c_sum = 0., s_sum = 0.;
    for (unsigned b = 0; b < s->n_blocks; b++) {
        ssim_sum += params->partials[b * 4 + 0];
        l_sum += params->partials[b * 4 + 1];
        c_sum += params->partials[b * 4 + 2];
        s_sum += params->partials[b * 4 + 3];
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

    params->err = err;
}

static int launch_conv(SsimStateCuda *s, CudaFunctions *cu_f, CUstream stream,
                        VmafCudaBuffer *in, VmafCudaBuffer *out)
{
    int w = s->sw, h = s->sh, dst_w = s->cw, dst_h = s->ch;
    {
        void *args[] = { (void*)in, (void*)s->cache, &w, &h, &dst_w };
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_conv_h,
                    DIV_ROUND_UP(dst_w, 16), DIV_ROUND_UP(h, 16), 1,
                    16, 16, 1, 0, stream, args, NULL));
    }
    {
        void *args[] = { (void*)s->cache, (void*)out, &w, &dst_w, &dst_h };
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_conv_v,
                    DIV_ROUND_UP(dst_w, 16), DIV_ROUND_UP(dst_h, 16), 1,
                    16, 16, 1, 0, stream, args, NULL));
    }
    return 0;
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

    // Wait for the previous callback using this slot, including skipped indices.
    const unsigned slot = index & 1;
    CUDA_FEX_CHECK(cu_f, cuEventSynchronize(s->cuda.slot_done[slot]));
    write_score_parameters_ssim *params =
        &((write_score_parameters_ssim *)s->write_score_parameters)[slot];
    if (params->err) return params->err;

    /* Wait for both producers before reading either picture. */
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(ref_pic),
                CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(dist_pic),
                CU_EVENT_WAIT_DEFAULT));

    unsigned w = s->w, h = s->h;
    float scaler = 4.0f;
    if (s->bpc == 12) scaler = 16.0f;
    if (s->bpc == 16) scaler = 256.0f;

    VmafCudaBuffer *ref_in, *cmp_in;
    if (s->factor > 1) {
        // fused normalize+decimate reads the pictures directly
        int iw = w, ih = h, sw = s->sw, sh = s->sh, factor = s->factor;
        if (s->bpc == 8) {
            void *a1[] = { (void*)ref_pic, (void*)s->refd, &iw, &ih, &sw, &sh, &factor };
            void *a2[] = { (void*)dist_pic, (void*)s->cmpd, &iw, &ih, &sw, &sh, &factor };
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_dec8, DIV_ROUND_UP(sw, 16),
                        DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, s->cuda.str, a1, NULL));
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_dec8, DIV_ROUND_UP(sw, 16),
                        DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, s->cuda.str, a2, NULL));
        } else {
            void *a1[] = { (void*)ref_pic, (void*)s->refd, &iw, &ih, &sw, &sh, &factor, &scaler };
            void *a2[] = { (void*)dist_pic, (void*)s->cmpd, &iw, &ih, &sw, &sh, &factor, &scaler };
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_dec16, DIV_ROUND_UP(sw, 16),
                        DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, s->cuda.str, a1, NULL));
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_dec16, DIV_ROUND_UP(sw, 16),
                        DIV_ROUND_UP(sh, 16), 1, 16, 16, 1, 0, s->cuda.str, a2, NULL));
        }
        ref_in = s->refd;
        cmp_in = s->cmpd;
    } else {
        if (s->bpc == 8) {
            void *a1[] = { (void*)ref_pic, (void*)s->ref_f, &w, &h };
            void *a2[] = { (void*)dist_pic, (void*)s->cmp_f, &w, &h };
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_norm8, DIV_ROUND_UP(w, 16),
                        DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, s->cuda.str, a1, NULL));
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_norm8, DIV_ROUND_UP(w, 16),
                        DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, s->cuda.str, a2, NULL));
        } else {
            void *a1[] = { (void*)ref_pic, (void*)s->ref_f, &w, &h, &scaler };
            void *a2[] = { (void*)dist_pic, (void*)s->cmp_f, &w, &h, &scaler };
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_norm16, DIV_ROUND_UP(w, 16),
                        DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, s->cuda.str, a1, NULL));
            CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_norm16, DIV_ROUND_UP(w, 16),
                        DIV_ROUND_UP(h, 16), 1, 16, 16, 1, 0, s->cuda.str, a2, NULL));
        }
        ref_in = s->ref_f;
        cmp_in = s->cmp_f;
    }

    /* Later kernels use private buffers, so the pictures can now be reused. */
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.consumed, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(dist_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));

    {
        int n = s->sw * s->sh;
        void *args[] = { (void*)ref_in, (void*)cmp_in, (void*)s->ref2,
                         (void*)s->cmp2, (void*)s->both, &n };
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_products,
                    DIV_ROUND_UP(n, REDUCE_BLOCK), 1, 1,
                    REDUCE_BLOCK, 1, 1, 0, s->cuda.str, args, NULL));
    }

    int err;
    err = launch_conv(s, cu_f, s->cuda.str, ref_in, s->mu1);
    if (err) return err;
    err = launch_conv(s, cu_f, s->cuda.str, cmp_in, s->mu2);
    if (err) return err;
    err = launch_conv(s, cu_f, s->cuda.str, s->ref2, s->cref2);
    if (err) return err;
    err = launch_conv(s, cu_f, s->cuda.str, s->cmp2, s->ccmp2);
    if (err) return err;
    err = launch_conv(s, cu_f, s->cuda.str, s->both, s->cboth);
    if (err) return err;

    {
        // _iqa_ssim: C1 = (K1*L)^2, C2 = (K2*L)^2, C3 = C2/2, L = 255
        float c1 = (0.01f * 255) * (0.01f * 255);
        float c2 = (0.03f * 255) * (0.03f * 255);
        float c3 = c2 / 2.0f;
        int n = s->cw * s->ch;
        void *args[] = { (void*)s->mu1, (void*)s->mu2, (void*)s->cref2,
                         (void*)s->ccmp2, (void*)s->cboth, (void*)s->partials,
                         &n, &c1, &c2, &c3 };
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(s->f_map_reduce,
                    s->n_blocks, 1, 1, REDUCE_BLOCK, 1, 1, 0,
                    s->cuda.str, args, NULL));
    }

    // Download block partials into this slot's readback segment
    double *partials_host = s->partials_host + slot * 4 * s->n_blocks;
    CUDA_FEX_CHECK(cu_f, cuMemcpyDtoHAsync(partials_host, s->partials->data,
                sizeof(double) * 4 * s->n_blocks, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.finished, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.host_stream, s->cuda.finished,
                CU_EVENT_WAIT_DEFAULT));

    params->feature_collector = feature_collector;
    params->partials = partials_host;
    params->index = index;
    CUDA_FEX_CHECK(cu_f, cuLaunchHostFunc(s->cuda.host_stream, write_scores,
                params));
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.slot_done[slot], s->cuda.host_stream));

    return 0;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Publish the final callback result before returning. */
    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.host_stream));
    write_score_parameters_ssim *params = s->write_score_parameters;
    for (unsigned i = 0; i < 2; i++)
        if (params[i].err) return params[i].err;
    return 1;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    int err = cuda_fex_resources_close(cu_f, &s->cuda);
    err |= cuda_fex_buffer_free(cu_f, &s->ref_f);
    err |= cuda_fex_buffer_free(cu_f, &s->cmp_f);
    err |= cuda_fex_buffer_free(cu_f, &s->refd);
    err |= cuda_fex_buffer_free(cu_f, &s->cmpd);
    err |= cuda_fex_buffer_free(cu_f, &s->ref2);
    err |= cuda_fex_buffer_free(cu_f, &s->cmp2);
    err |= cuda_fex_buffer_free(cu_f, &s->both);
    err |= cuda_fex_buffer_free(cu_f, &s->cache);
    err |= cuda_fex_buffer_free(cu_f, &s->mu1);
    err |= cuda_fex_buffer_free(cu_f, &s->mu2);
    err |= cuda_fex_buffer_free(cu_f, &s->cref2);
    err |= cuda_fex_buffer_free(cu_f, &s->ccmp2);
    err |= cuda_fex_buffer_free(cu_f, &s->cboth);
    err |= cuda_fex_buffer_free(cu_f, &s->partials);
    if (s->partials_host)
        err |= cuda_fex_error(cu_f, cu_f->cuMemFreeHost(s->partials_host));
    s->partials_host = NULL;
    free(s->write_score_parameters);
    s->write_score_parameters = NULL;
    err |= cuda_fex_error(cu_f, cu_f->cuCtxPopCurrent(NULL));
    return err;
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
