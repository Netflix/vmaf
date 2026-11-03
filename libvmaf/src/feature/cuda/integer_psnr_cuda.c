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
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "cuda/integer_psnr_cuda.h"
#include "opt.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"
#include "cuda/feature_common.h"

typedef struct PsnrStateCuda {
    CudaFeatureResources cuda;
    CUfunction funcbpc8, funcbpc16;
    VmafCudaBuffer *sse;
    uint64_t *sse_host;
    void *write_score_parameters;
    unsigned bpc;
    bool enable_chroma;
    bool enable_mse;
    bool enable_apsnr;
    bool reduced_hbd_peak;
    uint32_t peak;
    double psnr_max[3];
    double min_sse;
    struct {
        uint64_t sse[3];
        uint64_t n_pixels[3];
    } apsnr;
} PsnrStateCuda;

static const VmafOption options[] = {
    {
        .name = "enable_chroma",
        .help = "enable calculation for chroma channels",
        .offset = offsetof(PsnrStateCuda, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "enable_mse",
        .help = "enable MSE calculation",
        .offset = offsetof(PsnrStateCuda, enable_mse),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_apsnr",
        .help = "enable APSNR calculation",
        .offset = offsetof(PsnrStateCuda, enable_apsnr),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "reduced_hbd_peak",
        .help = "reduce hbd peak value to align with scaled 8-bit content",
        .offset = offsetof(PsnrStateCuda, reduced_hbd_peak),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "min_sse",
        .help = "constrain the minimum possible sse",
        .offset = offsetof(PsnrStateCuda, min_sse),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = DBL_MAX,
    },
    { 0 }
};

typedef struct write_score_parameters_psnr {
    VmafFeatureCollector *feature_collector;
    PsnrStateCuda *s;
    const uint64_t *sse;
    unsigned w[3], h[3];
    unsigned index;
    int err; /* Read only after slot_done or host_stream synchronization. */
} write_score_parameters_psnr;

static int close_fex_cuda(VmafFeatureExtractor *fex);

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    if (!w || !h || (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16))
        return -EINVAL;
    s->bpc = bpc;
    s->peak = s->reduced_hbd_peak ? 255U << (bpc - 8) : (1U << bpc) - 1;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;
    if (!s->enable_chroma)
        fex->flags &= ~VMAF_FEATURE_EXTRACTOR_CHROMA;
    for (unsigned i = 0; i < (s->enable_chroma ? 3U : 1U); i++) {
        if (s->min_sse != 0.0) {
            const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
            const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
            const double mse = s->min_sse /
                ((double)((i && ss_hor) ? w / 2 : w) * ((i && ss_ver) ? h / 2 : h));
            s->psnr_max[i] = ceil(10. * log10((double)s->peak * s->peak / mse));
        } else {
            s->psnr_max[i] = (6 * bpc) + 12;
        }
    }

    int err;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    err = cuda_fex_resources_init(cu_f, &s->cuda, psnr_ptx);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->funcbpc8, s->cuda.module, "psnr_kernel_8bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->funcbpc16, s->cuda.module, "psnr_kernel_16bpc"));
    s->write_score_parameters = calloc(2, sizeof(write_score_parameters_psnr));
    if (!s->write_score_parameters) { err = -ENOMEM; goto fail; }
    for (unsigned i = 0; i < 2; i++)
        ((write_score_parameters_psnr *)s->write_score_parameters)[i].s = s;
    err = cuda_fex_buffer_alloc(cu_f, &s->sse, sizeof(uint64_t) * 3);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuMemHostAlloc((void **)&s->sse_host,
                                     sizeof(uint64_t) * 3 * 2, 0x01));
    CUDA_FEX_INIT(cu_f, cuCtxPopCurrent(NULL));
    return 0;
fail:
    close_fex_cuda(fex);
    cu_f->cuCtxPopCurrent(NULL);
    return err;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static char *mse_name[3] = { "mse_y", "mse_cb", "mse_cr" };
static char *psnr_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };

static void CUDAAPI write_scores(void *opaque)
{
    write_score_parameters_psnr *params = opaque;
    PsnrStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    const double peak = (s->bpc == 8) ? 255. : (double) s->peak;
    const unsigned n = s->enable_chroma ? 3 : 1;

    int err = 0;
    for (unsigned p = 0; p < n; p++) {
        const uint64_t sse = params->sse[p];

        if (s->enable_apsnr) {
            s->apsnr.sse[p] += sse;
            s->apsnr.n_pixels[p] += (uint64_t)params->w[p] * params->h[p];
        }

        const double mse =
            ((double) sse) / ((double)params->w[p] * params->h[p]);
        const double psnr =
            MIN(10. * log10(peak * peak / MAX(mse, 1e-16)), s->psnr_max[p]);

        err |= vmaf_feature_collector_append(feature_collector, psnr_name[p],
                                             psnr, params->index);
        if (s->enable_mse) {
            err |= vmaf_feature_collector_append(feature_collector, mse_name[p],
                                                 mse, params->index);
        }
    }

    params->err = err;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) ref_pic_90;
    (void) dist_pic_90;

    // Wait for the previous callback using this slot, including skipped indices.
    const unsigned slot = index & 1;
    CUDA_FEX_CHECK(cu_f, cuEventSynchronize(s->cuda.slot_done[slot]));
    write_score_parameters_psnr *params =
        &((write_score_parameters_psnr *)s->write_score_parameters)[slot];
    if (params->err) return params->err;

    const unsigned n = s->enable_chroma ? 3 : 1;

    /* Wait for both producers before reading either picture. */
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(ref_pic),
                CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(dist_pic),
                CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuMemsetD8Async(s->sse->data, 0, sizeof(uint64_t) * 3,
                s->cuda.str));

    const CUfunction func = (ref_pic->bpc == 8) ? s->funcbpc8 : s->funcbpc16;
    const unsigned vec = (ref_pic->bpc == 8) ? 4 : 2;
    for (unsigned p = 0; p < n; p++) {
        unsigned plane = p;
        unsigned width = ref_pic->w[p];
        unsigned height = ref_pic->h[p];
        // 1-D grid-stride kernel over vectorized loads; cap the grid so
        // tail-of-wave blocks stay busy
        unsigned n_blocks = DIV_ROUND_UP(width / vec * height, 256);
        if (n_blocks > 2048) n_blocks = 2048;
        if (!n_blocks) n_blocks = 1;
        void *kernel_params[] = {
            (void*) ref_pic, (void*) dist_pic, (void*) s->sse,
            &plane, &width, &height,
        };
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(func,
                    n_blocks, 1, 1, 256, 1, 1, 0,
                    s->cuda.str, kernel_params, NULL));
    }

    /* Keep pooled pictures alive until the kernels finish reading them. */
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.consumed, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(dist_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));

    // Download sse into this slot's readback segment
    uint64_t *sse_host = s->sse_host + slot * 3;
    CUDA_FEX_CHECK(cu_f, cuMemcpyDtoHAsync(sse_host, s->sse->data,
                sizeof(uint64_t) * 3, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.finished, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.host_stream, s->cuda.finished,
                CU_EVENT_WAIT_DEFAULT));

    params->feature_collector = feature_collector;
    params->sse = sse_host;
    for (unsigned p = 0; p < n; p++) {
        params->w[p] = ref_pic->w[p];
        params->h[p] = ref_pic->h[p];
    }
    params->index = index;
    CUDA_FEX_CHECK(cu_f, cuLaunchHostFunc(s->cuda.host_stream, write_scores,
                params));
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.slot_done[slot], s->cuda.host_stream));

    return 0;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    const char *apsnr_name[3] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };

    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.host_stream));

    write_score_parameters_psnr *params = s->write_score_parameters;
    for (unsigned i = 0; i < 2; i++)
        if (params[i].err) return params[i].err;
    int err = 0;
    if (s->enable_apsnr) {
        for (unsigned i = 0; i < (s->enable_chroma ? 3U : 1U); i++) {

            double apsnr = 10 * (log10((double)s->peak * s->peak) +
                                 log10(s->apsnr.n_pixels[i]) -
                                 log10(s->apsnr.sse[i]));

            double max_apsnr =
                ceil(10 * log10((double)s->peak * s->peak *
                                s->apsnr.n_pixels[i] *
                                2));

            err |=
                vmaf_feature_collector_set_aggregate(feature_collector,
                                                     apsnr_name[i],
                                                     MIN(apsnr, max_apsnr));
        }
    }

    return (err < 0) ? err : !err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    int err = cuda_fex_resources_close(cu_f, &s->cuda);
    err |= cuda_fex_buffer_free(cu_f, &s->sse);
    if (s->sse_host)
        err |= cuda_fex_error(cu_f, cu_f->cuMemFreeHost(s->sse_host));
    s->sse_host = NULL;
    free(s->write_score_parameters);
    s->write_score_parameters = NULL;
    err |= cuda_fex_error(cu_f, cu_f->cuCtxPopCurrent(NULL));
    return err;
}

static const char *provided_features[] = {
    "psnr_y", "psnr_cb", "psnr_cr",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_psnr_cuda = {
    .name = "psnr_cuda",
    .options = options,
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(PsnrStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_CUDA |
             VMAF_FEATURE_EXTRACTOR_CHROMA,
};
