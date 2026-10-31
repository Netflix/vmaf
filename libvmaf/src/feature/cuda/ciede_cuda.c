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
#include "cuda/ciede_cuda.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"
#include "cuda/feature_common.h"

typedef struct CiedeStateCuda {
    CudaFeatureResources cuda;
    CUfunction funcbpc8, funcbpc16;
    VmafCudaBuffer *partials;
    double *partials_host;
    void *write_score_parameters;
    unsigned w, h;
    unsigned n_partials;
    int ss_hor, ss_ver;
} CiedeStateCuda;

typedef struct write_score_parameters_ciede {
    VmafFeatureCollector *feature_collector;
    CiedeStateCuda *s;
    const double *partials;
    unsigned index;
    int err; /* Read after callback completion. */
} write_score_parameters_ciede;

static int close_fex_cuda(VmafFeatureExtractor *fex);

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (!w || !h)
        return -EINVAL;
    /* Subsampled planes contain only complete pairs of luma samples. */
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        if (h & 1) return -EINVAL;
        /* fall through */
    case VMAF_PIX_FMT_YUV422P:
        if (w & 1) return -EINVAL;
        break;
    case VMAF_PIX_FMT_YUV444P:
        break;
    default:
        return -EINVAL;
    }
    switch (bpc) {
    case 8:
    case 10:
    case 12:
    case 16:
        break;
    default:
        return -EINVAL;
    }

    s->w = w;
    s->h = h;
    s->ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    s->ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
    s->n_partials = DIV_ROUND_UP(w, 16) * DIV_ROUND_UP(h, 16);

    int err;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    err = cuda_fex_resources_init(cu_f, &s->cuda, ciede_ptx);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->funcbpc8, s->cuda.module, "ciede_kernel_8bpc"));
    CUDA_FEX_INIT(cu_f, cuModuleGetFunction(&s->funcbpc16, s->cuda.module, "ciede_kernel_16bpc"));
    s->write_score_parameters = calloc(2, sizeof(write_score_parameters_ciede));
    if (!s->write_score_parameters) { err = -ENOMEM; goto fail; }
    for (unsigned i = 0; i < 2; i++)
        ((write_score_parameters_ciede *)s->write_score_parameters)[i].s = s;
    err = cuda_fex_buffer_alloc(cu_f, &s->partials, sizeof(double) * s->n_partials);
    if (err) goto fail;
    CUDA_FEX_INIT(cu_f, cuMemHostAlloc((void **)&s->partials_host,
            sizeof(double) * s->n_partials * 2, 0x01));
    CUDA_FEX_INIT(cu_f, cuCtxPopCurrent(NULL));
    return 0;
fail:
    close_fex_cuda(fex);
    cu_f->cuCtxPopCurrent(NULL);
    return err;
}

static void CUDAAPI write_scores(void *opaque)
{
    write_score_parameters_ciede *params = opaque;
    CiedeStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    // sequential sum over block partials keeps the result deterministic
    double de00_sum = 0.;
    for (unsigned b = 0; b < s->n_partials; b++)
        de00_sum += params->partials[b];

    // identical frames give de00_sum == 0 and score +inf, like the CPU fex
    const double score = 45. - 20. *
                         log10(de00_sum / ((double)s->w * s->h));
    params->err = vmaf_feature_collector_append(feature_collector, "ciede2000", score,
                                         params->index);
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) ref_pic_90;
    (void) dist_pic_90;

    // Wait for the previous callback using this slot, including skipped indices.
    const unsigned slot = index & 1;
    CUDA_FEX_CHECK(cu_f, cuEventSynchronize(s->cuda.slot_done[slot]));
    write_score_parameters_ciede *params =
        &((write_score_parameters_ciede *)s->write_score_parameters)[slot];
    if (params->err) return params->err;

    /* Wait for both producers before reading either picture. */
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(ref_pic),
                CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(s->cuda.str,
                vmaf_cuda_picture_get_ready_event(dist_pic),
                CU_EVENT_WAIT_DEFAULT));

    {
        unsigned width = s->w, height = s->h;
        int ss_hor = s->ss_hor, ss_ver = s->ss_ver;
        void *kernel_params[] = {
            (void*) ref_pic, (void*) dist_pic, (void*) s->partials,
            &width, &height, &ss_hor, &ss_ver,
        };
        const CUfunction func =
            (ref_pic->bpc == 8) ? s->funcbpc8 : s->funcbpc16;
        CUDA_FEX_CHECK(cu_f, cuLaunchKernel(func,
                    DIV_ROUND_UP(width, 16), DIV_ROUND_UP(height, 16), 1,
                    16, 16, 1, 0, s->cuda.str, kernel_params, NULL));
    }

    /* Keep pooled pictures alive until the kernels finish reading them. */
    CUDA_FEX_CHECK(cu_f, cuEventRecord(s->cuda.consumed, s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));
    CUDA_FEX_CHECK(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(dist_pic),
                s->cuda.consumed, CU_EVENT_WAIT_DEFAULT));

    // Download block partials into this slot's readback segment
    double *partials_host = s->partials_host + slot * s->n_partials;
    CUDA_FEX_CHECK(cu_f, cuMemcpyDtoHAsync(partials_host, s->partials->data,
                sizeof(double) * s->n_partials, s->cuda.str));
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
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Publish the final callback result before returning. */
    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.str));
    CUDA_FEX_CHECK(cu_f, cuStreamSynchronize(s->cuda.host_stream));
    write_score_parameters_ciede *params = s->write_score_parameters;
    for (unsigned i = 0; i < 2; i++)
        if (params[i].err) return params[i].err;
    return 1;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CUDA_FEX_CHECK(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    int err = cuda_fex_resources_close(cu_f, &s->cuda);
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
    "ciede2000",
    NULL
};

VmafFeatureExtractor vmaf_fex_ciede_cuda = {
    .name = "ciede_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(CiedeStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA | VMAF_FEATURE_EXTRACTOR_CHROMA,
};
