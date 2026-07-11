/**
 *
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

typedef struct CiedeStateCuda {
    CUevent finished, consumed;
    CUevent slot_done[2];
    CUfunction funcbpc8, funcbpc16;
    CUstream str, host_stream;
    VmafCudaBuffer *partials;
    double *partials_host;
    void *write_score_parameters;
    unsigned w, h;
    unsigned bpc;
    unsigned n_partials;
    int ss_hor, ss_ver;
} CiedeStateCuda;

typedef struct write_score_parameters_ciede {
    VmafFeatureCollector *feature_collector;
    CiedeStateCuda *s;
    const double *partials;
    unsigned index;
} write_score_parameters_ciede;

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;
    switch (bpc) {
    case 8:
    case 10:
    case 12:
    case 16:
        break;
    default:
        return -EINVAL;
    }

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    // the work stream is deliberately legacy-blocking: producers like the
    // ffmpeg libvmaf_cuda filter fill device pictures with synchronous-API
    // copies that are queued on the legacy NULL stream (device-to-device
    // memcpy does not block the host), and only blocking-flavor streams are
    // implicitly ordered after NULL-stream work. Other extractors' created
    // streams are unaffected, so kernel overlap with them is preserved.
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_DEFAULT, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->consumed, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->slot_done[0], CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->slot_done[1], CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, ciede_ptx));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "ciede_kernel_8bpc"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "ciede_kernel_16bpc"));

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    s->w = w;
    s->h = h;
    s->bpc = bpc;
    s->ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    s->ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
    s->n_partials = DIV_ROUND_UP(w, 16) * DIV_ROUND_UP(h, 16);

    int ret = 0;

    // two write_score slots + two pinned readback slots so frame i+1 never
    // has to wait for frame i's host callback (see slot_done in extract)
    s->write_score_parameters = malloc(sizeof(write_score_parameters_ciede) * 2);
    if (!s->write_score_parameters) return -ENOMEM;
    for (unsigned i = 0; i < 2; i++)
        ((write_score_parameters_ciede*)s->write_score_parameters)[i].s = s;

    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->partials,
                                  sizeof(double) * s->n_partials);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void**)&s->partials_host,
                                       sizeof(double) * s->n_partials * 2);
    if (ret) return -ENOMEM;

    return 0;
}

static int write_scores(write_score_parameters_ciede *params)
{
    CiedeStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    // sequential sum over block partials keeps the result deterministic
    double de00_sum = 0.;
    for (unsigned b = 0; b < s->n_partials; b++)
        de00_sum += params->partials[b];

    // identical frames give de00_sum == 0 and score +inf, like the CPU fex
    const double score = 45. - 20. *
                         log10(de00_sum / ((double)s->w * s->h));
    return vmaf_feature_collector_append(feature_collector, "ciede2000", score,
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

    // two slots: wait for frame index-2's host callback (effectively always
    // complete) instead of stalling on the whole previous frame's work
    const unsigned slot = index & 1;
    CHECK_CUDA(cu_f, cuEventSynchronize(s->slot_done[slot]));

    // kernels run on the extractor's own stream so they overlap with other
    // extractors' work on the picture streams; wait for both uploads first
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str,
                vmaf_cuda_picture_get_ready_event(ref_pic),
                CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str,
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
        CHECK_CUDA(cu_f, cuLaunchKernel(func,
                    DIV_ROUND_UP(width, 16), DIV_ROUND_UP(height, 16), 1,
                    16, 16, 1, 0, s->str, kernel_params, NULL));
    }

    // lifetime handshake: the pool recycles a picture once the `finished`
    // event its own stream records (after the fex loop) has completed, so
    // make both picture streams wait for our reads
    CHECK_CUDA(cu_f, cuEventRecord(s->consumed, s->str));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                s->consumed, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(dist_pic),
                s->consumed, CU_EVENT_WAIT_DEFAULT));

    // Download block partials into this slot's readback segment
    double *partials_host = s->partials_host + slot * s->n_partials;
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(partials_host, s->partials->data,
                sizeof(double) * s->n_partials, s->str));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->host_stream, s->finished,
                CU_EVENT_WAIT_DEFAULT));

    write_score_parameters_ciede *params =
        &((write_score_parameters_ciede*)s->write_score_parameters)[slot];
    params->feature_collector = feature_collector;
    params->partials = partials_host;
    params->index = index;
    CHECK_CUDA(cu_f, cuLaunchHostFunc(s->host_stream, (CUhostFn*)write_scores,
                params));
    CHECK_CUDA(cu_f, cuEventRecord(s->slot_done[slot], s->host_stream));

    return 0;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    // drain the pending write_scores host callback so the final frame's
    // score is in the collector before anything reads it
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    return 1;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cu_f, cuEventDestroy(s->finished));
    CHECK_CUDA(cu_f, cuEventDestroy(s->consumed));
    CHECK_CUDA(cu_f, cuEventDestroy(s->slot_done[0]));
    CHECK_CUDA(cu_f, cuEventDestroy(s->slot_done[1]));
    CHECK_CUDA(cu_f, cuStreamDestroy(s->str));
    CHECK_CUDA(cu_f, cuStreamDestroy(s->host_stream));

    int ret = 0;
    if (s->partials) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->partials);
        free(s->partials);
    }
    if (s->partials_host)
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->partials_host);
    if (s->write_score_parameters)
        free(s->write_score_parameters);

    return ret;
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
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA | VMAF_FEATURE_EXTRACTOR_CUDA_CHROMA,
};
