/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2022 NVIDIA Corporation.
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
#include <string.h>

#include "cuda_helper.cuh"
#include "libvmaf/vmaf_cuda_state.h"
#include "picture_cuda.h"

#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"


typedef struct PsnrStateCuda {
    bool enable_chroma;
    bool enable_mse;
    bool enable_apsnr;
    bool reduced_hbd_peak;
    uint32_t peak;
    CUevent event, finished;
    CUfunction func_psnr;
    CUstream str, host_stream;
    double psnr_max[3];
    double min_sse;
    VmafCudaBuffer *sse_device;
    uint64_t *sse_host;
    void* write_score_parameters;

    struct {
        uint64_t sse[3];
        uint64_t n_pixels[3];
    } apsnr;
} PsnrStateCuda;

typedef struct write_score_parameters_psnr {
    VmafFeatureCollector *feature_collector;
    PsnrStateCuda *s;
    unsigned h[3], w[3];
    unsigned index;
} write_score_parameters_psnr;

extern unsigned char src_psnr_ptx[];

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


static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    PsnrStateCuda *s = fex->priv;

    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, src_psnr_ptx));
    if (bpc > 8) {
        CHECK_CUDA(cuModuleGetFunction(&s->func_psnr, module, "psnr_hbd"));
    } else {
        CHECK_CUDA(cuModuleGetFunction(&s->func_psnr, module, "psnr"));
    }
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    s->write_score_parameters = malloc(sizeof(write_score_parameters_psnr));
    ((write_score_parameters_psnr*)s->write_score_parameters)->s = s;


    int ret = 0; 
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->sse_device, sizeof(uint64_t) * 3);
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->sse_host, sizeof(uint64_t) * 3);
    if (ret) goto free_ref;

    s->peak = s->reduced_hbd_peak ? 255 * 1 << (bpc - 8) : (1 << bpc) - 1;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;

    for (unsigned i = 0; i < 3; i++) {
        if (s->min_sse != 0.0) {
            const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
            const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
            const double mse = s->min_sse / 
                (((i && ss_hor) ? w / 2 : w) * ((i && ss_ver) ? h / 2 : h));
            s->psnr_max[i] = ceil(10. * log10(s->peak * s->peak / mse));
        } else {
            s->psnr_max[i] = (6 * bpc) + 12;
        }
    }

    return ret;
free_ref:
    if (s->sse_device) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sse_device);
        free(s->sse_device);
    }
    return -ENOMEM;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static char *mse_name[3] = { "mse_y", "mse_cb", "mse_cr" };
static char *psnr_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };

static int write_scores(write_score_parameters_psnr* params)
{
    PsnrStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;
    
    const unsigned n = s->enable_chroma ? 3 : 1;
    for (unsigned p = 0; p < n; p++) {
        if (s->enable_apsnr) {
            s->apsnr.sse[p] += s->sse_host[p];
            s->apsnr.n_pixels[p] += params->h[p] * params->w[p];
        }

        const double mse = ((double) s->sse_host[p]) / (params->w[p] * params->h[p]);
        const double psnr =
            MIN(10. * log10(s->peak * s->peak / MAX(mse, 1e-16)), s->psnr_max[p]);


        int err = 0;
        err |= vmaf_feature_collector_append(feature_collector, psnr_name[p],
                                                psnr, params->index);
        if (s->enable_mse) {
            err |= vmaf_feature_collector_append(feature_collector, mse_name[p],
                                                    mse, params->index);
        }
    }
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;

    (void) ref_pic_90;
    (void) dist_pic_90;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cuEventSynchronize(s->finished));

    // Reset device SSE
    CHECK_CUDA(cuMemsetD8Async(s->sse_device->data, 0, sizeof(uint64_t) * 3, s->str));


    const int width_y = ref_pic->w[0];
    const int height_y = ref_pic->h[0];
    const int stride_y =  ref_pic->stride[0];

    const int val_per_thread = 8;
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = DIV_ROUND_UP(width_y, block_dim_x * val_per_thread);
    const int grid_dim_y = DIV_ROUND_UP(height_y, block_dim_y);
    const int grid_dim_z = s->enable_chroma ? 3 : 1;

    void *kernelParams[] = {ref_pic,  dist_pic, s->sse_device};
    CHECK_CUDA(cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic), vmaf_cuda_picture_get_ready_event(dist_pic), CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cuLaunchKernel(s->func_psnr, 
                grid_dim_x, grid_dim_y, grid_dim_z, 
                block_dim_x, block_dim_y, 1, 
                0, vmaf_cuda_picture_get_stream(ref_pic), kernelParams, NULL));

    CHECK_CUDA(cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
    // This event ensures the input buffer is consumed
    CHECK_CUDA(cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
    // CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    // CHECK_CUDA(cuEventDestroy(s->event));
    // CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    // CHECK_CUDA(cuCtxPopCurrent(NULL));
    
    // Download sad
    // CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuMemcpyDtoHAsync(s->sse_host, (CUdeviceptr)s->sse_device->data,
                sizeof(uint64_t) * 3, s->str));
    CHECK_CUDA(cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));
    
    write_score_parameters_psnr* params = s->write_score_parameters;
    params->feature_collector = feature_collector;
    for (unsigned p = 0; p < grid_dim_z; p++) {
        params->h[p] = ref_pic->h[p];
        params->w[p] = ref_pic->w[p];
    }
    params->index = index;
    CHECK_CUDA(cuLaunchHostFunc(s->host_stream, write_scores, s->write_score_parameters));
    return 0;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;
    const char *apsnr_name[3] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };
    CHECK_CUDA(cuStreamSynchronize(s->str));
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));

    int err = 0;
    if (s->enable_apsnr) {
        for (unsigned i = 0; i < 3; i++) {

            double apsnr = 10 * (log10(s->peak * s->peak) +
                                 log10(s->apsnr.n_pixels[i]) -
                                 log10(s->apsnr.sse[i]));

            double max_apsnr =
                ceil(10 * log10(s->peak * s->peak *
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
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuStreamSynchronize(s->str));
    int ret = 0;

    if (s->sse_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->sse_host);
    }
    if (s->sse_device) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sse_device);
        free(s->sse_device);
    }
    if(s->write_score_parameters) {
        free(s->write_score_parameters);
    }

    return ret;
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
    .priv_size = sizeof(PsnrStateCuda),
    .close = close_fex_cuda,
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_CUDA,
};
