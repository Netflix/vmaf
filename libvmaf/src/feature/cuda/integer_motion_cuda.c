/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include "common.h"
#include "cpu.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_motion_cuda.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

typedef struct MotionStateCuda {
    CUevent event, finished;
    CUfunction funcbpc8, funcbpc16;
    CUstream str, host_stream;
    VmafCudaBuffer* blur[2];
    VmafCudaBuffer* sad;
    uint64_t* sad_host;
    void* write_score_parameters;
    unsigned index;
    double score;
    bool debug;
    bool motion_force_zero;
    void (*calculate_motion_score)(const VmafPicture* src, VmafCudaBuffer* src_blurred,
            const VmafCudaBuffer* prev_blurred, VmafCudaBuffer* sad,
            unsigned width, unsigned height,
            ptrdiff_t src_stride, ptrdiff_t blurred_stride, unsigned src_bpc,
            CUfunction funcbpc8, CUfunction funcbpc16, CUstream stream);
    VmafDictionary *feature_name_dict;
} MotionStateCuda;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionStateCuda, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

typedef struct write_score_parameters_moco {
    VmafFeatureCollector *feature_collector;
    MotionStateCuda *s;
    unsigned h, w;
    unsigned index;
} write_score_parameters_moco;

static int extract_force_zero(VmafFeatureExtractor *fex,
        VmafPicture *ref_pic, VmafPicture *ref_pic_90,
        VmafPicture *dist_pic, VmafPicture *dist_pic_90,
        unsigned index,
        VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;

    (void) fex;
    (void) ref_pic;
    (void) ref_pic_90;
    (void) dist_pic;
    (void) dist_pic_90;

    int err =
        vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict, "VMAF_integer_feature_motion2_score", 0.,
                index);

    if (!s->debug) return err;

    err = vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_motion_score", 0.,
            index);

    return err;
}

void calculate_motion_score(const VmafPicture* src, VmafCudaBuffer* src_blurred,
        const VmafCudaBuffer* prev_blurred, VmafCudaBuffer* sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t blurred_stride, unsigned src_bpc,
        CUfunction funcbpc8, CUfunction funcbpc16, CUstream stream)
{
    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    if (src_bpc == 8){
        void *kernelParams[] = {(void*)src,   (void*) src_blurred, (void*)prev_blurred, (void*)sad,
            &width, &height,     &src_stride,  &blurred_stride};
        CHECK_CUDA(cuLaunchKernel(funcbpc8, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    } else {
        void *kernelParams[] = {(void*)src,   (void*) src_blurred, (void*)prev_blurred, (void*)sad,
            &width, &height,     &src_stride,  &blurred_stride};
        CHECK_CUDA(cuLaunchKernel(funcbpc16, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    }
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
        unsigned bpc, unsigned w, unsigned h)
{
    MotionStateCuda *s = fex->priv;

    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, src_motion_score_ptx));

    CHECK_CUDA(cuModuleGetFunction(&s->funcbpc16, module, "calculate_motion_score_kernel_16bpc"));
    CHECK_CUDA(cuModuleGetFunction(&s->funcbpc8, module, "calculate_motion_score_kernel_8bpc"));

    CHECK_CUDA(cuCtxPopCurrent(NULL));

    if (s->motion_force_zero) {
        fex->extract = extract_force_zero;
        fex->flush = NULL;
        fex->close = NULL;
        return 0;
    }

    s->calculate_motion_score = calculate_motion_score;

    int ret = 0;

    s->score = 0;

    s->write_score_parameters = malloc(sizeof(write_score_parameters_moco));
    ((write_score_parameters_moco*)s->write_score_parameters)->s = s;

    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->blur[0], sizeof(uint16_t) * w * h);
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->blur[1], sizeof(uint16_t) * w * h);
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->sad, sizeof(uint64_t));
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->sad_host, sizeof(uint64_t));
    if (ret) goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto free_ref;

    return 0;


free_ref:
    if (s->blur[0]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[0]);
        free(s->blur[0]);
    }
    if (s->blur[1]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[1]);
        free(s->blur[1]);
    }
    if (s->sad) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);

    return -ENOMEM;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;
    int ret = 0;
    CHECK_CUDA(cuStreamSynchronize(s->str));
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));

    if (s->index > 0) {
        ret = vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion2_score",
                s->score, s->index);
    }

    return (ret < 0) ? ret : !ret;
}

static inline double normalize_and_scale_sad(uint64_t sad,
        unsigned w, unsigned h)
{
    return (float) (sad / 256.) / (w * h);
}


static int write_scores(write_score_parameters_moco* params)
{
    MotionStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    double score_prev = s->score;

    s->score = normalize_and_scale_sad(*s->sad_host, params->w, params->h);
    int err = 0;
    if (s->debug) {
        err |= vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion_score",
                s->score, params->index);
    }
    if (err) return err;

    if (params->index == 1)
        return 0;

    err = vmaf_feature_collector_append(feature_collector,
            "VMAF_integer_feature_motion2_score",
            score_prev < s->score ? score_prev : s->score, params->index - 1);
    return err;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cuStreamSynchronize(s->str));
    // CHECK_CUDA(cuEventSynchronize(s->finished));
    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuEventDestroy(s->finished));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    int err = 0;
    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    s->index = index;
    const unsigned src_blurred_idx = (index + 0) % 2;
    const unsigned prev_blurred_idx = (index + 1) % 2;

    // Reset device SAD
    CHECK_CUDA(cuMemsetD8Async(s->sad->data, 0, sizeof(uint64_t), s->str));

    // Compute motion score
    CHECK_CUDA(cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic), vmaf_cuda_picture_get_ready_event(dist_pic), CU_EVENT_WAIT_DEFAULT));
    s->calculate_motion_score(ref_pic, s->blur[src_blurred_idx], s->blur[prev_blurred_idx],
            s->sad, ref_pic->w[0], ref_pic->h[0], ref_pic->stride[0], sizeof(uint16_t) * ref_pic->w[0],
            ref_pic->bpc, s->funcbpc8, s->funcbpc16, vmaf_cuda_picture_get_stream(ref_pic));
    CHECK_CUDA(cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
    // This event ensures the input buffer is consumed
    CHECK_CUDA(cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuEventDestroy(s->event));
    CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    if (index == 0) {
        err = vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion2_score",
                0., index);
        if (s->debug) {
            err |= vmaf_feature_collector_append(feature_collector,
                    "VMAF_integer_feature_motion_score",
                    0., index);
        }
        return err;
    }

    // Download sad
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuMemcpyDtoHAsync(s->sad_host, (CUdeviceptr)s->sad->data,
                sizeof(*s->sad_host), s->str));
    CHECK_CUDA(cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));

    write_score_parameters_moco* params = s->write_score_parameters;
    params->feature_collector = feature_collector;
    params->h = ref_pic->h[0];
    params->w = ref_pic->w[0];
    params->index = index;
    CHECK_CUDA(cuLaunchHostFunc(s->host_stream, write_scores, s->write_score_parameters));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    MotionStateCuda *s = fex->priv;
    CHECK_CUDA(cuStreamSynchronize(s->str));
    int ret = 0;

    if (s->blur[0]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[0]);
        free(s->blur[0]);
    }
    if (s->blur[1]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[1]);
        free(s->blur[1]);
    }
    if (s->sad) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);

    if(s->write_score_parameters) {
        free(s->write_score_parameters);
    }
    return ret;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_score", "VMAF_integer_feature_motion2_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion_cuda = {
    .name = "motion_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(MotionStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_CUDA,
};
