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
#include "motion_blend_tools.h"

typedef struct MotionStateCuda {
    CUevent event, finished, sad_zeroed;
    CUfunction funcbpc8, funcbpc16;
    CUstream str, host_stream;
    // Ring of raw (unfiltered) luma planes of the three most recent frames;
    // the SAD filter is applied to the difference of two frames. The default
    // path compares frame i against frame i-1 (fex->prev_ref on the CPU), the
    // five frame window (motion_five_frame_window) against frame i-2
    // (fex->prev_prev_ref), so three slots cover every frame that is still
    // needed.
    VmafCudaBuffer* raw[3];
    VmafCudaBuffer* sad;
    uint64_t* sad_host;
    double score;
    bool debug;
    bool motion_force_zero;
    double *motion_scores;   // per-frame normalized SAD scores, indexed by picture index
    unsigned motion_scores_capacity;
    unsigned motion_scores_count;
    double motion_max_val;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    bool motion_five_frame_window;
    bool motion_moving_average;
    void (*calculate_motion_score)(const VmafPicture* src, VmafCudaBuffer* src_blurred,
            const VmafCudaBuffer* prev_blurred, VmafCudaBuffer* sad,
            unsigned width, unsigned height,
            ptrdiff_t src_stride, ptrdiff_t blurred_stride, unsigned src_bpc,
            CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions *cu_f, CUstream stream);
    VmafDictionary *feature_name_dict;
} MotionStateCuda;

#define DEFAULT_MOTION_MAX_VAL (10000.0)
#define DEFAULT_MOTION_BLEND_FACTOR (1.0)
#define DEFAULT_MOTION_BLEND_OFFSET (40.0)
#define DEFAULT_MOTION_FPS_WEIGHT (1.0)

/* Option table mirrors the CPU reference (integer_motion.c) name for name,
 * including the aliases and the FEATURE_PARAM flags, so that the derived
 * feature names (and therefore model integration) are identical. */
static const VmafOption options[] = {
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionStateCuda, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionStateCuda, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_BLEND_FACTOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionStateCuda, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_BLEND_OFFSET,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionStateCuda, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_FPS_WEIGHT,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionStateCuda, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window",
        .offset = offsetof(MotionStateCuda, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion scores after first frame",
        .offset = offsetof(MotionStateCuda, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

typedef struct write_score_parameters_moco {
    VmafFeatureCollector *feature_collector;
    MotionStateCuda *s;
    unsigned h, w;
    unsigned index;
} write_score_parameters_moco;

static int write_motion_scores(MotionStateCuda *s,
        VmafFeatureCollector *feature_collector, double score, unsigned idx);
static int motion_scores_store(MotionStateCuda *s, unsigned idx, double score);

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

    int err = write_motion_scores(s, feature_collector, 0., index);
    if (err) return err;

    // Keep the flush() path enabled: motion2/motion3 are derived from the
    // (zero) per-frame scores, exactly like the CPU reference.
    return motion_scores_store(s, index, 0.);
}

void calculate_motion_score(const VmafPicture* src, VmafCudaBuffer* cur_raw,
        const VmafCudaBuffer* prev_raw, VmafCudaBuffer* sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t raw_stride_ignored, unsigned src_bpc,
        CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions* cu_f, CUstream stream)
{
    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    // The raw frame buffers hold one uint16_t per sample, so the row stride in
    // elements is simply the width.
    ptrdiff_t raw_stride = (ptrdiff_t)width;
    (void)raw_stride_ignored;

    if (src_bpc == 8){
        void *kernelParams[] = {(void*)src,   (void*) cur_raw, (void*)prev_raw, (void*)sad,
            &width, &height,     &src_stride,  &raw_stride, &src_bpc};
        CHECK_CUDA(cu_f, cuLaunchKernel(funcbpc8, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    } else {
        void *kernelParams[] = {(void*)src,   (void*) cur_raw, (void*)prev_raw, (void*)sad,
            &width, &height,     &src_stride,  &raw_stride, &src_bpc};
        CHECK_CUDA(cu_f, cuLaunchKernel(funcbpc16, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    }
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
        unsigned bpc, unsigned w, unsigned h)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->sad_zeroed, CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, motion_score_ptx));

    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_motion_score_kernel_16bpc"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_motion_score_kernel_8bpc"));

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    if (s->motion_force_zero) {
        fex->extract = extract_force_zero;
        return 0;
    }

    s->calculate_motion_score = calculate_motion_score;

    int ret = 0;

    s->score = 0;

    for (unsigned i = 0; i < 3; i++) {
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->raw[i],
                                      sizeof(uint16_t) * w * h);
        if (ret) goto free_ref;
    }
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->sad, sizeof(uint64_t));
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void**)&s->sad_host, sizeof(uint64_t));
    if (ret) goto free_ref;

    // The first min_idx frames have no predecessor ({index-1} or {index-2}
    // respectively): zero the raw planes so that the first (unreported) SADs
    // are deterministic instead of reading stale memory.
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    for (unsigned i = 0; i < 3; i++)
        CHECK_CUDA(cu_f, cuMemsetD8Async(s->raw[i]->data, 0,
                                         sizeof(uint16_t) * w * h, s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto free_ref;

    s->motion_scores_capacity = 64;
    s->motion_scores = malloc(sizeof(*s->motion_scores) * s->motion_scores_capacity);
    if (!s->motion_scores) goto free_ref;
    s->motion_scores_count = 0;

    return 0;


free_ref:
    for (unsigned i = 0; i < 3; i++) {
        if (s->raw[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->raw[i]);
            free(s->raw[i]);
        }
    }
    if (s->sad) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    if (s->motion_scores) free(s->motion_scores);

    return -ENOMEM;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));

    if (!s->motion_scores || s->motion_scores_count == 0)
        return 0;

    const unsigned stride = s->motion_five_frame_window ? 2u : 1u;
    const unsigned min_idx = s->motion_five_frame_window ? 2u : 1u;
    const unsigned n = s->motion_scores_count;

    /* stamp_value: first valid processed motion2 value */
    double stamp_value = 0.;
    if (n > min_idx) {
        double sad_at_min_idx = s->motion_scores[min_idx];
        stamp_value = MIN(motion_blend(sad_at_min_idx,
                                       s->motion_blend_factor,
                                       s->motion_blend_offset),
                          s->motion_max_val);
    }

    double prev_processed = 0.;
    for (unsigned i = 0; i < n; i++) {
        double sad_i = s->motion_scores[i];

        double motion2;
        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const int hi_idx = (int)i + 1;
            double motion2_val;
            if (hi_idx >= (int)n) {
                motion2_val = sad_i;
            } else if (lo_idx >= (int)min_idx) {
                double lo = s->motion_scores[lo_idx];
                double hi = s->motion_scores[hi_idx];
                motion2_val = lo < hi ? lo : hi;
            } else {
                motion2_val = s->motion_scores[hi_idx];
            }
            motion2 = motion2_val;
        }

        int err = vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict,
                "VMAF_integer_feature_motion2_score", motion2, i);
        if (err) return err;

        double motion3;
        if (i < min_idx) {
            motion3 = stamp_value;
            prev_processed = stamp_value;
        } else {
            double processed = MIN(motion_blend(motion2,
                                                s->motion_blend_factor,
                                                s->motion_blend_offset),
                                   s->motion_max_val);
            motion3 = s->motion_moving_average
                    ? (processed + prev_processed) / 2.0
                    : processed;
            prev_processed = processed;
        }

        err = vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict,
                "VMAF_integer_feature_motion3_score", motion3, i);
        if (err) return err;
    }

    return (ret < 0) ? ret : !ret;
}

/* Per-frame SAD score, identical to the CPU reference:
 *   MIN((double)sad / 256. / (w * h) * motion_fps_weight, motion_max_val) */
static double motion_sad_score(MotionStateCuda *s, uint64_t sad,
        unsigned w, unsigned h)
{
    return MIN((double)sad / 256. / (w * h) * s->motion_fps_weight,
               s->motion_max_val);
}

static int motion_scores_store(MotionStateCuda *s, unsigned idx, double score)
{
    if (idx >= s->motion_scores_capacity) {
        unsigned new_cap = s->motion_scores_capacity * 2;
        double *tmp = realloc(s->motion_scores, sizeof(*tmp) * new_cap);
        if (!tmp) return -ENOMEM;
        s->motion_scores = tmp;
        s->motion_scores_capacity = new_cap;
    }
    s->motion_scores[idx] = score;
    if (idx >= s->motion_scores_count)
        s->motion_scores_count = idx + 1;
    return 0;
}

static int write_motion_scores(MotionStateCuda *s,
        VmafFeatureCollector *feature_collector, double score, unsigned idx)
{
    int err = vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion_sad_score", score, idx);
    if (err) return err;

    if (s->debug)
        err = vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict,
                "VMAF_integer_feature_motion_score", score, idx);

    return err;
}

static int write_scores(write_score_parameters_moco* params)
{
    MotionStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;
    const unsigned idx = params->index;
    const unsigned min_idx = s->motion_five_frame_window ? 2u : 1u;

    double score = 0.;
    if (idx >= min_idx)
        score = motion_sad_score(s, *s->sad_host, params->w, params->h);

    int err = write_motion_scores(s, feature_collector, score, idx);
    if (err) {
        free(params);
        return err;
    }

    /* Store per-frame normalized score; motion2/motion3 are computed in flush(). */
    err = motion_scores_store(s, idx, score);
    free(params);
    return err;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    // CHECK_CUDA(cu_f, cuEventSynchronize(s->finished));
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    int err = 0;
    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    const unsigned cur_raw_idx = index % 3;
    // motion_five_frame_window needs frame index - 2 (the CPU reads
    // fex->prev_prev_ref), the default path frame index - 1 (fex->prev_ref).
    // Both are the frame that a 3-slot ring last wrote to that slot:
    // index-2 == index+1 (mod 3) and index-1 == index+2 (mod 3).
    const unsigned prev_raw_idx =
        (index + (s->motion_five_frame_window ? 1u : 2u)) % 3;

    // Reset device SAD on s->str: this is serialized with the previous frame's
    // DtoH copy of the same buffer.
    CHECK_CUDA(cu_f, cuMemsetD8Async(s->sad->data, 0, sizeof(uint64_t), s->str));
    CHECK_CUDA(cu_f, cuEventRecord(s->sad_zeroed, s->str));

    // Compute motion score on the picture's stream, ordered after the picture
    // upload, the SAD zeroing, and the previous frame's kernel (which either
    // wrote the slot read here, or read the slot written here, so the ring is
    // always consumed before it is reused).
    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA(cu_f, cuStreamWaitEvent(pic_stream, vmaf_cuda_picture_get_ready_event(dist_pic), CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(pic_stream, s->sad_zeroed, CU_EVENT_WAIT_DEFAULT));
    if (index > 0)
        CHECK_CUDA(cu_f, cuStreamWaitEvent(pic_stream, s->event, CU_EVENT_WAIT_DEFAULT));
    s->calculate_motion_score(ref_pic, s->raw[cur_raw_idx], s->raw[prev_raw_idx],
            s->sad, ref_pic->w[0], ref_pic->h[0], ref_pic->stride[0], (ptrdiff_t)ref_pic->w[0],
            ref_pic->bpc, s->funcbpc8, s->funcbpc16, cu_f, pic_stream);
    CHECK_CUDA(cu_f, cuEventRecord(s->event, pic_stream));
    // This event ensures the input buffer is consumed
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    if (index == 0) {
        // Match the CPU reference: the first min_idx frames report a zero
        // score, there is no predecessor to compare against.
        err = write_motion_scores(s, feature_collector, 0., index);
        if (err) return err;
        return motion_scores_store(s, index, 0.);
    }

    // Download sad
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(s->sad_host, (CUdeviceptr)s->sad->data,
                sizeof(*s->sad_host), s->str));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));

    write_score_parameters_moco* params = malloc(sizeof(*params));
    if (!params) return -ENOMEM;
    params->s = s;
    params->feature_collector = feature_collector;
    params->h = ref_pic->h[0];
    params->w = ref_pic->w[0];
    params->index = index;
    CHECK_CUDA(cu_f, cuLaunchHostFunc(s->host_stream, (CUhostFn*)write_scores, params));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuEventDestroy(s->event));
    CHECK_CUDA(cu_f, cuEventDestroy(s->finished));
    CHECK_CUDA(cu_f, cuEventDestroy(s->sad_zeroed));

    int ret = 0;

    for (unsigned i = 0; i < 3; i++) {
        if (s->raw[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->raw[i]);
            free(s->raw[i]);
        }
    }
    if (s->sad) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    if (s->motion_scores) free(s->motion_scores);
    return ret;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_sad_score",
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_motion3_score",
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
