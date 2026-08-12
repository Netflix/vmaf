/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "cpu.h"
#include "cuda/speed_cuda.h"
#include "cuda_helper.cuh"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "speed.h"
#include "vif_tools.h"

/*
 * CUDA chroma SpEED-QA. Together with the CUDA CAMBI extractor this is what a
 * VMAF v1 model needs in order to load on libvmaf_cuda at all.
 *
 * WHAT RUNS WHERE, AND WHY
 * ------------------------
 * Profiling the CPU extractor on a 1080p clip put est_params at 28% of
 * speed_chroma and the filtering around it at the remaining ~72%. Within
 * est_params the time is dominated by a single 25x25 solve and one covariance
 * accumulation -- both O(1) per scale rather than per block, because
 * est_params estimates ONE covariance matrix across all blocks and runs ONE
 * eigen-decomposition on it.
 *
 * So the device runs the per-pixel front end -- picture_copy, the two
 * separable filters, decimation and the subtraction -- and est_params stays
 * on the host. That keeps the numerically awkward parts (a float reduction in
 * compute_mean whose summation order a parallel reduction would not preserve,
 * and an iterative QR eigensolver) exactly where they already work, and it
 * costs one download of the filtered plane per picture per channel. The
 * filtered plane is 1/16 the width and height of the source after dec16, so
 * that transfer is small.
 *
 * ORDERING
 * --------
 * speed_extract_score interleaves filter(ref), est_params(ref), filter(dis),
 * est_params(dis). est_params writes into SpeedState::buffers, so that order
 * is preserved here rather than batching both filters first.
 *
 * TRANSFERS
 * ---------
 * cuMemcpy{HtoD,DtoH}Async are used directly rather than
 * vmaf_cuda_buffer_{upload,download}_async, which substitute cu_state->str
 * for any non-zero stream argument and would therefore leave transfers
 * unsynchronised against kernels on our own stream.
 *
 * SpeedChromaState is the FIRST member of SpeedChromaStateCuda so every
 * offsetof(SpeedChromaState, ...) in speed_chroma_options[] stays valid and
 * both extractors share one option table.
 */

typedef struct SpeedChromaStateCuda {
    SpeedChromaState cpu;       /* MUST be first -- see speed_chroma_options */

    CUstream str;

    CUfunction func_copy_u8;
    CUfunction func_copy_u16;
    CUfunction func_filter_v;
    CUfunction func_filter_h;
    CUfunction func_dec16;
    CUfunction func_subtract;

    VmafCudaBuffer *d_frame;    /* working plane; holds the result */
    VmafCudaBuffer *d_curr;     /* curr_scale */
    VmafCudaBuffer *d_tmp;      /* separable-filter intermediate */
    VmafCudaBuffer *d_filt_aa;  /* antialias taps */
    VmafCudaBuffer *d_filt_sc;  /* scale taps */

    int filt_aa_width;
    int filt_sc_width;
    size_t stride_px;
    unsigned alloc_height;
} SpeedChromaStateCuda;

#define BLOCK_X 32
#define BLOCK_Y 8

static int launch_2d(CudaFunctions *cu_f, CUfunction f, CUstream str,
                     int width, int height, void **args)
{
    if (width <= 0 || height <= 0)
        return 0;
    CHECK_CUDA(cu_f, cuLaunchKernel(f,
                                    (width + BLOCK_X - 1) / BLOCK_X,
                                    (height + BLOCK_Y - 1) / BLOCK_Y, 1,
                                    BLOCK_X, BLOCK_Y, 1, 0, str, args, NULL));
    return 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    (void) bpc;

    switch (pix_fmt) {
    case VMAF_PIX_FMT_UNKNOWN:
    case VMAF_PIX_FMT_YUV400P:
        return -EINVAL;
    case VMAF_PIX_FMT_YUV420P: w /= 2; h /= 2; break;
    case VMAF_PIX_FMT_YUV422P: w /= 2;         break;
    case VMAF_PIX_FMT_YUV444P:                 break;
    }

    SpeedChromaStateCuda *sc = fex->priv;
    SpeedChromaState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* The prescale path uses vif_scale_frame_*, which has no device
     * counterpart here. Reject rather than silently running a different
     * pipeline than the option asks for. */
    if (!(fabs(s->speed_chroma_prescale - 1.0) < 1e-9)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "speed_chroma_cuda: speed_prescale != 1.0 is not supported "
                 "on the CUDA path\n");
        return -EINVAL;
    }

    s->speed_options = (SpeedOptions) {
        .speed_kernelscale = s->speed_chroma_kernelscale,
        .speed_prescale = s->speed_chroma_prescale,
        .speed_prescale_method = s->speed_chroma_prescale_method,
        .speed_sigma_nn = s->speed_chroma_sigma_nn,
        .speed_nn_floor = s->speed_chroma_nn_floor,
        .speed_weight_var_mode = s->speed_weight_var_mode,
    };

    int err = speed_init(&s->speed_state, &s->speed_options, w, h);
    if (err) return err;
    SpeedDimensions dim = s->speed_state.dimensions;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    s->frame_buffer_ref =
        aligned_malloc(s->speed_state.float_stride * dim.alloc_height, 32);
    s->frame_buffer_dis =
        aligned_malloc(s->speed_state.float_stride * dim.alloc_height, 32);
    if (!s->frame_buffer_ref || !s->frame_buffer_dis)
        return -ENOMEM;

    sc->stride_px = s->speed_state.float_stride / sizeof(float);
    sc->alloc_height = dim.alloc_height;

    /* Filter taps depend only on the kernelscale, so build them once. */
    float filt_aa[128], filt_sc[128];
    sc->filt_aa_width = vif_get_filter_size(1, s->speed_options.speed_kernelscale);
    speed_get_antialias_filter(filt_aa, NUM_SPEED_SCALES,
                               s->speed_options.speed_kernelscale);
    sc->filt_sc_width =
        vif_get_filter_size(NUM_SPEED_SCALES, s->speed_options.speed_kernelscale);
    vif_get_filter(filt_sc, NUM_SPEED_SCALES, s->speed_options.speed_kernelscale);

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&sc->str, CU_STREAM_NON_BLOCKING, 0));

    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, speed_filter_ptx));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_copy_u8, module,
                                         "speed_picture_copy_u8_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_copy_u16, module,
                                         "speed_picture_copy_u16_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_filter_v, module,
                                         "speed_filter1d_v_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_filter_h, module,
                                         "speed_filter1d_h_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_dec16, module,
                                         "speed_dec16_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_subtract, module,
                                         "speed_subtract_kernel"));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    const size_t plane_bytes =
        (size_t) s->speed_state.float_stride * dim.alloc_height;
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_frame, plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_curr, plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_tmp, plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_filt_aa, 128 * sizeof(float));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_filt_sc, 128 * sizeof(float));
    if (err) return err;

    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_filt_aa->data, filt_aa,
                                       128 * sizeof(float), sc->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_filt_sc->data, filt_sc,
                                       128 * sizeof(float), sc->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));

    return 0;
}

/*
 * Device equivalent of picture_copy() followed by filter_and_downscale(),
 * leaving the result in d_frame and downloading it into `host_out`.
 */
static int filter_plane_cuda(VmafFeatureExtractor *fex,
                             SpeedChromaStateCuda *sc, VmafPicture *pic,
                             int channel, float *host_out)
{
    SpeedChromaState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;
    const SpeedDimensions dim = s->speed_state.dimensions;
    const ptrdiff_t px = (ptrdiff_t) sc->stride_px;

    int w = dim.scaled_width, h = dim.scaled_height;
    ptrdiff_t sp, dp = px, tp = px;
    int offset = -128;
    int err;

    /* plane -> float, centred by -128 */
    if (pic->bpc == 8) {
        sp = pic->stride[channel];
        void *a[] = { &pic->data[channel], &sc->d_frame->data, &w, &h,
                      &sp, &dp, &offset };
        err = launch_2d(cu_f, sc->func_copy_u8, sc->str, w, h, a);
    } else {
        sp = pic->stride[channel] >> 1;
        float scaler = (pic->bpc == 10) ? 4.0f
                     : (pic->bpc == 12) ? 16.0f : 256.0f;
        void *a[] = { &pic->data[channel], &sc->d_frame->data, &w, &h,
                      &sp, &dp, &offset, &scaler };
        err = launch_2d(cu_f, sc->func_copy_u16, sc->str, w, h, a);
    }
    if (err) return err;

    /* antialias filter: d_frame -> d_curr */
    {
        int fw = sc->filt_aa_width;
        void *av[] = { &sc->d_filt_aa->data, &sc->d_frame->data,
                       &sc->d_tmp->data, &w, &h, &dp, &tp, &fw };
        err = launch_2d(cu_f, sc->func_filter_v, sc->str, w, h, av);
        if (err) return err;
        void *ah[] = { &sc->d_filt_aa->data, &sc->d_tmp->data,
                       &sc->d_curr->data, &w, &h, &tp, &dp, &fw };
        err = launch_2d(cu_f, sc->func_filter_h, sc->str, w, h, ah);
        if (err) return err;
    }

    /* decimate by 16: d_curr -> d_frame */
    {
        void *a[] = { &sc->d_curr->data, &sc->d_frame->data, &w, &h, &dp, &dp };
        err = launch_2d(cu_f, sc->func_dec16, sc->str, w / 16, h / 16, a);
        if (err) return err;
    }

    int dw = dim.scaled_width >> NUM_SPEED_SCALES;
    int dh = dim.scaled_height >> NUM_SPEED_SCALES;

    /* scale filter: d_frame -> d_curr, then d_frame -= d_curr */
    {
        int fw = sc->filt_sc_width;
        void *av[] = { &sc->d_filt_sc->data, &sc->d_frame->data,
                       &sc->d_tmp->data, &dw, &dh, &dp, &tp, &fw };
        err = launch_2d(cu_f, sc->func_filter_v, sc->str, dw, dh, av);
        if (err) return err;
        void *ah[] = { &sc->d_filt_sc->data, &sc->d_tmp->data,
                       &sc->d_curr->data, &dw, &dh, &tp, &dp, &fw };
        err = launch_2d(cu_f, sc->func_filter_h, sc->str, dw, dh, ah);
        if (err) return err;

        void *as[] = { &sc->d_frame->data, &sc->d_curr->data, &dw, &dh, &dp };
        err = launch_2d(cu_f, sc->func_subtract, sc->str, dw, dh, as);
        if (err) return err;
    }

    /* est_params only reads the dw x dh region, but the rows are strided, so
     * bring back everything up to the last row it touches. */
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(host_out, sc->d_frame->data,
                                       (size_t) px * dh * sizeof(float),
                                       sc->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));
    return 0;
}

static int extract_channel_cuda(VmafFeatureExtractor *fex,
                                SpeedChromaStateCuda *sc, VmafPicture *ref_pic,
                                VmafPicture *dist_pic, int channel,
                                float *score)
{
    SpeedChromaState *s = &sc->cpu;
    SpeedState *ss = &s->speed_state;
    SpeedOptions *opt = &s->speed_options;

    /* Same interleaving as speed_extract_score: est_params writes into
     * ss->buffers, so ref is filtered and estimated before dis is touched. */
    int err = filter_plane_cuda(fex, sc, ref_pic, channel, s->frame_buffer_ref);
    if (err) return err;
    int err_ref = est_params(ss, s->frame_buffer_ref, opt->speed_sigma_nn,
                             &(ss->ref_results));

    err = filter_plane_cuda(fex, sc, dist_pic, channel, s->frame_buffer_dis);
    if (err) return err;
    int err_dis = est_params(ss, s->frame_buffer_dis, opt->speed_sigma_nn,
                             &(ss->dis_results));

    if ((err_ref && !err_dis) || (!err_ref && err_dis)) {
        *score = 0.0f;
    } else {
        *score = get_speed_score(ss->dimensions, ss->ref_results,
                                 ss->dis_results, opt->speed_sigma_nn,
                                 opt->speed_nn_floor,
                                 opt->speed_weight_var_mode);
    }
    return err_ref || err_dis;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                            VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                            VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                            unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    (void) ref_pic_90;
    (void) dist_pic_90;

    SpeedChromaStateCuda *sc = fex->priv;
    SpeedChromaState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Order our stream after the framework's upload of these pictures. */
    CHECK_CUDA(cu_f, cuStreamWaitEvent(sc->str,
                                       vmaf_cuda_picture_get_ready_event(ref_pic), 0));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(sc->str,
                                       vmaf_cuda_picture_get_ready_event(dist_pic), 0));

    float score_u, score_v;
    int err_u = extract_channel_cuda(fex, sc, ref_pic, dist_pic, 1, &score_u);
    int err_v = extract_channel_cuda(fex, sc, ref_pic, dist_pic, 2, &score_v);

    /* Where exactly one channel had a singular covariance matrix, impute its
     * score from the other -- a better approximation than zero. Matches the
     * CPU extractor. */
    float score_uv;
    if (err_u && !err_v)      score_uv = score_v;
    else if (err_v && !err_u) score_uv = score_u;
    else                      score_uv = (score_u + score_v) / 2.0;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
        s->feature_name_dict, "Speed_chroma_feature_speed_chroma_u_score",
        MIN(score_u, s->speed_chroma_max_val), index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
        s->feature_name_dict, "Speed_chroma_feature_speed_chroma_v_score",
        MIN(score_v, s->speed_chroma_max_val), index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
        s->feature_name_dict, "Speed_chroma_feature_speed_chroma_uv_score",
        MIN(score_uv, s->speed_chroma_max_val), index);
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SpeedChromaStateCuda *sc = fex->priv;
    SpeedChromaState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;

    CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));
    CHECK_CUDA(cu_f, cuStreamDestroy(sc->str));

    VmafCudaBuffer *bufs[] = { sc->d_frame, sc->d_curr, sc->d_tmp,
                               sc->d_filt_aa, sc->d_filt_sc };
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (bufs[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, bufs[i]);
            free(bufs[i]);
        }
    }

    if (s->frame_buffer_ref) aligned_free(s->frame_buffer_ref);
    if (s->frame_buffer_dis) aligned_free(s->frame_buffer_dis);
    ret |= speed_close(&s->speed_state);
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_speed_chroma_cuda = {
    .name = "speed_chroma_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .close = close_fex_cuda,
    .options = speed_chroma_options,
    .priv_size = sizeof(SpeedChromaStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA | VMAF_FEATURE_EXTRACTOR_CHROMA,
};
