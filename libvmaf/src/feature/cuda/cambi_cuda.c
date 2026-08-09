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
#include "cambi.h"
#include "cuda/cambi_cuda.h"
#include "cuda_helper.cuh"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"

/*
 * CUDA CAMBI. Addresses the gap reported in #1567: the CUDA feature
 * extractors did not include CAMBI, so a VMAF v1 model could not be loaded
 * through libvmaf_cuda at all.
 *
 * SPLIT BETWEEN HOST AND DEVICE
 * -----------------------------
 * On device: the derivative map, the spatial mask, and per scale the
 * decimation, the mode filter and the c_values computation -- i.e. everything
 * that is per-pixel work over the whole frame.
 *
 * On host: setup (TVI/EOTF tables, contrast arrays), preprocessing, and the
 * spatial pooling.
 *
 * Pooling stays on the host deliberately rather than for convenience.
 * cambi_spatial_pooling() runs quick_select() and then sums the top-k *in
 * whatever order the partition happened to leave them*, accumulating into a
 * double. Float addition is not associative, so matching the CPU bit-for-bit
 * would mean reproducing quick_select's partition ordering -- which is not a
 * reasonable thing to do in parallel. The per-pixel window scan is where the
 * time goes; the reduction is one download per scale.
 *
 * KNOWN COST, NOT YET ADDRESSED
 * -----------------------------
 * cambi_preprocessing() (bitdepth conversion, generic decimation, anti
 * dithering) is host-only, so each picture makes a device->host->device round
 * trip before the pyramid starts. Moving preprocessing onto the device would
 * remove that and is the obvious next step.
 *
 * CambiState is the FIRST member of CambiStateCuda so that every
 * offsetof(CambiState, ...) in cambi_options[] stays valid and both
 * extractors share one option table.
 */

typedef struct CambiStateCuda {
    CambiState cpu;             /* MUST be first -- see cambi_options[] */

    CUstream str;
    CUevent event;

    CUfunction func_derivative;
    CUfunction func_decimate;
    CUfunction func_filter_h;
    CUfunction func_filter_v;
    CUfunction func_mask;
    CUfunction func_c_values;
    CUfunction func_pre_u8;
    CUfunction func_pre_u16;
    CUfunction func_antidither;

    VmafCudaBuffer *d_img[2];   /* ping-pong: decimate and filter_mode */
    VmafCudaBuffer *d_msk[2];
    VmafCudaBuffer *d_tmp;      /* filter_mode horizontal intermediate */
    VmafCudaBuffer *d_deriv;
    VmafCudaBuffer *d_cv;       /* float c_values */

    VmafCudaBuffer *d_tvi;
    VmafCudaBuffer *d_dw;
    VmafCudaBuffer *d_ad;
    VmafCudaBuffer *d_lut;
    VmafCudaBuffer *d_ori_x;   /* resample index tables, see the kernel */
    VmafCudaBuffer *d_ori_y;
    VmafCudaBuffer *d_ori_x_src;
    VmafCudaBuffer *d_ori_y_src;

    float *h_cv;                /* pinned, for the per-scale download */
    VmafPicture host_pic;       /* device picture downloaded for preprocessing */

    unsigned alloc_w, alloc_h;
} CambiStateCuda;

#define BLOCK_X 32
#define BLOCK_Y 8

/* Same as launch_2d but supplies dynamic shared memory. Block height is
 * reduced until the tile fits the device limit, which only matters for very
 * large windows. */
static int launch_2d_shared(CudaFunctions *cu_f, CUfunction f, CUstream str,
                            int width, int height, int pad_size, void **args)
{
    if (width <= 0 || height <= 0)
        return 0;

    unsigned by = BLOCK_Y;
    size_t shmem;
    for (;;) {
        shmem = (size_t)(BLOCK_X + 2 * pad_size)
              * (size_t)(by + 2 * pad_size) * sizeof(uint16_t);
        if (shmem <= 48000 || by == 1)
            break;
        by >>= 1;
    }

    CHECK_CUDA(cu_f, cuLaunchKernel(f,
                                    (width + BLOCK_X - 1) / BLOCK_X,
                                    (height + by - 1) / by, 1,
                                    BLOCK_X, by, 1,
                                    (unsigned) shmem, str, args, NULL));
    return 0;
}

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
    CambiStateCuda *sc = fex->priv;
    CambiState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;
    (void) pix_fmt;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    if (s->enc_bitdepth == 0)
        s->enc_bitdepth = bpc;
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->src_width == 0 || s->src_height == 0) {
        s->src_width = w;
        s->src_height = h;
    }
    if (s->enc_height > h || s->enc_width > w) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->enc_width < CAMBI_MIN_WIDTH_HEIGHT &&
        s->enc_height < CAMBI_MIN_WIDTH_HEIGHT)
        return -EINVAL;

    /* Same resolution gating as the CPU extractor: the option only takes
     * effect above its per-tier pixel threshold. */
    {
        const int enc_pix = s->enc_width * s->enc_height;
        switch (s->cambi_high_res_speedup) {
        case 1080:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p)
                s->cambi_high_res_speedup = 0;
            break;
        case 1440:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p)
                s->cambi_high_res_speedup = 0;
            break;
        case 2160:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p)
                s->cambi_high_res_speedup = 0;
            break;
        default:
            s->cambi_high_res_speedup = 0;
        }
    }

    cambi_adjust_window_size(&s->window_size, s->enc_width, s->enc_height,
                             (bool) s->cambi_high_res_speedup);
    cambi_adjust_window_size(&s->src_window_size, s->src_width, s->src_height,
                             (bool) s->cambi_high_res_speedup);

    const unsigned alloc_w = s->enc_width;
    const unsigned alloc_h = s->enc_height;
    sc->alloc_w = alloc_w;
    sc->alloc_h = alloc_h;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_alloc(&s->pics[i], VMAF_PIX_FMT_YUV400P, 10,
                                  alloc_w, alloc_h);
    if (err) return err;

    const uint16_t num_diffs = 1 << s->max_log_contrast;
    err = cambi_set_contrast_arrays(num_diffs, &s->buffers.diffs_to_consider,
                                    &s->buffers.diff_weights,
                                    &s->buffers.all_diffs);
    if (err) return err;

    VmafLumaRange range;
    /* TVI thresholds live in the PREPROCESSED 10-bit domain, not the
     * encode bitdepth -- cambi.c hardcodes 10 here for the same reason. */
    err = vmaf_luminance_init_luma_range(&range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err) return err;
    VmafEOTF eotf;
    const char *effective_eotf =
        (strcmp(s->cambi_eotf, DEFAULT_CAMBI_EOTF) != 0) ? s->cambi_eotf
                                                         : s->eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err) return err;

    s->buffers.tvi_for_diff = aligned_malloc(ALIGN_CEIL(num_diffs * sizeof(uint16_t)), 32);
    if (!s->buffers.tvi_for_diff) return -ENOMEM;
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.tvi_for_diff[d] =
            cambi_get_tvi_for_diff(s->buffers.diffs_to_consider[d],
                                   s->tvi_threshold, 10, range, eotf);
        s->buffers.tvi_for_diff[d] += num_diffs;
    }
    s->vlt_luma = cambi_get_vlt_luma(s->cambi_vis_lum_threshold, range, eotf);

    /* --- CUDA resources ------------------------------------------------- */
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&sc->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&sc->event, CU_EVENT_DEFAULT));

    CUmodule m_deriv, m_dec, m_filt, m_mask, m_cv;
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_deriv, cambi_derivative_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_dec, cambi_decimate_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_filt, cambi_filter_mode_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_mask, cambi_spatial_mask_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_cv, cambi_c_values_ptx));
    CUmodule m_pre;
    CHECK_CUDA(cu_f, cuModuleLoadData(&m_pre, cambi_preprocess_ptx));

    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_derivative, m_deriv,
                                         "cambi_derivative_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_decimate, m_dec,
                                         "cambi_decimate_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_filter_h, m_filt,
                                         "cambi_filter_mode_h_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_filter_v, m_filt,
                                         "cambi_filter_mode_v_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_mask, m_mask,
                                         "cambi_spatial_mask_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_c_values, m_cv,
                                         "cambi_c_values_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_pre_u8, m_pre,
                                         "cambi_preprocess_u8_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_pre_u16, m_pre,
                                         "cambi_preprocess_u16_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&sc->func_antidither, m_pre,
                                         "cambi_antidither_kernel"));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    /* The transfer helpers move the buffer's ENTIRE allocation, so the
     * plane buffers must match the picture's padded stride exactly. */
    const size_t plane_bytes = (size_t) s->pics[0].stride[0] * alloc_h;
    const size_t px = (size_t) alloc_w * alloc_h;
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_img[0], plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_img[1], plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_msk[0], plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_msk[1], plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_tmp,    plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_deriv,  plane_bytes);
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_cv,     px * sizeof(float));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_tvi, num_diffs * sizeof(uint16_t));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_dw,  num_diffs * sizeof(int));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_ad,
                                  (2 * num_diffs + 1) * sizeof(int));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_lut,
                                  CAMBI_RECIPROCAL_LUT_SIZE * sizeof(float));
    if (err) return err;

    /* Resample index tables. The CPU derives these from an ACCUMULATED float
     * (x += ratio_x), so computing start + j*ratio arithmetically rounds
     * differently and would not be bit-exact. Run the same accumulation here
     * and let the kernel gather. When the sizes already match, the CPU takes
     * a same-size fast path, which the identity mapping reproduces exactly. */
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_ori_x, alloc_w * sizeof(int));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_ori_y, alloc_h * sizeof(int));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_ori_x_src, alloc_w * sizeof(int));
    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &sc->d_ori_y_src, alloc_h * sizeof(int));
    if (err) return err;
    {
        int *tbl = malloc(MAX(alloc_w, alloc_h) * sizeof(int));
        if (!tbl) return -ENOMEM;

        /* one axis of one configuration */
        #define CAMBI_FILL_TABLE(in_n, out_n)                                 \
            do {                                                              \
                if ((unsigned)(in_n) == (unsigned)(out_n)) {                  \
                    for (unsigned k_ = 0; k_ < (unsigned)(out_n); k_++)       \
                        tbl[k_] = (int) k_;                                   \
                } else {                                                      \
                    const float ratio_ = (float)(in_n) / (out_n);             \
                    float v_ = ratio_ / 2 - 0.5f;                             \
                    for (unsigned k_ = 0; k_ < (unsigned)(out_n); k_++) {     \
                        tbl[k_] = (int)(v_ + 0.5f);                           \
                        v_ += ratio_;                                         \
                    }                                                         \
                }                                                             \
            } while (0)

        CAMBI_FILL_TABLE(w, s->enc_width);
        CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_ori_x->data, tbl,
                                           s->enc_width * sizeof(int), sc->str));
        CAMBI_FILL_TABLE(h, s->enc_height);
        CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_ori_y->data, tbl,
                                           s->enc_height * sizeof(int), sc->str));
        CAMBI_FILL_TABLE(w, s->src_width);
        CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_ori_x_src->data, tbl,
                                           s->src_width * sizeof(int), sc->str));
        CAMBI_FILL_TABLE(h, s->src_height);
        CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_ori_y_src->data, tbl,
                                           s->src_height * sizeof(int), sc->str));
        #undef CAMBI_FILL_TABLE

        CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));
        free(tbl);
    }

    err |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **) &sc->h_cv,
                                       px * sizeof(float));
    if (err) return err;

    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_tvi->data, s->buffers.tvi_for_diff,
                                       sc->d_tvi->size, sc->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_dw->data, s->buffers.diff_weights,
                                       sc->d_dw->size, sc->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_ad->data, s->buffers.all_diffs,
                                       sc->d_ad->size, sc->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(sc->d_lut->data, (void *) reciprocal_lut,
                                       sc->d_lut->size, sc->str));
    if (err) return err;
    CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));

    err = vmaf_picture_alloc(&sc->host_pic, VMAF_PIX_FMT_YUV400P,
                             s->enc_bitdepth, alloc_w, alloc_h);
    return err;
}

/* Runs the pyramid for one already-preprocessed picture sitting in
 * s->pics[0], and returns the pooled score. Mirrors cambi_score(). */
static int cambi_score_cuda(VmafFeatureExtractor *fex, CambiStateCuda *sc,
                            int width, int height, uint16_t window_size,
                            double topk, double *score)
{
    CambiState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;
    const uint16_t num_diffs = 1 << s->max_log_contrast;
    const int pad_mask = MASK_FILTER_SIZE >> 1;
    double scores_per_scale[NUM_SCALES];

    /* Preprocessing already left the 10-bit image in d_img[0], packed. */
    const ptrdiff_t src_stride = sc->alloc_w;

    int err = 0;
    if (err) return err;

    int ii = 0, mi = 0;
    ptrdiff_t img_stride = src_stride;
    ptrdiff_t msk_stride = width;

    /* --- spatial mask, once, at full resolution ------------------------- */
    {
        int w = width, h = height;
        ptrdiff_t ds = width;
        void *a1[] = { &sc->d_img[ii]->data, &sc->d_deriv->data,
                       &w, &h, &img_stride, &ds };
        err = launch_2d(cu_f, sc->func_derivative, sc->str, w, h, a1);
        if (err) return err;

        unsigned mask_index = cambi_get_mask_index(width, height,
                                                   MASK_FILTER_SIZE);
        int pad = pad_mask;
        void *a2[] = { &sc->d_deriv->data, &sc->d_msk[mi]->data,
                       &w, &h, &pad, &mask_index, &ds, &msk_stride };
        err = launch_2d(cu_f, sc->func_mask, sc->str, w, h, a2);
        if (err) return err;
    }

    int sw = width, sh = height;

    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || s->cambi_high_res_speedup) {
            const int nw = (sw + 1) >> 1, nh = (sh + 1) >> 1;
            ptrdiff_t dst_stride = nw;

            void *ai[] = { &sc->d_img[ii]->data, &sc->d_img[1 - ii]->data,
                           (void *) &nw, (void *) &nh,
                           &img_stride, &dst_stride };
            err = launch_2d(cu_f, sc->func_decimate, sc->str, nw, nh, ai);
            if (err) return err;

            void *am[] = { &sc->d_msk[mi]->data, &sc->d_msk[1 - mi]->data,
                           (void *) &nw, (void *) &nh,
                           &msk_stride, &dst_stride };
            err = launch_2d(cu_f, sc->func_decimate, sc->str, nw, nh, am);
            if (err) return err;

            ii = 1 - ii;
            mi = 1 - mi;
            img_stride = dst_stride;
            msk_stride = dst_stride;
            sw = nw;
            sh = nh;
        }

        /* --- mode filter, horizontal then vertical ---------------------- */
        {
            ptrdiff_t ts = sw;
            void *ah[] = { &sc->d_img[ii]->data, &sc->d_tmp->data,
                           &sw, &sh, &img_stride, &ts };
            err = launch_2d(cu_f, sc->func_filter_h, sc->str, sw, sh, ah);
            if (err) return err;

            ptrdiff_t os = sw;
            void *av[] = { &sc->d_img[ii]->data, &sc->d_tmp->data,
                           &sc->d_img[1 - ii]->data, &sw, &sh,
                           &img_stride, &ts, &os };
            err = launch_2d(cu_f, sc->func_filter_v, sc->str, sw, sh, av);
            if (err) return err;

            ii = 1 - ii;
            img_stride = os;
        }

        /* --- c_values --------------------------------------------------- */
        {
            const int pad = window_size >> 1;
            int nd = num_diffs;
            unsigned vbb = s->buffers.v_band_base;
            unsigned vbs = s->buffers.v_band_size;
            int vlt = s->vlt_luma;

            int v_lo = (int) s->vlt_luma - 3 * (int) num_diffs + 1;
            vbb = v_lo > 0 ? (unsigned) v_lo : 0u;
            vbs = (unsigned) (s->buffers.tvi_for_diff[num_diffs - 1] + 1 - vbb);

            void *ac[] = { &sc->d_img[ii]->data, &sc->d_msk[mi]->data,
                           &sc->d_cv->data, &sw, &sh, &img_stride,
                           (void *) &pad, &nd, &vbb, &vbs, &vlt,
                           &sc->d_tvi->data, &sc->d_dw->data,
                           &sc->d_ad->data, &sc->d_lut->data };
            err = launch_2d_shared(cu_f, sc->func_c_values, sc->str, sw, sh, pad, ac);
            if (err) return err;

        }

        /* The buffer helper transfers the whole allocation and did not
         * deliver the kernel's output intact here; copy exactly the bytes
         * this scale produced. */
        CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(sc->h_cv, sc->d_cv->data,
                                           (size_t) sw * sh * sizeof(float),
                                           sc->str));
        CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));

        scores_per_scale[scale] =
            cambi_spatial_pooling(sc->h_cv, topk, sw, sh);
    }

    *score = cambi_weight_scores_per_scale(scores_per_scale,
                                           cambi_get_pixels_in_window(window_size));
    return 0;
}

static int preprocess_and_extract_cuda(VmafFeatureExtractor *fex,
                                       CambiStateCuda *sc, VmafPicture *pic,
                                       double *score, bool is_src)
{
    CambiState *s = &sc->cpu;
    const int width = is_src ? s->src_width : s->enc_width;
    const int height = is_src ? s->src_height : s->enc_height;
    const uint16_t window_size = is_src ? s->src_window_size : s->window_size;

    /* Preprocessing is host-only; bring the picture back for it. */
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Order our stream after the framework's upload of this picture. */
    CHECK_CUDA(cu_f, cuStreamWaitEvent(sc->str,
                                       vmaf_cuda_picture_get_ready_event(pic), 0));

    /* Preprocess on the device: decimate + convert to 10 bit, then the
     * anti-dithering filter when the encode bitdepth is below 10. */
    {
        VmafCudaBuffer *tx = is_src ? sc->d_ori_x_src : sc->d_ori_x;
        VmafCudaBuffer *ty = is_src ? sc->d_ori_y_src : sc->d_ori_y;
        const ptrdiff_t dst_stride = sc->alloc_w;
        int w = width, h = height;
        ptrdiff_t ds = dst_stride;
        int err2;

        if (pic->bpc <= 8) {
            ptrdiff_t ss = pic->stride[0];
            int shl = 10 - (int) pic->bpc;
            void *a[] = { &pic->data[0], &sc->d_img[1]->data, &w, &h, &ss, &ds,
                          &shl, &tx->data, &ty->data };
            err2 = launch_2d(cu_f, sc->func_pre_u8, sc->str, w, h, a);
        } else {
            ptrdiff_t ss = pic->stride[0] >> 1;
            int shift = (pic->bpc >= 10) ? ((int) pic->bpc - 10)
                                         : -(10 - (int) pic->bpc);
            int rounding = (shift > 0) ? (1 << (shift - 1)) : 0;
            void *a[] = { &pic->data[0], &sc->d_img[1]->data, &w, &h, &ss, &ds,
                          &shift, &rounding, &tx->data, &ty->data };
            err2 = launch_2d(cu_f, sc->func_pre_u16, sc->str, w, h, a);
        }
        if (err2) return err2;

        if (s->enc_bitdepth < 10) {
            void *a[] = { &sc->d_img[1]->data, &sc->d_img[0]->data,
                          &w, &h, &ds, &ds };
            err2 = launch_2d(cu_f, sc->func_antidither, sc->str, w, h, a);
            if (err2) return err2;
        } else {
            CHECK_CUDA(cu_f, cuMemcpyDtoDAsync(sc->d_img[0]->data,
                                               sc->d_img[1]->data,
                                               (size_t) dst_stride * h * sizeof(uint16_t),
                                               sc->str));
        }
    }

    const double topk = (s->topk != DEFAULT_CAMBI_TOPK_POOLING)
                      ? s->topk : s->cambi_topk;

    return cambi_score_cuda(fex, sc, width, height, window_size, topk, score);
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                            VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                            VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                            unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    CambiStateCuda *sc = fex->priv;
    CambiState *s = &sc->cpu;
    (void) ref_pic_90;
    (void) dist_pic_90;

    double dist_score;
    int err = preprocess_and_extract_cuda(fex, sc, dist_pic, &dist_score, false);
    if (err) return err;

    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "Cambi_feature_cambi_score",
        MIN(dist_score, s->cambi_max_val), index);
    if (err) return err;

    if (s->full_ref) {
        double src_score;
        err = preprocess_and_extract_cuda(fex, sc, ref_pic, &src_score, true);
        if (err) return err;

        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_source",
            MIN(src_score, s->cambi_max_val), index);
        if (err) return err;

        const double combined = cambi_combine_dist_src_scores(dist_score, src_score);
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_full_reference",
            MIN(combined, s->cambi_max_val), index);
        if (err) return err;
    }

    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    CambiStateCuda *sc = fex->priv;
    CambiState *s = &sc->cpu;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;

    CHECK_CUDA(cu_f, cuStreamSynchronize(sc->str));
    CHECK_CUDA(cu_f, cuEventDestroy(sc->event));
    CHECK_CUDA(cu_f, cuStreamDestroy(sc->str));

    VmafCudaBuffer *bufs[] = {
        sc->d_img[0], sc->d_img[1], sc->d_msk[0], sc->d_msk[1], sc->d_tmp,
        sc->d_deriv, sc->d_cv, sc->d_tvi, sc->d_dw, sc->d_ad, sc->d_lut,
        sc->d_ori_x, sc->d_ori_y, sc->d_ori_x_src, sc->d_ori_y_src,
    };
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (bufs[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, bufs[i]);
            free(bufs[i]);
        }
    }

    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++)
        ret |= vmaf_picture_unref(&s->pics[i]);
    ret |= vmaf_picture_unref(&sc->host_pic);

    if (s->buffers.tvi_for_diff)      aligned_free(s->buffers.tvi_for_diff);
    if (s->buffers.diffs_to_consider) aligned_free(s->buffers.diffs_to_consider);
    if (s->buffers.diff_weights)      aligned_free(s->buffers.diff_weights);
    if (s->buffers.all_diffs)         aligned_free(s->buffers.all_diffs);

    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {
    "Cambi_feature_cambi_score", "cambi_source", "cambi_full_reference",
    NULL
};

VmafFeatureExtractor vmaf_fex_cambi_cuda = {
    .name = "cambi_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .close = close_fex_cuda,
    .options = cambi_options,
    .priv_size = sizeof(CambiStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
