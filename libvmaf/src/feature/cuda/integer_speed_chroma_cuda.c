/**
 *
 *  Copyright 2026 luoye-cpu
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

/*
 * CUDA SpEED chroma feature extractor (speed_chroma_cuda).
 *
 * The kernels live in cuda/integer_speed_chroma/speed_chroma.cu, which is
 * compiled to a fatbin and embedded as `speed_chroma_ptx` (bin2c). This file
 * owns the whole pipeline and loads the module through cuModuleLoadData().
 *
 * Numerical parity with src/feature/speed.c is the contract:
 *
 *  - The device kernels mirror the CPU loops, the accumulation order and the
 *    operand types (float vs double) of compute_mean(), compute_cov_kernel_*,
 *    compute_independent_term(), compute_eigenvalues(),
 *    solve_linear_system() and compute_pointwise_product_and_division().
 *    See speed_chroma.cu for the details.
 *
 *  - Everything the CPU computes with the platform libm is computed *here*,
 *    with the very same libm: the gaussian filter taps (vif_get_filter /
 *    speed_get_antialias_filter), the entropy update (log2) and
 *    get_speed_score() are verbatim copies of the CPU code. The kernel
 *    results that feed them (eigenvalues, S) and the error flag are copied
 *    back to the host, which then produces the per-frame scores exactly like
 *    extract_chroma() does.
 *
 *  - The prescale step of filter_and_downscale() is ported for the nearest
 *    and bilinear scaling methods (speed_chroma_scale_*_kernel); bicubic and
 *    lanczos4 return -EINVAL from init_fex_cuda. speed_prescale == 1.0 skips
 *    the step, exactly like the CPU.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"
#include "vif_tools.h"

/*
 * The SpEED fatbin, embedded by bin2c (see src/meson.build, cuda_cu_sources /
 * ptx_bin2c). cuda/integer_speed_chroma/speed_chroma.cu contains device code
 * only.
 */
extern const unsigned char speed_chroma_ptx[];

/* Device-side constants must match speed_chroma.cu. */
#define SPEED_BLOCK_SIZE (5)
#define SPEED_ELEMENTS_IN_BLOCK (SPEED_BLOCK_SIZE * SPEED_BLOCK_SIZE)
#define SPEED_NUM_SCALES (4)
#define SPEED_THREADS_X (16)
#define SPEED_THREADS_Y (16)
#define SPEED_SCAN_THREADS (256)
/* The QR kernel is one cooperative block; it uses ~70 registers per thread, so
 * 512 threads is the largest block that is always resident. */
#define SPEED_QR_THREADS (512)
#define SPEED_MAX_KERNEL_TAPS (128)

/* Defaults and constants copied from speed.c. */
#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")
#define ALMOST_EQUAL(x, c) (fabs((x) - (c)) < 1.0e-3)

typedef struct SpeedChromaStateCuda {
    /* Feature options; the offsets in `options` below point into this struct. */
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;

    /* Mirrors SpeedDimensions in speed.c (chroma dimensions). */
    int original_w, original_h;
    int scaled_w, scaled_h;
    int alloc_w, alloc_h;
    int operating_w, operating_h;
    int truncated_w, truncated_h;
    int num_blocks_horizontal, num_blocks_vertical, num_blocks;
    int submatrix_w, submatrix_h;
    size_t float_stride; /* bytes, ALIGN_CEIL(alloc_w * sizeof(float)) */
    int stride_px;       /* floats */

    VmafCudaState *cu_state;
    VmafDictionary *feature_name_dict;

    CUmodule module;
    CUfunction fn_copy8, fn_copy16, fn_filter_vert, fn_filter_horiz, fn_dec16,
               fn_subtract, fn_means, fn_cov, fn_eigen, fn_independent, fn_qr,
               fn_solve, fn_pointwise, fn_sum_columns, fn_scale_nearest,
               fn_scale_bilinear;

    /* Set in init when speed_prescale != 1.0 (see vif_get_scaling_method). */
    enum vif_scaling_method prescale_method;

    /* Device buffers. */
    VmafCudaBuffer *frame[2];
    VmafCudaBuffer *curr_scale;
    VmafCudaBuffer *tmpbuf;
    VmafCudaBuffer *means;
    VmafCudaBuffer *cov_mat;
    VmafCudaBuffer *eigenvalues;
    VmafCudaBuffer *independent_term;
    VmafCudaBuffer *linear_system_sol;
    VmafCudaBuffer *matrix_r;
    VmafCudaBuffer *tmp_rect;
    VmafCudaBuffer *sum_s;
    VmafCudaBuffer *err_flag;
    VmafCudaBuffer *filter_antialias;
    VmafCudaBuffer *filter_dec;

    /* Pinned host readback, one slot per picture (0 = ref, 1 = dis). */
    float *eig_host;
    float *S_host;
    int *err_host;

    /* Host side results (see update_entropy()/get_speed_score() in speed.c). */
    float *entropies[2];
    float *variances[2];
} SpeedChromaStateCuda;

static const VmafOption options[] = {
    {
        .name = "speed_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ks",
    },
    {
        .name = "speed_prescale",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ps",
    },
    {
        .name = "speed_prescale_method",
        .help = "scaling method for the frame, supported options: "
                 "[nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "psm",
    },
    {
        .name = "speed_sigma_nn",
        .help = "standard deviation of neural noise",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_sigma_nn),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_SIGMA_NN,
        .min = 0.1,
        .max = 2.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snn",
    },
    {
        .name = "speed_nn_floor",
        .help = "neural noise floor, expressed in percentage of sigma_nn",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_nn_floor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_NN_FLOOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "nnf",
    },
    {
        .name = "speed_max_val",
        .help = "maximum value allowed; "
                "larger values will be clipped to this value",
        .offset = offsetof(SpeedChromaStateCuda, speed_chroma_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mxv",
    },
    {
        .name = "speed_weight_var_mode",
        .help = "different approaches to perform variance-absed weighting",
        .offset = offsetof(SpeedChromaStateCuda, speed_weight_var_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.d = 0,
        .min = 0,
        .max = 6,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "wvm",
    },
    { 0 }
};

static const char *provided_features[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};

/* ------------------------------------------------------------------------- */
/* Module setup                                                              */
/* ------------------------------------------------------------------------- */

static int speed_chroma_load_module(SpeedChromaStateCuda *s)
{
    CudaFunctions *cu_f = s->cu_state->f;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(s->cu_state->ctx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&s->module, speed_chroma_ptx));

#define SPEED_GET_FN(field, symbol)                                             \
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->field, s->module, symbol))

    SPEED_GET_FN(fn_copy8, "speed_chroma_copy8_kernel");
    SPEED_GET_FN(fn_copy16, "speed_chroma_copy16_kernel");
    SPEED_GET_FN(fn_filter_vert, "speed_chroma_filter1d_vert_kernel");
    SPEED_GET_FN(fn_filter_horiz, "speed_chroma_filter1d_horiz_kernel");
    SPEED_GET_FN(fn_dec16, "speed_chroma_dec16_kernel");
    SPEED_GET_FN(fn_subtract, "speed_chroma_subtract_kernel");
    SPEED_GET_FN(fn_scale_nearest, "speed_chroma_scale_nearest_kernel");
    SPEED_GET_FN(fn_scale_bilinear, "speed_chroma_scale_bilinear_kernel");
    SPEED_GET_FN(fn_means, "speed_chroma_means_kernel");
    SPEED_GET_FN(fn_cov, "speed_chroma_cov_kernel");
    SPEED_GET_FN(fn_eigen, "speed_chroma_eigen_kernel");
    SPEED_GET_FN(fn_independent, "speed_chroma_independent_term_kernel");
    SPEED_GET_FN(fn_qr, "speed_chroma_qr_kernel");
    SPEED_GET_FN(fn_solve, "speed_chroma_solve_kernel");
    SPEED_GET_FN(fn_pointwise, "speed_chroma_pointwise_kernel");
    SPEED_GET_FN(fn_sum_columns, "speed_chroma_sum_columns_kernel");

#undef SPEED_GET_FN

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Buffers                                                                   */
/* ------------------------------------------------------------------------- */

static void speed_chroma_free_buffer(SpeedChromaStateCuda *s,
                                     VmafCudaBuffer **buf)
{
    if (*buf) {
        vmaf_cuda_buffer_free(s->cu_state, *buf);
        free(*buf);
        *buf = NULL;
    }
}

static void speed_chroma_free_buffers(SpeedChromaStateCuda *s)
{
    speed_chroma_free_buffer(s, &s->frame[0]);
    speed_chroma_free_buffer(s, &s->frame[1]);
    speed_chroma_free_buffer(s, &s->curr_scale);
    speed_chroma_free_buffer(s, &s->tmpbuf);
    speed_chroma_free_buffer(s, &s->means);
    speed_chroma_free_buffer(s, &s->cov_mat);
    speed_chroma_free_buffer(s, &s->eigenvalues);
    speed_chroma_free_buffer(s, &s->independent_term);
    speed_chroma_free_buffer(s, &s->linear_system_sol);
    speed_chroma_free_buffer(s, &s->matrix_r);
    speed_chroma_free_buffer(s, &s->tmp_rect);
    speed_chroma_free_buffer(s, &s->sum_s);
    speed_chroma_free_buffer(s, &s->err_flag);
    speed_chroma_free_buffer(s, &s->filter_antialias);
    speed_chroma_free_buffer(s, &s->filter_dec);
}

static void speed_chroma_free_host_buffers(SpeedChromaStateCuda *s)
{
    if (s->eig_host) vmaf_cuda_buffer_host_free(s->cu_state, s->eig_host);
    if (s->S_host) vmaf_cuda_buffer_host_free(s->cu_state, s->S_host);
    if (s->err_host) vmaf_cuda_buffer_host_free(s->cu_state, s->err_host);
    s->eig_host = NULL;
    s->S_host = NULL;
    s->err_host = NULL;

    for (int i = 0; i < 2; i++) {
        if (s->entropies[i]) free(s->entropies[i]);
        if (s->variances[i]) free(s->variances[i]);
        s->entropies[i] = NULL;
        s->variances[i] = NULL;
    }
}

/* Device buffers are never 0 bytes: the degenerate geometries below (num_blocks
 * == 0, empty frame) still launch the kernels that write into them. */
static int speed_chroma_device_alloc(SpeedChromaStateCuda *s,
                                     VmafCudaBuffer **buf, size_t bytes)
{
    if (bytes < sizeof(float)) bytes = sizeof(float);
    return vmaf_cuda_buffer_alloc(s->cu_state, buf, bytes);
}

/* ------------------------------------------------------------------------- */
/* Kernel launchers                                                          */
/* ------------------------------------------------------------------------- */

/* picture_copy(): ref/dis chroma plane -> float frame buffer. */
static int speed_chroma_copy_plane(SpeedChromaStateCuda *s, VmafPicture *pic,
                                   int channel, CUdeviceptr dst, CUstream str)
{
    CudaFunctions *cu_f = s->cu_state->f;
    const unsigned bpc = pic->bpc;
    const float offset = -128.0f;
    const int dst_stride = s->stride_px;

    int w = (int)pic->w[channel];
    int h = (int)pic->h[channel];
    if (w > s->original_w) w = s->original_w;
    if (h > s->original_h) h = s->original_h;
    if (w <= 0 || h <= 0) return 0;

    const unsigned grid_x = DIV_ROUND_UP(w, SPEED_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, SPEED_THREADS_Y);

    if (bpc == 10 || bpc == 12 || bpc == 16) {
        const float scaler = bpc == 10 ? 4.0f : (bpc == 12 ? 16.0f : 256.0f);
        const uint16_t *src = (const uint16_t *)(uintptr_t)pic->data[channel];
        const int src_stride = (int)(pic->stride[channel] / 2);
        void *kernel_params[] = { &src, &src_stride, &dst, &dst_stride,
                                  &w, &h, &scaler, &offset };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_copy16, grid_x, grid_y, 1,
                                        SPEED_THREADS_X, SPEED_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    } else {
        const uint8_t *src = (const uint8_t *)(uintptr_t)pic->data[channel];
        const int src_stride = (int)pic->stride[channel];
        void *kernel_params[] = { &src, &src_stride, &dst, &dst_stride,
                                  &w, &h, &offset };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_copy8, grid_x, grid_y, 1,
                                        SPEED_THREADS_X, SPEED_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    }
    return 0;
}

/* vif_filter1d_s(): vertical pass into the packed w x h scratch, then the
 * horizontal pass into dst. */
static int speed_chroma_filter1d(SpeedChromaStateCuda *s, CUfunction vert,
                                 CUfunction horiz, CUdeviceptr f,
                                 CUdeviceptr src, CUdeviceptr dst,
                                 CUdeviceptr tmp, int w, int h, int fwidth,
                                 CUstream str)
{
    CudaFunctions *cu_f = s->cu_state->f;
    if (w <= 0 || h <= 0 || fwidth <= 0) return 0;

    const int stride = s->stride_px;
    const unsigned grid_x = DIV_ROUND_UP(w, SPEED_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, SPEED_THREADS_Y);
    {
        void *kernel_params[] = { &f, &src, &tmp, &w, &h, &stride, &fwidth };
        CHECK_CUDA(cu_f, cuLaunchKernel(vert, grid_x, grid_y, 1,
                                        SPEED_THREADS_X, SPEED_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    }
    {
        void *kernel_params[] = { &f, &tmp, &dst, &w, &h, &stride, &fwidth };
        CHECK_CUDA(cu_f, cuLaunchKernel(horiz, grid_x, grid_y, 1,
                                        SPEED_THREADS_X, SPEED_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    }
    return 0;
}

/*
 * filter_and_downscale(): prescale (vif_scale_frame_s()), antialias filter,
 * decimation by 16, decimated filter and local mean subtraction, all on the
 * frame buffer in place.
 */
static int speed_chroma_filter_and_downscale(SpeedChromaStateCuda *s, int which,
                                             CUstream str)
{
    CudaFunctions *cu_f = s->cu_state->f;
    CUdeviceptr frame = s->frame[which]->data;
    CUdeviceptr curr_scale = s->curr_scale->data;
    CUdeviceptr tmpbuf = s->tmpbuf->data;

    /* if (!ALMOST_EQUAL(opt->speed_prescale, 1.0)) {
     *     memcpy(tmpbuf, frame_buffer,
     *            stride_px * dim.alloc_height * sizeof(float));
     *     vif_scale_frame_s(scaling_method, tmpbuf, frame_buffer,
     *                       dim.original_width, dim.original_height, stride_px,
     *                       dim.scaled_width, dim.scaled_height, stride_px);
     * }
     * vif_scale_frame_s() copies the frame when the dimensions match, which
     * leaves frame_buffer untouched, so the kernel launch is skipped there. */
    if (!ALMOST_EQUAL(s->speed_chroma_prescale, 1.0) &&
        (s->scaled_w != s->original_w || s->scaled_h != s->original_h)) {
        const size_t frame_bytes = s->float_stride * (size_t)s->alloc_h;
        CHECK_CUDA(cu_f, cuMemcpyDtoDAsync(tmpbuf, frame, frame_bytes, str));

        const int src_w = s->original_w, src_h = s->original_h;
        const int dst_w = s->scaled_w, dst_h = s->scaled_h;
        const int stride = s->stride_px;
        const unsigned grid_x = DIV_ROUND_UP(dst_w, SPEED_THREADS_X);
        const unsigned grid_y = DIV_ROUND_UP(dst_h, SPEED_THREADS_Y);
        void *kernel_params[] = { &tmpbuf, &frame, &src_w, &src_h, &stride,
                                  &dst_w, &dst_h, &stride };
        CUfunction scale = s->prescale_method == vif_scale_bilinear
                               ? s->fn_scale_bilinear
                               : s->fn_scale_nearest;
        CHECK_CUDA(cu_f, cuLaunchKernel(scale, grid_x, grid_y, 1,
                                        SPEED_THREADS_X, SPEED_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    }

    const float kernelscale = (float)s->speed_chroma_kernelscale;

    /* int filter_width_antialias = vif_get_filter_size(1, opt->speed_kernelscale);
     * float filter_antialias[128];
     * speed_get_antialias_filter(filter_antialias, NUM_SCALES, opt->speed_kernelscale);
     */
    const int fwidth_aa = vif_get_filter_size(1, kernelscale);
    float filter_antialias[SPEED_MAX_KERNEL_TAPS];
    speed_get_antialias_filter(filter_antialias, SPEED_NUM_SCALES, kernelscale);
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->filter_antialias->data,
                                       filter_antialias,
                                       sizeof(float) * fwidth_aa, str));

    int ret = speed_chroma_filter1d(s, s->fn_filter_vert, s->fn_filter_horiz,
                                    s->filter_antialias->data, frame,
                                    curr_scale, tmpbuf, s->scaled_w,
                                    s->scaled_h, fwidth_aa, str);
    if (ret) return ret;

    /* vif_dec16_s(curr_scale, frame_buffer, scaled_width, scaled_height, ...) */
    {
        const int dw = s->scaled_w >> SPEED_NUM_SCALES;
        const int dh = s->scaled_h >> SPEED_NUM_SCALES;
        if (s->scaled_w / 16 > 0 && s->scaled_h / 16 > 0) {
            const int src_w = s->scaled_w, src_h = s->scaled_h;
            const int stride = s->stride_px;
            const unsigned grid_x = DIV_ROUND_UP(s->scaled_w / 16,
                                                 SPEED_THREADS_X);
            const unsigned grid_y = DIV_ROUND_UP(s->scaled_h / 16,
                                                 SPEED_THREADS_Y);
            void *kernel_params[] = { &curr_scale, &frame, &src_w, &src_h,
                                      &stride, &stride };
            CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_dec16, grid_x, grid_y, 1,
                                            SPEED_THREADS_X, SPEED_THREADS_Y,
                                            1, 0, str, kernel_params, NULL));
        }

        /* int filter_width = vif_get_filter_size(NUM_SCALES, opt->speed_kernelscale);
         * float filter[128];
         * vif_get_filter(filter, NUM_SCALES, opt->speed_kernelscale); */
        const int fwidth = vif_get_filter_size(SPEED_NUM_SCALES, kernelscale);
        float filter[SPEED_MAX_KERNEL_TAPS];
        vif_get_filter(filter, SPEED_NUM_SCALES, kernelscale);
        CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->filter_dec->data, filter,
                                           sizeof(float) * fwidth, str));

        ret = speed_chroma_filter1d(s, s->fn_filter_vert, s->fn_filter_horiz,
                                    s->filter_dec->data, frame, curr_scale,
                                    tmpbuf, dw, dh, fwidth, str);
        if (ret) return ret;

        /* subtract_image(frame_buffer, curr_scale, downscaled_w, downscaled_h, ...) */
        if (dw > 0 && dh > 0) {
            const int stride = s->stride_px;
            const unsigned grid_x = DIV_ROUND_UP(dw, SPEED_THREADS_X);
            const unsigned grid_y = DIV_ROUND_UP(dh, SPEED_THREADS_Y);
            void *kernel_params[] = { &frame, &curr_scale, &dw, &dh, &stride };
            CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_subtract, grid_x, grid_y, 1,
                                            SPEED_THREADS_X, SPEED_THREADS_Y, 1,
                                            0, str, kernel_params, NULL));
        }
    }

    return 0;
}

/*
 * est_params(): covariance matrix, eigenvalues, independent term, linear
 * system, pointwise product and sum of columns. The eigenvalues, S and the
 * error flag are read back into `slot` for the host side entropy update.
 */
static int speed_chroma_est_params(SpeedChromaStateCuda *s, int which, int slot,
                                   CUstream str)
{
    CudaFunctions *cu_f = s->cu_state->f;
    const int nb = s->num_blocks;
    const int total = nb * SPEED_ELEMENTS_IN_BLOCK;
    const int stride = s->stride_px;
    const int block_size = SPEED_BLOCK_SIZE;
    const int elements_in_block = SPEED_ELEMENTS_IN_BLOCK;
    CUdeviceptr data = s->frame[which]->data;

    /* Step 1: covariance matrix (means are stored in the eigenvalues buffer on
     * the CPU; here they live in their own buffer, which is equivalent). */
    {
        const int sub_w = s->submatrix_w, sub_h = s->submatrix_h;
        const unsigned grid = DIV_ROUND_UP(elements_in_block,
                                           SPEED_SCAN_THREADS);
        void *kernel_params[] = { &data, &s->means->data, &stride, &block_size,
                                  &sub_w, &sub_h };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_means, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }
    {
        const int pairs = elements_in_block * (elements_in_block + 1) / 2;
        const int sub_w = s->submatrix_w, sub_h = s->submatrix_h;
        const unsigned grid = DIV_ROUND_UP(pairs, SPEED_SCAN_THREADS);
        void *kernel_params[] = { &data, &s->means->data, &s->cov_mat->data,
                                  &stride, &block_size, &elements_in_block,
                                  &sub_w, &sub_h };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_cov, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Step 2: eigenvalues of the covariance matrix. */
    {
        const int size = elements_in_block;
        void *kernel_params[] = { &s->cov_mat->data, &s->eigenvalues->data,
                                  &size };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_eigen, 1, 1, 1, 1, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Step 3: independent term. */
    {
        const int truncated_w = s->truncated_w, truncated_h = s->truncated_h;
        const int nb_h = s->num_blocks_horizontal;
        const unsigned grid = DIV_ROUND_UP(elements_in_block,
                                           SPEED_SCAN_THREADS);
        void *kernel_params[] = { &data, &s->independent_term->data, &stride,
                                  &block_size, &truncated_w, &truncated_h,
                                  &nb_h, &nb };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_independent, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Step 4: QR decomposition of the covariance matrix and the two matrix
     * products; the solution is zeroed if the system cannot be inverted. */
    void *qr_params[] = { &s->cov_mat->data, &s->eigenvalues->data,
                          &s->independent_term->data, &s->matrix_r->data,
                          &s->tmp_rect->data, &s->linear_system_sol->data,
                          (void *)&nb, &s->err_flag->data };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_qr, 1, 1, 1, SPEED_QR_THREADS, 1, 1,
                                    0, str, qr_params, NULL));

    if (nb > 0) {
        const unsigned grid = DIV_ROUND_UP(nb, SPEED_SCAN_THREADS);
        void *kernel_params[] = { &s->matrix_r->data, &s->tmp_rect->data,
                                  &s->linear_system_sol->data, (void *)&nb,
                                  &s->err_flag->data };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_solve, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Step 5: pointwise product with 1/B^2. */
    if (total > 0) {
        const float denominator = (float)elements_in_block;
        const unsigned grid = DIV_ROUND_UP(total, SPEED_SCAN_THREADS);
        void *kernel_params[] = { &s->linear_system_sol->data,
                                  &s->independent_term->data, &denominator,
                                  (void *)&total };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_pointwise, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Step 6: sum of columns. */
    if (nb > 0) {
        const unsigned grid = DIV_ROUND_UP(nb, SPEED_SCAN_THREADS);
        void *kernel_params[] = { &s->linear_system_sol->data,
                                  &s->sum_s->data, (void *)&nb,
                                  &elements_in_block };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_sum_columns, grid, 1, 1,
                                        SPEED_SCAN_THREADS, 1, 1, 0, str,
                                        kernel_params, NULL));
    }

    /* Read back the eigenvalues, S and the error flag of this picture. */
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(s->eig_host + slot * elements_in_block,
                                       s->eigenvalues->data,
                                       sizeof(float) * elements_in_block, str));
    if (nb > 0) {
        CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(s->S_host + (size_t)slot * nb,
                                           s->sum_s->data,
                                           sizeof(float) * nb, str));
    }
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(s->err_host + slot, s->err_flag->data,
                                       sizeof(int), str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(str));

    /* est_params(): report a singular covariance matrix like the CPU does. */
    if (s->err_host[slot])
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "est_params: covariance matrix is singular\n");

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Host side entropy / score (verbatim from speed.c)                         */
/* ------------------------------------------------------------------------- */

static void speed_chroma_update_entropy(SpeedChromaStateCuda *s, float *entropy,
                                        const float *S, float L, float sigma_nn)
{
    for (int i = 0; i < s->num_blocks_vertical; i++) {
        for (int j = 0; j < s->num_blocks_horizontal; j++) {
            entropy[i * s->num_blocks_horizontal + j] +=
                log2(L * S[i * s->num_blocks_horizontal + j] + sigma_nn) +
                log2(2 * M_PI * M_E);
        }
    }
}

static float speed_chroma_get_speed_score(SpeedChromaStateCuda *s,
                                          const float *ref_entropies,
                                          const float *ref_variances,
                                          const float *dis_entropies,
                                          const float *dis_variances,
                                          float sigma_nn, float nn_floor,
                                          int speed_weight_var_mode)
{
    const int elements_in_block = SPEED_ELEMENTS_IN_BLOCK;
    const int num_blocks = s->num_blocks;

    float score = 0;
    float base_entropy = elements_in_block * (log2((1 + nn_floor) * sigma_nn) + log2(2 * M_PI * M_E));
    for (int i = 0; i < num_blocks; i++) {
        if ((ref_entropies[i] < base_entropy) &&
            (dis_entropies[i] < base_entropy))
        {
            // If both entropies are below the base_entropy,
            // there is no visible difference
            score += 0;
        } else {
            float spatial_ref = 0.0;
            float spatial_dis = 0.0;
            if (speed_weight_var_mode == 0) {
                spatial_ref = ref_entropies[i] * log2(1 + ref_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + dis_variances[i]);
            } else if (speed_weight_var_mode == 1) {
                spatial_ref = ref_entropies[i] * log2(1 + ref_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + ref_variances[i]);
            } else if (speed_weight_var_mode == 2) {
                spatial_ref = ref_entropies[i] * log2(1 + dis_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + dis_variances[i]);
            } else if (speed_weight_var_mode == 3) {
                spatial_ref = ref_entropies[i] * log2(1 + (ref_variances[i] + dis_variances[i]) / 2.0);
                spatial_dis = dis_entropies[i] * log2(1 + (ref_variances[i] + dis_variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 4) {
                spatial_ref = ref_entropies[i] * log2(1 + ref_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + (ref_variances[i] + dis_variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 5) {
                spatial_ref = ref_entropies[i] * log2(1 + ref_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + (0.75 * ref_variances[i] + 0.25 * dis_variances[i]));
            } else if (speed_weight_var_mode == 6) {
                spatial_ref = ref_entropies[i] * log2(1 + ref_variances[i]);
                spatial_dis = dis_entropies[i] * log2(1 + (0.25 * ref_variances[i] + 0.75 * dis_variances[i]));
            } else {
                return -EINVAL;
            }
            score += fabsf(spatial_ref - spatial_dis);
        }

    }

    return score / num_blocks;
}

/*
 * est_params(), steps 7-10: entropies, S and the error flag -> the values
 * est_params() returns to speed_extract_score(). `slot` is the readback slot
 * (0 = ref, 1 = dis) of one picture pair.
 */
static int speed_chroma_results(SpeedChromaStateCuda *s, int slot,
                                const float *sigma_nn, int *err)
{
    const int nb = s->num_blocks;
    const float *eig = s->eig_host + slot * SPEED_ELEMENTS_IN_BLOCK;
    const float *S = s->S_host + (size_t)slot * MAX(nb, 1);
    float *entropies = s->entropies[slot];
    float *variances = s->variances[slot];

    /* Steps 8/9: zeroed entropy matrix, updated once per eigenvalue. */
    memset(entropies, 0, (size_t)nb * sizeof(float));
    for (int k = 0; k < SPEED_ELEMENTS_IN_BLOCK; k++) {
        float L = eig[k] < 0 ? 0 : eig[k];
        speed_chroma_update_entropy(s, entropies, S, L, *sigma_nn);
    }

    /* Step 10: S is the variance of each block. */
    memcpy(variances, S, (size_t)nb * sizeof(float));

    *err = s->err_host[slot] != 0;

    return 0;
}

/* Same flow as extract_channel(): picture_copy() of both pictures, then
 * speed_extract_score() (filter_and_downscale() + est_params() per picture). */
static int speed_chroma_extract_channel(SpeedChromaStateCuda *s,
                                        VmafPicture *ref_pic,
                                        VmafPicture *dist_pic, int channel,
                                        float *score, int *err_out,
                                        CUstream str)
{
    const float sigma_nn = (float)s->speed_chroma_sigma_nn;

    int ret = speed_chroma_copy_plane(s, ref_pic, channel, s->frame[0]->data,
                                      str);
    if (ret) return ret;
    ret = speed_chroma_filter_and_downscale(s, 0, str);
    if (ret) return ret;
    ret = speed_chroma_est_params(s, 0, 0, str);
    if (ret) return ret;
    int err_ref = s->err_host[0] != 0;

    ret = speed_chroma_copy_plane(s, dist_pic, channel, s->frame[1]->data,
                                  str);
    if (ret) return ret;
    ret = speed_chroma_filter_and_downscale(s, 1, str);
    if (ret) return ret;
    ret = speed_chroma_est_params(s, 1, 1, str);
    if (ret) return ret;
    int err_dis = s->err_host[1] != 0;

    int err0 = 0, err1 = 0;
    ret = speed_chroma_results(s, 0, &sigma_nn, &err0);
    if (ret) return ret;
    ret = speed_chroma_results(s, 1, &sigma_nn, &err1);
    if (ret) return ret;

    // If only one of ref and dis was numerically unstable (very rare)
    // we return 0 instead of an inflated score that may skew the average
    if ((err_ref && !err_dis) || (!err_ref && err_dis)) {
        *score = 0.0f;
    } else {
        *score = speed_chroma_get_speed_score(s, s->entropies[0],
                                              s->variances[0],
                                              s->entropies[1],
                                              s->variances[1], sigma_nn,
                                              (float)s->speed_chroma_nn_floor,
                                              s->speed_weight_var_mode);
    }

    *err_out = err_ref || err_dis;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Feature extractor                                                         */
/* ------------------------------------------------------------------------- */

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    SpeedChromaStateCuda *s = fex->priv;
    VmafCudaState *cs = fex->cu_state;
    int ret;

    (void)bpc;

    switch (pix_fmt) {
        case VMAF_PIX_FMT_UNKNOWN:
        case VMAF_PIX_FMT_YUV400P:
            return -EINVAL;
        case VMAF_PIX_FMT_YUV420P:
            w /= 2;
            h /= 2;
            break;
        case VMAF_PIX_FMT_YUV422P:
            w /= 2;
            break;
        case VMAF_PIX_FMT_YUV444P:
            break;
    }

    s->cu_state = cs;

    /* speed_init_dimensions() */
    s->original_w = w;
    s->original_h = h;
    s->scaled_w = (int)(s->original_w * s->speed_chroma_prescale + 0.5);
    s->scaled_h = (int)(s->original_h * s->speed_chroma_prescale + 0.5);
    s->alloc_w = MAX(s->original_w, s->scaled_w);
    s->alloc_h = MAX(s->original_h, s->scaled_h);
    s->operating_w = s->scaled_w >> SPEED_NUM_SCALES;
    s->operating_h = s->scaled_h >> SPEED_NUM_SCALES;
    s->truncated_w = (s->operating_w / SPEED_BLOCK_SIZE) * SPEED_BLOCK_SIZE;
    s->truncated_h = (s->operating_h / SPEED_BLOCK_SIZE) * SPEED_BLOCK_SIZE;
    s->num_blocks_horizontal = s->truncated_w / SPEED_BLOCK_SIZE;
    s->num_blocks_vertical = s->truncated_h / SPEED_BLOCK_SIZE;
    s->num_blocks = s->num_blocks_horizontal * s->num_blocks_vertical;
    s->submatrix_w = s->truncated_w - SPEED_BLOCK_SIZE + 1;
    s->submatrix_h = s->truncated_h - SPEED_BLOCK_SIZE + 1;

    if (s->scaled_h <= 0 || s->scaled_w <= 0 || s->alloc_w <= 0 ||
        s->alloc_h <= 0) {
        return -EINVAL;
    }

    if (s->truncated_w == 0 || s->truncated_h == 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SpEED: image too small, operating width or height is 0");
    }

    /* speed_init(): the kernelscale drives the gaussian filter sizes. */
    if (!vif_validate_kernelscale((float)s->speed_chroma_kernelscale)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "invalid speed_kernelscale");
        return -EINVAL;
    }

    /* The prescale step calls vif_scale_frame_s(); its nearest and bilinear
     * branches are ported (speed_chroma_scale_*_kernel). bicubic/lanczos4 are
     * not ported yet. The CPU only scales when prescale != 1.0, and both ports
     * are bit-exact with vif_tools.c. */
    s->prescale_method = vif_scale_nearest;
    if (!ALMOST_EQUAL(s->speed_chroma_prescale, 1.0)) {
        if (vif_get_scaling_method(s->speed_chroma_prescale_method,
                                   &s->prescale_method)) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "invalid speed_prescale_method\n");
            return -EINVAL;
        }
        if (s->prescale_method != vif_scale_nearest &&
            s->prescale_method != vif_scale_bilinear) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "speed_chroma_cuda: speed_prescale_method '%s' is not "
                     "supported yet (supported: nearest, bilinear)\n",
                     s->speed_chroma_prescale_method);
            return -EINVAL;
        }
    }

    s->float_stride = ALIGN_CEIL(s->alloc_w * sizeof(float));
    s->stride_px = (int)(s->float_stride / sizeof(float));

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    ret = speed_chroma_load_module(s);
    if (ret) goto fail;

    {
        const size_t frame_bytes = s->float_stride * (size_t)s->alloc_h;
        const size_t nb = (size_t)MAX(s->num_blocks, 1);
        const size_t total = nb * SPEED_ELEMENTS_IN_BLOCK;

        ret = speed_chroma_device_alloc(s, &s->frame[0], frame_bytes);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->frame[1], frame_bytes);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->curr_scale, frame_bytes);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->tmpbuf, frame_bytes);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->means,
                                        sizeof(float) * SPEED_ELEMENTS_IN_BLOCK);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->cov_mat,
                                        sizeof(float) * SPEED_ELEMENTS_IN_BLOCK *
                                        SPEED_ELEMENTS_IN_BLOCK);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->eigenvalues,
                                        sizeof(float) * SPEED_ELEMENTS_IN_BLOCK);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->independent_term,
                                        sizeof(float) * total);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->linear_system_sol,
                                        sizeof(float) * total);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->matrix_r,
                                        sizeof(float) * SPEED_ELEMENTS_IN_BLOCK *
                                        SPEED_ELEMENTS_IN_BLOCK);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->tmp_rect, sizeof(float) * total);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->sum_s, sizeof(float) * nb);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->err_flag, sizeof(int));
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->filter_antialias,
                                        sizeof(float) * SPEED_MAX_KERNEL_TAPS);
        if (ret) goto fail;
        ret = speed_chroma_device_alloc(s, &s->filter_dec,
                                        sizeof(float) * SPEED_MAX_KERNEL_TAPS);
        if (ret) goto fail;

        ret = vmaf_cuda_buffer_host_alloc(cs, (void **)&s->eig_host,
                                          sizeof(float) *
                                          SPEED_ELEMENTS_IN_BLOCK * 2);
        if (ret) goto fail;
        ret = vmaf_cuda_buffer_host_alloc(cs, (void **)&s->S_host,
                                          sizeof(float) * nb * 2);
        if (ret) goto fail;
        ret = vmaf_cuda_buffer_host_alloc(cs, (void **)&s->err_host,
                                          sizeof(int) * 2);
        if (ret) goto fail;
        memset(s->err_host, 0, sizeof(int) * 2);

        for (int i = 0; i < 2; i++) {
            s->entropies[i] = calloc(nb, sizeof(float));
            s->variances[i] = calloc(nb, sizeof(float));
            if (!s->entropies[i] || !s->variances[i]) {
                ret = -ENOMEM;
                goto fail;
            }
        }
    }

    return 0;

fail:
    speed_chroma_free_buffers(s);
    speed_chroma_free_host_buffers(s);
    vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    SpeedChromaStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    (void)ref_pic_90;
    (void)dist_pic_90;

    /*
     * Run on the reference picture's stream, ordered after both uploads. The
     * per-frame scratch buffers are reused by the next frame, but every
     * extract() drains the stream before it returns (the scores are computed on
     * the host), so there is no overlap to order against.
     */
    CUstream str = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(str,
                                       vmaf_cuda_picture_get_ready_event(ref_pic),
                                       CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(str,
                                       vmaf_cuda_picture_get_ready_event(dist_pic),
                                       CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    float score_u, score_v;
    int err_u = 0, err_v = 0;
    int ret = speed_chroma_extract_channel(s, ref_pic, dist_pic, 1, &score_u,
                                           &err_u, str);
    if (ret) return ret;
    ret = speed_chroma_extract_channel(s, ref_pic, dist_pic, 2, &score_v,
                                       &err_v, str);
    if (ret) return ret;

    // There are edge cases where one or both channels (U and V) have singular
    // covariance matrices. For example, when the channel is completely flat.
    // If only one channel is singular, we impute its score from the other
    // channel, and therefore the combined score_uv is equal to the other
    // channel. This is a better approximation than imputing it to be zero.

    float score_uv;
    if (err_u && !err_v) {
        score_uv = score_v;
    } else if (err_v && !err_u) {
        score_uv = score_u; }
    else {
        score_uv = (score_u + score_v) / 2.0;
    }

    int err = 0;

    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_u_score",
            MIN(score_u, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_v_score",
            MIN(score_v, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_uv_score",
            MIN(score_uv, s->speed_chroma_max_val), index);
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SpeedChromaStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;

    if (s->module) {
        CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
        CHECK_CUDA(cu_f, cuModuleUnload(s->module));
        CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
        s->module = NULL;
    }

    speed_chroma_free_buffers(s);
    speed_chroma_free_host_buffers(s);
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

VmafFeatureExtractor vmaf_fex_speed_chroma_cuda = {
    .name = "speed_chroma_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = NULL,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(SpeedChromaStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA | VMAF_FEATURE_EXTRACTOR_CHROMA
};
