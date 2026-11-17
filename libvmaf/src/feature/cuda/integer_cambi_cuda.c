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
 */

/*
 * CUDA CAMBI feature extractor.
 *
 * The kernels live in cuda/integer_cambi/cambi.cu, which is compiled to a
 * fatbin and embedded as `cambi_ptx` (bin2c). This file owns the whole
 * pipeline and loads the module through cuModuleLoadData().
 *
 * Numerical parity with src/feature/cambi.c is the contract:
 *   - the luma band, TVI thresholds and VLT luma come from the helpers
 *     exported by cambi.h, i.e. the same code the CPU runs;
 *   - the window is adjusted with the same cambi_adjust_window_size() rule;
 *   - the final score is sum(score[scale] * g_scale_weights[scale]) divided by
 *     the number of pixels in the window -- the CPU does *not* rescale by the
 *     sum of the scale weights.
 * The only deliberate divergence is floating point summation order in the topk
 * pooling (see cambi_pool()).
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "luminance_tools.h"
#include "cuda_helper.cuh"
#include "cuda/integer_cambi/cambi.h"
#include "cambi.h"
#include "mkdirp.h"

/*
 * Small helpers that the CPU CAMBI implementation keeps private to cambi.c.
 * They are duplicated here to keep the CUDA feature parameter derivation
 * bit-identical without changing the public (and already exported) CPU API.
 */
static FORCE_INLINE int cambi_clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

static FORCE_INLINE uint16_t cambi_ceil_log2(uint32_t num)
{
    if (num == 0)
        return 0;

    uint32_t tmp = num - 1;
    uint16_t shift = 0;
    while (tmp > 0) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static FORCE_INLINE uint16_t cambi_get_mask_index(unsigned input_width,
                                                  unsigned input_height,
                                                  uint16_t filter_size)
{
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (cambi_ceil_log2(shifted_wh) - 11) - 1) >> 1;
}

static FORCE_INLINE void cambi_adjust_window_size(uint16_t *window_size,
                                                  unsigned input_width,
                                                  unsigned input_height,
                                                  bool cambi_high_res_speedup)
{
    /* Adjustment weight: (input_width + input_height) /
     * (CAMBI_4K_WIDTH + CAMBI_4K_HEIGHT). */
    (*window_size) = (((*window_size) * (input_width + input_height)) / 375) >> 4;
    if (cambi_high_res_speedup)
        (*window_size) = ((*window_size) + 1) >> 1;
    /* round up to odd */
    *window_size |= 1;
}

static FORCE_INLINE uint16_t cambi_get_pixels_in_window(uint16_t window_length)
{
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

/* Device-side constants must match cambi.cu. */
#define CAMBI_THREADS_X 16
#define CAMBI_THREADS_Y 16
#define CAMBI_SCAN_THREADS 256
#define CAMBI_POOL_BLOCKS 512
#define CAMBI_POOL_THREADS 256

typedef struct CambiStateCuda {
    unsigned enc_width;
    unsigned enc_height;
    unsigned enc_bitdepth;
    unsigned src_width;
    unsigned src_height;
    uint16_t window_size;
    uint16_t src_window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    char *eotf;
    char *cambi_eotf;
    uint16_t vlt_luma;
    /* Read back as an int by vmaf_option_set()/feature-name generation, so it
     * must be int-sized: a uint16_t field would let a VMAF_OPT_TYPE_INT read
     * spill into the neighbouring member. */
    int max_log_contrast;
    int cambi_high_res_speedup;

    char *heatmaps_path;
    FILE *heatmaps_files[NUM_SCALES];

    unsigned num_diffs;

    /* Host side tables */
    uint16_t tvi_for_diff[32];
    int diff_weights[32];
    int all_diffs[65];

    /* Luma band */
    unsigned v_band_base;
    unsigned v_band_size;
    unsigned v_band_offset_val;

    /* Derived */
    int alloc_w;
    int alloc_h;
    int mask_pad;
    int z_pitch;
    unsigned mask_index;

    /* Module and kernels */
    CUmodule module;
    CUfunction fn_convert_10b;
    CUfunction fn_convert_10b_scaled;
    CUfunction fn_anti_dithering;
    CUfunction fn_row_prefix;
    CUfunction fn_integrate_rows;
    CUfunction fn_mask;
    CUfunction fn_decimate;
    CUfunction fn_filter_mode;
    CUfunction fn_calc_c_values;
    CUfunction fn_pool_count;
    CUfunction fn_pool_sum;
    CUfunction fn_pool_partials;

    /* Device buffers */
    VmafCudaBuffer *image[2];
    VmafCudaBuffer *mask[2];
    VmafCudaBuffer *integral;
    VmafCudaBuffer *c_values;
    VmafCudaBuffer *pool_partials;
    VmafCudaBuffer *pool_counts;
    VmafCudaBuffer *tvi_for_diff_dev;
    VmafCudaBuffer *diff_weights_dev;
    VmafCudaBuffer *all_diffs_dev;
    VmafCudaBuffer *recip_lut_dev;

    VmafDictionary *feature_name_dict;

    /* The CAMBI kernels run on the picture's own stream (which is what the
     * frame upload and the library's "finished" event use), so consecutive
     * frames are chained with an event: the scratch buffers and the pooling
     * counters are reused every frame and must not overlap. */
    CUevent finished;
} CambiStateCuda;

static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed",
        .offset = offsetof(CambiStateCuda, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateCuda, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiStateCuda, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiStateCuda, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI",
        .offset = offsetof(CambiStateCuda, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for spatial pooling",
        .offset = offsetof(CambiStateCuda, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for spatial pooling (cambi)",
        .offset = offsetof(CambiStateCuda, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold for luminance",
        .offset = offsetof(CambiStateCuda, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is not visible",
        .offset = offsetof(CambiStateCuda, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum contrast in log luma level",
        .offset = offsetof(CambiStateCuda, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF assumed for luminance conversion",
        .offset = offsetof(CambiStateCuda, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF assumed for the CAMBI feature",
        .offset = offsetof(CambiStateCuda, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_high_res_speedup",
        .help = "Speed up for high resolutions",
        .offset = offsetof(CambiStateCuda, cambi_high_res_speedup),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_HIGH_RES_SPEEDUP,
        .min = 0,
        .max = CAMBI_4K_HEIGHT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "hrs",
    },
    {
        .name = "heatmaps_path",
        .help = "Path where heatmaps will be dumped.",
        .offset = offsetof(CambiStateCuda, heatmaps_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
    },
    { 0 }
};

static const char *provided_features[] = {
    "Cambi_feature_cambi_score",
    NULL
};

/* ------------------------------------------------------------------------- */
/* Heatmap dump (same option semantics as the CPU extractor)                 */
/* ------------------------------------------------------------------------- */

#ifdef _WIN32
    #define PATH_SEPARATOR '\\'
#else
    #define PATH_SEPARATOR '/'
#endif

/*
 * Mirrors dump_c_values(): per-frame images are stacked vertically (frame f
 * starts at row f * height), max_c_value uses the *integer* window_size^2 / 4
 * and the value is truncated on the cast to uint16_t, so the dumped bytes are
 * directly comparable with the CPU reference.
 */
static int cambi_dump_c_values(CambiStateCuda *s, CudaFunctions *cu_f,
                               CUstream str, int w, int h, unsigned scale,
                               unsigned frame)
{
    FILE *file = s->heatmaps_files[scale];
    if (!file) return 0;

    const size_t n = (size_t)w * h;
    float *host = malloc(n * sizeof(float));
    uint16_t *row = malloc((size_t)w * sizeof(uint16_t));
    if (!host || !row) {
        free(host);
        free(row);
        return -ENOMEM;
    }
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(host, s->c_values->data,
                                       n * sizeof(float), str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(str));

    int max_diff_weight = s->diff_weights[0];
    for (unsigned d = 0; d < s->num_diffs; d++) {
        if (s->diff_weights[d] > max_diff_weight) max_diff_weight = s->diff_weights[d];
    }
    const int max_c_value = max_diff_weight * (int)s->window_size * (int)s->window_size / 4;
    const double scaling_value = 65535.0 / (double)max_c_value;

    int ret = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            row[j] = (uint16_t)(scaling_value * host[(size_t)i * w + j]);
        const ptrdiff_t offset = ((ptrdiff_t)frame * h + i) * w * sizeof(uint16_t);
        if (fseek(file, offset, SEEK_SET) ||
            fwrite(row, sizeof(uint16_t), (size_t)w, file) != (size_t)w) {
            ret = -EINVAL;
            break;
        }
    }
    free(host);
    free(row);
    return ret;
}

/* ------------------------------------------------------------------------- */
/* Module setup                                                             */
/* ------------------------------------------------------------------------- */

static int cambi_load_module(CambiStateCuda *s, VmafCudaState *cu_state)
{
    CudaFunctions *cu_f = cu_state->f;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(cu_state->ctx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&s->module, cambi_ptx));

#define CAMBI_GET_FN(field, symbol)                                             \
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->field, s->module, symbol))

    CAMBI_GET_FN(fn_convert_10b, "cambi_convert_10b_kernel");
    CAMBI_GET_FN(fn_convert_10b_scaled, "cambi_convert_10b_scaled_kernel");
    CAMBI_GET_FN(fn_anti_dithering, "cambi_anti_dithering_kernel");
    CAMBI_GET_FN(fn_row_prefix, "cambi_row_prefix_kernel");
    CAMBI_GET_FN(fn_integrate_rows, "cambi_integrate_rows_kernel");
    CAMBI_GET_FN(fn_mask, "cambi_mask_kernel");
    CAMBI_GET_FN(fn_decimate, "cambi_decimate_kernel");
    CAMBI_GET_FN(fn_filter_mode, "cambi_filter_mode_kernel");
    CAMBI_GET_FN(fn_calc_c_values, "cambi_calc_c_values_kernel");
    CAMBI_GET_FN(fn_pool_count, "cambi_pool_count_kernel");
    CAMBI_GET_FN(fn_pool_sum, "cambi_pool_sum_kernel");
    CAMBI_GET_FN(fn_pool_partials, "cambi_pool_partials_kernel");

#undef CAMBI_GET_FN

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Kernel launchers                                                         */
/* ------------------------------------------------------------------------- */

/*
 * Convert the distorted picture to the 10-bit working image.
 * Mirrors cambi_preprocessing() + decimate_generic_*_and_convert_to_10b().
 */
static int cambi_preprocess(CambiStateCuda *s, const VmafPicture *pic,
                            CUdeviceptr dst, CudaFunctions *cu_f, CUstream str)
{
    const int bpc = (int)pic->bpc;
    const int in_w = (int)pic->w[0];
    const int in_h = (int)pic->h[0];
    const int out_w = s->enc_width;
    const int out_h = s->enc_height;

    int mode, shift, rounding;
    if (bpc >= 10) {
        mode = 2;
        shift = bpc - 10;
        rounding = shift == 0 ? 0 : 1 << (shift - 1);
    } else if (bpc == 9) {
        mode = 1;
        shift = 1;
        rounding = 0;
    } else {
        mode = 0;
        shift = 10 - bpc;
        rounding = 0;
    }

    const int elem_size = (mode == 0) ? 1 : 2;
    const int src_stride = (int)(pic->stride[0] / (size_t)elem_size);
    CUdeviceptr src = (CUdeviceptr)(uintptr_t)pic->data[0];

    if (in_w == out_w && in_h == out_h) {
        const unsigned grid_x = DIV_ROUND_UP(out_w, CAMBI_THREADS_X);
        const unsigned grid_y = DIV_ROUND_UP(out_h, CAMBI_THREADS_Y);
        void *kernel_params[] = { &src, (void *)&src_stride, &dst,
                                  (void *)&out_w, (void *)&out_h,
                                  &mode, &shift, &rounding };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_convert_10b, grid_x, grid_y, 1,
                                        CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    } else {
        const unsigned grid_x = DIV_ROUND_UP(out_h, CAMBI_SCAN_THREADS);
        void *kernel_params[] = { &src, (void *)&src_stride, &dst,
                                  (void *)&out_w, (void *)&out_h,
                                  (void *)&in_w, (void *)&in_h,
                                  &mode, &shift, &rounding };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_convert_10b_scaled, grid_x, 1, 1,
                                        CAMBI_SCAN_THREADS, 1, 1, 0,
                                        str, kernel_params, NULL));
    }

    /*
     * anti_dithering_filter() runs when the encoding bitdepth is < 10; the
     * caller applies it out of place (cambi_anti_dithering), because the CPU
     * version is an in-place 2x2 box filter that a parallel kernel would race.
     */
    return 0;
}

static int cambi_anti_dithering(CambiStateCuda *s, CUdeviceptr src, CUdeviceptr dst,
                                int w, int h, CudaFunctions *cu_f, CUstream str)
{
    const unsigned grid_x = DIV_ROUND_UP(w, CAMBI_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, CAMBI_THREADS_Y);
    const int stride = s->alloc_w;
    void *kernel_params[] = { &src, &dst, (void *)&w, (void *)&h, (void *)&stride };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_anti_dithering, grid_x, grid_y, 1,
                                    CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                    str, kernel_params, NULL));
    return 0;
}

/*
 * Spatial mask: derivative -> summed-area table -> 7x7 thresholding.
 * Equivalent to get_spatial_mask() (see cambi.cu for the derivation).
 */
static int cambi_spatial_mask(CambiStateCuda *s, CUdeviceptr image, CUdeviceptr mask,
                              int w, int h, CudaFunctions *cu_f, CUstream str)
{
    const unsigned grid_x = DIV_ROUND_UP(w, CAMBI_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, CAMBI_THREADS_Y);
    CUdeviceptr integral = s->integral->data;
    const int z_pitch = s->z_pitch;
    const int w1 = w + 1;

    {
        const unsigned grid = h;
        void *kernel_params[] = { &image, (void *)&s->alloc_w, (void *)&w,
                                  (void *)&h, (void *)&z_pitch, &integral };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_row_prefix, grid, 1, 1,
                                        CAMBI_SCAN_THREADS, 1, 1, 0,
                                        str, kernel_params, NULL));
    }
    {
        const unsigned grid = DIV_ROUND_UP(w1, CAMBI_SCAN_THREADS);
        void *kernel_params[] = { &integral, (void *)&z_pitch, (void *)&h,
                                  (void *)&w1 };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_integrate_rows, grid, 1, 1,
                                        CAMBI_SCAN_THREADS, 1, 1, 0,
                                        str, kernel_params, NULL));
    }
    {
        const int pad = s->mask_pad;
        unsigned mask_index = s->mask_index;
        void *kernel_params[] = { &integral, &mask, (void *)&s->alloc_w,
                                  (void *)&z_pitch, (void *)&w, (void *)&h,
                                  (void *)&pad, &mask_index };
        CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_mask, grid_x, grid_y, 1,
                                        CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                        str, kernel_params, NULL));
    }
    return 0;
}

static int cambi_decimate(CambiStateCuda *s, CUdeviceptr src, CUdeviceptr dst,
                          int out_w, int out_h, CudaFunctions *cu_f, CUstream str)
{
    const unsigned grid_x = DIV_ROUND_UP(out_w, CAMBI_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(out_h, CAMBI_THREADS_Y);
    const int pitch = s->alloc_w;
    void *kernel_params[] = { &src, &dst, (void *)&out_w, (void *)&out_h,
                              (void *)&pitch };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_decimate, grid_x, grid_y, 1,
                                    CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                    str, kernel_params, NULL));
    return 0;
}

static int cambi_filter_mode(CambiStateCuda *s, CUdeviceptr src, CUdeviceptr dst,
                             int w, int h, CudaFunctions *cu_f, CUstream str)
{
    const unsigned grid_x = DIV_ROUND_UP(w, CAMBI_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, CAMBI_THREADS_Y);
    const int img_pitch = s->alloc_w;
    void *kernel_params[] = { &src, &dst, (void *)&w, (void *)&h,
                              (void *)&img_pitch };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_filter_mode, grid_x, grid_y, 1,
                                    CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                    str, kernel_params, NULL));
    return 0;
}

static int cambi_calc_c_values(CambiStateCuda *s, CUdeviceptr image, CUdeviceptr mask,
                               int w, int h, CudaFunctions *cu_f, CUstream str)
{
    const unsigned grid_x = DIV_ROUND_UP(w, CAMBI_THREADS_X);
    const unsigned grid_y = DIV_ROUND_UP(h, CAMBI_THREADS_Y);
    const int img_pitch = s->alloc_w;
    const int c_pitch = w;
    const int pad = (int)(s->window_size >> 1);
    CUdeviceptr c_values = s->c_values->data;
    CUdeviceptr tvi = s->tvi_for_diff_dev->data;
    CUdeviceptr all_diffs = s->all_diffs_dev->data;
    CUdeviceptr recip = s->recip_lut_dev->data;
    CUdeviceptr diff_weights = s->diff_weights_dev->data;
    unsigned num_diffs = s->num_diffs;
    unsigned vlt_luma = s->vlt_luma;
    unsigned v_band_base = s->v_band_base;
    unsigned v_band_size = s->v_band_size;
    unsigned v_band_offset = s->v_band_offset_val;

    void *kernel_params[] = { &image, &mask, &c_values,
                              (void *)&img_pitch, (void *)&c_pitch,
                              (void *)&w, (void *)&h, (void *)&pad,
                              &num_diffs, &vlt_luma, &tvi, &all_diffs, &recip,
                              &v_band_base, &v_band_size, &v_band_offset,
                              &diff_weights };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_calc_c_values, grid_x, grid_y, 1,
                                    CAMBI_THREADS_X, CAMBI_THREADS_Y, 1, 0,
                                    str, kernel_params, NULL));
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Spatial pooling (top-k average)                                          */
/* ------------------------------------------------------------------------- */

static int pool_zero_counter(CambiStateCuda *s, int idx, CudaFunctions *cu_f, CUstream str)
{
    CHECK_CUDA(cu_f, cuMemsetD32Async(s->pool_counts->data + idx * sizeof(uint64_t),
                                      0, 2, str));
    return 0;
}

static int pool_read_counter(CambiStateCuda *s, int idx, uint64_t *out,
                             CudaFunctions *cu_f, CUstream str)
{
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(out, s->pool_counts->data + idx * sizeof(uint64_t),
                                       sizeof(*out), str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(str));
    return 0;
}

static int pool_count_ge(CambiStateCuda *s, CUdeviceptr c_values, int n,
                         unsigned key_threshold, uint64_t *out,
                         CudaFunctions *cu_f, CUstream str)
{
    unsigned blocks = DIV_ROUND_UP(n, CAMBI_POOL_THREADS);
    if (blocks > CAMBI_POOL_BLOCKS) blocks = CAMBI_POOL_BLOCKS;
    if (blocks == 0) blocks = 1;

    CUdeviceptr counter = s->pool_counts->data;
    void *kernel_params[] = { &c_values, (void *)&n, &key_threshold, &counter };
    pool_zero_counter(s, 0, cu_f, str);
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_pool_count, blocks, 1, 1,
                                    CAMBI_POOL_THREADS, 1, 1, 0,
                                    str, kernel_params, NULL));
    pool_read_counter(s, 0, out, cu_f, str);
    return 0;
}

static int pool_sum_gt(CambiStateCuda *s, CUdeviceptr c_values, int n,
                       unsigned key_threshold, uint64_t *count_gt, double *sum_gt,
                       CudaFunctions *cu_f, CUstream str)
{
    unsigned blocks = DIV_ROUND_UP(n, CAMBI_POOL_THREADS);
    if (blocks > CAMBI_POOL_BLOCKS) blocks = CAMBI_POOL_BLOCKS;
    if (blocks == 0) blocks = 1;

    CUdeviceptr counter = s->pool_counts->data + sizeof(uint64_t);
    CUdeviceptr partials = s->pool_partials->data;
    CUdeviceptr sum_dev = s->pool_counts->data + 2 * sizeof(uint64_t);

    void *sum_params[] = { &c_values, (void *)&n, &key_threshold, &counter, &partials };
    pool_zero_counter(s, 1, cu_f, str);
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_pool_sum, blocks, 1, 1,
                                    CAMBI_POOL_THREADS, 1, 1, 0,
                                    str, sum_params, NULL));

    void *red_params[] = { &partials, (void *)&blocks, &sum_dev };
    CHECK_CUDA(cu_f, cuLaunchKernel(s->fn_pool_partials, 1, 1, 1, 1, 1, 1, 0,
                                    str, red_params, NULL));

    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(count_gt, counter, sizeof(*count_gt), str));
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(sum_gt, sum_dev, sizeof(*sum_gt), str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(str));
    return 0;
}

static float key_to_float(unsigned key)
{
    float f;
    memcpy(&f, &key, sizeof(f));
    return f;
}

/*
 * spatial_pooling(): mean of the topk_num largest c_values.
 *
 * The value is a float and every c_value is non-negative, so its IEEE-754 bit
 * pattern is monotone in the value and can be radix-searched as an unsigned
 * integer: after 32 halving steps `key` is exactly the k-th largest c_value.
 * The sum of the elements strictly above it (plus the required multiplicity of
 * the threshold itself) is that of the top-k multiset.
 *
 * The CPU accumulates this sum in the order produced by quick_select(); we use
 * a deterministic but different order, so the two sums may differ by a few ulp.
 * The multiset being summed is identical.
 */
static double cambi_pool(CambiStateCuda *s, CUdeviceptr c_values, int n, int topk_num,
                         CudaFunctions *cu_f, CUstream str)
{
    unsigned key = 0;
    for (int b = 31; b >= 0; b--) {
        const unsigned candidate = key | (1u << b);
        uint64_t count = 0;
        pool_count_ge(s, c_values, n, candidate, &count, cu_f, str);
        if (count >= (uint64_t)topk_num) key = candidate;
    }

    uint64_t count_gt = 0;
    double sum_gt = 0.0;
    pool_sum_gt(s, c_values, n, key, &count_gt, &sum_gt, cu_f, str);

    const double threshold = (double)key_to_float(key);
    const double sum = sum_gt + (double)((uint64_t)topk_num - count_gt) * threshold;
    return sum / (double)topk_num;
}

/* ------------------------------------------------------------------------- */
/* Feature extractor                                                        */
/* ------------------------------------------------------------------------- */

static int cambi_cleanup(VmafFeatureExtractor *fex);

static void cambi_free_buffer(VmafFeatureExtractor *fex, VmafCudaBuffer **buf)
{
    if (!*buf) return;
    if ((*buf)->data)
        vmaf_cuda_buffer_free(fex->cu_state, *buf);
    free(*buf);
    *buf = NULL;
}

static void cambi_free_buffers(VmafFeatureExtractor *fex)
{
    CambiStateCuda *s = fex->priv;

    for (int i = 0; i < 2; i++) {
        cambi_free_buffer(fex, &s->image[i]);
        cambi_free_buffer(fex, &s->mask[i]);
    }
    cambi_free_buffer(fex, &s->integral);
    cambi_free_buffer(fex, &s->c_values);
    cambi_free_buffer(fex, &s->pool_partials);
    cambi_free_buffer(fex, &s->pool_counts);
    cambi_free_buffer(fex, &s->tvi_for_diff_dev);
    cambi_free_buffer(fex, &s->diff_weights_dev);
    cambi_free_buffer(fex, &s->all_diffs_dev);
    cambi_free_buffer(fex, &s->recip_lut_dev);
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                         unsigned bpc, unsigned w, unsigned h)
{
    CambiStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret;

    (void)pix_fmt;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    if (s->enc_bitdepth == 0) s->enc_bitdepth = bpc;
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

    if (s->enc_width < CAMBI_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_MIN_WIDTH_HEIGHT) {
        ret = -EINVAL;
        goto fail;
    }
    if (s->src_width < CAMBI_MIN_WIDTH_HEIGHT && s->src_height < CAMBI_MIN_WIDTH_HEIGHT) {
        ret = -EINVAL;
        goto fail;
    }
    if (s->src_width > s->enc_width && s->src_height < s->enc_height) {
        ret = -EINVAL;
        goto fail;
    }
    if (s->src_width < s->enc_width && s->src_height > s->enc_height) {
        ret = -EINVAL;
        goto fail;
    }

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

    s->num_diffs = 1u << s->max_log_contrast;
    if (s->num_diffs > 32) {
        ret = -EINVAL;
        goto fail;
    }

    /* Contrast tables, exactly like set_contrast_arrays(). */
    for (unsigned d = 0; d < s->num_diffs; d++) {
        s->diff_weights[d] = g_contrast_weights[d];
    }
    for (int d = -(int)s->num_diffs; d <= (int)s->num_diffs; d++) {
        s->all_diffs[d + (int)s->num_diffs] = d;
    }

    VmafLumaRange luma_range;
    ret = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (ret) goto fail;

    /* use cambi_eotf if it has a non-default value, else use eotf */
    const char *effective_eotf;
    if (strcmp(s->cambi_eotf, DEFAULT_CAMBI_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = s->eotf;
    }

    VmafEOTF eotf;
    ret = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (ret) goto fail;

    for (unsigned d = 0; d < s->num_diffs; d++) {
        s->tvi_for_diff[d] = get_tvi_for_diff((int)d + 1, s->tvi_threshold, 10,
                                              luma_range, eotf);
        s->tvi_for_diff[d] += s->num_diffs;
    }

    s->vlt_luma = get_vlt_luma(s->cambi_vis_lum_threshold, luma_range, eotf);

    s->src_window_size = s->window_size;
    cambi_adjust_window_size(&s->window_size, s->enc_width, s->enc_height,
                       (bool)s->cambi_high_res_speedup);
    cambi_adjust_window_size(&s->src_window_size, s->src_width, s->src_height,
                       (bool)s->cambi_high_res_speedup);

    const int max_window = MAX(s->window_size, s->src_window_size);
    if (max_window * max_window >= CAMBI_RECIPROCAL_LUT_SIZE) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
            "cambi: window_size %d too large for reciprocal LUT\n", max_window);
        ret = -EINVAL;
        goto fail;
    }

    {
        const int v_lo_signed = (int)s->vlt_luma - 3 * (int)s->num_diffs + 1;
        s->v_band_base = v_lo_signed > 0 ? (unsigned)v_lo_signed : 0;
        s->v_band_size = (unsigned)s->tvi_for_diff[s->num_diffs - 1] + 1 - s->v_band_base;
        s->v_band_offset_val = s->v_band_base + s->num_diffs;
    }

    s->alloc_w = (int)s->enc_width;
    s->alloc_h = (int)s->enc_height;
    s->mask_pad = MASK_FILTER_SIZE >> 1;
    s->z_pitch = s->alloc_w + 1;
    s->mask_index = cambi_get_mask_index(s->enc_width, s->enc_height, MASK_FILTER_SIZE);

    if (s->heatmaps_path) {
        if (mkdirp(s->heatmaps_path, 0770)) {
            ret = -EINVAL;
            goto fail;
        }
        char path[1024] = { 0 };
        int scaled_w = (int)s->enc_width;
        int scaled_h = (int)s->enc_height;
        for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
            snprintf(path, sizeof(path), "%s%ccambi_heatmap_scale_%u_%dx%d_16b.gray",
                     s->heatmaps_path, PATH_SEPARATOR, scale, scaled_w, scaled_h);
            s->heatmaps_files[scale] = fopen(path, "wb");
            if (!s->heatmaps_files[scale]) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR,
                    "cambi_cuda: could not open heatmaps_path: %s\n", path);
                ret = -EINVAL;
                goto fail;
            }
            scaled_w = (scaled_w + 1) >> 1;
            scaled_h = (scaled_h + 1) >> 1;
        }
    }

    ret = cambi_load_module(s, fex->cu_state);
    if (ret) goto fail;

    const size_t image_bytes = sizeof(uint16_t) * (size_t)s->alloc_w * s->alloc_h;
    const size_t integral_bytes = sizeof(int32_t) * (size_t)s->z_pitch * s->alloc_h;

    for (int i = 0; i < 2; i++) {
        if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->image[i], image_bytes))
            { ret = -ENOMEM; goto fail; }
        if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->mask[i], image_bytes))
            { ret = -ENOMEM; goto fail; }
    }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->integral, integral_bytes))
        { ret = -ENOMEM; goto fail; }
    const size_t c_values_bytes = sizeof(float) * (size_t)s->alloc_w * s->alloc_h;
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->c_values, c_values_bytes))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->pool_partials,
                               sizeof(double) * CAMBI_POOL_BLOCKS))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->pool_counts,
                               sizeof(uint64_t) * 2 + sizeof(double)))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->tvi_for_diff_dev,
                               sizeof(uint16_t) * s->num_diffs))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->diff_weights_dev,
                               sizeof(int) * s->num_diffs))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->all_diffs_dev,
                               sizeof(int) * (2 * s->num_diffs + 1)))
        { ret = -ENOMEM; goto fail; }
    if (vmaf_cuda_buffer_alloc(fex->cu_state, &s->recip_lut_dev,
                               sizeof(float) * CAMBI_RECIPROCAL_LUT_SIZE))
        { ret = -ENOMEM; goto fail; }

    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->tvi_for_diff_dev->data, s->tvi_for_diff,
                                       sizeof(uint16_t) * s->num_diffs,
                                       fex->cu_state->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->diff_weights_dev->data, s->diff_weights,
                                       sizeof(int) * s->num_diffs,
                                       fex->cu_state->str));
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->all_diffs_dev->data, s->all_diffs,
                                       sizeof(int) * (2 * s->num_diffs + 1),
                                       fex->cu_state->str));
    /* Upload the CPU's own table so both sides divide identically. */
    CHECK_CUDA(cu_f, cuMemcpyHtoDAsync(s->recip_lut_dev->data, reciprocal_lut,
                                       sizeof(float) * CAMBI_RECIPROCAL_LUT_SIZE,
                                       fex->cu_state->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(fex->cu_state->str));

    /* Cross-frame dependency event: recording it here makes the first frame's
     * wait a no-op while still establishing the chain. (The library's context
     * must be current for event creation, as in integer_adm_cuda.c.) */
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, fex->cu_state->str));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    return 0;

fail:
    cambi_cleanup(fex);
    return ret;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                            VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                            VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                            unsigned index, VmafFeatureCollector *feature_collector)
{
    CambiStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    /* Run on the distorted picture's stream: that is the stream the frame was
     * uploaded on and the one the library records its "finished" event on, so
     * ordering with the upload and with the ring-buffer reuse is guaranteed. */
    CUstream str = vmaf_cuda_picture_get_stream(dist_pic);

    /* The scratch buffers, the c_values buffer and the pooling counters are
     * reused every frame: make this frame wait for the previous one. */
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(str, s->finished, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;

    VmafCudaBuffer *image = s->image[0];
    VmafCudaBuffer *image_scratch = s->image[1];
    VmafCudaBuffer *mask = s->mask[0];
    VmafCudaBuffer *mask_scratch = s->mask[1];

    /* Preprocess the distorted picture to 10-bit. */
    cambi_preprocess(s, dist_pic, image->data, cu_f, str);
    if ((int)s->enc_bitdepth < 10) {
        cambi_anti_dithering(s, image->data, image_scratch->data,
                             s->alloc_w, s->alloc_h, cu_f, str);
        VmafCudaBuffer *tmp = image;
        image = image_scratch;
        image_scratch = tmp;
    }

    /* The spatial mask is computed once, at full resolution. */
    cambi_spatial_mask(s, image->data, mask->data, s->alloc_w, s->alloc_h,
                       cu_f, str);

    double score = 0.0;
    int scaled_w = (int)s->enc_width;
    int scaled_h = (int)s->enc_height;
    double scores_per_scale[NUM_SCALES];

    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || s->cambi_high_res_speedup) {
            scaled_w = (scaled_w + 1) >> 1;
            scaled_h = (scaled_h + 1) >> 1;

            cambi_decimate(s, image->data, image_scratch->data,
                           scaled_w, scaled_h, cu_f, str);
            VmafCudaBuffer *tmp = image;
            image = image_scratch;
            image_scratch = tmp;

            cambi_decimate(s, mask->data, mask_scratch->data,
                           scaled_w, scaled_h, cu_f, str);
            tmp = mask;
            mask = mask_scratch;
            mask_scratch = tmp;
        }

        cambi_filter_mode(s, image->data, image_scratch->data,
                          scaled_w, scaled_h, cu_f, str);
        VmafCudaBuffer *tmp = image;
        image = image_scratch;
        image_scratch = tmp;

        cambi_calc_c_values(s, image->data, mask->data, scaled_w, scaled_h,
                            cu_f, str);

        if (s->heatmaps_files[scale]) {
            int err = cambi_dump_c_values(s, cu_f, str, scaled_w, scaled_h,
                                          scale, index);
            if (err) return err;
        }

        const int num_elements = scaled_w * scaled_h;
        const double topk = (s->topk != DEFAULT_CAMBI_TOPK_POOLING) ?
                            s->topk : s->cambi_topk;
        const int topk_num = cambi_clip(topk * num_elements, 1, num_elements);
        scores_per_scale[scale] =
            cambi_pool(s, s->c_values->data, num_elements, topk_num, cu_f, str);
    }

    const uint16_t pixels_in_window = cambi_get_pixels_in_window(s->window_size);
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += scores_per_scale[scale] * g_scale_weights[scale];
    score /= (double)pixels_in_window;

    if (score > s->cambi_max_val)
        score = s->cambi_max_val;

    /* Signal that this frame's device work is done: the next frame waits on
     * this event, and it also orders CAMBI's kernels before the library's
     * "finished" event (recorded on the same picture stream afterwards). */
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, str));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict,
        "Cambi_feature_cambi_score", score, index);
}

static int cambi_cleanup(VmafFeatureExtractor *fex)
{
    CambiStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;

    if (s->heatmaps_path) {
        for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
            if (s->heatmaps_files[scale])
                fclose(s->heatmaps_files[scale]);
        }
    }

    if (s->module) {
        CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
        CHECK_CUDA(cu_f, cuModuleUnload(s->module));
        CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
        s->module = NULL;
    }

    if (s->finished) {
        CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
        CHECK_CUDA(cu_f, cuEventDestroy(s->finished));
        CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
        s->finished = NULL;
    }

    cambi_free_buffers(fex);
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    return cambi_cleanup(fex);
}

VmafFeatureExtractor vmaf_fex_integer_cambi_cuda = {
    .name = "cambi_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = NULL,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(CambiStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA
};
