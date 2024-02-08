/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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
#include <stdio.h>
#include <string.h>

#include "common/macros.h"
#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"
#include "mkdirp.h"
#include "picture.h"

#if ARCH_X86
#include "x86/cambi_avx2.h"
#endif

/* Ratio of pixels for computation, must be 0 < topk <= 1.0 */
#define DEFAULT_CAMBI_TOPK_POOLING (0.6)

/* Window size to compute CAMBI: 65 corresponds to approximately 1 degree at 4k scale */
#define DEFAULT_CAMBI_WINDOW_SIZE (65)

/* Visibility threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886 */
#define DEFAULT_CAMBI_TVI (0.019)

/* Max log contrast luma levels */
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)

/* If true, CAMBI will be run in full-reference mode and will use both the reference and distorted inputs */
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)

/* EOTF to use for the visibility threshold calculations. One of ['bt1886', 'pq']. Default: 'bt1886'. */
#define DEFAULT_CAMBI_EOTF ("bt1886")

#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)

#define NUM_SCALES 5
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

/* Suprathreshold contrast response */
static const int g_contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                           8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define SWAP_FLOATS(x, y) \
    {                     \
        float temp = x;   \
        x = y;            \
        y = temp;         \
    }

#define PICS_BUFFER_SIZE 2
#define MASK_FILTER_SIZE 7

typedef struct CambiBuffers {
    float *c_values;
    uint32_t *mask_dp;
    uint16_t *c_values_histograms;
    uint16_t *filter_mode_buffer;
    uint16_t *diffs_to_consider;
    uint16_t *tvi_for_diff;
    uint16_t *derivative_buffer;
    int *diff_weights;
    int *all_diffs;
} CambiBuffers;

typedef void (*VmafRangeUpdater)(uint16_t *arr, int left, int right);
typedef void (*VmafDerivativeCalculator)(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride);

typedef struct CambiState {
    VmafPicture pics[PICS_BUFFER_SIZE];
    unsigned enc_width;
    unsigned enc_height;
    unsigned enc_bitdepth;
    unsigned src_width;
    unsigned src_height;
    uint16_t window_size;
    uint16_t src_window_size;
    double topk;
    double tvi_threshold;
    uint16_t max_log_contrast;
    char *heatmaps_path;
    char *eotf;
    bool full_ref;
    FILE *heatmaps_files[NUM_SCALES];
    VmafRangeUpdater inc_range_callback;
    VmafRangeUpdater dec_range_callback;
    VmafDerivativeCalculator derivative_callback;
    CambiBuffers buffers;
} CambiState;

static const VmafOption options[] = {
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiState, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
    },
    {
        .name = "src_width",
        .help = "Source width. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
    },
    {
        .name = "src_height",
        .help = "Source height. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 200,
        .max = 4320,
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 63 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < topk <= 1.0",
        .offset = offsetof(CambiState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean",
        .offset = offsetof(CambiState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum contrast in log luma level (2^max_log_contrast) at 10-bits, "
                "e.g., 2 is equivalent to 4 luma levels at 10-bit and 1 luma level at 8-bit. "
                "From 0 to 5: default 2 is recommended for banding from compression.",
        .offset = offsetof(CambiState, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
    },
    {
        .name = "heatmaps_path",
        .help = "Path where heatmaps will be dumped.",
        .offset = offsetof(CambiState, heatmaps_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
    },
    {
        .name = "full_ref",
        .help = "If true, CAMBI will be run in full-reference mode and will be computed on both the reference and distorted inputs",
        .offset = offsetof(CambiState, full_ref),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = DEFAULT_CAMBI_FULL_REF_FLAG,
    },
    {
        .name = "eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'",
        .offset = offsetof(CambiState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
    },
    { 0 }
};

enum CambiTVIBisectFlag {
    CAMBI_TVI_BISECT_TOO_SMALL,
    CAMBI_TVI_BISECT_CORRECT,
    CAMBI_TVI_BISECT_TOO_BIG
};

static FORCE_INLINE inline int clip(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

static bool tvi_condition(int sample, int diff, double tvi_threshold,
                          VmafLumaRange luma_range, VmafEOTF eotf) {
    double mean_luminance = vmaf_luminance_get_luminance(sample, luma_range, eotf);
    double diff_luminance = vmaf_luminance_get_luminance(sample + diff, luma_range, eotf);
    double delta_luminance = diff_luminance - mean_luminance;
    return (delta_luminance > tvi_threshold * mean_luminance);
}

static enum CambiTVIBisectFlag tvi_hard_threshold_condition(int sample, int diff, double tvi_threshold,
                                                            VmafLumaRange luma_range, VmafEOTF eotf) {
    bool condition;
    condition = tvi_condition(sample, diff, tvi_threshold, luma_range, eotf);
    if (!condition) return CAMBI_TVI_BISECT_TOO_BIG;

    condition = tvi_condition(sample + 1, diff, tvi_threshold, luma_range, eotf);
    if (condition) return CAMBI_TVI_BISECT_TOO_SMALL;

    return CAMBI_TVI_BISECT_CORRECT;
}

static int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth, VmafLumaRange luma_range, VmafEOTF eotf) {
    enum CambiTVIBisectFlag tvi_bisect;
    const int max_val = (1 << bitdepth) - 1;

    int foot = luma_range.foot;
    int head = luma_range.head;
    head = head - diff - 1;

    tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG) return 0;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return foot;

    tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL) return max_val;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return head;

    // bisect
    while (1) {
        int mid = foot + (head - foot) / 2;
        tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, luma_range, eotf);
        if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG)
            head = mid;
        else if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL)
            foot = mid;
        else if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT)
            return mid;
        else // Should never get here (todo: add assert)
            (void)0;
    }
}

static FORCE_INLINE inline void adjust_window_size(uint16_t *window_size,
                                                   unsigned input_width,
                                                   unsigned input_height)
{
    // Adjustment weight: (input_width + input_height) / (CAMBI_4K_WIDTH + CAMBI_4K_HEIGHT)
    (*window_size) = (((*window_size) * (input_width+input_height)) / 375) >> 4;
}

static int set_contrast_arrays(const uint16_t num_diffs, uint16_t **diffs_to_consider,
                               int **diffs_weights, int **all_diffs)
{
    *diffs_to_consider = aligned_malloc(ALIGN_CEIL(sizeof(uint16_t)) * num_diffs, 16);
    if (!(*diffs_to_consider)) return -ENOMEM;

    *diffs_weights = aligned_malloc(ALIGN_CEIL(sizeof(int)) * num_diffs, 32);
    if (!(*diffs_weights)) return -ENOMEM;

    *all_diffs = aligned_malloc(ALIGN_CEIL(sizeof(int)) * (2 * num_diffs + 1), 32);
    if (!(*all_diffs)) return -ENOMEM;

    for (int d = 0; d < num_diffs; d++) {
        (*diffs_to_consider)[d] = d + 1;
        (*diffs_weights)[d] = g_contrast_weights[d];
    }

    for (int d = -num_diffs; d <= num_diffs; d++)
        (*all_diffs)[d + num_diffs] = d;

    return 0;
}

static void increment_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]++;
    }
}

static void decrement_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]--;
    }
}

static void get_derivative_data_for_row(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride) {
    for (int col = 0; col < width; col++) {
        bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] == image_data[row * stride + col + 1]);
        bool vertical_derivative = (row == height - 1 || image_data[row * stride + col] == image_data[(row + 1) * stride + col]);
        derivative_buffer[col] = horizontal_derivative && vertical_derivative;
    }
}

#ifdef _WIN32
    #define PATH_SEPARATOR '\\'
#else
    #define PATH_SEPARATOR '/'
#endif

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h) {
    (void)pix_fmt;

    CambiState *s = fex->priv;

    if (s->enc_bitdepth == 0) {
        s->enc_bitdepth = bpc;
    }
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->src_width == 0 || s->src_height == 0) {
        s->src_width = w;
        s->src_height = h;
    }

    if (s->enc_width < CAMBI_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_MIN_WIDTH_HEIGHT) {
        return -EINVAL;
    }
    if (s->src_width < CAMBI_MIN_WIDTH_HEIGHT && s->src_height < CAMBI_MIN_WIDTH_HEIGHT) {
        return -EINVAL;
    }
    if (s->src_width > s->enc_width && s->src_height < s->enc_height) {
        return -EINVAL;
    }
    if (s->src_width < s->enc_width && s->src_height > s->enc_height) {
        return -EINVAL;
    }

    int alloc_w = s->full_ref ? MAX(s->src_width, s->enc_width) : s->enc_width;
    int alloc_h = s->full_ref ? MAX(s->src_height, s->enc_height) : s->enc_height;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        err |= vmaf_picture_alloc(&s->pics[i], VMAF_PIX_FMT_YUV400P, 10, alloc_w, alloc_h);
    }
    if (err) return err;

    const int num_diffs = 1 << s->max_log_contrast;

    set_contrast_arrays(num_diffs, &s->buffers.diffs_to_consider, &s->buffers.diff_weights, &s->buffers.all_diffs);

    VmafLumaRange luma_range;
    err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err) return err;

    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, s->eotf);
    if (err) return err;

    s->buffers.tvi_for_diff = aligned_malloc(ALIGN_CEIL(sizeof(uint16_t)) * num_diffs, 16);
    if (!s->buffers.tvi_for_diff) return -ENOMEM;
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.tvi_for_diff[d] = get_tvi_for_diff(s->buffers.diffs_to_consider[d], s->tvi_threshold, 10, luma_range, eotf);
        s->buffers.tvi_for_diff[d] += num_diffs;
    }

    s->src_window_size = s->window_size;
    adjust_window_size(&s->window_size, s->enc_width, s->enc_height);
    adjust_window_size(&s->src_window_size, s->src_width, s->src_height);
    s->buffers.c_values = aligned_malloc(ALIGN_CEIL(alloc_w * sizeof(float)) * alloc_h, 32);
    if (!s->buffers.c_values) return -ENOMEM;

    const uint16_t num_bins = 1024 + (s->buffers.all_diffs[2 * num_diffs] - s->buffers.all_diffs[0]);
    s->buffers.c_values_histograms = aligned_malloc(ALIGN_CEIL(alloc_w * num_bins * sizeof(uint16_t)), 32);
    if (!s->buffers.c_values_histograms) return -ENOMEM;

    int pad_size = MASK_FILTER_SIZE >> 1;
    int dp_width = alloc_w + 2 * pad_size + 1;
    int dp_height = 2 * pad_size + 2;

    s->buffers.mask_dp = aligned_malloc(ALIGN_CEIL(dp_height * dp_width * sizeof(uint32_t)), 32);
    if (!s->buffers.mask_dp) return -ENOMEM;
    s->buffers.filter_mode_buffer = aligned_malloc(ALIGN_CEIL(3 * alloc_w * sizeof(uint16_t)), 32);
    if (!s->buffers.filter_mode_buffer) return -ENOMEM;
    s->buffers.derivative_buffer = aligned_malloc(ALIGN_CEIL(alloc_w * sizeof(uint16_t)), 32);
    if (!s->buffers.derivative_buffer) return -ENOMEM;

    if (s->heatmaps_path) {
        int err = mkdirp(s->heatmaps_path, 0770);
        if (err) return -EINVAL;
        char path[1024] = { 0 };
        int scaled_w = s->enc_width;
        int scaled_h = s->enc_height;
        for (int scale = 0; scale < NUM_SCALES; scale++) {
            snprintf(path, sizeof(path), "%s%ccambi_heatmap_scale_%d_%dx%d_16b.gray",
                     s->heatmaps_path, PATH_SEPARATOR, scale, scaled_w, scaled_h);
            s->heatmaps_files[scale] = fopen(path, "w");
            if (!s->heatmaps_files[scale]) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR,
                "cambi: could not open heatmaps_path: %s\n", path);
                return -EINVAL;
            }
            scaled_w = (scaled_w + 1) >> 1;
            scaled_h = (scaled_h + 1) >> 1;
        }
    }

    s->inc_range_callback = increment_range;
    s->dec_range_callback = decrement_range;
    s->derivative_callback = get_derivative_data_for_row;

#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        s->inc_range_callback = cambi_increment_range_avx2;
        s->dec_range_callback = cambi_decrement_range_avx2;
        s->derivative_callback = get_derivative_data_for_row_avx2;
    }
#endif

    return err;
}

/* Preprocessing functions */

// For bitdepths <= 8.
static void decimate_generic_uint8_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic, unsigned out_w, unsigned out_h) {
    uint8_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0];
    ptrdiff_t out_stride = out_pic->stride[0] >> 1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];

    int shift_factor = 10 - pic->bpc;

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h){
        for (unsigned i = 0; i < out_h; i++) {
            for (unsigned j = 0; j < out_w; j++) {
                out_data[i * out_stride + j] = data[i * stride + j] << shift_factor;
            }
        }
        return;
    }

    float ratio_x = (float)in_w / out_w;
    float ratio_y = (float)in_h / out_h;

    float start_x = ratio_x / 2 - 0.5;
    float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for (unsigned i = 0; i < out_h; i++) {
        unsigned ori_y = (int)(y + 0.5);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)(x + 0.5);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x] << shift_factor;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

// For the special case of bitdepth 9, which doesn't fit into uint8_t but has to be upscaled to 10b.
static void decimate_generic_9b_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic, unsigned out_w, unsigned out_h) {
    uint16_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    ptrdiff_t out_stride = out_pic->stride[0] >> 1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h) {
        for (unsigned i = 0; i < out_h; i++) {
            for (unsigned j = 0; j < out_w; j++) {
                out_data[i * out_stride + j] = data[i * stride + j] << 1;
            }
        }
        return;
    }

    float ratio_x = (float)in_w / out_w;
    float ratio_y = (float)in_h / out_h;

    float start_x = ratio_x / 2 - 0.5;
    float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for (unsigned i = 0; i < out_h; i++) {
        unsigned ori_y = (int)(y + 0.5);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)(x + 0.5);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x] << 1;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

// For bitdepths >= 10.
static void decimate_generic_uint16_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic, unsigned out_w, unsigned out_h) {
    uint16_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    ptrdiff_t out_stride = out_pic->stride[0] >> 1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];

    int shift_factor = pic->bpc - 10;
    int rounding_offset = shift_factor == 0 ? 0 : 1 << (shift_factor - 1);

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h) {
        if (pic->bpc == 10) {
            // memcpy is faster in case the original bitdepth is already 10
            memcpy(out_data, data, stride * pic->h[0] * sizeof(uint16_t));
        }
        else {
            for (unsigned i = 0; i < out_h; i++) {
                for (unsigned j = 0; j < out_w; j++) {
                    out_data[i * out_stride + j] = (data[i * stride + j] + rounding_offset) >> shift_factor;
                }
            }
        }
        return;
    }

    float ratio_x = (float)in_w / out_w;
    float ratio_y = (float)in_h / out_h;

    float start_x = ratio_x / 2 - 0.5;
    float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for (unsigned i = 0; i < out_h; i++) {
        unsigned ori_y = (int)(y + 0.5);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)(x + 0.5);
            out_data[i * out_stride + j] = (data[ori_y * stride + ori_x] + rounding_offset) >> shift_factor;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

static void anti_dithering_filter(VmafPicture *pic, unsigned width, unsigned height) {
    uint16_t *data = pic->data[0];
    int stride = pic->stride[0] >> 1;

    for (unsigned i = 0; i < height - 1; i++) {
        for (unsigned j = 0; j < width - 1; j++) {
            data[i * stride + j] = (data[i * stride + j] +
                                    data[i * stride + j + 1] +
                                    data[(i + 1) * stride + j] +
                                    data[(i + 1) * stride + j + 1]) >> 2;
        }

        // Last column
        unsigned j = width - 1;
        data[i * stride + j] = (data[i * stride + j] +
                                data[(i + 1) * stride + j]) >> 1;
    }

    // Last row
    unsigned i = height - 1;
    for (unsigned j = 0; j < width - 1; j++) {
        data[i * stride + j] = (data[i * stride + j] +
                                data[i * stride + j + 1]) >> 1;
    }
}

static int cambi_preprocessing(const VmafPicture *image, VmafPicture *preprocessed, int width, int height, int enc_bitdepth) {
    if (image->bpc >= 10) {
        decimate_generic_uint16_and_convert_to_10b(image, preprocessed, width, height);
    }
    else {
        if (image->bpc <= 8) {
            decimate_generic_uint8_and_convert_to_10b(image, preprocessed, width, height);
        }
        else {
            decimate_generic_9b_and_convert_to_10b(image, preprocessed, width, height);
        }
    }
    if (enc_bitdepth < 10) {
        anti_dithering_filter(preprocessed, width, height);
    }

    return 0;
}

/* Banding detection functions */
static void decimate(VmafPicture *image, unsigned width, unsigned height) {
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            data[i * stride + j] = data[(i << 1) * stride + (j << 1)];
        }
    }
}

static inline uint16_t min3(uint16_t a, uint16_t b, uint16_t c) {
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}

static inline uint16_t mode3(uint16_t a, uint16_t b, uint16_t c) {
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min3(a, b, c);
}

static void filter_mode(const VmafPicture *image, int width, int height, uint16_t *buffer) {
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        buffer[curr_line * width + 0] = data[i * stride + 0];
        for (int j = 1; j < width - 1; j++) {
            buffer[curr_line * width + j] = mode3(data[i * stride + j - 1], data[i * stride + j], data[i * stride + j + 1]);
        }
        buffer[curr_line * width + width - 1] = data[i * stride + width - 1];

        if (i > 1) {
            for (int j = 0; j < width; j++) {
                data[(i - 1) * stride + j] = mode3(buffer[0 * width + j], buffer[1 * width + j], buffer[2 * width + j]);
            }
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

static FORCE_INLINE inline uint16_t ceil_log2(uint32_t num) {
    if (num==0)
        return 0;

    uint32_t tmp = num - 1;
    uint16_t shift = 0;
    while (tmp>0) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static FORCE_INLINE inline uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                                                   uint16_t filter_size) {
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1)>>1;
}

/*
* This function calculates the horizontal and vertical derivatives of the image using 2x1 and 1x2 kernels.
* We say a pixel has zero_derivative=1 if it's equal to its right and bottom neighbours, and =0 otherwise (edges also count as "equal").
* This function then computes the sum of zero_derivative on the filter_size x filter_size square around each pixel
* and stores 1 into the corresponding mask index iff this number is larger than mask_index.
* To calculate the square sums, it uses a dynamic programming algorithm based on inclusion-exclusion.
* To save memory, it uses a DP matrix of only the necessary size, rather than the full matrix, and indexes its rows cyclically.
*/
static void get_spatial_mask_for_index(const VmafPicture *image, VmafPicture *mask,
                                       uint32_t *dp, uint16_t *derivative_buffer, uint16_t mask_index,
                                       uint16_t filter_size, int width, int height, VmafDerivativeCalculator derivative_callback) {
    uint16_t pad_size = filter_size >> 1;
    uint16_t *image_data = image->data[0];
    uint16_t *mask_data = mask->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;

    int dp_width = width + 2 * pad_size + 1;
    int dp_height = 2 * pad_size + 2;
    memset(dp, 0, dp_width * dp_height * sizeof(uint32_t));

    // Initial computation: fill dp except for the last row
    for (int i = 0; i < pad_size; i++) {
        if (i < height) {
            derivative_callback(image_data, derivative_buffer, width, height, i, stride);
        }
        for (int j = 0; j < width + pad_size; j++) {
            int value = (i < height && j < width ? derivative_buffer[j] : 0);
            int curr_row = i + pad_size + 1;
            int curr_col = j + pad_size + 1;
            dp[curr_row * dp_width + curr_col] =
                value
                + dp[(curr_row - 1) * dp_width + curr_col]
                + dp[curr_row * dp_width + curr_col - 1]
                - dp[(curr_row - 1) * dp_width + curr_col - 1];
        }
    }

    // Start from the last row in the dp matrix
    int prev_row = dp_height - 2;
    int curr_row = dp_height - 1;
    int curr_compute = pad_size + 1;
    int bottom = (curr_compute + pad_size) % dp_height;
    int top = (curr_compute + dp_height - pad_size - 1) % dp_height;
    for (int i = pad_size; i < height + pad_size; i++) {
        if (i < height) {
            derivative_callback(image_data, derivative_buffer, width, height, i, stride);
        }
        // First compute the values of dp for curr_row
        for (int j = 0; j < width + pad_size; j++) {
            int value = (i < height && j < width ? derivative_buffer[j] : 0);
            int curr_col = j + pad_size + 1;
            dp[curr_row * dp_width + curr_col] =
                value
                + dp[prev_row * dp_width + curr_col]
                + dp[curr_row * dp_width + curr_col - 1]
                - dp[prev_row * dp_width + curr_col - 1];
        }
        prev_row = curr_row;
        curr_row = (curr_row + 1 == dp_height ? 0 : curr_row + 1);

        // Then use the values to compute the square sum for the curr_compute row.
        for (int j = 0; j < width; j++) {
            int curr_col = j + pad_size + 1;
            int right = curr_col + pad_size;
            int left = curr_col - pad_size - 1;
            int result =
                dp[bottom * dp_width + right]
                - dp[bottom * dp_width + left]
                - dp[top * dp_width + right]
                + dp[top * dp_width + left];
            mask_data[(i - pad_size) * stride + j] = (result > mask_index);
        }
        curr_compute = (curr_compute + 1 == dp_height ? 0 : curr_compute + 1);
        bottom = (bottom + 1 == dp_height ? 0 : bottom + 1);
        top = (top + 1 == dp_height ? 0 : top + 1);
    }
}

static void get_spatial_mask(const VmafPicture *image, VmafPicture *mask,
                             uint32_t *dp, uint16_t *derivative_buffer, unsigned width, unsigned height,
                             VmafDerivativeCalculator derivative_callback) {
    uint16_t mask_index = get_mask_index(width, height, MASK_FILTER_SIZE);
    get_spatial_mask_for_index(image, mask, dp, derivative_buffer, mask_index, MASK_FILTER_SIZE, width, height, derivative_callback);
}

static float c_value_pixel(const uint16_t *histograms, uint16_t value, const int *diff_weights,
                           const int *diffs, uint16_t num_diffs, const uint16_t *tvi_thresholds, int histogram_col, int histogram_width) {
    uint16_t p_0 = histograms[value * histogram_width + histogram_col];
    float val, c_value = 0.0;
    for (uint16_t d = 0; d < num_diffs; d++) {
        if (value <= tvi_thresholds[d]) {
            uint16_t p_1 = histograms[(value + diffs[num_diffs + d + 1]) * histogram_width + histogram_col];
            uint16_t p_2 = histograms[(value + diffs[num_diffs - d - 1]) * histogram_width + histogram_col];
            if (p_1 > p_2) {
                val = (float)(diff_weights[d] * p_0 * p_1) / (p_1 + p_0);
            }
            else {
                val = (float)(diff_weights[d] * p_0 * p_2) / (p_2 + p_0);
            }

            if (val > c_value) {
                c_value = val;
            }
        }
    }

    return c_value;
}

static FORCE_INLINE inline void update_histogram_subtract_edge(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                          int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                          const uint16_t num_diffs, VmafRangeUpdater dec_range_callback) {
    uint16_t mask_val = mask[(i - pad_size - 1) * stride + j];
    if (mask_val) {
        uint16_t val = image[(i - pad_size - 1) * stride + j] + num_diffs;
        dec_range_callback(&histograms[val * width], MAX(j - pad_size, 0), MIN(j + pad_size + 1, width));
    }
}

static FORCE_INLINE inline void update_histogram_subtract(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                          int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                          const uint16_t num_diffs, VmafRangeUpdater dec_range_callback) {
    uint16_t mask_val = mask[(i - pad_size - 1) * stride + j];
    if (mask_val) {
        uint16_t val = image[(i - pad_size - 1) * stride + j] + num_diffs;
        dec_range_callback(&histograms[val * width], j - pad_size, j + pad_size + 1);
    }
}

static FORCE_INLINE inline void update_histogram_add_edge(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                     int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                     const uint16_t num_diffs, VmafRangeUpdater inc_range_callback) {
    uint16_t mask_val = mask[(i + pad_size) * stride + j];
    if (mask_val) {
        uint16_t val = image[(i + pad_size) * stride + j] + num_diffs;
        inc_range_callback(&histograms[val * width], MAX(j - pad_size, 0), MIN(j + pad_size + 1, width));
    }
}

static FORCE_INLINE inline void update_histogram_add(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                     int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                     const uint16_t num_diffs, VmafRangeUpdater inc_range_callback) {
    uint16_t mask_val = mask[(i + pad_size) * stride + j];
    if (mask_val) {
        uint16_t val = image[(i + pad_size) * stride + j] + num_diffs;
        inc_range_callback(&histograms[val * width], j - pad_size, j + pad_size + 1);
    }
}

static FORCE_INLINE inline void update_histogram_add_edge_first_pass(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                     int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                     const uint16_t num_diffs, VmafRangeUpdater inc_range_callback) {
    uint16_t mask_val = mask[i * stride + j];
    if (mask_val) {
        uint16_t val = image[i * stride + j] + num_diffs;
        inc_range_callback(&histograms[val * width], MAX(j - pad_size, 0), MIN(j + pad_size + 1, width));
    }
}

static FORCE_INLINE inline void update_histogram_add_first_pass(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                                     int i, int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                                     const uint16_t num_diffs, VmafRangeUpdater inc_range_callback) {
    uint16_t mask_val = mask[i * stride + j];
    if (mask_val) {
        uint16_t val = image[i * stride + j] + num_diffs;
        inc_range_callback(&histograms[val * width], j - pad_size, j + pad_size + 1);
    }
}

static FORCE_INLINE inline void calculate_c_values_row(float *c_values, uint16_t *histograms, uint16_t *image,
                                                       uint16_t *mask, int row, int width, ptrdiff_t stride,
                                                       const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                                       const int *diff_weights, const int *all_diffs) {
    for (int col = 0; col < width; col++) {
        if (mask[row * stride + col]) {
            c_values[row * width + col] = c_value_pixel(
                histograms, image[row * stride + col] + num_diffs, diff_weights, all_diffs, num_diffs, tvi_for_diff, col, width
            );
        }
    }
}

static void calculate_c_values(VmafPicture *pic, const VmafPicture *mask_pic,
                               float *c_values, uint16_t *histograms, uint16_t window_size,
                               const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                               const int *diff_weights, const int *all_diffs, int width, int height,
                               VmafRangeUpdater inc_range_callback, VmafRangeUpdater dec_range_callback) {

    uint16_t pad_size = window_size >> 1;
    const uint16_t num_bins = 1024 + (all_diffs[2*num_diffs] - all_diffs[0]);

    uint16_t *image = pic->data[0];
    uint16_t *mask = mask_pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;

    memset(c_values, 0.0, sizeof(float) * width * height);

    // Use a histogram for each pixel in width
    // histograms[i * width + j] accesses the j'th histogram, i'th value
    // This is done for cache optimization reasons
    memset(histograms, 0, width * num_bins * sizeof(uint16_t));

    // First pass: first pad_size rows
    for (int i = 0; i < pad_size; i++) {
        for (int j = 0; j < pad_size; j++) {
            update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
        for (int j = pad_size; j < width - pad_size - 1; j++) {
            update_histogram_add_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
        for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
            update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
    }

    // Iterate over all rows, unrolled into 3 loops to avoid conditions
    for (int i = 0; i < pad_size + 1; i++) {
        if (i + pad_size < height) {
            for (int j = 0; j < pad_size; j++) {
                update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
            }
            for (int j = pad_size; j < width - pad_size - 1; j++) {
                update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
            }
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
                update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
            }
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
    }
    for (int i = pad_size + 1; i < height - pad_size; i++) {
        for (int j = 0; j < pad_size; j++) {
            update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
        for (int j = pad_size; j < width - pad_size - 1; j++) {
            update_histogram_subtract(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
        for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
            update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, inc_range_callback);
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
    }
    for (int i = height - pad_size; i < height; i++) {
        if (i - pad_size - 1 >= 0) {
            for (int j = 0; j < pad_size; j++) {
                update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            }
            for (int j = pad_size; j < width - pad_size - 1; j++) {
                update_histogram_subtract(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            }
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
                update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, dec_range_callback);
            }
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
    }
}

static double average_topk_elements(const float *arr, int topk_elements) {
    double sum = 0;
    for (int i = 0; i < topk_elements; i++)
        sum += arr[i];

    return (double)sum / topk_elements;
}

static void quick_select(float *arr, int n, int k) {
    if (n == k) return;
    int left = 0;
    int right = n - 1;
    while (left < right) {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do {
            while (arr[i] > pivot) {
                i++;
            }
            while (arr[j] < pivot) {
                j--;
            }
            if (i <= j) {
                SWAP_FLOATS(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) {
            left = i;
        }
        if (k < i) {
            right = j;
        }
    }
}

static double spatial_pooling(float *c_values, double topk, unsigned width, unsigned height) {
    int num_elements = height * width;
    int topk_num_elements = clip(topk * num_elements, 1, num_elements);
    quick_select(c_values, num_elements, topk_num_elements);
    return average_topk_elements(c_values, topk_num_elements);
}

static FORCE_INLINE inline uint16_t get_pixels_in_window(uint16_t window_length) {
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

// Inner product weighting scores for each scale
static FORCE_INLINE inline double weight_scores_per_scale(double *scores_per_scale, uint16_t normalization) {
    double score = 0.0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);

    return score / normalization;
}

static int dump_c_values(FILE *heatmaps_files[], const float *c_values, int width, int height, int scale,
                         int window_size, const uint16_t num_diffs, const int *diff_weights, int frame) {
    int max_diff_weight = diff_weights[0];
    for (int i = 0; i < num_diffs; i++) {
        if (diff_weights[i] > max_diff_weight) {
            max_diff_weight = diff_weights[i];
        }
    }
    int max_c_value = max_diff_weight * window_size * window_size / 4;
    int max_16bit_value = (1 << 16) - 1;
    double scaling_value = (double)max_16bit_value / max_c_value;
    FILE *file = heatmaps_files[scale];
    uint16_t *to_write = malloc(width * sizeof(uint16_t));
    if (!to_write) return -ENOMEM;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            to_write[j] = (uint16_t)(scaling_value * c_values[i * width + j]);
        }
        ptrdiff_t offset = (frame * height + i) * width * sizeof(uint16_t);
        fseek(file, offset, SEEK_SET);
        fwrite((void*)to_write, sizeof(uint16_t), width, file);
    }
    free(to_write);
    return 0;
}

static int cambi_score(VmafPicture *pics, uint16_t window_size, double topk,
                       const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                       CambiBuffers buffers, VmafRangeUpdater inc_range_callback, VmafRangeUpdater dec_range_callback,
                       VmafDerivativeCalculator derivative_callback, double *score, bool write_heatmaps, FILE *heatmaps_files[],
                       int width, int height, int frame) {
    double scores_per_scale[NUM_SCALES];
    VmafPicture *image = &pics[0];
    VmafPicture *mask = &pics[1];

    int scaled_width = width;
    int scaled_height = height;

    get_spatial_mask(image, mask, buffers.mask_dp, buffers.derivative_buffer, width, height, derivative_callback);
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0) {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
            decimate(image, scaled_width, scaled_height);
            decimate(mask, scaled_width, scaled_height);
        }

        filter_mode(image, scaled_width, scaled_height, buffers.filter_mode_buffer);

        calculate_c_values(image, mask, buffers.c_values, buffers.c_values_histograms, window_size,
                           num_diffs, tvi_for_diff, buffers.diff_weights, buffers.all_diffs, scaled_width, scaled_height,
                           inc_range_callback, dec_range_callback);

        if (write_heatmaps) {
            int err = dump_c_values(heatmaps_files, buffers.c_values, scaled_width, scaled_height, scale, window_size,
                                    num_diffs, buffers.diff_weights, frame);
            if (err) return err;
        }

        scores_per_scale[scale] =
            spatial_pooling(buffers.c_values, topk, scaled_width, scaled_height);
    }

    uint16_t pixels_in_window = get_pixels_in_window(window_size);
    *score = weight_scores_per_scale(scores_per_scale, pixels_in_window);
    return 0;
}

static int preprocess_and_extract_cambi(CambiState *s, VmafPicture *pic, double *score, bool is_src, int frame) {
    int width = is_src ? s->src_width : s->enc_width;
    int height = is_src ? s->src_height : s->enc_height;
    int window_size = is_src ? s->src_window_size : s->window_size;
    int num_diffs = 1 << s->max_log_contrast;

    int err = cambi_preprocessing(pic, &s->pics[0], width, height, s->enc_bitdepth);
    if (err) return err;

    bool write_heatmaps = s->heatmaps_path && !is_src;
    err = cambi_score(s->pics, window_size, s->topk, num_diffs, s->buffers.tvi_for_diff,
                      s->buffers, s->inc_range_callback, s->dec_range_callback, s->derivative_callback, score, write_heatmaps, s->heatmaps_files, width, height, frame);
    if (err) return err;

    return 0;
}

static double combine_dist_src_scores(double dist_score, double src_score) {
    return MAX(0, dist_score - src_score);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector) {
    (void)ref_pic_90;
    (void)dist_pic_90;

    CambiState *s = fex->priv;
    double dist_score;
    int err = preprocess_and_extract_cambi(s, dist_pic, &dist_score, false, index);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector, "cambi", dist_score, index);
    if (err) return err;

    if (s->full_ref) {
        double src_score;
        int err = preprocess_and_extract_cambi(s, ref_pic, &src_score, true, index);
        if (err) return err;

        err = vmaf_feature_collector_append(feature_collector, "cambi_source", src_score, index);
        if (err) return err;

        double combined_score = combine_dist_src_scores(dist_score, src_score);
        err = vmaf_feature_collector_append(feature_collector, "cambi_full_reference", combined_score, index);
        if (err) return err;
    }

    return 0;
}

static int close_cambi(VmafFeatureExtractor *fex) {
    CambiState *s = fex->priv;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_unref(&s->pics[i]);

    aligned_free(s->buffers.tvi_for_diff);
    aligned_free(s->buffers.c_values);
    aligned_free(s->buffers.c_values_histograms);
    aligned_free(s->buffers.mask_dp);
    aligned_free(s->buffers.filter_mode_buffer);
    aligned_free(s->buffers.diffs_to_consider);
    aligned_free(s->buffers.diff_weights);
    aligned_free(s->buffers.all_diffs);
    aligned_free(s->buffers.derivative_buffer);

    if (s->heatmaps_path) {
        for (int scale = 0; scale < NUM_SCALES; scale++) {
            fclose(s->heatmaps_files[scale]);
        }
    }

    return err;
}

static const char *provided_features[] = {
    "cambi",
    NULL
};

VmafFeatureExtractor vmaf_fex_cambi = {
    .name = "cambi",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close_cambi,
    .priv_size = sizeof(CambiState),
    .provided_features = provided_features,
};
