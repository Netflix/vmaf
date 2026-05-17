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
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "common/macros.h"
#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"
#include "mkdirp.h"
#include "picture.h"

#if ARCH_X86
#include "x86/cambi_avx2.h"
#include "x86/cambi_avx512.h"
#elif ARCH_AARCH64
#include "arm64/cambi_neon.h"
#endif

/* Ratio of pixels for computation, must be 0 < topk <= 1.0 */
#define DEFAULT_CAMBI_TOPK_POOLING (0.6)

/* Window size to compute CAMBI: 65 corresponds to approximately 1 degree at 4k scale */
#define DEFAULT_CAMBI_WINDOW_SIZE (65)

/* Visibility threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886 */
#define DEFAULT_CAMBI_TVI (0.019)

/* Luminance value below which we assume any banding is not visible */
#define DEFAULT_CAMBI_VLT (0.0)

/* Max log contrast luma levels */
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)

/* If true, CAMBI will be run in full-reference mode and will use both the reference and distorted inputs */
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)

/* EOTF to use for the visibility threshold calculations. One of ['bt1886', 'pq']. Default: 'bt1886'. */
#define DEFAULT_CAMBI_EOTF ("bt1886")

/* CAMBI speed-up for resolutions >=1080p by down-scaling right after the sptial mask */
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)

#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)

/* Default maximum value allowed for CAMBI */
#define DEFAULT_CAMBI_MAX_VAL (1000.0)

#define NUM_SCALES 5
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

/* Suprathreshold contrast response */
static const int g_contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                           8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define SWAP_FLOATS(x, y)                                                                          \
    {                                                                                              \
        float temp = x;                                                                            \
        x = y;                                                                                     \
        y = temp;                                                                                  \
    }

#define PICS_BUFFER_SIZE 2
#define MASK_FILTER_SIZE 7

#include "cambi_reciprocal_lut.h"

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
    uint16_t v_band_base;
    uint16_t v_band_size;
} CambiBuffers;

typedef void (*VmafRangeUpdater)(uint16_t *arr, int left, int right);
typedef void (*VmafDerivativeCalculator)(const uint16_t *image_data, uint16_t *derivative_buffer,
                                         int width, int height, int row, int stride);
typedef void (*VmafFilterMode)(const VmafPicture *image, int width, int height, uint16_t *buffer);
typedef void (*VmafDecimate)(VmafPicture *image, unsigned width, unsigned height);
typedef void (*VmafCalcCValues)(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                uint16_t *histograms, uint16_t window_size,
                                const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                uint16_t vlt_luma, const int *diff_weights, const int *all_diffs,
                                int width, int height);

static void filter_mode(const VmafPicture *image, int width, int height, uint16_t *buffer);
static void decimate(VmafPicture *image, unsigned width, unsigned height);
static void calculate_c_values(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                               uint16_t *histograms, uint16_t window_size, const uint16_t num_diffs,
                               const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                               const int *diff_weights, const int *all_diffs, int width,
                               int height);
#if ARCH_X86
static void calculate_c_values_avx2(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                    uint16_t *histograms, uint16_t window_size,
                                    const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                    uint16_t vlt_luma, const int *diff_weights,
                                    const int *all_diffs, int width, int height);
#endif
#if ARCH_AARCH64
static void calculate_c_values_neon(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                    uint16_t *histograms, uint16_t window_size,
                                    const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                    uint16_t vlt_luma, const int *diff_weights,
                                    const int *all_diffs, int width, int height);
#endif

static void calculate_c_values_row(float *c_values, const uint16_t *histograms,
                                   const uint16_t *image, const uint16_t *mask, int row, int width,
                                   ptrdiff_t stride, const uint16_t num_diffs,
                                   const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                                   const int *diff_weights, const int *all_diffs,
                                   const float *reciprocal_lut);

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
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    uint16_t vlt_luma;
    uint16_t max_log_contrast;
    char *heatmaps_path;
    char *eotf;
    char *cambi_eotf;
    bool full_ref;
    int cambi_high_res_speedup;

    FILE *heatmaps_files[NUM_SCALES];
    VmafDerivativeCalculator derivative_callback;
    VmafCalcCValues calc_c_values_callback;
    VmafFilterMode filter_mode_callback;
    VmafDecimate decimate_callback;
    CambiBuffers buffers;
    VmafDictionary *feature_name_dict;
} CambiState;

static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(CambiState, cambi_max_val),
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
        .offset = offsetof(CambiState, enc_width),
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
        .offset = offsetof(CambiState, enc_height),
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
        .offset = offsetof(CambiState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "src_width",
        .help = "Source width. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srcw",
    },
    {
        .name = "src_height",
        .help = "Source height. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 200,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srch",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < topk <= 1.0",
        .offset = offsetof(CambiState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help =
            "Ratio of pixels for the spatial pooling computation, must be 0 < cambi_topk <= 1.0",
        .offset = offsetof(CambiState, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean",
        .offset = offsetof(CambiState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which we assume any banding is not visible",
        .offset = offsetof(CambiState, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
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
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
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
        .help =
            "If true, CAMBI will be run in full-reference mode and will be computed on both the reference and distorted inputs",
        .offset = offsetof(CambiState, full_ref),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = DEFAULT_CAMBI_FULL_REF_FLAG,
    },
    {
        .name = "eotf",
        .help =
            "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'",
        .offset = offsetof(CambiState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help =
            "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'. If both eotf and cambi_eotf are set, cambi_eotf takes precedence.",
        .offset = offsetof(CambiState, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {
        .name = "cambi_high_res_speedup",
        .help =
            "Speed up the processing by downsampling post spatial mask for resolutions >= 1080p. "
            "Min speed-up resolution possible values: [1080, 1440, 2160, 0]. Default: 0 (not applied)"
            "Note some loss of accuracy is expected with this speedup.",
        .offset = offsetof(CambiState, cambi_high_res_speedup),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_HIGH_RES_SPEEDUP,
        .min = 0,
        .max = CAMBI_4K_HEIGHT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "hrs",
    },
    {0}};

enum CambiTVIBisectFlag {
    CAMBI_TVI_BISECT_TOO_SMALL,
    CAMBI_TVI_BISECT_CORRECT,
    CAMBI_TVI_BISECT_TOO_BIG
};

static FORCE_INLINE inline int clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

static bool tvi_condition(int sample, int diff, double tvi_threshold, VmafLumaRange luma_range,
                          VmafEOTF eotf)
{
    double mean_luminance = vmaf_luminance_get_luminance(sample, luma_range, eotf);
    double diff_luminance = vmaf_luminance_get_luminance(sample + diff, luma_range, eotf);
    double delta_luminance = diff_luminance - mean_luminance;
    return (delta_luminance > tvi_threshold * mean_luminance);
}

static enum CambiTVIBisectFlag tvi_hard_threshold_condition(int sample, int diff,
                                                            double tvi_threshold,
                                                            VmafLumaRange luma_range, VmafEOTF eotf)
{
    bool condition;
    condition = tvi_condition(sample, diff, tvi_threshold, luma_range, eotf);
    if (!condition)
        return CAMBI_TVI_BISECT_TOO_BIG;

    condition = tvi_condition(sample + 1, diff, tvi_threshold, luma_range, eotf);
    if (condition)
        return CAMBI_TVI_BISECT_TOO_SMALL;

    return CAMBI_TVI_BISECT_CORRECT;
}

static int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth, VmafLumaRange luma_range,
                            VmafEOTF eotf)
{
    enum CambiTVIBisectFlag tvi_bisect;
    const int max_val = (1 << bitdepth) - 1;

    int foot = luma_range.foot;
    int head = luma_range.head;
    head = head - diff - 1;

    tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG)
        return 0;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT)
        return foot;

    tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL)
        return max_val;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT)
        return head;

    // bisect
    while (1) {
        int mid = foot + (head - foot) / 2;
        tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, luma_range, eotf);
        if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG) {
            head = mid;
        } else if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL) {
            foot = mid;
        } else if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) {
            return mid;
        } else { // Should never get here (todo: add assert)
            (void)0;
        }
    }
}

static int get_vlt_luma(double visibility_luminance_threshold, VmafLumaRange luma_range,
                        VmafEOTF eotf)
{
    // find the smallest luma value above the visibility_luminance_threshold

    uint16_t sample = luma_range.foot;

    while (vmaf_luminance_get_luminance(sample, luma_range, eotf) <
           visibility_luminance_threshold) {
        sample++;
    }
    if (sample == luma_range.foot) {
        return 0;
    } else {
        return sample;
    }
}

static FORCE_INLINE inline void adjust_window_size(uint16_t *window_size, unsigned input_width,
                                                   unsigned input_height,
                                                   bool cambi_high_res_speedup)
{
    // Adjustment weight: (input_width + input_height) / (CAMBI_4K_WIDTH + CAMBI_4K_HEIGHT)
    (*window_size) = (((*window_size) * (input_width + input_height)) / 375) >> 4;
    if (cambi_high_res_speedup) {
        (*window_size) = ((*window_size) + 1) >> 1;
    }
    // round up to odd
    *window_size |= 1;
}

static int set_contrast_arrays(const uint16_t num_diffs, uint16_t **diffs_to_consider,
                               int **diffs_weights, int **all_diffs)
{
    *diffs_to_consider = aligned_malloc(ALIGN_CEIL(sizeof(uint16_t)) * num_diffs, 16);
    if (!(*diffs_to_consider))
        return -ENOMEM;

    *diffs_weights = aligned_malloc(ALIGN_CEIL(sizeof(int)) * num_diffs, 32);
    if (!(*diffs_weights))
        return -ENOMEM;

    *all_diffs = aligned_malloc(ALIGN_CEIL(sizeof(int)) * (2 * num_diffs + 1), 32);
    if (!(*all_diffs))
        return -ENOMEM;

    for (int d = 0; d < num_diffs; d++) {
        (*diffs_to_consider)[d] = d + 1;
        (*diffs_weights)[d] = g_contrast_weights[d];
    }

    for (int d = -num_diffs; d <= num_diffs; d++)
        (*all_diffs)[d + num_diffs] = d;

    return 0;
}

static void increment_range(uint16_t *arr, int left, int right)
{
    for (int i = left; i < right; i++) {
        arr[i]++;
    }
}

static void decrement_range(uint16_t *arr, int left, int right)
{
    for (int i = left; i < right; i++) {
        arr[i]--;
    }
}

static void get_derivative_data_for_row(const uint16_t *image_data, uint16_t *derivative_buffer,
                                        int width, int height, int row, int stride)
{
    for (int col = 0; col < width; col++) {
        bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] ==
                                                              image_data[row * stride + col + 1]);
        bool vertical_derivative = (row == height - 1 || image_data[row * stride + col] ==
                                                             image_data[(row + 1) * stride + col]);
        derivative_buffer[col] = horizontal_derivative && vertical_derivative;
    }
}

#ifdef _WIN32
#define PATH_SEPARATOR '\\'
#else
#define PATH_SEPARATOR '/'
#endif

/**
 * `VmafFeatureExtractor::init` for the CAMBI banding-detection feature.
 *
 * Body groups four phases that are stacked here for upstream parity
 * with Netflix/vmaf's CAMBI implementation:
 *
 *   1. Fix-up encode/source dimensions. `enc_*` defaults to picture
 *      dimensions when unset; `src_*` likewise. The `enc > pic`
 *      down-clamp implements the "encode was downscaled, do not
 *      upscale back" behaviour — without it, CAMBI would report
 *      banding from upscale ringing instead of source banding.
 *      The mixed-direction `src vs enc` rejections (one larger, one
 *      smaller) catch user CLI input that combines a downscale and
 *      an upscale axis, which has no defined CAMBI interpretation.
 *   2. Resolve `cambi_high_res_speedup` against the encode pixel
 *      count. The selected tier (1080 / 1440 / 2160) only takes
 *      effect when the encode is *at least* that resolution; below
 *      the threshold it silently disables (set to 0). This matches
 *      the user-visible documentation in `docs/metrics/cambi.md`.
 *   3. Allocate the picture pyramid and per-pixel C-value scratch.
 *      `alloc_w / alloc_h` is `MAX(src, enc)` only when `full_ref`
 *      is set, otherwise enc-only — `full_ref` enables the source-
 *      reference pyramid that the FR-CAMBI variant requires.
 *      `num_bins` derives from `max_log_contrast` and gates the size
 *      of the per-row histogram buffer.
 *   4. Bind range-update + derivative callbacks, then upgrade them
 *      to AVX2 / AVX-512 / NEON via `vmaf_get_cpu_flags()`. The
 *      callbacks are the inner-loop hot path; the SIMD upgrade is
 *      runtime-selected because CAMBI ships in CPU-only builds.
 *
 * `s->vlt_luma` is set from the user-tunable `cambi_vis_lum_threshold`
 * via `get_vlt_luma()`; this is the "darkest luma to consider for
 * banding" cut-off and is critical for HDR inputs (see ADR section in
 * `docs/metrics/cambi.md`).
 *
 * Returns `0` on success, `-EINVAL` for invalid encode/source
 * dimensions or heatmap-path failures, `-ENOMEM` on allocation
 * failure. Allocations are NOT freed here on partial failure — the
 * matching `close()` walks the same buffer set and is null-tolerant.
 */
static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;

    CambiState *s = fex->priv;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

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

    // if the encode had been downscaled, there is no need to upscale it back to the encoding resolution
    if (s->enc_height > h || s->enc_width > w) {
        s->enc_width = w;
        s->enc_height = h;
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

    int enc_pix = s->enc_width * s->enc_height;
    switch (s->cambi_high_res_speedup) {
    case 1080:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p) {
            s->cambi_high_res_speedup = 0;
        }
        break;
    case 1440:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p) {
            s->cambi_high_res_speedup = 0;
        }
        break;
    case 2160:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p) {
            s->cambi_high_res_speedup = 0;
        }
        break;
    default:
        s->cambi_high_res_speedup = 0;
    }

    int alloc_w = s->full_ref ? MAX(s->src_width, s->enc_width) : s->enc_width;
    int alloc_h = s->full_ref ? MAX(s->src_height, s->enc_height) : s->enc_height;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        err |= vmaf_picture_alloc(&s->pics[i], VMAF_PIX_FMT_YUV400P, 10, alloc_w, alloc_h);
    }
    if (err)
        return err;

    const int num_diffs = 1 << s->max_log_contrast;

    set_contrast_arrays(num_diffs, &s->buffers.diffs_to_consider, &s->buffers.diff_weights,
                        &s->buffers.all_diffs);

    VmafLumaRange luma_range;
    err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;

    /* use cambi_eotf if it has a non-default value, else use eotf */
    const char *effective_eotf;
    if (strcmp(s->cambi_eotf, DEFAULT_CAMBI_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = s->eotf;
    }

    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err)
        return err;

    s->buffers.tvi_for_diff = aligned_malloc(ALIGN_CEIL(sizeof(uint16_t)) * num_diffs, 16);
    if (!s->buffers.tvi_for_diff)
        return -ENOMEM;
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.tvi_for_diff[d] = get_tvi_for_diff(s->buffers.diffs_to_consider[d],
                                                      s->tvi_threshold, 10, luma_range, eotf);
        s->buffers.tvi_for_diff[d] += num_diffs;
    }

    // get the largest luma value below cambi_vis_lum_threshold
    s->vlt_luma = get_vlt_luma(s->cambi_vis_lum_threshold, luma_range, eotf);

    s->src_window_size = s->window_size;
    adjust_window_size(&s->window_size, s->enc_width, s->enc_height,
                       (bool)s->cambi_high_res_speedup);
    adjust_window_size(&s->src_window_size, s->src_width, s->src_height,
                       (bool)s->cambi_high_res_speedup);

    int max_window = MAX(s->window_size, s->src_window_size);
    if (max_window * max_window >= CAMBI_RECIPROCAL_LUT_SIZE) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi: window_size %d too large for reciprocal LUT\n",
                 max_window);
        return -EINVAL;
    }

    s->buffers.c_values = aligned_malloc(ALIGN_CEIL(alloc_w * sizeof(float)) * alloc_h, 32);
    if (!s->buffers.c_values)
        return -ENOMEM;

    {
        int v_lo_signed = (int)s->vlt_luma - 3 * (int)num_diffs + 1;
        s->buffers.v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
        s->buffers.v_band_size =
            s->buffers.tvi_for_diff[num_diffs - 1] + 1 - s->buffers.v_band_base;
    }
    s->buffers.c_values_histograms =
        aligned_malloc(ALIGN_CEIL(alloc_w * s->buffers.v_band_size * sizeof(uint16_t)), 32);
    if (!s->buffers.c_values_histograms)
        return -ENOMEM;

    int pad_size = MASK_FILTER_SIZE >> 1;
    int dp_width = alloc_w + 2 * pad_size + 1;
    int dp_height = 2 * pad_size + 2;

    s->buffers.mask_dp =
        aligned_malloc(ALIGN_CEIL((size_t)dp_height * dp_width * sizeof(uint32_t)), 32);
    if (!s->buffers.mask_dp)
        return -ENOMEM;
    s->buffers.filter_mode_buffer = aligned_malloc(ALIGN_CEIL(3 * alloc_w * sizeof(uint16_t)), 32);
    if (!s->buffers.filter_mode_buffer)
        return -ENOMEM;
    s->buffers.derivative_buffer = aligned_malloc(ALIGN_CEIL(alloc_w * sizeof(uint16_t)), 32);
    if (!s->buffers.derivative_buffer)
        return -ENOMEM;

    if (s->heatmaps_path) {
        int mkdir_err = mkdirp(s->heatmaps_path, 0770);
        if (mkdir_err)
            return -EINVAL;
        char path[1024] = {0};
        int scaled_w = s->enc_width;
        int scaled_h = s->enc_height;
        for (int scale = 0; scale < NUM_SCALES; scale++) {
            (void)snprintf(path, sizeof(path), "%s%ccambi_heatmap_scale_%d_%dx%d_16b.gray",
                           s->heatmaps_path, PATH_SEPARATOR, scale, scaled_w, scaled_h);
            /* Mode 0644: owner-rw, group-r, other-r. Pinning the mode at
             * open(2) avoids the world-writable surface fopen(3) inherits
             * from a permissive umask. CodeQL cpp/world-writable-file-creation. */
#ifdef _WIN32
            int hfd = _open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0644);
#else
            int hfd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
            if (hfd < 0) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi: could not open heatmaps_path: %s\n", path);
                return -EINVAL;
            }
#ifdef _WIN32
            s->heatmaps_files[scale] = _fdopen(hfd, "wb");
#else
            s->heatmaps_files[scale] = fdopen(hfd, "w");
#endif
            if (!s->heatmaps_files[scale]) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi: could not open heatmaps_path: %s\n", path);
#ifdef _WIN32
                _close(hfd);
#else
                close(hfd);
#endif
                return -EINVAL;
            }
            scaled_w = (scaled_w + 1) >> 1;
            scaled_h = (scaled_h + 1) >> 1;
        }
    }

    s->derivative_callback = get_derivative_data_for_row;
    s->calc_c_values_callback = calculate_c_values;
    s->filter_mode_callback = filter_mode;
    s->decimate_callback = decimate;

#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        s->derivative_callback = get_derivative_data_for_row_avx2;
        s->calc_c_values_callback = calculate_c_values_avx2;
        s->filter_mode_callback = filter_mode_avx2;
        s->decimate_callback = decimate_avx2;
    }
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        s->derivative_callback = get_derivative_data_for_row_avx512;
    }
#elif ARCH_AARCH64
    {
        unsigned flags = vmaf_get_cpu_flags();
        if (flags & VMAF_ARM_CPU_FLAG_NEON) {
            s->derivative_callback = get_derivative_data_for_row_neon;
            s->calc_c_values_callback = calculate_c_values_neon;
        }
    }
#endif

    return err;
}

/* Preprocessing functions */

// For bitdepths <= 8.
static void decimate_generic_uint8_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic,
                                                      unsigned out_w, unsigned out_h)
{
    uint8_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0];
    ptrdiff_t out_stride = out_pic->stride[0] >> 1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];

    int shift_factor = 10 - pic->bpc;

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h) {
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
        unsigned ori_y = (int)lroundf(y);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)lroundf(x);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x] << shift_factor;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

// For the special case of bitdepth 9, which doesn't fit into uint8_t but has to be upscaled to 10b.
static void decimate_generic_9b_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic,
                                                   unsigned out_w, unsigned out_h)
{
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
        unsigned ori_y = (int)lroundf(y);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)lroundf(x);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x] << 1;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

// For bitdepths >= 10.
static void decimate_generic_uint16_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic,
                                                       unsigned out_w, unsigned out_h)
{
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
        } else {
            for (unsigned i = 0; i < out_h; i++) {
                for (unsigned j = 0; j < out_w; j++) {
                    out_data[i * out_stride + j] =
                        (data[i * stride + j] + rounding_offset) >> shift_factor;
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
        unsigned ori_y = (int)lroundf(y);
        float x = start_x;
        for (unsigned j = 0; j < out_w; j++) {
            unsigned ori_x = (int)lroundf(x);
            out_data[i * out_stride + j] =
                (data[ori_y * stride + ori_x] + rounding_offset) >> shift_factor;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

static void anti_dithering_filter(VmafPicture *pic, unsigned width, unsigned height)
{
    uint16_t *data = pic->data[0];
    int stride = pic->stride[0] >> 1;

    for (unsigned i = 0; i < height - 1; i++) {
        for (unsigned j = 0; j < width - 1; j++) {
            data[i * stride + j] = (data[i * stride + j] + data[i * stride + j + 1] +
                                    data[(i + 1) * stride + j] + data[(i + 1) * stride + j + 1]) >>
                                   2;
        }

        // Last column
        unsigned j = width - 1;
        data[i * stride + j] = (data[i * stride + j] + data[(i + 1) * stride + j]) >> 1;
    }

    // Last row
    unsigned i = height - 1;
    for (unsigned j = 0; j < width - 1; j++) {
        data[i * stride + j] = (data[i * stride + j] + data[i * stride + j + 1]) >> 1;
    }
}

static int validate_image_lbd(const VmafPicture *pic)
{
    int bpc = pic->bpc;
    if (bpc == 8)
        return 0;
    uint8_t max_val = (1 << bpc) - 1;
    int channel = 0;
    uint8_t *data = (uint8_t *)pic->data[channel];
    size_t stride = pic->stride[channel];
    for (unsigned i = 0; i < pic->h[channel]; i++) {
        for (unsigned j = 0; j < pic->w[channel]; j++) {
            if (data[i * stride + j] > max_val) {
                vmaf_log(
                    VMAF_LOG_LEVEL_ERROR,
                    "Invalid input. The input contains values greater than %d, which exceeds the maximum value for a %d-bit depth format.",
                    max_val, bpc);
                return -EINVAL;
            }
        }
    }
    return 0;
}

static int validate_image_hbd(const VmafPicture *pic)
{
    int bpc = pic->bpc;
    if (bpc == 16)
        return 0;
    uint16_t max_val = (1 << bpc) - 1;
    int channel = 0;
    uint16_t *data = (uint16_t *)pic->data[channel];
    size_t stride = pic->stride[channel] / 2;
    for (unsigned i = 0; i < pic->h[channel]; i++) {
        for (unsigned j = 0; j < pic->w[channel]; j++) {
            if (data[i * stride + j] > max_val) {
                vmaf_log(
                    VMAF_LOG_LEVEL_ERROR,
                    "Invalid input. The input contains values greater than %d, which exceeds the maximum value for a %d-bit depth format.",
                    max_val, bpc);
                return -EINVAL;
            }
        }
    }
    return 0;
}

static int validate_image(const VmafPicture *pic)
{
    if (pic->bpc <= 8) {
        return validate_image_lbd(pic);
    } else {
        return validate_image_hbd(pic);
    }
}

static int cambi_preprocessing(const VmafPicture *image, VmafPicture *preprocessed, int width,
                               int height, int enc_bitdepth)
{
    if (validate_image(image)) {
        return -EINVAL;
    }
    if (image->bpc >= 10) {
        decimate_generic_uint16_and_convert_to_10b(image, preprocessed, width, height);
    } else {
        if (image->bpc <= 8) {
            decimate_generic_uint8_and_convert_to_10b(image, preprocessed, width, height);
        } else {
            decimate_generic_9b_and_convert_to_10b(image, preprocessed, width, height);
        }
    }
    if (enc_bitdepth < 10) {
        anti_dithering_filter(preprocessed, width, height);
    }

    return 0;
}

/* Banding detection functions */
static void decimate(VmafPicture *image, unsigned width, unsigned height)
{
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            data[i * stride + j] = data[(i << 1) * stride + (j << 1)];
        }
    }
}

static inline uint16_t min3(uint16_t a, uint16_t b, uint16_t c)
{
    if (a <= b && a <= c)
        return a;
    if (b <= c)
        return b;
    return c;
}

static inline uint16_t mode3(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c)
        return a;
    if (b == c)
        return b;
    return min3(a, b, c);
}

static void filter_mode(const VmafPicture *image, int width, int height, uint16_t *buffer)
{
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        buffer[curr_line * width + 0] = data[i * stride + 0];
        for (int j = 1; j < width - 1; j++) {
            buffer[curr_line * width + j] =
                mode3(data[i * stride + j - 1], data[i * stride + j], data[i * stride + j + 1]);
        }
        buffer[curr_line * width + width - 1] = data[i * stride + width - 1];

        if (i > 1) {
            for (int j = 0; j < width; j++) {
                data[(i - 1) * stride + j] =
                    mode3(buffer[0 * width + j], buffer[1 * width + j], buffer[2 * width + j]);
            }
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

static FORCE_INLINE inline uint16_t ceil_log2(uint32_t num)
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

static FORCE_INLINE inline uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                                                   uint16_t filter_size)
{
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1) >> 1;
}

/*
* This function calculates the horizontal and vertical derivatives of the image using 2x1 and 1x2 kernels.
* We say a pixel has zero_derivative=1 if it's equal to its right and bottom neighbours, and =0 otherwise (edges also count as "equal").
* This function then computes the sum of zero_derivative on the filter_size x filter_size square around each pixel
* and stores 1 into the corresponding mask index iff this number is larger than mask_index.
* To calculate the square sums, it uses a dynamic programming algorithm based on inclusion-exclusion.
* To save memory, it uses a DP matrix of only the necessary size, rather than the full matrix, and indexes its rows cyclically.
*/
/*
 * Computes one DP row using a 1D row prefix sum + element-wise add of the previous DP row.
 * Equivalent to the SAT recurrence:
 *   dp[r][c] = a[r][c] + dp[r-1][c] + dp[r][c-1] - dp[r-1][c-1]
 * but rewritten as:
 *   R[c] = a[r][c] + R[c-1]                  (1D prefix sum of derivative)
 *   dp[r][c] = dp[r-1][c] + R[c]             (element-wise add, no inter-column dep)
 */
static FORCE_INLINE void compute_dp_row(uint32_t *dp_curr, const uint32_t *dp_prev,
                                        const uint16_t *deriv, int width, int pad_size,
                                        bool deriv_valid)
{
    uint32_t prefix = 0;
    int dp_offset = pad_size + 1;
    int actual_width = deriv_valid ? width : 0;
    int j;
    for (j = 0; j < actual_width; j++) {
        prefix += deriv[j];
        dp_curr[dp_offset + j] = dp_prev[dp_offset + j] + prefix;
    }
    int n = width + pad_size;
    for (; j < n; j++) {
        dp_curr[dp_offset + j] = dp_prev[dp_offset + j] + prefix;
    }
}

/*
 * For each output column j in [0, width), computes
 *   result = dp_bottom[j + delta] - dp_bottom[j] - dp_top[j + delta] + dp_top[j]
 *   mask_row[j] = (result > mask_index)
 * where delta = 2*pad_size + 1.
 */
static FORCE_INLINE void compute_mask_row(uint16_t *mask_row, const uint32_t *dp_bottom,
                                          const uint32_t *dp_top, int width, int pad_size,
                                          uint32_t mask_index)
{
    const int delta = 2 * pad_size + 1;
    for (int j = 0; j < width; j++) {
        uint32_t result = dp_bottom[j + delta] + dp_top[j] - dp_bottom[j] - dp_top[j + delta];
        mask_row[j] = (uint16_t)(result > mask_index);
    }
}

static void get_spatial_mask_for_index(const VmafPicture *image, VmafPicture *mask, uint32_t *dp,
                                       uint16_t *derivative_buffer, uint16_t mask_index,
                                       uint16_t filter_size, int width, int height,
                                       VmafDerivativeCalculator derivative_callback)
{
    uint16_t pad_size = filter_size >> 1;
    uint16_t *image_data = image->data[0];
    uint16_t *mask_data = mask->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;

    int dp_width = width + 2 * pad_size + 1;
    int dp_height = 2 * pad_size + 2;
    memset(dp, 0, (size_t)dp_width * dp_height * sizeof(uint32_t));

    // Initial computation: fill dp except for the last row
    for (int i = 0; i < pad_size; i++) {
        bool deriv_valid = (i < height);
        if (deriv_valid) {
            derivative_callback(image_data, derivative_buffer, width, height, i, stride);
        }
        int curr_row = i + pad_size + 1;
        compute_dp_row(&dp[curr_row * dp_width], &dp[(curr_row - 1) * dp_width], derivative_buffer,
                       width, pad_size, deriv_valid);
    }

    // Start from the last row in the dp matrix
    int prev_row = dp_height - 2;
    int curr_row = dp_height - 1;
    int curr_compute = pad_size + 1;
    int bottom = (curr_compute + pad_size) % dp_height;
    int top = (curr_compute + dp_height - pad_size - 1) % dp_height;
    for (int i = pad_size; i < height + pad_size; i++) {
        bool deriv_valid = (i < height);
        if (deriv_valid) {
            derivative_callback(image_data, derivative_buffer, width, height, i, stride);
        }
        // First compute the values of dp for curr_row
        compute_dp_row(&dp[curr_row * dp_width], &dp[prev_row * dp_width], derivative_buffer, width,
                       pad_size, deriv_valid);
        prev_row = curr_row;
        curr_row = (curr_row + 1 == dp_height ? 0 : curr_row + 1);

        // Then use the values to compute the square sum for the curr_compute row.
        compute_mask_row(&mask_data[(i - pad_size) * stride], &dp[bottom * dp_width],
                         &dp[top * dp_width], width, pad_size, mask_index);
        curr_compute = (curr_compute + 1 == dp_height ? 0 : curr_compute + 1);
        bottom = (bottom + 1 == dp_height ? 0 : bottom + 1);
        top = (top + 1 == dp_height ? 0 : top + 1);
    }
}

static void get_spatial_mask(const VmafPicture *image, VmafPicture *mask, uint32_t *dp,
                             uint16_t *derivative_buffer, unsigned width, unsigned height,
                             VmafDerivativeCalculator derivative_callback)
{
    uint16_t mask_index = get_mask_index(width, height, MASK_FILTER_SIZE);
    get_spatial_mask_for_index(image, mask, dp, derivative_buffer, mask_index, MASK_FILTER_SIZE,
                               width, height, derivative_callback);
}

static float c_value_pixel(const uint16_t *histograms, uint16_t value, const int *diff_weights,
                           const int *diffs, uint16_t num_diffs, const uint16_t *tvi_thresholds,
                           uint16_t vlt_luma, uint16_t v_band_offset_val, uint16_t v_band_size,
                           int histogram_col, int histogram_width)
{
    int compact_v_signed = (int)value - (int)v_band_offset_val;
    if ((unsigned)compact_v_signed >= v_band_size)
        return 0.0f;
    uint16_t compact_v = (uint16_t)compact_v_signed;
    uint16_t p_0 = histograms[compact_v * histogram_width + histogram_col];
    float val;
    float c_value = 0.0;
    for (uint16_t d = 0; d < num_diffs; d++) {
        if ((value <= tvi_thresholds[d]) && ((value + diffs[num_diffs + d + 1]) > vlt_luma)) {
            int idx1 = compact_v_signed + diffs[num_diffs + d + 1];
            int idx2 = compact_v_signed + diffs[num_diffs - d - 1];
            uint16_t p_1 = histograms[idx1 * histogram_width + histogram_col];
            uint16_t p_2 = (idx2 >= 0) ? histograms[idx2 * histogram_width + histogram_col] : 0;
            if (p_1 > p_2) {
                val = (float)(diff_weights[d] * p_0 * p_1) * reciprocal_lut[p_1 + p_0];
            } else {
                val = (float)(diff_weights[d] * p_0 * p_2) * reciprocal_lut[p_2 + p_0];
            }

            if (val > c_value) {
                c_value = val;
            }
        }
    }

    return c_value;
}

// A pixel with raw value v contributes to a histogram cell that is queried by
// some output pixel only if some d-iteration in c_value_pixel can fire for an
// output value v_out related to v. Tracing the gating condition (v_out <=
// tvi_thresholds[d]) && (v_out + (d+1) > vlt_luma) over all d and all three
// query roles (p_0, p_1, p_2) shows v needs to satisfy
//     vlt_luma - 3*num_diffs < v <= tvi_for_diff[num_diffs - 1]
// (where tvi_for_diff is already in adjusted = raw + num_diffs space).
// Pixels outside this band can be skipped: the cells they would update are
// only ever read by output pixels whose c_value would be 0 regardless. The
// skip is therefore bit-identical.
//
// The two-sided bounds check is collapsed to a single unsigned compare:
//     ((uint16_t)(v - v_band_base) >= v_band_size)
// where v_band_base = max(0, vlt_luma - 3*num_diffs + 1) and v_band_size is
// the count of values in the useful band. Compiler emits ~1 sub + 1 cmp per
// pixel with no second branch, which costs less when the skip rarely fires.

static FORCE_INLINE inline void
update_histogram_subtract_edge(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j,
                               int width, ptrdiff_t stride, uint16_t pad_size,
                               const uint16_t num_diffs, uint16_t v_band_base, uint16_t v_band_size,
                               VmafRangeUpdater dec_range_callback)
{
    if (!mask[(i - pad_size - 1) * stride + j])
        return;
    uint16_t v = image[(i - pad_size - 1) * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    dec_range_callback(&histograms[row * width], MAX(j - pad_size, 0),
                       MIN(j + pad_size + 1, width));
}

static FORCE_INLINE inline void
update_histogram_subtract(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j,
                          int width, ptrdiff_t stride, uint16_t pad_size, const uint16_t num_diffs,
                          uint16_t v_band_base, uint16_t v_band_size,
                          VmafRangeUpdater dec_range_callback)
{
    if (!mask[(i - pad_size - 1) * stride + j])
        return;
    uint16_t v = image[(i - pad_size - 1) * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    dec_range_callback(&histograms[row * width], j - pad_size, j + pad_size + 1);
}

static FORCE_INLINE inline void
update_histogram_add_edge(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j,
                          int width, ptrdiff_t stride, uint16_t pad_size, const uint16_t num_diffs,
                          uint16_t v_band_base, uint16_t v_band_size,
                          VmafRangeUpdater inc_range_callback)
{
    if (!mask[(i + pad_size) * stride + j])
        return;
    uint16_t v = image[(i + pad_size) * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    inc_range_callback(&histograms[row * width], MAX(j - pad_size, 0),
                       MIN(j + pad_size + 1, width));
}

static FORCE_INLINE inline void update_histogram_add(uint16_t *histograms, uint16_t *image,
                                                     uint16_t *mask, int i, int j, int width,
                                                     ptrdiff_t stride, uint16_t pad_size,
                                                     const uint16_t num_diffs, uint16_t v_band_base,
                                                     uint16_t v_band_size,
                                                     VmafRangeUpdater inc_range_callback)
{
    if (!mask[(i + pad_size) * stride + j])
        return;
    uint16_t v = image[(i + pad_size) * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    inc_range_callback(&histograms[row * width], j - pad_size, j + pad_size + 1);
}

static FORCE_INLINE inline void
update_histogram_add_edge_first_pass(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i,
                                     int j, int width, ptrdiff_t stride, uint16_t pad_size,
                                     const uint16_t num_diffs, uint16_t v_band_base,
                                     uint16_t v_band_size, VmafRangeUpdater inc_range_callback)
{
    if (!mask[i * stride + j])
        return;
    uint16_t v = image[i * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    inc_range_callback(&histograms[row * width], MAX(j - pad_size, 0),
                       MIN(j + pad_size + 1, width));
}

static FORCE_INLINE inline void
update_histogram_add_first_pass(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j,
                                int width, ptrdiff_t stride, uint16_t pad_size,
                                const uint16_t num_diffs, uint16_t v_band_base,
                                uint16_t v_band_size, VmafRangeUpdater inc_range_callback)
{
    if (!mask[i * stride + j])
        return;
    uint16_t v = image[i * stride + j];
    if ((uint16_t)(v - v_band_base) >= v_band_size)
        return;
    (void)num_diffs;
    uint16_t row = v - v_band_base;
    inc_range_callback(&histograms[row * width], j - pad_size, j + pad_size + 1);
}

// Fused subtract-then-add for interior columns. Skips when both pixels are
// in-band with the same value: the decrement and increment cancel exactly.
static FORCE_INLINE inline void uh_slide(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                         int i, int j, int width, ptrdiff_t stride,
                                         uint16_t pad_size, uint16_t v_band_base,
                                         uint16_t v_band_size, VmafRangeUpdater inc_fn,
                                         VmafRangeUpdater dec_fn)
{
    bool sub_valid = mask[(i - pad_size - 1) * stride + j];
    bool add_valid = mask[(i + pad_size) * stride + j];
    uint16_t v_sub = sub_valid ? image[(i - pad_size - 1) * stride + j] : 0;
    uint16_t v_add = add_valid ? image[(i + pad_size) * stride + j] : 0;
    bool sub_in = sub_valid && (uint16_t)(v_sub - v_band_base) < v_band_size;
    bool add_in = add_valid && (uint16_t)(v_add - v_band_base) < v_band_size;
    if (sub_in && add_in && v_sub == v_add)
        return;
    if (sub_in)
        dec_fn(&histograms[(v_sub - v_band_base) * width], j - pad_size, j + pad_size + 1);
    if (add_in)
        inc_fn(&histograms[(v_add - v_band_base) * width], j - pad_size, j + pad_size + 1);
}

// Same as uh_slide but with clamped (edge) column ranges.
static FORCE_INLINE inline void uh_slide_edge(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                              int i, int j, int width, ptrdiff_t stride,
                                              uint16_t pad_size, uint16_t v_band_base,
                                              uint16_t v_band_size, VmafRangeUpdater inc_fn,
                                              VmafRangeUpdater dec_fn)
{
    bool sub_valid = mask[(i - pad_size - 1) * stride + j];
    bool add_valid = mask[(i + pad_size) * stride + j];
    uint16_t v_sub = sub_valid ? image[(i - pad_size - 1) * stride + j] : 0;
    uint16_t v_add = add_valid ? image[(i + pad_size) * stride + j] : 0;
    bool sub_in = sub_valid && (uint16_t)(v_sub - v_band_base) < v_band_size;
    bool add_in = add_valid && (uint16_t)(v_add - v_band_base) < v_band_size;
    if (sub_in && add_in && v_sub == v_add)
        return;
    int left = MAX(j - pad_size, 0), right = MIN(j + pad_size + 1, width);
    if (sub_in)
        dec_fn(&histograms[(v_sub - v_band_base) * width], left, right);
    if (add_in)
        inc_fn(&histograms[(v_add - v_band_base) * width], left, right);
}

static void calculate_c_values_row(float *c_values, const uint16_t *histograms,
                                   const uint16_t *image, const uint16_t *mask, int row, int width,
                                   ptrdiff_t stride, const uint16_t num_diffs,
                                   const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                                   const int *diff_weights, const int *all_diffs,
                                   const float *reciprocal_lut)
{
    (void)reciprocal_lut; // scalar version uses the file-scope LUT directly
    int v_lo_signed = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    uint16_t v_band_size = tvi_for_diff[num_diffs - 1] + 1 - v_band_base;
    uint16_t v_band_offset_val = v_band_base + num_diffs;
    for (int col = 0; col < width; col++) {
        if (mask[row * stride + col]) {
            c_values[row * width + col] = c_value_pixel(
                histograms, image[row * stride + col] + num_diffs, diff_weights, all_diffs,
                num_diffs, tvi_for_diff, vlt_luma, v_band_offset_val, v_band_size, col, width);
        }
    }
}

/* Frame-level c-values driver. Templated by macro on the increment/decrement/calc-row
 * triple so each ISA dispatch lowers to a flat call (not an indirect one) per cell. */
#define CAMBI_CALC_C_VALUES_BODY(INC, DEC, ROW)                                                    \
    do {                                                                                           \
        uint16_t pad_size = window_size >> 1;                                                      \
                                                                                                   \
        int v_lo_signed = (int)vlt_luma - 3 * (int)num_diffs + 1;                                  \
        uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;                        \
        uint16_t v_band_size = tvi_for_diff[num_diffs - 1] + 1 - v_band_base;                      \
                                                                                                   \
        uint16_t *image = pic->data[0];                                                            \
        uint16_t *mask = mask_pic->data[0];                                                        \
        ptrdiff_t stride = pic->stride[0] >> 1;                                                    \
                                                                                                   \
        memset(c_values, 0.0, sizeof(float) * width * height);                                     \
        memset(histograms, 0, width * v_band_size * sizeof(uint16_t));                             \
                                                                                                   \
        for (int i = 0; i < pad_size; i++) {                                                       \
            for (int j = 0; j < pad_size; j++) {                                                   \
                update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, \
                                                     pad_size, num_diffs, v_band_base,             \
                                                     v_band_size, (INC));                          \
            }                                                                                      \
            for (int j = pad_size; j < width - pad_size - 1; j++) {                                \
                update_histogram_add_first_pass(histograms, image, mask, i, j, width, stride,      \
                                                pad_size, num_diffs, v_band_base, v_band_size,     \
                                                (INC));                                            \
            }                                                                                      \
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {                    \
                update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, \
                                                     pad_size, num_diffs, v_band_base,             \
                                                     v_band_size, (INC));                          \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        for (int i = 0; i < pad_size + 1; i++) {                                                   \
            if (i + pad_size < height) {                                                           \
                for (int j = 0; j < pad_size; j++) {                                               \
                    update_histogram_add_edge(histograms, image, mask, i, j, width, stride,        \
                                              pad_size, num_diffs, v_band_base, v_band_size,       \
                                              (INC));                                              \
                }                                                                                  \
                for (int j = pad_size; j < width - pad_size - 1; j++) {                            \
                    update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size,   \
                                         num_diffs, v_band_base, v_band_size, (INC));              \
                }                                                                                  \
                for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {                \
                    update_histogram_add_edge(histograms, image, mask, i, j, width, stride,        \
                                              pad_size, num_diffs, v_band_base, v_band_size,       \
                                              (INC));                                              \
                }                                                                                  \
            }                                                                                      \
            (ROW)(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff,    \
                  vlt_luma, diff_weights, all_diffs, reciprocal_lut);                              \
        }                                                                                          \
        for (int i = pad_size + 1; i < height - pad_size; i++) {                                   \
            for (int j = 0; j < pad_size; j++)                                                     \
                uh_slide_edge(histograms, image, mask, i, j, width, stride, pad_size, v_band_base, \
                              v_band_size, (INC), (DEC));                                          \
            for (int j = pad_size; j < width - pad_size - 1; j++)                                  \
                uh_slide(histograms, image, mask, i, j, width, stride, pad_size, v_band_base,      \
                         v_band_size, (INC), (DEC));                                               \
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++)                      \
                uh_slide_edge(histograms, image, mask, i, j, width, stride, pad_size, v_band_base, \
                              v_band_size, (INC), (DEC));                                          \
            (ROW)(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff,    \
                  vlt_luma, diff_weights, all_diffs, reciprocal_lut);                              \
        }                                                                                          \
        for (int i = height - pad_size; i < height; i++) {                                         \
            if (i - pad_size - 1 >= 0) {                                                           \
                for (int j = 0; j < pad_size; j++) {                                               \
                    update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride,   \
                                                   pad_size, num_diffs, v_band_base, v_band_size,  \
                                                   (DEC));                                         \
                }                                                                                  \
                for (int j = pad_size; j < width - pad_size - 1; j++) {                            \
                    update_histogram_subtract(histograms, image, mask, i, j, width, stride,        \
                                              pad_size, num_diffs, v_band_base, v_band_size,       \
                                              (DEC));                                              \
                }                                                                                  \
                for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {                \
                    update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride,   \
                                                   pad_size, num_diffs, v_band_base, v_band_size,  \
                                                   (DEC));                                         \
                }                                                                                  \
            }                                                                                      \
            (ROW)(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff,    \
                  vlt_luma, diff_weights, all_diffs, reciprocal_lut);                              \
        }                                                                                          \
    } while (0)

static void calculate_c_values(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                               uint16_t *histograms, uint16_t window_size, const uint16_t num_diffs,
                               const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                               const int *diff_weights, const int *all_diffs, int width, int height)
{
    CAMBI_CALC_C_VALUES_BODY(increment_range, decrement_range, calculate_c_values_row);
}

#if ARCH_X86
static void calculate_c_values_avx2(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                    uint16_t *histograms, uint16_t window_size,
                                    const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                    uint16_t vlt_luma, const int *diff_weights,
                                    const int *all_diffs, int width, int height)
{
    CAMBI_CALC_C_VALUES_BODY(cambi_increment_range_avx2, cambi_decrement_range_avx2,
                             calculate_c_values_row_avx2);
}
#endif

#if ARCH_AARCH64
/* NEON has bit-exact range updaters but no calculate_c_values_row_neon yet; the
 * row stage stays scalar (pending fork follow-up T7-NN). */
static void calculate_c_values_neon(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                    uint16_t *histograms, uint16_t window_size,
                                    const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                    uint16_t vlt_luma, const int *diff_weights,
                                    const int *all_diffs, int width, int height)
{
    CAMBI_CALC_C_VALUES_BODY(cambi_increment_range_neon, cambi_decrement_range_neon,
                             calculate_c_values_row);
}
#endif

#undef CAMBI_CALC_C_VALUES_BODY

static double average_topk_elements(const float *arr, int topk_elements)
{
    double sum = 0;
    for (int i = 0; i < topk_elements; i++)
        sum += arr[i];

    return (double)sum / topk_elements;
}

static void quick_select(float *arr, int n, int k)
{
    if (n == k)
        return;
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

static double spatial_pooling(float *c_values, double topk, unsigned width, unsigned height)
{
    int num_elements = height * width;
    int topk_num_elements = clip(topk * num_elements, 1, num_elements);
    quick_select(c_values, num_elements, topk_num_elements);
    return average_topk_elements(c_values, topk_num_elements);
}

static FORCE_INLINE inline uint16_t get_pixels_in_window(uint16_t window_length)
{
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

// Inner product weighting scores for each scale
static FORCE_INLINE inline double weight_scores_per_scale(double *scores_per_scale,
                                                          uint16_t normalization)
{
    double score = 0.0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);

    return score / normalization;
}

static int dump_c_values(FILE *heatmaps_files[], const float *c_values, int width, int height,
                         int scale, int window_size, const uint16_t num_diffs,
                         const int *diff_weights, int frame)
{
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
    if (!to_write)
        return -ENOMEM;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            to_write[j] = (uint16_t)(scaling_value * c_values[i * width + j]);
        }
        ptrdiff_t offset = ((ptrdiff_t)frame * height + i) * width * sizeof(uint16_t);
        (void)fseek(file, offset, SEEK_SET);
        (void)fwrite((void *)to_write, sizeof(uint16_t), width, file);
    }
    free(to_write);
    return 0;
}

static int cambi_score(VmafPicture *pics, uint16_t window_size, double topk,
                       const uint16_t num_diffs, const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                       CambiBuffers buffers, VmafDerivativeCalculator derivative_callback,
                       VmafCalcCValues calc_c_values_callback, VmafFilterMode filter_mode_callback,
                       VmafDecimate decimate_callback, double *score, bool write_heatmaps,
                       FILE *heatmaps_files[], int width, int height, int frame,
                       bool cambi_high_res_speedup)
{

    double scores_per_scale[NUM_SCALES];
    VmafPicture *image = &pics[0];
    VmafPicture *mask = &pics[1];

    int scaled_width = width;
    int scaled_height = height;

    get_spatial_mask(image, mask, buffers.mask_dp, buffers.derivative_buffer, width, height,
                     derivative_callback);
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || cambi_high_res_speedup) {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
            decimate_callback(image, scaled_width, scaled_height);
            decimate_callback(mask, scaled_width, scaled_height);
        }

        filter_mode_callback(image, scaled_width, scaled_height, buffers.filter_mode_buffer);

        calc_c_values_callback(image, mask, buffers.c_values, buffers.c_values_histograms,
                               window_size, num_diffs, tvi_for_diff, vlt_luma, buffers.diff_weights,
                               buffers.all_diffs, scaled_width, scaled_height);

        if (write_heatmaps) {
            int err = dump_c_values(heatmaps_files, buffers.c_values, scaled_width, scaled_height,
                                    scale, window_size, num_diffs, buffers.diff_weights, frame);
            if (err)
                return err;
        }

        scores_per_scale[scale] =
            spatial_pooling(buffers.c_values, topk, scaled_width, scaled_height);
    }

    uint16_t pixels_in_window = get_pixels_in_window(window_size);
    *score = weight_scores_per_scale(scores_per_scale, pixels_in_window);
    return 0;
}

static int preprocess_and_extract_cambi(CambiState *s, VmafPicture *pic, double *score, bool is_src,
                                        int frame)
{
    int width = is_src ? s->src_width : s->enc_width;
    int height = is_src ? s->src_height : s->enc_height;
    int window_size = is_src ? s->src_window_size : s->window_size;
    int num_diffs = 1 << s->max_log_contrast;

    int err = cambi_preprocessing(pic, &s->pics[0], width, height, s->enc_bitdepth);
    if (err)
        return err;

    bool write_heatmaps = s->heatmaps_path && !is_src;
    double topk;
    /* use the original topk setting if it has a non-default value, else use the cambi_topk one */
    if (s->topk != DEFAULT_CAMBI_TOPK_POOLING) {
        topk = s->topk;
    } else {
        topk = s->cambi_topk;
    }
    err = cambi_score(s->pics, window_size, topk, num_diffs, s->buffers.tvi_for_diff, s->vlt_luma,
                      s->buffers, s->derivative_callback, s->calc_c_values_callback,
                      s->filter_mode_callback, s->decimate_callback, score, write_heatmaps,
                      s->heatmaps_files, width, height, frame, (bool)s->cambi_high_res_speedup);

    if (err)
        return err;

    return 0;
}

static double combine_dist_src_scores(double dist_score, double src_score)
{
    return MAX(0, dist_score - src_score);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;

    CambiState *s = fex->priv;
    double dist_score;
    int err = preprocess_and_extract_cambi(s, dist_pic, &dist_score, false, index);
    if (err)
        return err;

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "Cambi_feature_cambi_score",
                                                  MIN(dist_score, s->cambi_max_val), index);
    if (err)
        return err;

    if (s->full_ref) {
        double src_score;
        int src_err = preprocess_and_extract_cambi(s, ref_pic, &src_score, true, index);
        if (src_err)
            return src_err;

        src_err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "cambi_source",
                                                          MIN(src_score, s->cambi_max_val), index);
        if (src_err)
            return src_err;

        double combined_score = combine_dist_src_scores(dist_score, src_score);
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "cambi_full_reference",
                                                      MIN(combined_score, s->cambi_max_val), index);
        if (err)
            return err;
    }

    return 0;
}

static int close_cambi(VmafFeatureExtractor *fex)
{
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
            (void)fclose(s->heatmaps_files[scale]);
        }
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return err;
}

static const char *provided_features[] = {"Cambi_feature_cambi_score", NULL};

VmafFeatureExtractor vmaf_fex_cambi = {
    .name = "cambi",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close_cambi,
    .priv_size = sizeof(CambiState),
    .provided_features = provided_features,
};

/* ------------------------------------------------------------------ */
/*  Internal helpers exposed to GPU twins (T7-36 / ADR-0205).         */
/*  Header: cambi_internal.h. The wrappers are thin trampolines so    */
/*  the file-static helpers above stay unchanged and CPU SIMD diff    */
/*  remains a no-op against upstream.                                 */
/* ------------------------------------------------------------------ */
#include "cambi_internal.h"

void vmaf_cambi_get_spatial_mask(const VmafPicture *image, VmafPicture *mask, uint32_t *dp,
                                 uint16_t *derivative_buffer, unsigned width, unsigned height,
                                 VmafCambiDerivativeCalculator derivative_callback)
{
    get_spatial_mask(image, mask, dp, derivative_buffer, width, height,
                     (VmafDerivativeCalculator)derivative_callback);
}

void vmaf_cambi_decimate(VmafPicture *image, unsigned width, unsigned height)
{
    decimate(image, width, height);
}

void vmaf_cambi_filter_mode(const VmafPicture *image, int width, int height, uint16_t *buffer)
{
    filter_mode(image, width, height, buffer);
}

void vmaf_cambi_calculate_c_values(VmafPicture *pic, const VmafPicture *mask_pic, float *c_values,
                                   uint16_t *histograms, uint16_t window_size,
                                   const uint16_t num_diffs, const uint16_t *tvi_for_diff,
                                   uint16_t vlt_luma, const int *diff_weights, const int *all_diffs,
                                   int width, int height, VmafCambiRangeUpdater inc_range_callback,
                                   VmafCambiRangeUpdater dec_range_callback)
{
    /* GPU-twin shim always runs the scalar c-values driver on the host residual; ADR-0205
     * Strategy II keeps the precision-sensitive sequential stage on host. The
     * inc/dec_range_callback parameters are retained for ABI compatibility with
     * cambi_internal.h but are unused here — calculate_c_values dispatches its own
     * scalar updaters. Callers that want SIMD updaters should call
     * VmafCambiHostBuffers-based GPU paths through the public CPU extractor instead. */
    (void)inc_range_callback;
    (void)dec_range_callback;
    calculate_c_values(pic, mask_pic, c_values, histograms, window_size, num_diffs, tvi_for_diff,
                       vlt_luma, diff_weights, all_diffs, width, height);
}

double vmaf_cambi_spatial_pooling(float *c_values, double topk, unsigned width, unsigned height)
{
    return spatial_pooling(c_values, topk, width, height);
}

double vmaf_cambi_weight_scores_per_scale(double *scores_per_scale, uint16_t normalization)
{
    return weight_scores_per_scale(scores_per_scale, normalization);
}

uint16_t vmaf_cambi_get_pixels_in_window(uint16_t window_length)
{
    return get_pixels_in_window(window_length);
}

void vmaf_cambi_default_callbacks(VmafCambiRangeUpdater *inc, VmafCambiRangeUpdater *dec,
                                  VmafCambiDerivativeCalculator *deriv)
{
    *inc = (VmafCambiRangeUpdater)increment_range;
    *dec = (VmafCambiRangeUpdater)decrement_range;
    *deriv = (VmafCambiDerivativeCalculator)get_derivative_data_for_row;
}

int vmaf_cambi_preprocessing(const VmafPicture *image, VmafPicture *preprocessed, int width,
                             int height, int enc_bitdepth)
{
    return cambi_preprocessing(image, preprocessed, width, height, enc_bitdepth);
}
