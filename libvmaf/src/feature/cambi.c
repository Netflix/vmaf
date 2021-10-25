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
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "picture.h"
#include "mem.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "common/macros.h"

/* Ratio of pixels for computation, must be 0 > topk >= 1.0 */
#define DEFAULT_CAMBI_TOPK_POOLING (0.6)

/* Window size to compute CAMBI: 63 corresponds to approximately 1 degree at 4k scale */
#define DEFAULT_CAMBI_WINDOW_SIZE (63)

/* Visibilty threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886 */
#define DEFAULT_CAMBI_TVI (0.019)

#define CAMBI_MIN_WIDTH (320)
#define CAMBI_MAX_WIDTH (4096)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)

#define NUM_SCALES 5
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

#define NUM_DIFFS 4
static const int g_diffs_to_consider[NUM_DIFFS] = {1, 2, 3, 4};
static const int g_diffs_weights[NUM_DIFFS] = {1, 2, 3, 4};

#define NUM_ALL_DIFFS (2*NUM_DIFFS+1)
static const int g_all_diffs[NUM_ALL_DIFFS] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
static const uint16_t g_c_value_histogram_offset = 4; // = -g_all_diffs[0]

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

#define PICS_BUFFER_SIZE 4

typedef struct CambiState {
    VmafPicture pics[PICS_BUFFER_SIZE];
    unsigned enc_width;
    unsigned enc_height;
    uint16_t tvi_for_diff[NUM_DIFFS];
    uint16_t window_size;
    double topk;
    double tvi_threshold;
    float *c_values;
    uint16_t *c_values_histograms;
} CambiState;

static const VmafOption options[] = {
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiState, enc_height),
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
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 > topk >= 1.0",
        .offset = offsetof(CambiState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886",
        .offset = offsetof(CambiState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
    },
    { 0 }
};

/* Visibility threshold functions */
#define BT1886_GAMMA (2.4)

enum CambiTVIBisectFlag {
    CAMBI_TVI_BISECT_TOO_SMALL,
    CAMBI_TVI_BISECT_CORRECT,
    CAMBI_TVI_BISECT_TOO_BIG
};

FORCE_INLINE inline int clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

static FORCE_INLINE inline double bt1886_eotf(double V, double gamma, double Lw, double Lb)
{
    double a = pow(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma), gamma);
    double b = pow(Lb, 1.0 / gamma) /(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma));
    double L = a * pow(MAX(V + b, 0), gamma);
    return L;
}

static FORCE_INLINE inline void range_foot_head(int bitdepth, const char *pix_range,
                                                int *foot, int *head)
{
    int foot_8b = 0;
    int head_8b = 255;
    if (!strcmp(pix_range, "standard"))
    {
        foot_8b = 16;
        head_8b = 235;
    }
    *foot = foot_8b * (pow(2, bitdepth - 8));
    *head = head_8b * (pow(2, bitdepth - 8));
}

static double normalize_range(int sample, int bitdepth, const char *pix_range)
{
    int foot, head, clipped_sample;
    range_foot_head(bitdepth, pix_range, &foot, &head);
    clipped_sample = clip(sample, foot, head);
    return (double)(clipped_sample-foot)/(head-foot);
}

static double luminance_bt1886(int sample, int bitdepth,
                               double Lw, double Lb, const char *pix_range)
{
    double normalized;
    normalized = normalize_range(sample, bitdepth, pix_range);
    return bt1886_eotf(normalized, BT1886_GAMMA, Lw, Lb);
}

static bool tvi_condition(int sample, int diff, double tvi_threshold,
                          int bitdepth, double Lw, double Lb, const char *pix_range)
{
    double mean_luminance = luminance_bt1886(sample, bitdepth, Lw, Lb, pix_range);
    double diff_luminance = luminance_bt1886(sample+diff, bitdepth, Lw, Lb, pix_range);
    double delta_luminance = diff_luminance - mean_luminance;
    return (delta_luminance > tvi_threshold * mean_luminance);
}

static enum CambiTVIBisectFlag tvi_hard_threshold_condition(int sample, int diff,
                                                            double tvi_threshold,
                                                            int bitdepth, double Lw, double Lb,
                                                            const char *pix_range)
{
    bool condition;
    condition = tvi_condition(sample, diff, tvi_threshold, bitdepth, Lw, Lb, pix_range);
    if (!condition) return CAMBI_TVI_BISECT_TOO_BIG;

    condition = tvi_condition(sample+1, diff, tvi_threshold, bitdepth, Lw, Lb, pix_range);
    if (condition) return CAMBI_TVI_BISECT_TOO_SMALL;

    return CAMBI_TVI_BISECT_CORRECT;
}

static int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth,
                            double Lw, double Lb, const char *pix_range)
{
    int foot, head, mid;
    enum CambiTVIBisectFlag tvi_bisect;
    const int max_val = (1 << bitdepth) - 1;

    range_foot_head(bitdepth, pix_range, &foot, &head);
    head = head - diff - 1;

    tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, bitdepth,
                                              Lw, Lb, pix_range);
    if(tvi_bisect==CAMBI_TVI_BISECT_TOO_BIG) return 0;
    if(tvi_bisect==CAMBI_TVI_BISECT_CORRECT) return foot;

    tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, bitdepth,
                                              Lw, Lb, pix_range);
    if(tvi_bisect==CAMBI_TVI_BISECT_TOO_SMALL) return max_val;
    if(tvi_bisect==CAMBI_TVI_BISECT_CORRECT) return head;

    // bisect
    while(1) {
        mid = foot + (head - foot) / 2;
        tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, bitdepth,
                                                  Lw, Lb, pix_range);
        if(tvi_bisect==CAMBI_TVI_BISECT_TOO_BIG)
            head = mid;
        else if(tvi_bisect==CAMBI_TVI_BISECT_TOO_SMALL)
            foot = mid;
        else if(tvi_bisect==CAMBI_TVI_BISECT_CORRECT)
            return mid;
        else // Should never get here (todo: add assert)
            (void) 0;
    }
}

static FORCE_INLINE inline void adjust_window_size(uint16_t *window_size, unsigned input_width)
{
    (*window_size) = ((*window_size) * input_width) / CAMBI_4K_WIDTH;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) pix_fmt;
    (void) bpc;

    CambiState *s = fex->priv;

    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = w;
        s->enc_height = h;
    }

    w = s->enc_width;
    h = s->enc_height;

    if (w < CAMBI_MIN_WIDTH || w > CAMBI_MAX_WIDTH)
        return -EINVAL;
    int err = 0;
    for (unsigned i=0; i<PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_alloc(&s->pics[i], VMAF_PIX_FMT_YUV400P, 10, w, h);

    for (int d=0; d<NUM_DIFFS; d++) {
        // BT1886 parameters
        s->tvi_for_diff[d] = get_tvi_for_diff(g_diffs_to_consider[d], s->tvi_threshold, 10,
                                              300.0, 0.01, "standard");
        s->tvi_for_diff[d] += g_c_value_histogram_offset;
    }

    adjust_window_size(&s->window_size, w);
    s->c_values = aligned_malloc(ALIGN_CEIL(w * sizeof(float)) * h, 32);
    
    const uint16_t num_bins = 1024 + (g_all_diffs[NUM_ALL_DIFFS-1] - g_all_diffs[0]);
    s->c_values_histograms = aligned_malloc(ALIGN_CEIL(w * num_bins * sizeof(uint16_t)), 32);

    return err;
}

/* Preprocessing functions */
static void decimate_generic_10b(const VmafPicture *pic, VmafPicture *out_pic)
{
    uint16_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0]>>1;
    ptrdiff_t out_stride = out_pic->stride[0]>>1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];
    unsigned out_w = out_pic->w[0];
    unsigned out_h = out_pic->h[0];

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h){
        memcpy(out_data, data, (size_t) stride * pic->h[0] * sizeof(uint16_t));
        return;
    }

    float ratio_x = (float)in_w / out_w;
    float ratio_y = (float)in_h / out_h;

    float start_x = ratio_x / 2 - 0.5;
    float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for(unsigned i=0; i<out_h; i++){
        unsigned ori_y = (int)(y+0.5);
        float x = start_x;
        for (unsigned j=0; j<out_w; j++){
            unsigned ori_x = (int)(x+0.5);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x];
            x += ratio_x;
        }
        y += ratio_y;
    }
}

static void decimate_generic_8b_and_convert_to_10b(const VmafPicture *pic, VmafPicture *out_pic)
{
    uint8_t *data = pic->data[0];
    uint16_t *out_data = out_pic->data[0];
    ptrdiff_t stride = pic->stride[0];
    ptrdiff_t out_stride = out_pic->stride[0]>>1;
    unsigned in_w = pic->w[0];
    unsigned in_h = pic->h[0];
    unsigned out_w = out_pic->w[0];
    unsigned out_h = out_pic->h[0];

    // if the input and output sizes are the same
    if (in_w == out_w && in_h == out_h){
        for (unsigned i=0; i<out_h; i++)
            for (unsigned j=0; j<out_w; j++)
                out_data[i * out_stride + j] = data[i * stride + j] << 2;
        return;
    }

    float ratio_x = (float)in_w / out_w;
    float ratio_y = (float)in_h / out_h;

    float start_x = ratio_x / 2 - 0.5;
    float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for(unsigned i=0; i<out_h; i++){
        unsigned ori_y = (int)(y+0.5);
        float x = start_x;
        for (unsigned j=0; j<out_w; j++){
            unsigned ori_x = (int)(x+0.5);
            out_data[i * out_stride + j] = data[ori_y * stride + ori_x] << 2;
            x += ratio_x;
        }
        y += ratio_y;
    }
}

static void copy_10b_luma(const VmafPicture *pic, VmafPicture *out_pic)
{
    ptrdiff_t stride = pic->stride[0] >> 1;
    memcpy(out_pic->data[0], pic->data[0], stride * pic->h[0] * sizeof(uint16_t));
}

static void anti_dithering_filter(VmafPicture *pic)
{
    uint16_t *data = pic->data[0];
    int stride = pic->stride[0] >> 1;

    for (unsigned i=0; i<pic->h[0]-1; i++) {
        for (unsigned j=0; j<pic->w[0]-1; j++)
            data[i * stride + j] = (data[i * stride + j] +
                                    data[i * stride + j+1] +
                                    data[(i+1) * stride + j] +
                                    data[(i+1) * stride + j+1]) >> 2;

        // Last column
        unsigned j = pic->w[0]-1;
        data[i * stride + j] = (data[i * stride + j] +
                                data[(i+1) * stride + j]) >> 1;
    }

    // Last row
    unsigned i = pic->h[0]-1;
    for (unsigned j=0; j<pic->w[0]-1; j++)
        data[i * stride + j] = (data[i * stride + j] +
                                data[i * stride + j+1]) >> 1;
}

static int cambi_preprocessing(const VmafPicture *image, VmafPicture *preprocessed)
{
    if (image->bpc==8) {
        decimate_generic_8b_and_convert_to_10b(image, preprocessed);
        anti_dithering_filter(preprocessed);
    }
    else {
        decimate_generic_10b(image, preprocessed);
    }

    return 0;
}

/* Banding detection functions */
static void decimate(VmafPicture *image, unsigned width, unsigned height)
{
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0]>>1;
    for (unsigned i=0; i<height; i++)
        for (unsigned j=0; j<width; j++)
            data[i * stride + j] = data[(i<<1) * stride + (j<<1)];
}

static FORCE_INLINE inline uint16_t mode_selection(uint16_t *elems, uint8_t *hist)
{
    unsigned max_counts = 0;
    uint16_t max_mode = 1024;
    // Set the 9 entries to 0
    for (int i = 0; i < 9; i++) {
        hist[elems[i]] = 0;
    }
    // Increment the 9 entries and find the mode
    for (int i = 0; i < 9; i++) {
        uint16_t value = elems[i];
        hist[value]++;
        uint8_t count = hist[value];
        if (count >= 5) {
            return value;
        }
        if (count > max_counts || (count == max_counts && value < max_mode)) {
            max_counts = count;
            max_mode = value;
        }
    }
    return max_mode;
}

static void filter_mode(const VmafPicture *image, VmafPicture *filtered_image,
                        int width, int height)
{
    uint8_t *hist = malloc(1024 * sizeof(uint8_t));
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0]>>1;    
    uint16_t *filtered_data = filtered_image->data[0];
    ptrdiff_t out_stride = filtered_image->stride[0]>>1;
    uint16_t curr[9];
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            // Get the 9 elements into an array for cache coherence
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    int clamped_row = CLAMP(i + row - 1, 0, height - 1);
                    int clamped_col = CLAMP(j + col - 1, 0, width - 1);
                    curr[3 * row + col] = data[clamped_row * stride + clamped_col];
                }
            }
            filtered_data[i * out_stride + j] = mode_selection(curr, hist);
        }
    }

    free(hist);
}

static FORCE_INLINE inline uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                                                   uint16_t filter_size)
{
    const int slope = 3;
    double resolution_ratio =
        sqrt((CAMBI_4K_WIDTH*CAMBI_4K_HEIGHT) / (input_width*input_height));

    return (uint16_t) (floor(pow(filter_size, 2)/2) - slope*(resolution_ratio-1));
}

static void get_zero_derivative(const VmafPicture *pic, VmafPicture *zero_derivative,
                                unsigned width, unsigned height)
{
    uint16_t *data = pic->data[0];
    uint16_t *output_data = zero_derivative->data[0];
    ptrdiff_t stride = pic->stride[0]>>1;

    for (unsigned i=0; i<height-1; i++) {
        for (unsigned j=0; j<width-1; j++) {
            output_data[i * stride + j] = 
                (data[i * stride + j] == data[i * stride + j+1] 
                && data[i * stride + j] == data[(i+1) * stride + j]);
        }
        // Last column
        unsigned j = width-1;
        output_data[i * stride + j] = 
            (data[i * stride + j] == data[(i+1) * stride + j]);
    }
    // Last row
    unsigned i = height-1;
    for (unsigned j=0; j<width-1; j++) {
        output_data[i * stride + j] = 
            (data[i * stride + j] == data[i * stride + j+1]);
    }
    output_data[(height-1) * stride + (width-1)] = 1;
}

static void get_spatial_mask_for_index(const VmafPicture *zero_derivative, VmafPicture *mask,
                                       uint16_t mask_index, uint16_t filter_size,
                                       int width, int height)
{
    uint16_t pad_size = filter_size >> 1;
    uint16_t *derivative_data = zero_derivative->data[0];
    uint16_t *mask_data = mask->data[0];
    ptrdiff_t stride = zero_derivative->stride[0]>>1;

    for (int i=0; i<height; i++) {
        uint16_t count = 0;
        for (int r=-pad_size; r<=pad_size; r++)
            for (int c=0; c<=pad_size; c++)
                if (i+r>=0 && i+r<height)
                    count += derivative_data[(i+r) * stride + c];

        mask_data[i * stride] = (count>mask_index);

        for (int j=1; j<width; j++) {
            if (j-pad_size-1>=0)
                for (int r=-pad_size; r<=pad_size; r++)
                    if (i+r>=0 && i+r<height)
                        count -= derivative_data[(i+r) * stride + j-pad_size-1];

            if (j+pad_size<width)
                for (int r=-pad_size; r<=pad_size; r++)
                    if (i+r>=0 && i+r<height)
                        count += derivative_data[(i+r) * stride + j+pad_size];

            mask_data[i * stride + j] = (count>mask_index);
        }
    }
}

static void get_spatial_mask(const VmafPicture *image, VmafPicture *mask, VmafPicture *zero_derivative,
                             unsigned width, unsigned height)
{
    const uint16_t mask_filter_size = 7;
    unsigned input_width = image->w[0];
    unsigned input_height = image->h[0];
    uint16_t mask_index = get_mask_index(input_width, input_height, mask_filter_size);
    get_zero_derivative(image, zero_derivative, width, height);
    get_spatial_mask_for_index(zero_derivative, mask, mask_index, mask_filter_size, width, height);
}

static void pic_add_offset(VmafPicture *pic, uint16_t offset,
                           unsigned width, unsigned height)
{
    uint16_t *data = pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    for (unsigned i=0; i<height; i++)
        for (unsigned j=0; j<width; j++)
            data[i * stride + j] += offset;
}

static float c_value_pixel(const uint16_t *histograms, uint16_t value, const int *diff_weights,
                           const int *diffs, uint16_t num_diffs, const uint16_t *tvi_thresholds, int histogram_col, int histogram_width)
{
    uint16_t p_0 = histograms[value * histogram_width + histogram_col];
    float val, c_value = 0.0;
    for (uint16_t d=0; d<num_diffs; d++) {
        if (value <= tvi_thresholds[d]) {
            uint16_t p_1 = histograms[(value + diffs[num_diffs+d+1]) * histogram_width + histogram_col];
            uint16_t p_2 = histograms[(value + diffs[num_diffs-d-1]) * histogram_width + histogram_col];
            if (p_1 > p_2)
                val = (float)(diff_weights[d] * p_0 * p_1) / (p_1 + p_0);
            else
                val = (float)(diff_weights[d] * p_0 * p_2) / (p_2 + p_0);

            if (val > c_value)
                c_value = val;
        }
    }

    return c_value;
}

static FORCE_INLINE inline void update_histogram_subtract(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j, int width, ptrdiff_t stride, uint16_t pad_size) {
    uint16_t mask_val = mask[(i - pad_size - 1) * stride + j];
    if (mask_val) {
        uint16_t top_val = image[(i - pad_size - 1) * stride + j];
        for (int col = MAX(j - pad_size, 0); col < MIN(j + pad_size + 1, width); col++) {
            histograms[top_val * width + col]--;
        }
    }
}

static FORCE_INLINE inline void update_histogram_add(uint16_t *histograms, uint16_t *image, uint16_t *mask, int i, int j, int width, ptrdiff_t stride, uint16_t pad_size) {
    uint16_t mask_val = mask[(i + pad_size) * stride + j];
    if (mask_val) {
        uint16_t val = image[(i + pad_size) * stride + j];
        for (int col = MAX(j - pad_size, 0); col < MIN(j + pad_size + 1, width); col++) {
            histograms[val * width + col]++;
        }
    }
}

static FORCE_INLINE inline void calculate_c_values_row(float *c_values, uint16_t *histograms, uint16_t *image, uint16_t *mask, int row, int width, ptrdiff_t stride, const uint16_t *tvi_for_diff) {
    for (int col = 0; col < width; col++) {
        if (mask[row * stride + col]) {
            c_values[row * width + col] = c_value_pixel(
                histograms, image[row * stride + col], g_diffs_weights, g_all_diffs, NUM_DIFFS, tvi_for_diff, col, width
            );
        }
    }
}

static void calculate_c_values(VmafPicture *pic, const VmafPicture *mask_pic,
                               float *c_values, uint16_t *histograms, uint16_t window_size,
                               const uint16_t *tvi_for_diff, int width, int height)
{
    uint16_t pad_size = window_size>>1;
    const uint16_t num_bins = 1024 + (g_all_diffs[NUM_ALL_DIFFS-1] - g_all_diffs[0]);

    uint16_t *image = pic->data[0];
    uint16_t *mask = mask_pic->data[0];
    ptrdiff_t stride = pic->stride[0]>>1;

    pic_add_offset(pic, g_c_value_histogram_offset, width, height);
    memset(c_values, 0.0, sizeof(float) * width * height);

    // Use a histogram for each pixel in width
    // histograms[i * width + j] accesses the j'th histogram, i'th value
    // This is done for cache optimization reasons
    memset(histograms, 0, width * num_bins * sizeof(uint16_t));

    // First pass: first pad_size rows
    for (int i = 0; i < pad_size; i++) {
        for (int j = 0; j < width; j++) {
            uint16_t mask_val = mask[i * stride + j];
            if (mask_val) {
                uint16_t val = image[i * stride + j];
                for (int col = MAX(j - pad_size, 0); col < MIN(j + pad_size + 1, width); col++) {
                    histograms[val * width + col]++;
                }
            }
        }
    }

    // Iterate over all rows, unrolled into 3 loops to avoid conditions
    for (int i = 0; i < pad_size + 1; i++) {
        if (i + pad_size < height) {
            for (int j = 0; j < width; j++) {
                update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size);
            }
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, tvi_for_diff);
    }
    for (int i = pad_size + 1; i < height - pad_size; i++) {
        for (int j = 0; j < width; j++) {
            update_histogram_subtract(histograms, image, mask, i, j, width, stride, pad_size);
            update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size);
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, tvi_for_diff);
    }
    for (int i = height - pad_size; i < height; i++) {
        if (i - pad_size - 1 >= 0) {
            for (int j = 0; j < width; j++) {
                update_histogram_subtract(histograms, image, mask, i, j, width, stride, pad_size);
            }
        }
        calculate_c_values_row(c_values, histograms, image, mask, i, width, stride, tvi_for_diff);
    }
}

static double average_topk_elements(const float *arr, int topk_elements)
{
    double sum = 0;
    for (int i=0; i < topk_elements; i++)
        sum += arr[i];

    return (double) sum / topk_elements;
}

#define SWAP(x, y)      \
    {                   \
        float temp = x; \
        x = y;          \
        y = temp;       \
    }

static void quick_select(float *arr, int n, int k)
{
    int left = 0;
    int right = n - 1;
    while (left < right)
    {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do
        {
            while (arr[i] > pivot)
            {
                i++;
            }
            while (arr[j] < pivot)
            {
                j--;
            }
            if (i <= j)
            {
                SWAP(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k)
        {
            left = i;
        }
        if (k < i)
        {
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
    return (uint16_t) pow(2*(window_length>>1)+1, 2);
}

// Inner product weighting scores for each scale
static FORCE_INLINE inline
    double weight_scores_per_scale(double *scores_per_scale, uint16_t normalization)
{
    double score = 0.0;
    for (unsigned scale=0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);

    return score/normalization;
}

static int cambi_score(VmafPicture *pics, uint16_t window_size, double topk,
                       const uint16_t *tvi_for_diff, float *c_values, uint16_t *c_values_histograms, double *score)
{
    double scores_per_scale[NUM_SCALES];
    VmafPicture *image = &pics[0];
    VmafPicture *filtered_image = &pics[1];
    VmafPicture *mask = &pics[2];
    VmafPicture *temp_image = &pics[3];

    unsigned scaled_width = image->w[0];
    unsigned scaled_height = image->h[0];
    for (unsigned scale=0; scale < NUM_SCALES; scale++) {
        if (scale>0) {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
            decimate(image, scaled_width, scaled_height);
            decimate(mask, scaled_width, scaled_height);
        } else {
            get_spatial_mask(image, mask, temp_image, scaled_width, scaled_height);
        }

        filter_mode(image, filtered_image, scaled_width, scaled_height);
        if (scale < NUM_SCALES-1)
            copy_10b_luma(filtered_image, image);

        calculate_c_values(filtered_image, mask, c_values, c_values_histograms, window_size,
                           tvi_for_diff, scaled_width, scaled_height);

        scores_per_scale[scale] =
            spatial_pooling(c_values, topk, scaled_width, scaled_height);
    }

    uint16_t pixels_in_window = get_pixels_in_window(window_size);
    *score = weight_scores_per_scale(scores_per_scale, pixels_in_window);
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    (void) ref_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    CambiState *s = fex->priv;

    int err = cambi_preprocessing(dist_pic, &s->pics[0]);
    if (err) return err;

    double score;
    err = cambi_score(s->pics, s->window_size, s->topk, s->tvi_for_diff, s->c_values, s->c_values_histograms, &score);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector, "cambi", score, index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    CambiState *s = fex->priv;

    int err = 0;
    for (unsigned i=0; i<PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_unref(&s->pics[i]);

    aligned_free(s->c_values);
    aligned_free(s->c_values_histograms);
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
    .close = close,
    .priv_size = sizeof(CambiState),
    .provided_features = provided_features,
};