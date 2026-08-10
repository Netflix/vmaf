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

#ifndef __VMAF_SRC_FEATURE_SPEED_H__
#define __VMAF_SRC_FEATURE_SPEED_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "opt.h"

typedef double (*compute_cov_kernel_fn)(const float *data_x, const float *data_y,
                                        size_t stride_px, size_t height,
                                        size_t width, double mean_x,
                                        double mean_y);

/* ------------------------------------------------------------------ */
/* Shared between the CPU extractors (speed.c) and the CUDA one        */
/* (cuda/speed_cuda.c).                                                */
/* ------------------------------------------------------------------ */

typedef struct SpeedDimensions {
    size_t original_height;
    size_t original_width;
    size_t scaled_height;
    size_t scaled_width;
    size_t alloc_height;
    size_t alloc_width;
    size_t operating_height;
    size_t operating_width;
    size_t block_size;
    size_t truncated_width;
    size_t truncated_height;
    size_t num_blocks_horizontal;
    size_t num_blocks_vertical;
    size_t num_blocks;
    size_t elements_in_block;
    size_t submatrix_width;
    size_t submatrix_height;
} SpeedDimensions;

typedef struct SpeedResultBuffers {
    float *entropies;
    float *variances;
} SpeedResultBuffers;

typedef struct SpeedBuffers {
    float *independent_term;
    float *linear_system_sol;
    float *cov_mat;
    float *eigenvalues;
    float *tmp_buffer;
    // Bilinear column table for the (fixed, resolution-derived) prescale of
    // this feature extractor instance. Populated once in speed_init() and
    // reused for every frame instead of being recomputed per call. NULL
    // when the configured scaling method isn't bilinear.
    int *bilinear_x1a;
    int *bilinear_x2a;
    float *bilinear_dxa;
} SpeedBuffers;

typedef struct SpeedOptions {
    double speed_kernelscale;
    double speed_prescale;
    char *speed_prescale_method;
    double speed_sigma_nn;
    double speed_nn_floor;
    int speed_weight_var_mode;
} SpeedOptions;

typedef struct SpeedState {
    SpeedDimensions dimensions;
    SpeedResultBuffers ref_results;
    SpeedResultBuffers dis_results;
    SpeedBuffers buffers;
    size_t float_stride;
    compute_cov_kernel_fn compute_cov_kernel;
} SpeedState;

typedef struct SpeedChromaState {
    SpeedState speed_state;
    SpeedOptions speed_options;
    float *frame_buffer_ref;
    float *frame_buffer_dis;
    VmafDictionary *feature_name_dict;
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;
} SpeedChromaState;

/* Already external in speed.c; declared here so the CUDA extractor can
 * reach them. */
int speed_init(SpeedState *s, SpeedOptions *opt, int w, int h);
int speed_extract_score(SpeedState *s, SpeedOptions *opt, float *ref,
                        float *dis, float *score);
int speed_close(SpeedState *s);

/* The CUDA extractor runs filter_and_downscale on the device and then calls
 * est_params on the host, so it needs this half of speed_extract_score. */
int est_params(SpeedState *s, const float *data, float sigma_nn,
               SpeedResultBuffers *output);

/* Scoring from the two result buffers. The CUDA extractor runs the
 * filters on the device and then scores on the host.
 * (speed_get_antialias_filter is already public in vif_tools.h.) */
float get_speed_score(SpeedDimensions dim, SpeedResultBuffers ref_results, SpeedResultBuffers dis_results, float sigma_nn, float nn_floor, int speed_weight_var_mode);

extern const VmafOption speed_chroma_options[];

#endif /* __VMAF_SRC_FEATURE_SPEED_H__ */
