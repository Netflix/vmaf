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

#ifndef FEATURE_CAMBI_CUDA_H_
#define FEATURE_CAMBI_CUDA_H_

#include "config.h"

/*
 * CAMBI constants. These must stay identical to the CPU reference
 * implementation in feature/cambi.c: they participate in the score itself
 * (window sizing, scale weights, contrast weights), so a divergence here is a
 * numerical parity break.
 */
#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)
#define NUM_SCALES (5)

#define MASK_FILTER_SIZE (7)

#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)

#define DEFAULT_CAMBI_TOPK_POOLING (0.6)
#define DEFAULT_CAMBI_WINDOW_SIZE (65)
#define DEFAULT_CAMBI_TVI (0.019)
#define DEFAULT_CAMBI_VLT (0.0)
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)
#define DEFAULT_CAMBI_MAX_VAL (1000.0)
#define DEFAULT_CAMBI_EOTF ("bt1886")
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)

/*
 * Suprathreshold contrast response weights. Same table as g_contrast_weights in
 * feature/cambi.c.
 */
static const int g_contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                           8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

/* Suprathreshold contrast response weights, per scale. */
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

/*
 * The CAMBI fatbin, embedded by bin2c (see src/meson.build, cuda_cu_sources /
 * ptx_bin2c). See cambi.cu: that translation unit contains device code only.
 */
extern const unsigned char cambi_ptx[];

#endif /* FEATURE_CAMBI_CUDA_H_ */
