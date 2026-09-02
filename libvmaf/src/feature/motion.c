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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "offset.h"
#include "motion_options.h"
#include "mem.h"
#include "common/alignment.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "motion_tools.h"
#include "vif_tools.h"

#define convolution_f32_c convolution_f32_c_s
#define FILTER_3           FILTER_3_s
#define FILTER_5           FILTER_5_s
#define FILTER_5_NO_OP     FILTER_5_NO_OP_s
#define offset_image       offset_image_s

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(float) bytes)
 */
float vmaf_image_sad_c(const float *img1, const float *img2, int width, int height, int img1_stride, int img2_stride,
                       int motion_add_scale1)
{
    float motion_scale0 = 0.0;
    float accum = (float)0.0;

    for (int i = 0; i < height; ++i) {
        float accum_line = (float)0.0;
        for (int j = 0; j < width; ++j) {
            float img1px = img1[i * img1_stride + j];
            float img2px = img2[i * img2_stride + j];
            accum_line += fabs(img1px - img2px);
        }
        accum += accum_line;
    }
    motion_scale0 = (float) (accum / (width * height));

    if (motion_add_scale1 == 1) {
        float motion_scale1 = 0.0;
        float accum_scale1 = (float)0.0;
        int scaled_width = (int)(width * 0.5 + 0.5);
        int scaled_height = (int)(height * 0.5 + 0.5);
        int float_stride = ALIGN_CEIL(width * sizeof(float));
        int scaled_float_stride = ALIGN_CEIL(scaled_width * sizeof(float));
        float *img1_scaled = aligned_malloc(scaled_float_stride * scaled_height, 32);
        float *img2_scaled = aligned_malloc(scaled_float_stride * scaled_height, 32);

        vif_scale_frame_s(vif_scale_bilinear, img1, img1_scaled, width, height, float_stride / sizeof(float), scaled_width, scaled_height, scaled_float_stride / sizeof(float));
        vif_scale_frame_s(vif_scale_bilinear, img2, img2_scaled, width, height, float_stride / sizeof(float), scaled_width, scaled_height, scaled_float_stride / sizeof(float));

        for (int i = 0; i < scaled_height; ++i) {
            float accum_line = (float)0.0;
            for (int j = 0; j < scaled_width; ++j) {
                float img1px = img1_scaled[i * scaled_float_stride / sizeof(float) + j];
                float img2px = img2_scaled[i * scaled_float_stride / sizeof(float) + j];
                accum_line += fabs(img1px - img2px);
            }
            accum_scale1 += accum_line;
        }

        aligned_free(img1_scaled);
        aligned_free(img2_scaled);

        motion_scale1 = (float) (accum_scale1 / (scaled_width * scaled_height));

        return motion_scale0 + motion_scale1;
    } else {
        return motion_scale0;
    }
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score,
                   int motion_decimate)
{
    if (ref_stride % sizeof(float) != 0)
    {
        printf("error: ref_stride %% sizeof(float) != 0, ref_stride = %d, sizeof(float) = %zu.\n", ref_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(float) != 0)
    {
        printf("error: dis_stride %% sizeof(float) != 0, dis_stride = %d, sizeof(float) = %zu.\n", dis_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    // stride for vmaf_image_sad_c is in terms of (sizeof(float) bytes)
    *score = vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(float), dis_stride / sizeof(float), motion_decimate);

    return 0;

fail:
    return 1;
}
