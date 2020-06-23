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
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "motion_tools.h"

#define convolution_f32_c convolution_f32_c_s
#define FILTER_5           FILTER_5_s
#define offset_image       offset_image_s

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(float) bytes)
 */
float vmaf_image_sad_c(const float *img1, const float *img2, int width, int height, int img1_stride, int img2_stride)
{
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

    return (float) (accum / (width * height));
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score)
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
    *score = vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(float), dis_stride / sizeof(float));

    return 0;

fail:
    return 1;
}
