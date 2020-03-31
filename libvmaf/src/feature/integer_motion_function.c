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

/**
 * Works similar to floating-point function vmaf_image_sad_c
 */
FORCE_INLINE inline float integer_vmaf_image_sad_c(const int16_t *img1, const int16_t *img2, int width, int height, int img1_stride, int img2_stride)
{
    int64_t accum = 0;


    for (int i = 0; i < height; ++i) {
    	int32_t accum_inner = 0;
        for (int j = 0; j < width; ++j) {
            int16_t img1px = img1[i * img1_stride + j];
            int16_t img2px = img2[i * img2_stride + j];

            accum_inner += (int32_t) abs(img1px - img2px);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (int64_t) accum_inner;
        //assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }
    float f_accum = (float) accum/256.0;
    return (float) (f_accum / (width * height));
}

/**
 * Works similar to floating-point function compute_motion
 */
int integer_compute_motion(const int16_t *ref, const int16_t *dis, int w, int h, int ref_stride, int dis_stride, double *score)
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
    // stride for integer_vmaf_image_sad_c is in terms of (sizeof(float) bytes)
    *score = integer_vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(float), dis_stride / sizeof(float));

    return 0;

fail:
    return 1;
}
