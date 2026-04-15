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

#include <arm_neon.h>
#include "float_motion_neon.h"

float float_sad_line_neon(const float *img1, const float *img2, int w)
{
    float accum = 0.0f;
    int j = 0;

    for (; j + 4 <= w; j += 4) {
        float32x4_t a = vld1q_f32(img1 + j);
        float32x4_t b = vld1q_f32(img2 + j);
        float32x4_t diff = vsubq_f32(a, b);
        float32x4_t abs_diff = vabsq_f32(diff);

        /*
         * Sequential accumulation in float to match scalar ordering.
         * NEON is used for vectorised load/subtract/abs; the reduction
         * mirrors the scalar loop so that rounding is bit-identical.
         */
        accum += vgetq_lane_f32(abs_diff, 0);
        accum += vgetq_lane_f32(abs_diff, 1);
        accum += vgetq_lane_f32(abs_diff, 2);
        accum += vgetq_lane_f32(abs_diff, 3);
    }

    for (; j < w; j++) {
        float diff = img1[j] - img2[j];
        accum += diff < 0 ? -diff : diff;
    }

    return accum;
}
