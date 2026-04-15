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
#include "float_psnr_neon.h"

double float_psnr_noise_line_neon(const float *ref, const float *dis, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    float64x2_t dsum0 = vdupq_n_f64(0.0);
    float64x2_t dsum1 = vdupq_n_f64(0.0);
    int j = 0;

    for (; j + 4 <= w; j += 4) {
        float32x4_t r = vld1q_f32(ref + j);
        float32x4_t d = vld1q_f32(dis + j);
        float32x4_t diff = vsubq_f32(r, d);
        float32x4_t sq = vmulq_f32(diff, diff);

        dsum0 = vaddq_f64(dsum0, vcvt_f64_f32(vget_low_f32(sq)));
        dsum1 = vaddq_f64(dsum1, vcvt_f64_f32(vget_high_f32(sq)));
    }

    double result = vaddvq_f64(vaddq_f64(dsum0, dsum1));

    for (; j < w; j++) {
        float diff = ref[j] - dis[j];
        result += (double)(diff * diff);
    }

    return result;
}
