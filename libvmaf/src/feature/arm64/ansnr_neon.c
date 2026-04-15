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

#include <arm_neon.h>
#include "ansnr_neon.h"

void ansnr_mse_line_neon(const float *ref, const float *dis,
                          float *sig_accum, float *noise_accum, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    float64x2_t sig_dsum0 = vdupq_n_f64(0.0);
    float64x2_t sig_dsum1 = vdupq_n_f64(0.0);
    float64x2_t noise_dsum0 = vdupq_n_f64(0.0);
    float64x2_t noise_dsum1 = vdupq_n_f64(0.0);
    int j = 0;

    for (; j + 4 <= w; j += 4) {
        float32x4_t r = vld1q_f32(ref + j);
        float32x4_t d = vld1q_f32(dis + j);
        float32x4_t diff = vsubq_f32(r, d);
        float32x4_t sig_val = vmulq_f32(r, r);
        float32x4_t noise_val = vmulq_f32(diff, diff);

        sig_dsum0 = vaddq_f64(sig_dsum0, vcvt_f64_f32(vget_low_f32(sig_val)));
        sig_dsum1 = vaddq_f64(sig_dsum1, vcvt_f64_f32(vget_high_f32(sig_val)));

        noise_dsum0 = vaddq_f64(noise_dsum0, vcvt_f64_f32(vget_low_f32(noise_val)));
        noise_dsum1 = vaddq_f64(noise_dsum1, vcvt_f64_f32(vget_high_f32(noise_val)));
    }

    float sig_result = (float)vaddvq_f64(vaddq_f64(sig_dsum0, sig_dsum1));
    float noise_result = (float)vaddvq_f64(vaddq_f64(noise_dsum0, noise_dsum1));

    for (; j < w; j++) {
        float r = ref[j];
        float d = dis[j];
        float diff = r - d;
        sig_result += r * r;
        noise_result += diff * diff;
    }

    *sig_accum += sig_result;
    *noise_accum += noise_result;
}
