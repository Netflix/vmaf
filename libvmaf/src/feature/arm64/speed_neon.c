/**
 * Copyright 2026 Dan Trapp.
 *
 * Licensed under the BSD+Patent License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://opensource.org/licenses/BSDplusPatent
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arm_neon.h>

#include "speed_neon.h"

double compute_cov_kernel_neon(const float *data_x, const float *data_y,
                               size_t stride_px, size_t height, size_t width,
                               double mean_x, double mean_y)
{
    const float64x2_t mx = vdupq_n_f64(mean_x);
    const float64x2_t my = vdupq_n_f64(mean_y);
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);
    double tail = 0.0;

    for (size_t i = 0; i < height; i++) {
        const float *row_x = data_x + i * stride_px;
        const float *row_y = data_y + i * stride_px;
        size_t j = 0;
        for (; j + 7 < width; j += 8) {
            const float32x4_t x0 = vld1q_f32(row_x + j);
            const float32x4_t y0 = vld1q_f32(row_y + j);
            const float32x4_t x1 = vld1q_f32(row_x + j + 4);
            const float32x4_t y1 = vld1q_f32(row_y + j + 4);
            acc0 = vfmaq_f64(acc0,
                vsubq_f64(vcvt_f64_f32(vget_low_f32(x0)), mx),
                vsubq_f64(vcvt_f64_f32(vget_low_f32(y0)), my));
            acc1 = vfmaq_f64(acc1,
                vsubq_f64(vcvt_high_f64_f32(x0), mx),
                vsubq_f64(vcvt_high_f64_f32(y0), my));
            acc2 = vfmaq_f64(acc2,
                vsubq_f64(vcvt_f64_f32(vget_low_f32(x1)), mx),
                vsubq_f64(vcvt_f64_f32(vget_low_f32(y1)), my));
            acc3 = vfmaq_f64(acc3,
                vsubq_f64(vcvt_high_f64_f32(x1), mx),
                vsubq_f64(vcvt_high_f64_f32(y1), my));
        }
        for (; j + 1 < width; j += 2) {
            const float64x2_t x = vcvt_f64_f32(vld1_f32(row_x + j));
            const float64x2_t y = vcvt_f64_f32(vld1_f32(row_y + j));
            acc0 = vfmaq_f64(acc0, vsubq_f64(x, mx), vsubq_f64(y, my));
        }
        if (j < width) {
            const double x = row_x[j];
            const double y = row_y[j];
            tail += (x - mean_x) * (y - mean_y);
        }
    }
    return vaddvq_f64(vaddq_f64(vaddq_f64(acc0, acc1),
                               vaddq_f64(acc2, acc3))) + tail;
}
