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

#include <immintrin.h>
#include <stddef.h>

// AVX2 4-wide-double accumulator with two parallel chains to hide FMA latency.
// Per-element math (mul + add) is fused via vfmadd231pd; the per-row sub is
// kept separate to mirror scalar's two-rounding sub more closely.
// Loop processes 8 elements per iteration when possible, then 4, then scalar.
double compute_cov_kernel_avx2(const float *data_x, const float *data_y,
                               size_t stride_px, size_t height, size_t width,
                               double mean_x, double mean_y)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    const __m256d mx = _mm256_set1_pd(mean_x);
    const __m256d my = _mm256_set1_pd(mean_y);
    double scalar_tail = 0.0;

    for (size_t i = 0; i < height; i++) {
        const float *row_x = data_x + i * stride_px;
        const float *row_y = data_y + i * stride_px;
        size_t j = 0;

        for (; j + 7 < width; j += 8) {
            __m128 fx0 = _mm_loadu_ps(row_x + j);
            __m128 fx1 = _mm_loadu_ps(row_x + j + 4);
            __m128 fy0 = _mm_loadu_ps(row_y + j);
            __m128 fy1 = _mm_loadu_ps(row_y + j + 4);
            __m256d cx0 = _mm256_sub_pd(_mm256_cvtps_pd(fx0), mx);
            __m256d cx1 = _mm256_sub_pd(_mm256_cvtps_pd(fx1), mx);
            __m256d cy0 = _mm256_sub_pd(_mm256_cvtps_pd(fy0), my);
            __m256d cy1 = _mm256_sub_pd(_mm256_cvtps_pd(fy1), my);
            acc0 = _mm256_fmadd_pd(cx0, cy0, acc0);
            acc1 = _mm256_fmadd_pd(cx1, cy1, acc1);
        }

        for (; j + 3 < width; j += 4) {
            __m128 fx = _mm_loadu_ps(row_x + j);
            __m128 fy = _mm_loadu_ps(row_y + j);
            __m256d cx = _mm256_sub_pd(_mm256_cvtps_pd(fx), mx);
            __m256d cy = _mm256_sub_pd(_mm256_cvtps_pd(fy), my);
            acc0 = _mm256_fmadd_pd(cx, cy, acc0);
        }

        for (; j < width; j++) {
            double val_x = row_x[j];
            double val_y = row_y[j];
            scalar_tail += (val_x - mean_x) * (val_y - mean_y);
        }
    }

    __m256d acc = _mm256_add_pd(acc0, acc1);
    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + scalar_tail;
}
