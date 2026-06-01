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

// AVX-512 8-wide-double accumulator with two parallel chains for FMA pipelining.
// Loop processes 16 elements per iteration, then 8, then scalar tail.
double compute_cov_kernel_avx512(const float *data_x, const float *data_y,
                                 size_t stride_px, size_t height, size_t width,
                                 double mean_x, double mean_y)
{
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    const __m512d mx = _mm512_set1_pd(mean_x);
    const __m512d my = _mm512_set1_pd(mean_y);
    double scalar_tail = 0.0;

    for (size_t i = 0; i < height; i++) {
        const float *row_x = data_x + i * stride_px;
        const float *row_y = data_y + i * stride_px;
        size_t j = 0;

        for (; j + 15 < width; j += 16) {
            __m256 fx0 = _mm256_loadu_ps(row_x + j);
            __m256 fx1 = _mm256_loadu_ps(row_x + j + 8);
            __m256 fy0 = _mm256_loadu_ps(row_y + j);
            __m256 fy1 = _mm256_loadu_ps(row_y + j + 8);
            __m512d cx0 = _mm512_sub_pd(_mm512_cvtps_pd(fx0), mx);
            __m512d cx1 = _mm512_sub_pd(_mm512_cvtps_pd(fx1), mx);
            __m512d cy0 = _mm512_sub_pd(_mm512_cvtps_pd(fy0), my);
            __m512d cy1 = _mm512_sub_pd(_mm512_cvtps_pd(fy1), my);
            acc0 = _mm512_fmadd_pd(cx0, cy0, acc0);
            acc1 = _mm512_fmadd_pd(cx1, cy1, acc1);
        }

        for (; j + 7 < width; j += 8) {
            __m256 fx = _mm256_loadu_ps(row_x + j);
            __m256 fy = _mm256_loadu_ps(row_y + j);
            __m512d cx = _mm512_sub_pd(_mm512_cvtps_pd(fx), mx);
            __m512d cy = _mm512_sub_pd(_mm512_cvtps_pd(fy), my);
            acc0 = _mm512_fmadd_pd(cx, cy, acc0);
        }

        for (; j < width; j++) {
            double val_x = row_x[j];
            double val_y = row_y[j];
            scalar_tail += (val_x - mean_x) * (val_y - mean_y);
        }
    }

    __m512d acc = _mm512_add_pd(acc0, acc1);
    return _mm512_reduce_add_pd(acc) + scalar_tail;
}
