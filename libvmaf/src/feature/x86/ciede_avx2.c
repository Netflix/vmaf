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

#include <immintrin.h>
#include <stdint.h>

#include "ciede_avx2.h"

void ciede_preprocess_8_avx2(const uint8_t *y_buf, const uint8_t *u_buf,
                              const uint8_t *v_buf, float *out_y, float *out_u,
                              float *out_v, int w)
{
    int j = 0;

    for (; j + 8 <= w; j += 8) {
        /* Load 8 uint8 values from each plane */
        __m128i y8 = _mm_loadl_epi64((const __m128i *)(y_buf + j));
        __m128i u8 = _mm_loadl_epi64((const __m128i *)(u_buf + j));
        __m128i v8 = _mm_loadl_epi64((const __m128i *)(v_buf + j));

        /* Zero-extend uint8 -> int32 (8 values fit in 256-bit register) */
        __m256i y32 = _mm256_cvtepu8_epi32(y8);
        __m256i u32 = _mm256_cvtepu8_epi32(u8);
        __m256i v32 = _mm256_cvtepu8_epi32(v8);

        /* Convert int32 -> float */
        __m256 yf = _mm256_cvtepi32_ps(y32);
        __m256 uf = _mm256_cvtepi32_ps(u32);
        __m256 vf = _mm256_cvtepi32_ps(v32);

        /* Store 8 floats */
        _mm256_storeu_ps(out_y + j, yf);
        _mm256_storeu_ps(out_u + j, uf);
        _mm256_storeu_ps(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}

void ciede_preprocess_16_avx2(const uint16_t *y_buf, const uint16_t *u_buf,
                               const uint16_t *v_buf, float *out_y,
                               float *out_u, float *out_v, int w)
{
    int j = 0;

    for (; j + 8 <= w; j += 8) {
        /* Load 8 uint16 values from each plane */
        __m128i y16 = _mm_loadu_si128((const __m128i *)(y_buf + j));
        __m128i u16 = _mm_loadu_si128((const __m128i *)(u_buf + j));
        __m128i v16 = _mm_loadu_si128((const __m128i *)(v_buf + j));

        /* Zero-extend uint16 -> int32 (8 values fit in 256-bit register) */
        __m256i y32 = _mm256_cvtepu16_epi32(y16);
        __m256i u32 = _mm256_cvtepu16_epi32(u16);
        __m256i v32 = _mm256_cvtepu16_epi32(v16);

        /* Convert int32 -> float */
        __m256 yf = _mm256_cvtepi32_ps(y32);
        __m256 uf = _mm256_cvtepi32_ps(u32);
        __m256 vf = _mm256_cvtepi32_ps(v32);

        /* Store 8 floats */
        _mm256_storeu_ps(out_y + j, yf);
        _mm256_storeu_ps(out_u + j, uf);
        _mm256_storeu_ps(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}
