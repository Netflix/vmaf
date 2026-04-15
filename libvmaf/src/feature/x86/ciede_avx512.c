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
#include <stdint.h>

#include "ciede_avx512.h"

void ciede_preprocess_8_avx512(const uint8_t *y_buf, const uint8_t *u_buf,
                                const uint8_t *v_buf, float *out_y,
                                float *out_u, float *out_v, int w)
{
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        /* Load 16 uint8 values from each plane */
        __m128i y8 = _mm_loadu_si128((const __m128i *)(y_buf + j));
        __m128i u8 = _mm_loadu_si128((const __m128i *)(u_buf + j));
        __m128i v8 = _mm_loadu_si128((const __m128i *)(v_buf + j));

        /* Zero-extend uint8 -> int32 (16 values fit in 512-bit register) */
        __m512i y32 = _mm512_cvtepu8_epi32(y8);
        __m512i u32 = _mm512_cvtepu8_epi32(u8);
        __m512i v32 = _mm512_cvtepu8_epi32(v8);

        /* Convert int32 -> float */
        __m512 yf = _mm512_cvtepi32_ps(y32);
        __m512 uf = _mm512_cvtepi32_ps(u32);
        __m512 vf = _mm512_cvtepi32_ps(v32);

        /* Store 16 floats */
        _mm512_storeu_ps(out_y + j, yf);
        _mm512_storeu_ps(out_u + j, uf);
        _mm512_storeu_ps(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}

void ciede_preprocess_16_avx512(const uint16_t *y_buf, const uint16_t *u_buf,
                                 const uint16_t *v_buf, float *out_y,
                                 float *out_u, float *out_v, int w)
{
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        /* Load 16 uint16 values from each plane */
        __m256i y16 = _mm256_loadu_si256((const __m256i *)(y_buf + j));
        __m256i u16 = _mm256_loadu_si256((const __m256i *)(u_buf + j));
        __m256i v16 = _mm256_loadu_si256((const __m256i *)(v_buf + j));

        /* Zero-extend uint16 -> int32 (16 values fit in 512-bit register) */
        __m512i y32 = _mm512_cvtepu16_epi32(y16);
        __m512i u32 = _mm512_cvtepu16_epi32(u16);
        __m512i v32 = _mm512_cvtepu16_epi32(v16);

        /* Convert int32 -> float */
        __m512 yf = _mm512_cvtepi32_ps(y32);
        __m512 uf = _mm512_cvtepi32_ps(u32);
        __m512 vf = _mm512_cvtepi32_ps(v32);

        /* Store 16 floats */
        _mm512_storeu_ps(out_y + j, yf);
        _mm512_storeu_ps(out_u + j, uf);
        _mm512_storeu_ps(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}
