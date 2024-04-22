/**
 *
 *  Copyright (C) 2022 Intel Corporation.
 *  Copyright (c) 2022-2024 Meta, Inc.
 *
 *     Licensed under the BSD 3-Clause License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/license/bsd-3-clause
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if ARCH_AARCH64
#include <arm_neon.h>
#endif

#include "resizer_avx512.h"
#include <immintrin.h>
#include <emmintrin.h>

#if !OPTIMISED_COEFF
static void interpolateCubic(float x, float *coeffs)
{
    const float A = -0.75f;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}
#endif

#if OPTIMISED_COEFF
void hresize_avx512(const unsigned char **src, int **dst, int count,
             const short *alpha,
             int swidth, int dwidth, int cn, int xmin, int xmax)
#else
void hresize_avx512(const unsigned char **src, int **dst, int count,
             const int *xofs, const short *alpha,
             int swidth, int dwidth, int cn, int xmin, int xmax)
#endif
{
    int xmax_64 = xmax - (xmax % 64);
    int xmax_32 = xmax - (xmax % 32);
    int xmax_16 = xmax - (xmax % 16);
    int xmax_8 = xmax - (xmax % 8);
    int xmax_4 = xmax - (xmax % 4);

    __m512i coef0_512 = _mm512_set1_epi32(alpha[0] + (alpha[1] << 16) + (1 << 16));
    __m512i coef2_512 = _mm512_set1_epi32(alpha[2] + (alpha[3] << 16));
    __m512i permlo_512 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    __m512i permhi_512 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    __m512i zero_512 = _mm512_setzero_si512();

    __m256i coef0_256 = _mm256_set1_epi32(alpha[0] + (alpha[1] << 16) + (1 << 16));
    __m256i coef2_256 = _mm256_set1_epi32(alpha[2] + (alpha[3] << 16));

    __m128i coef0_128 = _mm_set1_epi32(alpha[0] + (alpha[1] << 16) + (1 << 16));
    __m128i coef2_128 = _mm_set1_epi32(alpha[2] + (alpha[3] << 16));

    for (int k = 0; k < count; k++)
    {
        const unsigned char *S = src[k];
        int *D = dst[k];
        int dx = 0, limit = xmin;
        for (;;)
        {
#if OPTIMISED_COEFF
            for (; dx < limit; dx++)
            {
                int j;
                int sx = (dx * 2) - cn;
#else
            for (; dx < limit; dx++, alpha += 4)
            {
                int j;
                int sx = xofs[dx] - cn;
#endif
                int v = 0;
                for (j = 0; j < 4; j++)
                {
                    int sxj = sx + j * cn;
                    if ((unsigned)sxj >= (unsigned)swidth)
                    {
                        while (sxj < 0)
                            sxj += cn;
                        while (sxj >= swidth)
                            sxj -= cn;
                    }
                    v += S[sxj] * alpha[j];
                }
                D[dx] = v;
            }
            if (limit == dwidth)
                break;
#if OPTIMISED_COEFF
            for (; dx < xmax_64; dx+=64)
            {
                int sx = dx * 2;
#else
            for (; dx < xmax; dx++, alpha += 4)
            {
                int sx = xofs[dx]; // sx - 2, 4, 6, 8....
#endif
                __m512i val0 = _mm512_loadu_si512((__m512i*)(S + sx - 1));
                __m512i val2 = _mm512_loadu_si512((__m512i*)(S + sx + 1));
                __m512i val64 = _mm512_loadu_si512((__m512i*)(S + sx - 1 + 64));
                __m512i val66 = _mm512_loadu_si512((__m512i*)(S + sx + 1 + 64));

                __m512i val0_lo = _mm512_unpacklo_epi8(val0, zero_512);
                __m512i val0_hi = _mm512_unpackhi_epi8(val0, zero_512);
                __m512i val2_lo = _mm512_unpacklo_epi8(val2, zero_512);
                __m512i val2_hi = _mm512_unpackhi_epi8(val2, zero_512);

                __m512i val64_lo = _mm512_unpacklo_epi8(val64, zero_512);
                __m512i val64_hi = _mm512_unpackhi_epi8(val64, zero_512);
                __m512i val66_lo = _mm512_unpacklo_epi8(val66, zero_512);
                __m512i val66_hi = _mm512_unpackhi_epi8(val66, zero_512);

                __m512i res0_lo = _mm512_madd_epi16(val0_lo, coef0_512);
                __m512i res0_hi = _mm512_madd_epi16(val0_hi, coef0_512);
                __m512i res2_lo = _mm512_madd_epi16(val2_lo, coef2_512);
                __m512i res2_hi = _mm512_madd_epi16(val2_hi, coef2_512);

                __m512i res64_lo = _mm512_madd_epi16(val64_lo, coef0_512);
                __m512i res64_hi = _mm512_madd_epi16(val64_hi, coef0_512);
                __m512i res66_lo = _mm512_madd_epi16(val66_lo, coef2_512);
                __m512i res66_hi = _mm512_madd_epi16(val66_hi, coef2_512);

                __m512i r0_lo = _mm512_add_epi32(res0_lo, res2_lo);
                __m512i r0_hi = _mm512_add_epi32(res0_hi, res2_hi);
                __m512i r1_lo = _mm512_add_epi32(res64_lo, res66_lo);
                __m512i r1_hi = _mm512_add_epi32(res64_hi, res66_hi);
                __m512i tmp0 = r0_lo;
                __m512i tmp1 = r1_lo;

                r0_lo = _mm512_permutex2var_epi64(r0_lo, permlo_512, r0_hi);
                r0_hi = _mm512_permutex2var_epi64(tmp0, permhi_512, r0_hi);
                r1_lo = _mm512_permutex2var_epi64(r1_lo, permlo_512, r1_hi);
                r1_hi = _mm512_permutex2var_epi64(tmp1, permhi_512, r1_hi);

                _mm512_storeu_si512((__m512i*)(D + dx), r0_lo);
                _mm512_storeu_si512((__m512i*)(D + dx + 16), r0_hi);
                _mm512_storeu_si512((__m512i*)(D + dx + 32), r1_lo);
                _mm512_storeu_si512((__m512i*)(D + dx + 48), r1_hi);
            }   
            for (; dx < xmax_32; dx+=32)
            {
                int sx = dx * 2;
                __m512i val0 = _mm512_loadu_si512((__m512i*)(S + sx - 1));
                __m512i val2 = _mm512_loadu_si512((__m512i*)(S + sx + 1));

                __m512i val0_lo = _mm512_unpacklo_epi8(val0, zero_512);
                __m512i val0_hi = _mm512_unpackhi_epi8(val0, zero_512);

                __m512i val2_lo = _mm512_unpacklo_epi8(val2, zero_512);
                __m512i val2_hi = _mm512_unpackhi_epi8(val2, zero_512);

                __m512i res0_lo = _mm512_madd_epi16(val0_lo, coef0_512);
                __m512i res0_hi = _mm512_madd_epi16(val0_hi, coef0_512);
                __m512i res2_lo = _mm512_madd_epi16(val2_lo, coef2_512);
                __m512i res2_hi = _mm512_madd_epi16(val2_hi, coef2_512);

                __m512i res_lo = _mm512_add_epi32(res0_lo, res2_lo);
                __m512i res_hi = _mm512_add_epi32(res0_hi, res2_hi);
                __m512i tmp = res_lo;

                res_lo = _mm512_permutex2var_epi64(res_lo, permlo_512, res_hi);
                res_hi = _mm512_permutex2var_epi64(tmp, permhi_512, res_hi);

                _mm512_storeu_si512((__m512i*)(D + dx), res_lo);
                _mm512_storeu_si512((__m512i*)(D + dx + 16), res_hi);
            }
            for (; dx < xmax_16; dx+=16)
            {
                int sx = dx * 2;
                __m512i val0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(S + sx - 1)));
                __m512i val2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(S + sx + 1)));

                __m512i res0_lo = _mm512_madd_epi16(val0, coef0_512);
                __m512i res0_hi = _mm512_madd_epi16(val0, coef0_512);
                __m512i res2_lo = _mm512_madd_epi16(val2, coef2_512);
                __m512i res2_hi = _mm512_madd_epi16(val2, coef2_512);

                __m512i res_lo = _mm512_add_epi32(res0_lo, res2_lo);
                __m512i res_hi = _mm512_add_epi32(res0_hi, res2_hi);

                _mm512_storeu_si512((__m512i*)(D + dx), res_lo);
                _mm512_storeu_si512((__m512i*)(D + dx + 16), res_hi);
            }
            for (; dx < xmax_8; dx+=8)
            {
                int sx = dx * 2;

                __m128i val0 = _mm_loadu_si128((__m128i*)(S + sx - 1));
                __m128i val2 = _mm_loadu_si128((__m128i*)(S + sx + 1));

                __m256i val0_16 = _mm256_cvtepu8_epi16(val0);
                __m256i val2_16 = _mm256_cvtepu8_epi16(val2);

                __m256i res0 = _mm256_madd_epi16(val0_16, coef0_256);
                __m256i res2 = _mm256_madd_epi16(val2_16, coef2_256);

                __m256i res = _mm256_add_epi32(res0, res2);
                _mm256_storeu_si256((__m256i*)(D + dx), res);
            }
            for (; dx < xmax_4; dx+=4)
            {
                int sx = dx * 2;

                __m128i val0 = _mm_loadu_si128((__m128i*)(S + sx - 1));
                __m128i val2 = _mm_loadu_si128((__m128i*)(S + sx + 1));

                __m128i val0_16 = _mm_cvtepu8_epi16(val0);
                __m128i val2_16 = _mm_cvtepu8_epi16(val2);

                __m128i res0 = _mm_madd_epi16(val0_16, coef0_128);
                __m128i res2 = _mm_madd_epi16(val2_16, coef2_128);

                __m128i res = _mm_add_epi32(res0, res2);
                _mm_storeu_si128((__m128i*)(D + dx), res);
            }
            for (; dx < xmax; dx++)
            {
                int sx = dx * 2;
                D[dx] = S[sx - 1] * alpha[0] + S[sx] * alpha[1] + S[sx + 1] * alpha[2] + S[sx + 2] * alpha[3];
            }
            limit = dwidth;
        }
#if !OPTIMISED_COEFF
        alpha -= dwidth * 4;
#endif
    }
}

void vresize_avx512(const int **src, unsigned char *dst, const short *beta, int width)
{
    int b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
    const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
    int bits = 22;
    
    __m512i sh_32_to_8_512 = _mm512_set_epi64(0x8080808080808080, 0x808080800C080400, 0x8080808080808080, 0x808080800C080400, 0x8080808080808080, 0x808080800C080400, 0x8080808080808080, 0x808080800C080400);
    __m512i perm0_512 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 8, 4, 0);
    __m512i perm8_512 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 12, 8, 4, 0, 1, 1, 1, 1);
    __m512i coef0_512 = _mm512_set1_epi32(beta[0]);
    __m512i coef1_512 = _mm512_set1_epi32(beta[1]);
    __m512i delta_512 = _mm512_set1_epi32(1 << (bits - 1));
    __m512i max_char_512 = _mm512_set1_epi32(255);
    __m512i zero_512 = _mm512_setzero_si512();

    __m256i sh_32_to_8_256 = _mm256_set_epi64x(0x8080808080808080, 0x808080800C080400, 0x8080808080808080, 0x808080800C080400);
    __m256i perm0_256 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 4, 0);
    __m256i perm8_256 = _mm256_set_epi32(1, 1, 1, 1, 4, 0, 1, 1);
    __m256i coef0_256 = _mm256_set1_epi32(beta[0]);
    __m256i coef1_256 = _mm256_set1_epi32(beta[1]);
    __m256i delta_256 = _mm256_set1_epi32(1 << (bits - 1));
    __m256i max_char_256 = _mm256_set1_epi32(255);
    __m256i zero_256 = _mm256_setzero_si256();

    __m128i sh_32_to_8_128 =  _mm_set_epi64x(0x8080808080808080, 0x808080800C080400);
    __m128i coef0_128 = _mm_set1_epi32(beta[0]);
    __m128i coef1_128 = _mm_set1_epi32(beta[1]);
    __m128i delta_128 = _mm_set1_epi32(1 << (bits - 1));
    __m128i max_char_128 = _mm_set1_epi32(255);
    __m128i zero_128 = _mm_setzero_si128();

    int width_32 = width - (width % 32);
    int width_16 = width - (width % 16);
    int width_8 = width - (width % 8);
    int width_4 = width - (width % 4);
    int x = 0;

    for (; x < width_32; x+=32)
    {
        __m512i src0_0 = _mm512_loadu_si512((__m512i*)(S0 + x));
        __m512i src1_0 = _mm512_loadu_si512((__m512i*)(S1 + x));
        __m512i src2_0 = _mm512_loadu_si512((__m512i*)(S2 + x));
        __m512i src3_0 = _mm512_loadu_si512((__m512i*)(S3 + x));

        __m512i src0_16 = _mm512_loadu_si512((__m512i*)(S0 + x + 16));
        __m512i src1_16 = _mm512_loadu_si512((__m512i*)(S1 + x + 16));
        __m512i src2_16 = _mm512_loadu_si512((__m512i*)(S2 + x + 16));
        __m512i src3_16 = _mm512_loadu_si512((__m512i*)(S3 + x + 16));

        __m512i mul0_0 = _mm512_mullo_epi32(src0_0, coef0_512);
        __m512i mul1_0 = _mm512_mullo_epi32(src1_0, coef1_512);
        __m512i mul2_0 = _mm512_mullo_epi32(src2_0, coef1_512);
        __m512i mul3_0 = _mm512_mullo_epi32(src3_0, coef0_512);

        __m512i mul0_8 = _mm512_mullo_epi32(src0_16, coef0_512);
        __m512i mul1_8 = _mm512_mullo_epi32(src1_16, coef1_512);
        __m512i mul2_8 = _mm512_mullo_epi32(src2_16, coef1_512);
        __m512i mul3_8 = _mm512_mullo_epi32(src3_16, coef0_512);

        __m512i accum_01_0 = _mm512_add_epi32(mul0_0, mul1_0);
        __m512i accum_23_0 = _mm512_add_epi32(mul2_0, mul3_0);
        __m512i accum_01_8 = _mm512_add_epi32(mul0_8, mul1_8);
        __m512i accum_23_8 = _mm512_add_epi32(mul2_8, mul3_8);
        __m512i accum_0123_0 = _mm512_add_epi32(accum_01_0, accum_23_0);
        __m512i accum_0123_8 = _mm512_add_epi32(accum_01_8, accum_23_8);

        accum_0123_0 = _mm512_add_epi32(accum_0123_0, delta_512);
        accum_0123_8 = _mm512_add_epi32(accum_0123_8, delta_512);
        accum_0123_0 = _mm512_srai_epi32(accum_0123_0, bits);
        accum_0123_8 = _mm512_srai_epi32(accum_0123_8, bits);

        accum_0123_0 = _mm512_max_epi32(accum_0123_0, zero_512);
        accum_0123_8 = _mm512_max_epi32(accum_0123_8, zero_512);
        accum_0123_0 = _mm512_min_epi32(accum_0123_0, max_char_512);
        accum_0123_8 = _mm512_min_epi32(accum_0123_8,max_char_512);

        accum_0123_0 = _mm512_shuffle_epi8(accum_0123_0, sh_32_to_8_512);
        accum_0123_8 = _mm512_shuffle_epi8(accum_0123_8, sh_32_to_8_512);
        
        accum_0123_0 = _mm512_permutexvar_epi32(perm0_512, accum_0123_0);
        accum_0123_8 = _mm512_permutexvar_epi32(perm8_512, accum_0123_8);
        __m256i accum = _mm512_extracti32x8_epi32(_mm512_or_si512(accum_0123_0, accum_0123_8), 0);
        _mm256_storeu_si256((__m256i*)(dst + x), accum);
    }
    for (; x < width_16; x+=16)
    {
        __m256i src0_0 = _mm256_loadu_si256((__m256i*)(S0 + x));
        __m256i src1_0 = _mm256_loadu_si256((__m256i*)(S1 + x));
        __m256i src2_0 = _mm256_loadu_si256((__m256i*)(S2 + x));
        __m256i src3_0 = _mm256_loadu_si256((__m256i*)(S3 + x));

        __m256i src0_8 = _mm256_loadu_si256((__m256i*)(S0 + x + 8));
        __m256i src1_8 = _mm256_loadu_si256((__m256i*)(S1 + x + 8));
        __m256i src2_8 = _mm256_loadu_si256((__m256i*)(S2 + x + 8));
        __m256i src3_8 = _mm256_loadu_si256((__m256i*)(S3 + x + 8));

        __m256i mul0_0 = _mm256_mullo_epi32(src0_0, coef0_256);
        __m256i mul1_0 = _mm256_mullo_epi32(src1_0, coef1_256);
        __m256i mul2_0 = _mm256_mullo_epi32(src2_0, coef1_256);
        __m256i mul3_0 = _mm256_mullo_epi32(src3_0, coef0_256);

        __m256i mul0_8 = _mm256_mullo_epi32(src0_8, coef0_256);
        __m256i mul1_8 = _mm256_mullo_epi32(src1_8, coef1_256);
        __m256i mul2_8 = _mm256_mullo_epi32(src2_8, coef1_256);
        __m256i mul3_8 = _mm256_mullo_epi32(src3_8, coef0_256);

        __m256i accum_01_0 = _mm256_add_epi32(mul0_0, mul1_0);
        __m256i accum_23_0 = _mm256_add_epi32(mul2_0, mul3_0);
        __m256i accum_01_8 = _mm256_add_epi32(mul0_8, mul1_8);
        __m256i accum_23_8 = _mm256_add_epi32(mul2_8, mul3_8);
        __m256i accum_0123_0 = _mm256_add_epi32(accum_01_0, accum_23_0);
        __m256i accum_0123_8 = _mm256_add_epi32(accum_01_8, accum_23_8);

        accum_0123_0 = _mm256_add_epi32(accum_0123_0, delta_256);
        accum_0123_8 = _mm256_add_epi32(accum_0123_8, delta_256);
        accum_0123_0 = _mm256_srai_epi32(accum_0123_0, bits);
        accum_0123_8 = _mm256_srai_epi32(accum_0123_8, bits);

        accum_0123_0 = _mm256_max_epi32(accum_0123_0, zero_256);
        accum_0123_8 = _mm256_max_epi32(accum_0123_8, zero_256);
        accum_0123_0 = _mm256_min_epi32(accum_0123_0, max_char_256);
        accum_0123_8 = _mm256_min_epi32(accum_0123_8, max_char_256);

        accum_0123_0 = _mm256_shuffle_epi8(accum_0123_0, sh_32_to_8_256);
        accum_0123_8 = _mm256_shuffle_epi8(accum_0123_8, sh_32_to_8_256);
        accum_0123_0 = _mm256_permutevar8x32_epi32(accum_0123_0, perm0_256);
        accum_0123_8 = _mm256_permutevar8x32_epi32(accum_0123_8, perm8_256);

        __m128i accum = _mm256_extracti128_si256(_mm256_or_si256(accum_0123_0, accum_0123_8), 0);
        _mm_storeu_si128((__m128i*)(dst + x), accum);
    }
    for (; x < width_8; x+=8)
    {
        __m256i src0_0 = _mm256_loadu_si256((__m256i*)(S0 + x));
        __m256i src1_0 = _mm256_loadu_si256((__m256i*)(S1 + x));
        __m256i src2_0 = _mm256_loadu_si256((__m256i*)(S2 + x));
        __m256i src3_0 = _mm256_loadu_si256((__m256i*)(S3 + x));

        __m256i mul0_0 = _mm256_mullo_epi32(src0_0, coef0_256);
        __m256i mul1_0 = _mm256_mullo_epi32(src1_0, coef1_256);
        __m256i mul2_0 = _mm256_mullo_epi32(src2_0, coef1_256);
        __m256i mul3_0 = _mm256_mullo_epi32(src3_0, coef0_256);

        __m256i accum_01_0 = _mm256_add_epi32(mul0_0, mul1_0);
        __m256i accum_23_0 = _mm256_add_epi32(mul2_0, mul3_0);
        __m256i accum_0123_0 = _mm256_add_epi32(accum_01_0, accum_23_0);

        accum_0123_0 = _mm256_add_epi32(accum_0123_0, delta_256);
        accum_0123_0 = _mm256_srai_epi32(accum_0123_0, bits);

        accum_0123_0 = _mm256_max_epi32(accum_0123_0, zero_256);
        accum_0123_0 = _mm256_min_epi32(accum_0123_0, max_char_256);

        accum_0123_0 = _mm256_shuffle_epi8(accum_0123_0, sh_32_to_8_256);
        accum_0123_0 = _mm256_permutevar8x32_epi32(accum_0123_0, perm0_256);

        __m128i accum = _mm256_castsi256_si128(accum_0123_0);
        _mm_storel_epi64((__m128i*)(dst + x), accum);
    }
    for (; x < width_4; x+=4)
    {
        __m128i src0_0 = _mm_loadu_si128((__m128i*)(S0 + x));
        __m128i src1_0 = _mm_loadu_si128((__m128i*)(S1 + x));
        __m128i src2_0 = _mm_loadu_si128((__m128i*)(S2 + x));
        __m128i src3_0 = _mm_loadu_si128((__m128i*)(S3 + x));

        __m128i mul0_0 = _mm_mullo_epi32(src0_0, coef0_128);
        __m128i mul1_0 = _mm_mullo_epi32(src1_0, coef1_128);
        __m128i mul2_0 = _mm_mullo_epi32(src2_0, coef1_128);
        __m128i mul3_0 = _mm_mullo_epi32(src3_0, coef0_128);

        __m128i accum_01_0 = _mm_add_epi32(mul0_0, mul1_0);
        __m128i accum_23_0 = _mm_add_epi32(mul2_0, mul3_0);
        __m128i accum_0123_0 = _mm_add_epi32(accum_01_0, accum_23_0);

        accum_0123_0 = _mm_add_epi32(accum_0123_0, delta_128);
        accum_0123_0 = _mm_srai_epi32(accum_0123_0, bits);

        accum_0123_0 = _mm_max_epi32(accum_0123_0, zero_128);
        accum_0123_0 = _mm_min_epi32(accum_0123_0, max_char_128);

        accum_0123_0 = _mm_shuffle_epi8(accum_0123_0, sh_32_to_8_128);
        _mm_maskstore_epi32((int*)(dst + x), _mm_set_epi32(0, 0, 0, 0x80000000), accum_0123_0);
    }

    for (; x < width; x++)
        dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
}

static int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

#if OPTIMISED_COEFF
void step_avx512(const unsigned char *_src, unsigned char *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax)
#else
void step_avx512(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
#endif
{
    int dy, cn = channels;

    int bufstep = (int)((dwidth + 16 - 1) & -16);
    int *_buffer = (int *)malloc(bufstep * ksize * sizeof(int));
    if (_buffer == NULL)
    {
        printf("resizer: malloc fails\n");
        return;
    }
    const unsigned char *srows[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int *rows[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int prev_sy[MAX_ESIZE];

    for (int k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = _buffer + bufstep * k;
    }

#if !OPTIMISED_COEFF
    const short *beta = _beta + ksize * start;
#endif

#if OPTIMISED_COEFF
    for (dy = start; dy < end; dy++)
    {
        int sy0 = dy * 2;
#else
    for (dy = start; dy < end; dy++, beta += ksize)
    {
        int sy0 = yofs[dy];
#endif
        int k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (int k = 0; k < ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, iheight);
            for (k1 = MAX(k1, k); k1 < ksize; k1++)
            {
                if (k1 < MAX_ESIZE && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = MIN(k0, k); // remember the first row that needs to be computed
            srows[k] = _src + (sy * iwidth);
            prev_sy[k] = sy;
        }



        // regular c
#if OPTIMISED_COEFF
        if (k0 < ksize)
        {
            hresize_avx512((srows + k0), (rows + k0), ksize - k0, _alpha,
                    iwidth, dwidth, cn, xmin, xmax);
        }
        vresize_avx512((const int **)rows, (_dst + dwidth * dy), _beta, dwidth);
#else
        if (k0 < ksize)
        {
            hresize_avx512((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                    iwidth, dwidth, cn, xmin, xmax);
        }
        vresize_avx512((const int **)rows, (_dst + dwidth * dy), beta, dwidth);
#endif
    }
    free(_buffer);
}