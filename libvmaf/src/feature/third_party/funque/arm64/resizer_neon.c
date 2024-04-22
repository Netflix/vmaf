/**
 *
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
#include <arm_neon.h>
#include <time.h>
#include "../resizer.h"
#include "resizer_neon.h"

#if OPTIMISED_COEFF
void hresize_neon(const unsigned char **src, int **dst, int count,
                  const short *alpha,
                  int swidth, int dwidth, int cn, int xmin, int xmax)
#else
void hresize_neon(const unsigned char **src, int **dst, int count,
                  const int *xofs, const short *alpha,
                  int swidth, int dwidth, int cn, int xmin, int xmax)
#endif
{
    // int first_col_count = 0;
    uint8x8_t src1_8x8, src2_8x8, src3_8x8;
    int simd_loop = (xmax / 8) * 8;
    int num_pix = 8;

#if OPTIMISED_COEFF
    int sx_start = 2;
#else
    int sx_start = xofs[1];
#endif

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

            int start = sx_start - cn;
            src1_8x8 = vld1_u8(S + start);
#if OPTIMISED_COEFF
            for (; dx < simd_loop;)
            {
#else
            for (; dx < simd_loop; alpha += 32)
            {
#endif
                start += num_pix;
                src2_8x8 = vld1_u8(S + start);
                start += num_pix;
                src3_8x8 = vld1_u8(S + start);

                uint16x8_t movl1_16x8 = vmovl_u8(src1_8x8);
                uint16x8_t movl2_16x8 = vmovl_u8(src2_8x8);
                uint16x8_t movl3_16x8 = vmovl_u8(src3_8x8);
                int16x8_t s_movl1_16x8 = vreinterpretq_s16_u16(movl1_16x8);
                int16x8_t s_movl2_16x8 = vreinterpretq_s16_u16(movl2_16x8);
                int16x8_t s_movl3_16x8 = vreinterpretq_s16_u16(movl3_16x8);
                int16x8x2_t t1 = vuzpq_s16(s_movl1_16x8, s_movl2_16x8); // 0 odd, 1 even
                int16x8x2_t t2 = vuzpq_s16(s_movl3_16x8, s_movl3_16x8);
                int16x8_t vx1 = vextq_s16(t1.val[0], t2.val[0], 1); // s_movl3_16x8,1);
                int16x8_t vx2 = vextq_s16(t1.val[1], t2.val[1], 1);
                int32x4_t m1_l = vmull_n_s16(vget_low_s16(t1.val[0]), alpha[0]);
                int32x4_t m1_h = vmull_n_s16(vget_high_s16(t1.val[0]), alpha[0]);
                int32x4_t m2_l = vmlal_n_s16(m1_l, vget_low_s16(vx1), alpha[1]);
                int32x4_t m2_h = vmlal_n_s16(m1_h, vget_high_s16(vx1), alpha[1]);
                int32x4_t m3_l = vmlal_n_s16(m2_l, vget_low_s16(t1.val[1]), alpha[2]);
                int32x4_t m3_h = vmlal_n_s16(m2_h, vget_high_s16(t1.val[1]), alpha[2]);
                int32x4_t out_l = vmlal_n_s16(m3_l, vget_low_s16(vx2), alpha[3]);  // final out
                int32x4_t out_h = vmlal_n_s16(m3_h, vget_high_s16(vx2), alpha[3]); // final out

                vst1q_s32(D + dx, out_l);
                dx += 4;
                vst1q_s32(D + dx, out_h);
                dx += 4;
                src1_8x8 = src3_8x8;
            }

#if OPTIMISED_COEFF
            for (; dx < xmax; dx++)
            {
                int sx2 = dx * 2;
#else
            for (; dx < xmax; dx++, alpha += 4)
            {
                int sx2 = xofs[dx]; // sx - 2, 4, 6, 8....
#endif
                D[dx] = S[sx2 - 1] * alpha[0] + S[sx2] * alpha[1] + S[sx2 + 1] * alpha[2] + S[sx2 + 2] * alpha[3];
            }
            limit = dwidth;
        }
#if !OPTIMISED_COEFF
        alpha -= dwidth * 4;
#endif
    }
}

void vresize_neon(const int **src, unsigned char *dst, const short *beta, int width)
{
    int32x4_t src_1, src_2, src_3, src_4, src_1_mul;
    int32x4_t d4_q;
    int32x4_t add_1;
    int32x4_t add_delta;
    int32x4_t shift_right_32x4;
    uint16x4_t shift_right_16x4;
    uint16x8_t shift_right_16x8;
    int32x4_t dt;
    uint8x8_t dt2;


#define BITS 22
    int bits = BITS;

    // int32x4_t SHIFT = vdupq_n_s32(bits);
    int DELTA = (1 << (bits - 1));
    // b1_vq = vdupq_n_s32(beta[0]);
    // b2_vq = vdupq_n_s32(beta[1]);
    // b3_vq = vdupq_n_s32(beta[2]);
    // b4_vq = vdupq_n_s32(beta[3]);
    d4_q = vdupq_n_s32(DELTA);
    src_1_mul = vdupq_n_s32(0);

    int32x4_t lower = vdupq_n_s32(0);
    int32x4_t higher = vdupq_n_s32(255);

    for (int x = 0; x < width; x += 4)
    {
        src_1 = vld1q_s32(src[0] + x);
        src_2 = vld1q_s32(src[1] + x);
        src_3 = vld1q_s32(src[2] + x);
        src_4 = vld1q_s32(src[3] + x);

        add_1 = vmlaq_n_s32(src_1_mul, src_1, beta[0]);
        add_1 = vmlaq_n_s32(add_1, src_2, beta[1]);
        add_1 = vmlaq_n_s32(add_1, src_3, beta[2]);
        add_1 = vmlaq_n_s32(add_1, src_4, beta[3]);

        add_delta = vaddq_s32(add_1, d4_q);

        shift_right_32x4 = vshrq_n_s32(add_delta, BITS); // 32x4

        dt = vminq_s32(shift_right_32x4, higher);
        dt = vmaxq_s32(dt, lower);

        // shift_right_32x4 = vshrq_n_s32(add_delta, BITS); // 32x4

        shift_right_16x4 = vqmovun_s32(dt);                                  // 16x4
        shift_right_16x8 = vcombine_u16(shift_right_16x4, shift_right_16x4); // 16x8
        dt2 = vqmovn_u16(shift_right_16x8);                                  // 8x8

        vst1_lane_u32((unsigned int *)(dst + x), vreinterpret_u32_u8(dt2), 0);
    }

#undef BITS
}

static int clip_neon(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

#if OPTIMISED_COEFF
void step_neon(const unsigned char *_src, unsigned char *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax)
#else
void step_neon(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
#endif
{
    int dy, cn = channels;

    int bufstep = (int)((dwidth + 16 - 1) & -16);
    int *_buffer = (int *)malloc(bufstep * ksize * sizeof(int));
    if (_buffer == NULL)
    {
        printf("malloc fails\n");
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
            int sy = clip_neon(sy0 - ksize2 + 1 + k, 0, iheight);
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

        

#if OPTIMISED_COEFF
        if (k0 < ksize)
        {
            hresize_neon((srows + k0), (rows + k0), ksize - k0, _alpha,
                         iwidth, dwidth, cn, xmin, xmax);
        }
#if USE_C_VRESIZE
        vresize((const int **)rows, (_dst + dwidth * dy), _beta, dwidth);
#elif !USE_C_VRESIZE
        vresize_neon((const int **)rows, (_dst + dwidth * dy), _beta, dwidth);
#endif
#else
        if (k0 < ksize)
        {
            hresize_neon((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                         iwidth, dwidth, cn, xmin, xmax);
        }
#if USE_C_VRESIZE
        vresize((const int **)rows, (_dst + dwidth * dy), beta, dwidth);
#elif !USE_C_VRESIZE
        vresize_neon((const int **)rows, (_dst + dwidth * dy), beta, dwidth);
#endif
#endif

    }

    free(_buffer);
}
