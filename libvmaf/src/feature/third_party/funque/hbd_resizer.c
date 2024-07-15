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

#include "resizer.h"

//const int HBD_INTER_RESIZE_COEF_SCALE = 2048;
//static const int HBD_MAX_ESIZE = 16;

//#define CLIP3(X, MIN, MAX) ((X < MIN) ? MIN : (X > MAX) ? MAX \
//                                                        : X)
//#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
//#define MIN(LEFT, RIGHT) (LEFT < RIGHT ? LEFT : RIGHT)

// enabled by default for funque since resize factor is always 0.5, disabled otherwise
//#define OPTIMISED_COEFF 1

//#define USE_C_VRESIZE 0

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
void hbd_hresize(const unsigned short **src, int **dst, int count,
                 const short *alpha,
                 int swidth, int dwidth, int cn, int xmin, int xmax)
#else
void hbd_hresize(const unsigned short **src, int **dst, int count,
                 const int *xofs, const short *alpha,
                 int swidth, int dwidth, int cn, int xmin, int xmax)
#endif
{
    for (int k = 0; k < count; k++)
    {
        const unsigned short *S = src[k];
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
            for (; dx < xmax; dx++)
            {
                int sx = dx * 2;
#else
            for (; dx < xmax; dx++, alpha += 4)
            {
                int sx = xofs[dx]; // sx - 2, 4, 6, 8....
#endif
                D[dx] = S[sx - 1] * alpha[0] + S[sx] * alpha[1] + S[sx + 1] * alpha[2] + S[sx + 2] * alpha[3];
            }
            limit = dwidth;
        }
#if !OPTIMISED_COEFF
        alpha -= dwidth * 4;
#endif
    }
}

unsigned short hbd_castOp(int64_t val, int bitdepth)
{
    int bits = 22;
    int SHIFT = bits;
    int DELTA = (1 << (bits - 1));
    return CLIP3((val + DELTA) >> SHIFT, 0, ((1 << bitdepth) - 1));
}

static int hbd_clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

void hbd_vresize(const int **src, unsigned short *dst, const short *beta, int width, int bitdepth)
{
    int b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
    const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];

    for (int x = 0; x < width; x++)
        dst[x] = hbd_castOp((int64_t)S0[x] * b0 + (int64_t)S1[x] * b1 + (int64_t)S2[x] * b2 + (int64_t)S3[x] * b3, bitdepth);
}

#if OPTIMISED_COEFF
void hbd_step(const unsigned short *_src, unsigned short *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax, int bitdepth)
#else
void hbd_step(const unsigned short *_src, unsigned short *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax, int bitdepth)
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
    const unsigned short *srows[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
            int sy = hbd_clip(sy0 - ksize2 + 1 + k, 0, iheight);
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
            hbd_hresize((srows + k0), (rows + k0), ksize - k0, _alpha,
                        iwidth, dwidth, cn, xmin, xmax);
        }
        hbd_vresize((const int **)rows, (_dst + dwidth * dy), _beta, dwidth, bitdepth);
#else
        if (k0 < ksize)
        {
            hbd_hresize((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                        iwidth, dwidth, cn, xmin, xmax);
        }
        hbd_vresize((const int **)rows, (_dst + dwidth * dy), beta, dwidth, bitdepth);
#endif
    }
    free(_buffer);
}

void hbd_resize(ResizerState m, const unsigned short *_src, unsigned short *_dst, int iwidth, int iheight, int dwidth, int dheight, int bitdepth)
{
    // int depth = 0;
    int cn = 1;
    double inv_scale_x = (double)dwidth / iwidth;

    int ksize = 4, ksize2;
    ksize2 = ksize / 2;

    int xmin = 0, xmax = dwidth;

#if OPTIMISED_COEFF
    const short ibeta[] = {-192, 1216, 1216, -192};
    const short ialpha[] = {-192, 1216, 1216, -192};
    double scale_x = 1. / inv_scale_x;
    float fx;
    int sx;

    for (int dx = 0; dx < dwidth; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = (int)floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1)
        {
            xmin = dx + 1;
        }

        if (sx + ksize2 >= iwidth)
        {
            xmax = MIN(xmax, dx);
        }
    }
    m.hbd_resizer_step(_src, _dst, ialpha, ibeta, iwidth, iheight, dwidth, cn, ksize, 0, dheight, xmin, xmax, bitdepth);
#else
    double inv_scale_y = (double)dheight / iheight;
    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
    int width = dwidth * cn;

    int iscale_x = (int)scale_x;
    int iscale_y = (int)scale_y;

    int k, sx, sy, dx, dy;

    float fx, fy;

    unsigned short *_buffer = (unsigned short *)malloc((width + dheight) * (sizeof(int) + sizeof(float) * ksize));

    int *xofs = (int *)_buffer;
    int *yofs = xofs + width;
    float *alpha = (float *)(yofs + dheight);
    short *ialpha = (short *)alpha;
    float *beta = alpha + width * ksize;
    short *ibeta = ialpha + width * ksize;
    float cbuf[4] = {0};

    for (dx = 0; dx < dwidth; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = (int)floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1)
        {
            xmin = dx + 1;
        }

        if (sx + ksize2 >= iwidth)
        {
            xmax = MIN(xmax, dx);
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;

        interpolateCubic(fx, cbuf);
        for (k = 0; k < ksize; k++)
            ialpha[dx * cn * ksize + k] = (short)(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for (; k < cn * ksize; k++)
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
    }

    for (dy = 0; dy < dheight; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = (int)floor(fy);
        fy -= sy;

        yofs[dy] = sy;

        interpolateCubic(fy, cbuf);
        for (k = 0; k < ksize; k++)
            ibeta[dy * ksize + k] = (short)(cbuf[k] * INTER_RESIZE_COEF_SCALE);
    }
    m.hbd_resizer_step(_src, _dst, xofs, yofs, ialpha, ibeta, iwidth, iheight, dwidth, dheight, cn, ksize, 0, dheight, xmin, xmax, bitdepth);
#endif

}
