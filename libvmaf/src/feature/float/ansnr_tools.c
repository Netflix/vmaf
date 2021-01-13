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

#include <stddef.h>
#include "mem.h"
#include "ansnr_options.h"
#include "ansnr_tools.h"

#if 1
const float ansnr_filter1d_ref_s[3] = { 0.250138193, 0.499723613, 0.250138193 };

const float ansnr_filter1d_dis_s[5] = { 0.054488685, 0.244201347, 0.402619958, 0.244201347, 0.054488685 };

#else

const float ansnr_filter1d_ref_s[3] = { 0x1.00243ap-2, 0x1.ffb78cp-2, 0x1.00243ap-2 };

const float ansnr_filter1d_dis_s[5] = { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 };

#endif

const int ansnr_filter1d_ref_width = 3;
const int ansnr_filter1d_dis_width = 5;

const float ansnr_filter2d_ref_s[3*3] = {
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

const float ansnr_filter2d_dis_s[5*5] = {
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    12.0 / 571.0, 52.0 / 571.0, 127.0 / 571.0, 52.0 / 571.0, 12.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0
};

const int ansnr_filter2d_ref_width = 3;
const int ansnr_filter2d_dis_width = 5;

void ansnr_mse_s(const float *ref, const float *dis, float *sig, float *noise, int w, int h, int ref_stride, int dis_stride)
{
    int ref_px_stride = ref_stride / sizeof(float);
    int dis_px_stride = dis_stride / sizeof(float);
    int i, j;

    float ref_val, dis_val;

    float sig_accum = 0;
    float noise_accum = 0;

    for (i = 0; i < h; ++i) {
        float sig_accum_inner = 0;
        float noise_accum_inner = 0;

        for (j = 0; j < w; ++j) {
            ref_val = ref[i * ref_px_stride + j];
            dis_val = dis[i * dis_px_stride + j];

            sig_accum_inner   += ref_val * ref_val;
            noise_accum_inner += (ref_val - dis_val) * (ref_val - dis_val);
        }

        sig_accum   += sig_accum_inner;
        noise_accum += noise_accum_inner;
    }

    if (sig)
        *sig = sig_accum;
    if (noise)
        *noise = noise_accum;
}

void ansnr_filter1d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = f[fi];

                ii = i - fwidth / 2 + fi;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                ii = ii < 0 ? 0 : (ii > h - 1 ? h - 1 : ii);
                imgcoeff = src[ii * src_px_stride + j];
#else
                if (ii < 0) ii = -ii;
                else if (ii >= h) ii = 2 * h - ii - 1;
                imgcoeff = src[ii * src_px_stride + j];
#endif
                accum += fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = f[fj];

                jj = j - fwidth / 2 + fj;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                jj = jj < 0 ? 0 : (jj > w - 1 ? w - 1 : jj);
                imgcoeff = tmp[jj];
#else
                if (jj < 0) jj = -jj;
                else if (jj >= w) jj = 2 * w - jj - 1;
                imgcoeff = tmp[jj];
#endif
                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);
}

void ansnr_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float fcoeff, imgcoeff;
    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                float accum_inner = 0;

                for (fj = 0; fj < fwidth; ++fj) {
                    fcoeff = f[fi * fwidth + fj];

                    ii = i - fwidth / 2 + fi;
                    jj = j - fwidth / 2 + fj;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                    ii = ii < 0 ? 0 : (ii > h - 1 ? h - 1 : ii);
                    jj = jj < 0 ? 0 : (jj > w - 1 ? w - 1 : jj);
                    imgcoeff = src[ii * src_px_stride + jj];
#else
                    if (ii < 0) ii = -ii;
                    else if (ii >= h) ii = 2 * h - ii - 1;
                    if (jj < 0) jj = -jj;
                    else if (jj >= w) jj = 2 * w - jj - 1;
                    imgcoeff = src[ii * src_px_stride + jj];
#endif
                    accum_inner += fcoeff * imgcoeff;
                }

                accum += accum_inner;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}

