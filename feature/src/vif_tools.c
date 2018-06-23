/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "common/alloc.h"
#include "vif_options.h"
#include "vif_tools.h"
#include "common/cpu.h"

extern enum vmaf_cpu cpu;

#ifdef VIF_OPT_FAST_LOG2 // option to replace log2 calculation with faster speed

static const float log2_poly_s[9] = { -0.012671635276421, 0.064841182402670, -0.157048836463065, 0.257167726303123, -0.353800560300520, 0.480131410397451, -0.721314327952201, 1.442694803896991, 0 };

static float horner_s(const float *poly, float x, int n)
{
    float var = 0;
    int i;

    for (i = 0; i < n; ++i) {
        var = var * x + poly[i];
    }
    return var;
}

static float log2f_approx(float x)
{
    const uint32_t exp_zero_const = 0x3F800000UL;

    const uint32_t exp_expo_mask = 0x7F800000UL;
    const uint32_t exp_mant_mask = 0x007FFFFFUL;

    float remain;
    float log_base, log_remain;
    uint32_t u32, u32remain;
    uint32_t exponent, mant;

    if (x == 0)
        return -INFINITY;
    if (x < 0)
        return NAN;

    memcpy(&u32, &x, sizeof(float));

    exponent  = (u32 & exp_expo_mask) >> 23;
    mant      = (u32 & exp_mant_mask) >> 0;
    u32remain = mant | exp_zero_const;

    memcpy(&remain, &u32remain, sizeof(float));

    log_base = (int32_t)exponent - 127;
    log_remain = horner_s(log2_poly_s, (remain - 1.0f), sizeof(log2_poly_s) / sizeof(float));

    return log_base + log_remain;
}

#define log2f log2f_approx

#endif /* VIF_FAST_LOG2 */

#if 1 //defined(_MSC_VER)
const float vif_filter1d_table_s[4][17] = {
    { 0.00745626912, 0.0142655009, 0.0250313189, 0.0402820669, 0.0594526194, 0.0804751068, 0.0999041125, 0.113746084, 0.118773937, 0.113746084, 0.0999041125, 0.0804751068, 0.0594526194, 0.0402820669, 0.0250313189, 0.0142655009, 0.00745626912 },
    { 0.0189780835, 0.0558981746, 0.120920904, 0.192116052, 0.224173605, 0.192116052, 0.120920904, 0.0558981746, 0.0189780835 },
    { 0.054488685, 0.244201347, 0.402619958, 0.244201347, 0.054488685 },
    { 0.166378498, 0.667243004, 0.166378498 }
};
#else
const float vif_filter1d_table_s[4][17] = {
    { 0x1.e8a77p-8,  0x1.d373b2p-7, 0x1.9a1cf6p-6, 0x1.49fd9ep-5, 0x1.e7092ep-5, 0x1.49a044p-4, 0x1.99350ep-4, 0x1.d1e76ap-4, 0x1.e67f8p-4, 0x1.d1e76ap-4, 0x1.99350ep-4, 0x1.49a044p-4, 0x1.e7092ep-5, 0x1.49fd9ep-5, 0x1.9a1cf6p-6, 0x1.d373b2p-7, 0x1.e8a77p-8 },
    { 0x1.36efdap-6, 0x1.c9eaf8p-5, 0x1.ef4ac2p-4, 0x1.897424p-3, 0x1.cb1b88p-3, 0x1.897424p-3, 0x1.ef4ac2p-4, 0x1.c9eaf8p-5, 0x1.36efdap-6 },
    { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 },
    { 0x1.54be4p-3,  0x1.55a0ep-1,  0x1.54be4p-3 }
};
#endif

const int vif_filter1d_width[4] = { 17, 9, 5, 3 };

const float vif_filter2d_table_s[4][17*17] = {
    {
        5.55959459278305e-05, 0.000106367407656738, 0.000186640246416797, 0.000300353914609461, 0.000443294728908000, 0.000600044018801974, 0.000744911927278875, 0.000848121393654296, 0.000885609734747598, 0.000848121393654296, 0.000744911927278875, 0.000600044018801974, 0.000443294728908000, 0.000300353914609461, 0.000186640246416797, 0.000106367407656738, 5.55959459278305e-05,
        0.000106367407656738, 0.000203504504200748, 0.000357084295346644, 0.000574643829570478, 0.000848121393654296, 0.00114801764219911,  0.00142518216598191,  0.00162264482626724,  0.00169436835921373,  0.00162264482626724,  0.00142518216598191,  0.00114801764219911,  0.000848121393654296, 0.000574643829570478, 0.000357084295346644, 0.000203504504200748, 0.000106367407656738,
        0.000186640246416797, 0.000357084295346644, 0.000626566937591844, 0.00100831324477739,  0.00148817752909640,  0.00201439802239364,  0.00250073172325580,  0.00284721454525617,  0.00297306604580432,  0.00284721454525617,  0.00250073172325580,  0.00201439802239364,  0.00148817752909640,  0.00100831324477739,  0.000626566937591844, 0.000357084295346644, 0.000186640246416797,
        0.000300353914609461, 0.000574643829570478, 0.00100831324477739,  0.00162264482626724,  0.00239487439113087,  0.00324170345476482,  0.00402434403558712,  0.00458192726820001,  0.00478445588448085,  0.00458192726820001,  0.00402434403558712,  0.00324170345476482,  0.00239487439113087,  0.00162264482626724,  0.00100831324477739,  0.000574643829570478, 0.000300353914609461,
        0.000443294728908000, 0.000848121393654296, 0.00148817752909640,  0.00239487439113087,  0.00353461414133880,  0.00478445588448085,  0.00593956133585855,  0.00676250285891533,  0.00706141645279030,  0.00676250285891533,  0.00593956133585855,  0.00478445588448085,  0.00353461414133880,  0.00239487439113087,  0.00148817752909640,  0.000848121393654296, 0.000443294728908000,
        0.000600044018801974, 0.00114801764219911,  0.00201439802239364,  0.00324170345476482,  0.00478445588448085,  0.00647624243982486,  0.00803979389213322,  0.00915372804594197,  0.00955833767120214,  0.00915372804594197,  0.00803979389213322,  0.00647624243982486,  0.00478445588448085,  0.00324170345476482,  0.00201439802239364,  0.00114801764219911,  0.000600044018801974,
        0.000744911927278875, 0.00142518216598191,  0.00250073172325580,  0.00402434403558712,  0.00593956133585855,  0.00803979389213322,  0.00998083169809971,  0.0113637016399286,   0.0118659956822055,   0.0113637016399286,   0.00998083169809971,  0.00803979389213322,  0.00593956133585855,  0.00402434403558712,  0.00250073172325580,  0.00142518216598191,  0.000744911927278875,
        0.000848121393654296, 0.00162264482626724,  0.00284721454525617,  0.00458192726820001,  0.00676250285891533,  0.00915372804594197,  0.0113637016399286,   0.0129381717743925,   0.0135100599501079,   0.0129381717743925,   0.0113637016399286,   0.00915372804594197,  0.00676250285891533,  0.00458192726820001,  0.00284721454525617,  0.00162264482626724,  0.000848121393654296,
        0.000885609734747598, 0.00169436835921373,  0.00297306604580432,  0.00478445588448085,  0.00706141645279030,  0.00955833767120214,  0.0118659956822055,   0.0135100599501079,   0.0141072265106852,   0.0135100599501079,   0.0118659956822055,   0.00955833767120214,  0.00706141645279030,  0.00478445588448085,  0.00297306604580432,  0.00169436835921373,  0.000885609734747598,
        0.000848121393654296, 0.00162264482626724,  0.00284721454525617,  0.00458192726820001,  0.00676250285891533,  0.00915372804594197,  0.0113637016399286,   0.0129381717743925,   0.0135100599501079,   0.0129381717743925,   0.0113637016399286,   0.00915372804594197,  0.00676250285891533,  0.00458192726820001,  0.00284721454525617,  0.00162264482626724,  0.000848121393654296,
        0.000744911927278875, 0.00142518216598191,  0.00250073172325580,  0.00402434403558712,  0.00593956133585855,  0.00803979389213322,  0.00998083169809971,  0.0113637016399286,   0.0118659956822055,   0.0113637016399286,   0.00998083169809971,  0.00803979389213322,  0.00593956133585855,  0.00402434403558712,  0.00250073172325580,  0.00142518216598191,  0.000744911927278875,
        0.000600044018801974, 0.00114801764219911,  0.00201439802239364,  0.00324170345476482,  0.00478445588448085,  0.00647624243982486,  0.00803979389213322,  0.00915372804594197,  0.00955833767120214,  0.00915372804594197,  0.00803979389213322,  0.00647624243982486,  0.00478445588448085,  0.00324170345476482,  0.00201439802239364,  0.00114801764219911,  0.000600044018801974,
        0.000443294728908000, 0.000848121393654296, 0.00148817752909640,  0.00239487439113087,  0.00353461414133880,  0.00478445588448085,  0.00593956133585855,  0.00676250285891533,  0.00706141645279030,  0.00676250285891533,  0.00593956133585855,  0.00478445588448085,  0.00353461414133880,  0.00239487439113087,  0.00148817752909640,  0.000848121393654296, 0.000443294728908000,
        0.000300353914609461, 0.000574643829570478, 0.00100831324477739,  0.00162264482626724,  0.00239487439113087,  0.00324170345476482,  0.00402434403558712,  0.00458192726820001,  0.00478445588448085,  0.00458192726820001,  0.00402434403558712,  0.00324170345476482,  0.00239487439113087,  0.00162264482626724,  0.00100831324477739,  0.000574643829570478, 0.000300353914609461,
        0.000186640246416797, 0.000357084295346644, 0.000626566937591844, 0.00100831324477739,  0.00148817752909640,  0.00201439802239364,  0.00250073172325580,  0.00284721454525617,  0.00297306604580432,  0.00284721454525617,  0.00250073172325580,  0.00201439802239364,  0.00148817752909640,  0.00100831324477739,  0.000626566937591844, 0.000357084295346644, 0.000186640246416797,
        0.000106367407656738, 0.000203504504200748, 0.000357084295346644, 0.000574643829570478, 0.000848121393654296, 0.00114801764219911,  0.00142518216598191,  0.00162264482626724,  0.00169436835921373,  0.00162264482626724,  0.00142518216598191,  0.00114801764219911,  0.000848121393654296, 0.000574643829570478, 0.000357084295346644, 0.000203504504200748, 0.000106367407656738,
        5.55959459278305e-05, 0.000106367407656738, 0.000186640246416797, 0.000300353914609461, 0.000443294728908000, 0.000600044018801974, 0.000744911927278875, 0.000848121393654296, 0.000885609734747598, 0.000848121393654296, 0.000744911927278875, 0.000600044018801974, 0.000443294728908000, 0.000300353914609461, 0.000186640246416797, 0.000106367407656738, 5.55959459278305e-05
    },
    {
        0.000360167651076071, 0.00106084022060085, 0.00229484704297441, 0.00364599445829020, 0.00425438469465621, 0.00364599445829020, 0.00229484704297441, 0.00106084022060085, 0.000360167651076071,
        0.00106084022060085,  0.00312460591694495, 0.00675925790681296, 0.0107389365865762,  0.0125308932784936,  0.0107389365865762,  0.00675925790681296, 0.00312460591694495, 0.00106084022060085,
        0.00229484704297441,  0.00675925790681296, 0.0146218654976766,  0.0232308470133570,  0.0271072710362468,  0.0232308470133570,  0.0146218654976766,  0.00675925790681296, 0.00229484704297441,
        0.00364599445829020,  0.0107389365865762,  0.0232308470133570,  0.0369085772977295,  0.0430673409280585,  0.0369085772977295,  0.0232308470133570,  0.0107389365865762,  0.00364599445829020,
        0.00425438469465621,  0.0125308932784936,  0.0271072710362468,  0.0430673409280585,  0.0502537889675776,  0.0430673409280585,  0.0271072710362468,  0.0125308932784936,  0.00425438469465621,
        0.00364599445829020,  0.0107389365865762,  0.0232308470133570,  0.0369085772977295,  0.0430673409280585,  0.0369085772977295,  0.0232308470133570,  0.0107389365865762,  0.00364599445829020,
        0.00229484704297441,  0.00675925790681296, 0.0146218654976766,  0.0232308470133570,  0.0271072710362468,  0.0232308470133570,  0.0146218654976766,  0.00675925790681296, 0.00229484704297441,
        0.00106084022060085,  0.00312460591694495, 0.00675925790681296, 0.0107389365865762,  0.0125308932784936,  0.0107389365865762,  0.00675925790681296, 0.00312460591694495, 0.00106084022060085,
        0.000360167651076071, 0.00106084022060085, 0.00229484704297441, 0.00364599445829020, 0.00425438469465621, 0.00364599445829020, 0.00229484704297441, 0.00106084022060085, 0.000360167651076071
    },
    {
        0.00296901674395050, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.00296901674395050,
        0.0133062098910137,  0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137,
        0.0219382312797146,  0.0983203313488458, 0.162102821637127,  0.0983203313488458, 0.0219382312797146,
        0.0133062098910137,  0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137,
        0.00296901674395050, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.00296901674395050
    },
    {
        0.0276818087794658, 0.111014893010991, 0.0276818087794658,
        0.111014893010991,  0.445213192838173, 0.111014893010991,
        0.0276818087794658, 0.111014893010991, 0.0276818087794658
    }
};

const int vif_filter2d_width[4] = { 17, 9, 5, 3 };

void vif_dec2_s(const float *src, float *dst, int src_w, int src_h, int src_stride, int dst_stride)
{
    int src_px_stride = src_stride / sizeof(float); // src_stride is in bytes
    int dst_px_stride = dst_stride / sizeof(float);

    int i, j;

    // decimation by 2 in each direction (after gaussian blur? check)
    for (i = 0; i < src_h / 2; ++i) {
        for (j = 0; j < src_w / 2; ++j) {
            dst[i * dst_px_stride + j] = src[(i * 2) * src_px_stride + (j * 2)];
        }
    }
}

float vif_sum_s(const float *x, int w, int h, int stride)
{
    int px_stride = stride / sizeof(float);
    int i, j;

    float accum = 0;

    for (i = 0; i < h; ++i) {
        float accum_inner = 0;

        for (j = 0; j < w; ++j) {
            accum_inner += x[i * px_stride + j];
        } // having an inner accumulator help reduce numerical error (no accumulation of near-0 terms)

        accum += accum_inner;
    }

    return accum;
}

void vif_xx_yy_xy_s(const float *x, const float *y, float *xx, float *yy, float *xy, int w, int h, int xstride, int ystride, int xxstride, int yystride, int xystride)
{
    int x_px_stride = xstride / sizeof(float);
    int y_px_stride = ystride / sizeof(float);
    int xx_px_stride = xxstride / sizeof(float);
    int yy_px_stride = yystride / sizeof(float);
    int xy_px_stride = xystride / sizeof(float);

    int i, j;

    // L1 is 32 - 64 KB
    // L2 is 2-4 MB
    // w, h = 1920 x 1080 at floating point is 8 MB
    // not going to fit into L2

    float xval, yval, xxval, yyval, xyval;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            xval = x[i * x_px_stride + j];
            yval = y[i * y_px_stride + j];

            xxval = xval * xval;
            yyval = yval * yval;
            xyval = xval * yval;

            xx[i * xx_px_stride + j] = xxval;
            yy[i * yy_px_stride + j] = yyval;
            xy[i * xy_px_stride + j] = xyval;
        }
    }
}

void vif_statistic_s(const float *mu1_sq, const float *mu2_sq, const float *mu1_mu2, const float *xx_filt, const float *yy_filt, const float *xy_filt, float *num, float *den,
                     int w, int h, int mu1_sq_stride, int mu2_sq_stride, int mu1_mu2_stride, int xx_filt_stride, int yy_filt_stride, int xy_filt_stride, int num_stride, int den_stride)
{
    static const float sigma_nsq = 2;
    static const float sigma_max_inv = 4.0/(255.0*255.0);

    int mu1_sq_px_stride  = mu1_sq_stride / sizeof(float);
    int mu2_sq_px_stride  = mu2_sq_stride / sizeof(float);
    int mu1_mu2_px_stride = mu1_mu2_stride / sizeof(float);
    int xx_filt_px_stride = xx_filt_stride / sizeof(float);
    int yy_filt_px_stride = yy_filt_stride / sizeof(float);
    int xy_filt_px_stride = xy_filt_stride / sizeof(float);
    int num_px_stride = num_stride / sizeof(float);
    int den_px_stride = den_stride / sizeof(float);

    float mu1_sq_val, mu2_sq_val, mu1_mu2_val, xx_filt_val, yy_filt_val, xy_filt_val;
    float sigma1_sq, sigma2_sq, sigma12, g, sv_sq;
    float num_val, den_val;
    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            mu1_sq_val  = mu1_sq[i * mu1_sq_px_stride + j]; // same name as the Matlab code vifp_mscale.m
            mu2_sq_val  = mu2_sq[i * mu2_sq_px_stride + j];
            mu1_mu2_val = mu1_mu2[i * mu1_mu2_px_stride + j];
            xx_filt_val = xx_filt[i * xx_filt_px_stride + j];
            yy_filt_val = yy_filt[i * yy_filt_px_stride + j];
            xy_filt_val = xy_filt[i * xy_filt_px_stride + j];

            sigma1_sq = xx_filt_val - mu1_sq_val;
            sigma2_sq = yy_filt_val - mu2_sq_val;
            sigma12   = xy_filt_val - mu1_mu2_val;

            if (sigma1_sq < sigma_nsq) {
                num_val = 1.0 - sigma2_sq*sigma_max_inv;
                den_val = 1.0;
            }
            else {
                    sv_sq = (sigma2_sq + sigma_nsq) * sigma1_sq;
                                if( sigma12 < 0 )
                                {
                                    num_val = 0.0;
                                }
                                else
                                {
                        g = sv_sq - sigma12 * sigma12;
                    num_val = log2f(sv_sq / g);
                                }
                den_val = log2f(1.0f + sigma1_sq / sigma_nsq);
            }

            num[i * num_px_stride + j] = num_val;
            den[i * den_px_stride + j] = den_val;
        }
    }
}

void vif_filter1d_s(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth)
{

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    /* if support avx */

    if (cpu >= VMAF_CPU_AVX)
    {
        convolution_f32_avx_s(f, fwidth, src, dst, tmpbuf, w, h, src_px_stride, dst_px_stride);
        return;
    }

    /* fall back */

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
                ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

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
                jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

                imgcoeff = tmp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);
}

void vif_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth)
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

                    ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);
                    jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

                    imgcoeff = src[ii * src_px_stride + jj];

                    accum_inner += fcoeff * imgcoeff;
                }

                accum += accum_inner;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}
