/*
Copyright 2001-2012 Xiph.Org and contributors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "log.h"

typedef int32_t od_coeff;

#define OD_DCT_OVERFLOW_CHECK(val, scale, offset, idx)

#define OD_UNBIASED_RSHIFT32(_a, _b) \
    (((int32_t)(((uint32_t)(_a) >> (32 - (_b))) + (_a))) >> (_b))

#define OD_DCT_RSHIFT(_a, _b) OD_UNBIASED_RSHIFT32(_a, _b)

static void od_bin_fdct8(od_coeff y[8], const od_coeff *x, int xstride)
{
    int t0;
    int t1;
    int t1h;
    int t2;
    int t3;
    int t4;
    int t4h;
    int t5;
    int t6;
    int t6h;
    int t7;
    /*Initial permutation:*/
    t0 = *(x + 0 * xstride);
    t4 = *(x + 1 * xstride);
    t2 = *(x + 2 * xstride);
    t6 = *(x + 3 * xstride);
    t7 = *(x + 4 * xstride);
    t3 = *(x + 5 * xstride);
    t5 = *(x + 6 * xstride);
    t1 = *(x + 7 * xstride);
    /*+1/-1 butterflies:*/
    t1 = t0 - t1;
    t1h = OD_DCT_RSHIFT(t1, 1);
    t0 -= t1h;
    t4 += t5;
    t4h = OD_DCT_RSHIFT(t4, 1);
    t5 -= t4h;
    t3 = t2 - t3;
    t2 -= OD_DCT_RSHIFT(t3, 1);
    t6 += t7;
    t6h = OD_DCT_RSHIFT(t6, 1);
    t7 = t6h - t7;
    /*+ Embedded 4-point type-II DCT.*/
    t0 += t6h;
    t6 = t0 - t6;
    t2 = t4h - t2;
    t4 = t2 - t4;
    /*|-+ Embedded 2-point type-II DCT.*/
    /*13573/32768 ~= \sqrt{2} - 1 ~= 0.41421356237309504880168872420970*/
    OD_DCT_OVERFLOW_CHECK(t4, 13573, 16384, 3);
    t0 -= (t4 * 13573 + 16384) >> 15;
    /*11585/16384 ~= \sqrt{\frac{1}{2}} ~= 0.70710678118654752440084436210485*/
    OD_DCT_OVERFLOW_CHECK(t0, 11585, 8192, 4);
    t4 += (t0 * 11585 + 8192) >> 14;
    /*13573/32768 ~= \sqrt{2} - 1 ~= 0.41421356237309504880168872420970*/
    OD_DCT_OVERFLOW_CHECK(t4, 13573, 16384, 5);
    t0 -= (t4 * 13573 + 16384) >> 15;
    /*|-+ Embedded 2-point type-IV DST.*/
    /*21895/32768 ~= \frac{1 - cos(\frac{3\pi}{8})}{\sin(\frac{3\pi}{8})} ~=
       0.66817863791929891999775768652308*/
    OD_DCT_OVERFLOW_CHECK(t2, 21895, 16384, 6);
    t6 -= (t2 * 21895 + 16384) >> 15;
    /*15137/16384~=sin(\frac{3\pi}{8})~=0.92387953251128675612818318939679*/
    OD_DCT_OVERFLOW_CHECK(t6, 15137, 8192, 7);
    t2 += (t6 * 15137 + 8192) >> 14;
    /*21895/32768 ~= \frac{1 - cos(\frac{3\pi}{8})}{\sin(\frac{3\pi}{8})}~=
       0.66817863791929891999775768652308*/
    OD_DCT_OVERFLOW_CHECK(t2, 21895, 16384, 8);
    t6 -= (t2 * 21895 + 16384) >> 15;
    /*+ Embedded 4-point type-IV DST.*/
    /*19195/32768 ~= 2 - \sqrt{2} ~= 0.58578643762690495119831127579030*/
    OD_DCT_OVERFLOW_CHECK(t5, 19195, 16384, 9);
    t3 += (t5 * 19195 + 16384) >> 15;
    /*11585/16384 ~= \sqrt{\frac{1}{2}} ~= 0.70710678118654752440084436210485*/
    OD_DCT_OVERFLOW_CHECK(t3, 11585, 8192, 10);
    t5 += (t3 * 11585 + 8192) >> 14;
    /*7489/8192 ~= \sqrt{2}-\frac{1}{2} ~= 0.91421356237309504880168872420970*/
    OD_DCT_OVERFLOW_CHECK(t5, 7489, 4096, 11);
    t3 -= (t5 * 7489 + 4096) >> 13;
    t7 = OD_DCT_RSHIFT(t5, 1) - t7;
    t5 -= t7;
    t3 = t1h - t3;
    t1 -= t3;
    /*3227/32768 ~= \frac{1 - cos(\frac{\pi}{16})}{sin(\frac{\pi}{16})} ~=
       0.098491403357164253077197521291327*/
    OD_DCT_OVERFLOW_CHECK(t1, 3227, 16384, 12);
    t7 += (t1 * 3227 + 16384) >> 15;
    /*6393/32768 ~= sin(\frac{\pi}{16}) ~= 0.19509032201612826784828486847702*/
    OD_DCT_OVERFLOW_CHECK(t7, 6393, 16384, 13);
    t1 -= (t7 * 6393 + 16384) >> 15;
    /*3227/32768 ~= \frac{1 - cos(\frac{\pi}{16})}{sin(\frac{\pi}{16})} ~=
       0.098491403357164253077197521291327*/
    OD_DCT_OVERFLOW_CHECK(t1, 3227, 16384, 14);
    t7 += (t1 * 3227 + 16384) >> 15;
    /*2485/8192 ~= \frac{1 - cos(\frac{3\pi}{16})}{sin(\frac{3\pi}{16})} ~=
       0.30334668360734239167588394694130*/
    OD_DCT_OVERFLOW_CHECK(t3, 2485, 4096, 15);
    t5 += (t3 * 2485 + 4096) >> 13;
    /*18205/32768 ~= sin(\frac{3\pi}{16}) ~=
     * 0.55557023301960222474283081394853*/
    OD_DCT_OVERFLOW_CHECK(t5, 18205, 16384, 16);
    t3 -= (t5 * 18205 + 16384) >> 15;
    /*2485/8192 ~= \frac{1 - cos(\frac{3\pi}{16})}{sin(\frac{3\pi}{16})} ~=
       0.30334668360734239167588394694130*/
    OD_DCT_OVERFLOW_CHECK(t3, 2485, 4096, 17);
    t5 += (t3 * 2485 + 4096) >> 13;
    y[0] = (od_coeff)t0;
    y[1] = (od_coeff)t1;
    y[2] = (od_coeff)t2;
    y[3] = (od_coeff)t3;
    y[4] = (od_coeff)t4;
    y[5] = (od_coeff)t5;
    y[6] = (od_coeff)t6;
    y[7] = (od_coeff)t7;
}

static void od_bin_fdct8x8(od_coeff *y, int ystride, const od_coeff *x,
                           int xstride)
{
    od_coeff z[8 * 8];
    for (int i = 0; i < 8; i++)
        od_bin_fdct8(z + 8 * i, x + i, xstride);
    for (int i = 0; i < 8; i++)
        od_bin_fdct8(y + ystride * i, z + i, 8);
}

/*
Normalized inverse quantization matrix for 8x8 DCT at the point of
transparency. This is not the JPEG based matrix from the paper,
this one gives a slightly higher MOS agreement.
*/
static float csf_y[8][8] = {
    {1.6193873005, 2.2901594831, 2.08509755623, 1.48366094411, 1.00227514334, 0.678296995242, 0.466224900598, 0.3265091542},
    {2.2901594831, 1.94321815382, 2.04793073064, 1.68731108984, 1.2305666963, 0.868920337363, 0.61280991668, 0.436405793551},
    {2.08509755623, 2.04793073064, 1.34329019223, 1.09205635862, 0.875748795257, 0.670882927016, 0.501731932449, 0.372504254596},
    {1.48366094411, 1.68731108984, 1.09205635862, 0.772819797575, 0.605636379554, 0.48309405692, 0.380429446972, 0.295774038565},
    {1.00227514334, 1.2305666963, 0.875748795257, 0.605636379554, 0.448996256676, 0.352889268808, 0.283006984131, 0.226951348204},
    {0.678296995242, 0.868920337363, 0.670882927016, 0.48309405692, 0.352889268808, 0.27032073436, 0.215017739696, 0.17408067321},
    {0.466224900598, 0.61280991668, 0.501731932449, 0.380429446972, 0.283006984131, 0.215017739696, 0.168869545842, 0.136153931001},
    {0.3265091542, 0.436405793551, 0.372504254596, 0.295774038565, 0.226951348204, 0.17408067321, 0.136153931001, 0.109083846276}
};

static float csf_cb420[8][8] = {
    {1.91113096927, 2.46074210438, 1.18284184739, 1.14982565193, 1.05017074788, 0.898018824055, 0.74725392039, 0.615105596242},
    {2.46074210438, 1.58529308355, 1.21363250036, 1.38190029285, 1.33100189972, 1.17428548929, 0.996404342439, 0.830890433625},
    {1.18284184739, 1.21363250036, 0.978712413627, 1.02624506078, 1.03145147362, 0.960060382087, 0.849823426169, 0.731221236837},
    {1.14982565193, 1.38190029285, 1.02624506078, 0.861317501629, 0.801821139099, 0.751437590932, 0.685398513368, 0.608694761374},
    {1.05017074788, 1.33100189972, 1.03145147362, 0.801821139099, 0.676555426187, 0.605503172737, 0.55002013668, 0.495804539034},
    {0.898018824055, 1.17428548929, 0.960060382087, 0.751437590932, 0.605503172737, 0.514674450957, 0.454353482512, 0.407050308965},
    {0.74725392039, 0.996404342439, 0.849823426169, 0.685398513368, 0.55002013668, 0.454353482512, 0.389234902883, 0.342353999733},
    {0.615105596242, 0.830890433625, 0.731221236837, 0.608694761374, 0.495804539034, 0.407050308965, 0.342353999733, 0.295530605237}
};

static float csf_cr420[8][8] = {
    {2.03871978502, 2.62502345193, 1.26180942886, 1.11019789803, 1.01397751469, 0.867069376285, 0.721500455585, 0.593906509971},
    {2.62502345193, 1.69112867013, 1.17180569821, 1.3342742857, 1.28513006198, 1.13381474809, 0.962064122248, 0.802254508198},
    {1.26180942886, 1.17180569821, 0.944981930573, 0.990876405848, 0.995903384143, 0.926972725286, 0.820534991409, 0.706020324706},
    {1.11019789803, 1.3342742857, 0.990876405848, 0.831632933426, 0.77418706195, 0.725539939514, 0.661776842059, 0.587716619023},
    {1.01397751469, 1.28513006198, 0.995903384143, 0.77418706195, 0.653238524286, 0.584635025748, 0.531064164893, 0.478717061273},
    {0.867069376285, 1.13381474809, 0.926972725286, 0.725539939514, 0.584635025748, 0.496936637883, 0.438694579826, 0.393021669543},
    {0.721500455585, 0.962064122248, 0.820534991409, 0.661776842059, 0.531064164893, 0.438694579826, 0.375820256136, 0.330555063063},
    {0.593906509971, 0.802254508198, 0.706020324706, 0.587716619023, 0.478717061273, 0.393021669543, 0.330555063063, 0.285345396658}
};

static double calc_psnrhvs(const unsigned char *_src, int _systride,
                           const unsigned char *_dst, int _dystride,
                           double _par, int depth, int _w, int _h, int _step,
                           float _csf[8][8])
{
    float ret;
    od_coeff dct_s[8 * 8];
    od_coeff dct_d[8 * 8];
    float mask[8][8];
    int pixels;
    int x;
    int y;
    int32_t samplemax;
    (void)_par;
    ret = pixels = 0;
    /*
     In the PSNR-HVS-M paper[1] the authors describe the construction of
     their masking table as "we have used the quantization table for the
     color component Y of JPEG [6] that has been also obtained on the
     basis of CSF. Note that the values in quantization table JPEG have
     been normalized and then squared." Their CSF matrix (from PSNR-HVS)
     was also constructed from the JPEG matrices. I can not find any obvious
     scheme of normalizing to produce their table, but if I multiply their
     CSF by 0.38857 and square the result I get their masking table.
     I have no idea where this constant comes from, but deviating from it
     too greatly hurts MOS agreement.

     [1] Nikolay Ponomarenko, Flavia Silvestri, Karen Egiazarian, Marco Carli,
     Jaakko Astola, Vladimir Lukin, "On between-coefficient contrast masking
     of DCT basis functions", CD-ROM Proceedings of the Third International
     Workshop on Video Processing and Quality Metrics for Consumer
     Electronics VPQM-07, Scottsdale, Arizona, USA, 25-26 January, 2007, 4p.
    */
    for (x = 0; x < 8; x++)
        for (y = 0; y < 8; y++)
            mask[x][y] = (_csf[x][y] * 0.3885746225901003) *
                         (_csf[x][y] * 0.3885746225901003);

    for (y = 0; y < _h - 7; y += _step) {
        for (x = 0; x < _w - 7; x += _step) {
            int i;
            int j;
            float s_means[4];
            float d_means[4];
            float s_vars[4];
            float d_vars[4];
            float s_gmean = 0;
            float d_gmean = 0;
            float s_gvar = 0;
            float d_gvar = 0;
            float s_mask = 0;
            float d_mask = 0;
            for (i = 0; i < 4; i++)
                s_means[i] = d_means[i] = s_vars[i] = d_vars[i] = 0;
            for (i = 0; i < 8; i++) {
                for (j = 0; j < 8; j++) {
                    int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    if (depth > 8) {
                        dct_s[i * 8 + j] =
                            _src[(y + i) * _systride + (j + x) * 2] +
                            (_src[(y + i) * _systride + (j + x) * 2 + 1] << 8);
                        dct_d[i * 8 + j] =
                            _dst[(y + i) * _dystride + (j + x) * 2] +
                            (_dst[(y + i) * _dystride + (j + x) * 2 + 1] << 8);
                    } else {
                        dct_s[i * 8 + j] = _src[(y + i) * _systride + (j + x)];
                        dct_d[i * 8 + j] = _dst[(y + i) * _dystride + (j + x)];
                    }
                    s_gmean += dct_s[i * 8 + j];
                    d_gmean += dct_d[i * 8 + j];
                    s_means[sub] += dct_s[i * 8 + j];
                    d_means[sub] += dct_d[i * 8 + j];
                }
            }
            s_gmean /= 64.f;
            d_gmean /= 64.f;
            for (i = 0; i < 4; i++)
                s_means[i] /= 16.f;
            for (i = 0; i < 4; i++)
                d_means[i] /= 16.f;
            for (i = 0; i < 8; i++) {
                for (j = 0; j < 8; j++) {
                    int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    s_gvar += (dct_s[i * 8 + j] - s_gmean) *
                              (dct_s[i * 8 + j] - s_gmean);
                    d_gvar += (dct_d[i * 8 + j] - d_gmean) *
                              (dct_d[i * 8 + j] - d_gmean);
                    s_vars[sub] += (dct_s[i * 8 + j] - s_means[sub]) *
                                   (dct_s[i * 8 + j] - s_means[sub]);
                    d_vars[sub] += (dct_d[i * 8 + j] - d_means[sub]) *
                                   (dct_d[i * 8 + j] - d_means[sub]);
                }
            }
            s_gvar *= 1 / 63.f * 64;
            d_gvar *= 1 / 63.f * 64;
            for (i = 0; i < 4; i++)
                s_vars[i] *= 1 / 15.f * 16;
            for (i = 0; i < 4; i++)
                d_vars[i] *= 1 / 15.f * 16;
            if (s_gvar > 0)
                s_gvar =
                    (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
            if (d_gvar > 0)
                d_gvar =
                    (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;
            od_bin_fdct8x8(dct_s, 8, dct_s, 8);
            od_bin_fdct8x8(dct_d, 8, dct_d, 8);
            for (i = 0; i < 8; i++)
                for (j = (i == 0); j < 8; j++)
                    s_mask += dct_s[i * 8 + j] * dct_s[i * 8 + j] * mask[i][j];
            for (i = 0; i < 8; i++)
                for (j = (i == 0); j < 8; j++)
                    d_mask += dct_d[i * 8 + j] * dct_d[i * 8 + j] * mask[i][j];
            s_mask = sqrt(s_mask * s_gvar) / 32.f;
            d_mask = sqrt(d_mask * d_gvar) / 32.f;
            if (d_mask > s_mask)
                s_mask = d_mask;
            for (i = 0; i < 8; i++) {
                for (j = 0; j < 8; j++) {
                    float err;
                    err = abs(dct_s[i * 8 + j] - dct_d[i * 8 + j]);
                    if (i != 0 || j != 0)
                        err = err < s_mask / mask[i][j]
                                  ? 0
                                  : err - s_mask / mask[i][j];
                    ret += (err * _csf[i][j]) * (err * _csf[i][j]);
                    pixels++;
                }
            }
        }
    }
    ret /= pixels;
    samplemax = (1 << depth) - 1;
    ret /= samplemax * samplemax;
    return ret;
}

static double convert_score_db(double _score, double _weight)
{
    return 10 * (-1 * log10(_weight * _score));
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) fex;
    (void) bpc;
    (void) w;
    (void) h;

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "%s: invalid bitdepth (%d), "
                 "bpc must be less than or equal to 12\n",
                 fex->name, bpc);
        return -EINVAL;
    }

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;
    else
        return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                   VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                   VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    double score[3];
    for (unsigned i = 0; i < 3; i++) {
        score[i] =
            calc_psnrhvs(ref_pic->data[i], ref_pic->stride[i],
                         dist_pic->data[i], dist_pic->stride[i], 1.0,
                         ref_pic->bpc, ref_pic->w[i], ref_pic->h[i], 7,
                         i == 0 ? csf_y : i == 1 ? csf_cb420 : csf_cr420);

        err |= vmaf_feature_collector_append(feature_collector,
                                             fex->provided_features[i],
                                             convert_score_db(score[i], 1.0),
                                             index);
    }

    const double psnr_hvs = (score[0]) * .8 + .1 * (score[1] + score[2]);
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs",
                                         convert_score_db(psnr_hvs, 1.0),
                                         index);
    return err;
}

static const char *provided_features[] = {
    "psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs",
     NULL
};

VmafFeatureExtractor vmaf_fex_psnr_hvs = {
    .name = "psnr_hvs",
    .init = init,
    .extract = extract,
    .provided_features = provided_features,
};
