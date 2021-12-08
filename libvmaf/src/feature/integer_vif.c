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

#include <errno.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "cpu.h"
#include "dict.h"
#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "picture.h"
#include "integer_vif.h"

#if ARCH_X86
#include "x86/vif_avx2.h"
#if HAVE_AVX512
#include "x86/vif_avx512.h"
#endif
#endif

typedef struct VifState {
    VifBuffer buf;
    uint16_t log2_table[65537];
    bool debug;
    double vif_enhn_gain_limit;
    void (*filter1d_8)(VifBuffer buf, unsigned w, unsigned h);
    void (*filter1d_16)(VifBuffer buf, unsigned w, unsigned h, int scale,
                        int bpc);
    void (*filter1d_rd_8)(VifBuffer buf, unsigned w, unsigned h);
    void (*filter1d_rd_16)(VifBuffer buf, unsigned w, unsigned h, int scale,
                           int bpc);
    VmafDictionary *feature_name_dict;
} VifState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

static FORCE_INLINE inline void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    void *ref = buf.ref; void *dis = buf.dis;
    for (unsigned i = 1; i <= fwidth_half; ++i) {
        size_t offset = buf.stride * i;
        memcpy(ref - offset, ref + offset, buf.stride);
        memcpy(dis - offset, dis + offset, buf.stride);
        memcpy(ref + buf.stride * (h - 1) + buf.stride * i,
               ref + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
               dis + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
    }
}

static FORCE_INLINE inline void
decimate_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[(i * 2) * mu_stride + (j * 2)];
            dis[i * stride + j] = buf.mu2[(i * 2) * mu_stride + (j * 2)];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}

static FORCE_INLINE inline uint16_t
get_best16_from32(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);
    k = 16 - k;
    temp = temp >> k;
    *x = -k;
    return temp;
}

static FORCE_INLINE inline uint16_t
get_best16_from64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);
    if (k > 48) {
        k -= 48;
        temp = temp << k;
        *x = k;
    } else if (k < 47) {
        k = 48 - k;
        temp = temp >> k;
        *x = -k;
    } else {
        *x = 0;
        if (temp >> 16) {
            temp = temp >> 1;
            *x = -1;
        }
    }
    return (uint16_t)temp;
}

static void filter1d_8(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[0];
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];

    for (unsigned i = 0; i < h; ++i) {
        //VERTICAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            uint32_t accum_ref_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s0[fi];
                const uint8_t *ref = (uint8_t*)buf.ref;
                const uint8_t *dis = (uint8_t*)buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * buf.stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * buf.stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint32_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint32_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint32_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] = (accum_mu1 + 128) >> 8;
            buf.tmp.mu2[j] = (accum_mu2 + 128) >> 8;
            buf.tmp.ref[j] = accum_ref;
            buf.tmp.dis[j] = accum_dis;
            buf.tmp.ref_dis[j] = accum_ref_dis;
        }

        PADDING_SQ_DATA(buf, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt_s0[fj];
                accum_mu1 += fcoeff * ((uint32_t)buf.tmp.mu1[jj_check]);
                accum_mu2 += fcoeff * ((uint32_t)buf.tmp.mu2[jj_check]);
                accum_ref += fcoeff * ((uint64_t)buf.tmp.ref[jj_check]);
                accum_dis += fcoeff * ((uint64_t)buf.tmp.dis[jj_check]);
                accum_ref_dis += fcoeff * ((uint64_t)buf.tmp.ref_dis[jj_check]);
            }
            const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
            buf.mu1_32[i * dst_stride + j] = accum_mu1;
            buf.mu2_32[i * dst_stride + j] = accum_mu2;
            buf.ref_sq[i * dst_stride + j] = (uint32_t)((accum_ref + 32768) >> 16);
            buf.dis_sq[i * dst_stride + j] = (uint32_t)((accum_dis + 32768) >> 16);
            buf.ref_dis[i * dst_stride + j] = (uint32_t)((accum_ref_dis + 32768) >> 16);
        }
    }
}

static void filter1d_16(VifBuffer buf, unsigned w, unsigned h, int scale,
                        int bpc)
{
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];

    int32_t add_shift_round_HP, shift_HP;
    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    if (scale == 0) {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    } else {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }

    for (unsigned i = 0; i < h; ++i) {
        //VERTICAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt[fi];
                const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
                uint16_t *ref = buf.ref;
                uint16_t *dis = buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] = (uint16_t)((accum_mu1 + add_shift_round_VP) >> shift_VP);
            buf.tmp.mu2[j] = (uint16_t)((accum_mu2 + add_shift_round_VP) >> shift_VP);
            buf.tmp.ref[j] = (uint32_t)((accum_ref + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.dis[j] = (uint32_t)((accum_dis + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.ref_dis[j] = (uint32_t)((accum_ref_dis + add_shift_round_VP_sq) >> shift_VP_sq);
        }

        PADDING_SQ_DATA(buf, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt[fj];
                accum_mu1 += fcoeff * ((uint32_t)buf.tmp.mu1[jj_check]);
                accum_mu2 += fcoeff * ((uint32_t)buf.tmp.mu2[jj_check]);
                accum_ref += fcoeff * ((uint64_t)buf.tmp.ref[jj_check]);
                accum_dis += fcoeff * ((uint64_t)buf.tmp.dis[jj_check]);
                accum_ref_dis += fcoeff * ((uint64_t)buf.tmp.ref_dis[jj_check]);
            }
            const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
            buf.mu1_32[i * dst_stride + j] = accum_mu1;
            buf.mu2_32[i * dst_stride + j] = accum_mu2;
            buf.ref_sq[i * dst_stride + j] = (uint32_t)((accum_ref + add_shift_round_HP) >> shift_HP);
            buf.dis_sq[i * dst_stride + j] = (uint32_t)((accum_dis + add_shift_round_HP) >> shift_HP);
            buf.ref_dis[i * dst_stride + j] = (uint32_t)((accum_ref_dis + add_shift_round_HP) >> shift_HP);
        }
    }
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static void vif_statistic(VifBuffer buf, float *num, float *den,
                          unsigned w, unsigned h, uint16_t *log2_table,
                          double vif_enhn_gain_limit)
{
    uint32_t *xx_filt = buf.ref_sq;
    uint32_t *yy_filt = buf.dis_sq;
    uint32_t *xy_filt = buf.ref_dis;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t num_val, den_val;
    int64_t accum_x = 0, accum_x2 = 0;
    int64_t num_accum_x = 0;
    int64_t accum_num_log = 0.0;
    int64_t accum_den_log = 0.0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    /**
        * In floating-point there are two types of numerator scores and denominator scores
        * 1. num = 1 - sigma1_sq * constant den =1  when sigma1_sq<2  here constant=4/(255*255)
        * 2. num = log2(((sigma2_sq+2)*sigma1_sq)/((sigma2_sq+2)*sigma1_sq-sigma12*sigma12) den=log2(1+(sigma1_sq/2)) else
        *
        * In fixed-point separate accumulator is used for non-log score accumulations and log-based score accumulation
        * For non-log accumulator of numerator, only sigma1_sq * constant in fixed-point is accumulated
        * log based values are separately accumulated.
        * While adding both accumulator values the non-log accumulator is converted such that it is equivalent to 1 - sigma1_sq * constant(1's are accumulated with non-log denominator accumulator)
    */
    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < w; ++j) {
            const ptrdiff_t stride = buf.stride_32 / sizeof(uint32_t);
            uint32_t mu1_val = buf.mu1_32[i * stride + j];
            uint32_t mu2_val = buf.mu2_32[i * stride + j];
            uint32_t mu1_sq_val = (uint32_t)((((uint64_t)mu1_val * mu1_val)
                                    + 2147483648) >> 32);
            uint32_t mu2_sq_val = (uint32_t)((((uint64_t)mu2_val * mu2_val)
                                    + 2147483648) >> 32);
            uint32_t mu1_mu2_val = (uint32_t)((((uint64_t)mu1_val * mu2_val)
                                    + 2147483648) >> 32);

            uint32_t xx_filt_val = xx_filt[i * stride + j];
            uint32_t yy_filt_val = yy_filt[i * stride + j];
            uint32_t xy_filt_val = xy_filt[i * stride + j];

            int32_t sigma1_sq = (int32_t)(xx_filt_val - mu1_sq_val);
            int32_t sigma2_sq = (int32_t)(yy_filt_val - mu2_sq_val);
            int32_t sigma12 = (int32_t)(xy_filt_val - mu1_mu2_val);

            sigma1_sq = MAX(sigma1_sq, 0.0);
            sigma2_sq = MAX(sigma2_sq, 0.0);

            //eps is zero, an int will not be less then 1.0e-10, it can be changed to one
            const double eps = 65536 * 1.0e-10;
            double g = sigma12 / (sigma1_sq + eps);
            int32_t sv_sq = sigma2_sq - g * sigma12;

			sv_sq = sigma2_sq - g * sigma12;


			if (sigma1_sq < eps) {
			    g = 0.0;
                sv_sq = sigma2_sq;
                sigma1_sq = 0.0;
			}

			if (sigma2_sq < eps) {
			    g = 0.0;
			    sv_sq = 0.0;
			}

			if (g < 0.0) {
			    sv_sq = sigma2_sq;
			    g = 0.0;
			}

			sv_sq = (uint32_t)(MAX(sv_sq, eps));

            g = MIN(g, vif_enhn_gain_limit);

            if (sigma1_sq >= sigma_nsq) {
                uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                int x;
                uint16_t log_den1 = get_best16_from32(log_den_stage1, &x);

                /**
                * log values are taken from the look-up table generated by
                * log_generate() function which is called in integer_combo_threadfunc
                * den_val in float is log2(1 + sigma1_sq/2)
                * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                * x because best 16 bits are taken
                */
                num_accum_x++;
                accum_x += x;
                den_val = log2_table[log_den1];

                if (sigma12 >= 0) {
                    // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                    /**
                    * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                    *
                    * In Fixed-point the above is converted to
                    * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                    */
                    int x1, x2;
                    uint32_t numer1 = (sv_sq + sigma_nsq);
                    int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                    uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                    if (numer1 > 0) {
                        uint16_t denlog = get_best16_from64((uint64_t)numer1, &x2);
                        accum_x2 += (x2 - x1);
                        num_val = log2_table[numlog] - log2_table[denlog];
                        accum_num_log += num_val;
                        accum_den_log += den_val;
                    } else {
                        den_val = 1;
                        accum_num_non_log += sigma2_sq;
                        accum_den_non_log += den_val;
                    }

                }
                else {
                    num_val = 0;
                    accum_num_log += num_val;
                    accum_den_log += den_val;
                }
            }
            else {
                den_val = 1;
                accum_num_non_log += sigma2_sq;
                accum_den_non_log += den_val;
            }
        }
    }
    //log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    //num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log / 65536.0) / (255.0*255.0));
    //den[0] = accum_den_log / 2048.0 + accum_den_non_log;

    //changed calculation to increase performance
    num[0] = accum_num_log / 2048.0  + accum_x2 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0  - (accum_x + (num_accum_x * 17)) + accum_den_non_log;
}

static void filter1d_rd_8(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];

    for (unsigned i = 0; i < h; ++i) {
        //VERTICAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s1[fi];
                const uint8_t *ref = (uint8_t*)buf.ref;
                const uint8_t *dis = (uint8_t*)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis[ii_check * buf.stride + j];
            }
            buf.tmp.ref_convol[j] = (accum_ref + 128) >> 8;
            buf.tmp.dis_convol[j] = (accum_dis + 128) >> 8;
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
            buf.mu1[i * stride + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
}

static void filter1d_rd_16(VifBuffer buf, unsigned w, unsigned h, int scale,
                           int bpc)
{
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;

    if (scale == 0) {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    } else {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    for (unsigned i = 0; i < h; ++i) {
        //VERTICAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt[fi];
                const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
                uint16_t *ref = buf.ref;
                uint16_t *dis = buf.dis;
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride + j]);
            }
            buf.tmp.ref_convol[j] = (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] = (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt[fj];
                accum_ref += fcoeff * ((uint32_t)buf.tmp.ref_convol[jj_check]);
                accum_dis += fcoeff * ((uint32_t)buf.tmp.dis_convol[jj_check]);
            }
            const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
            buf.mu1[i * stride + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
}

static inline void log_generate(uint16_t *log2_table)
{
    for (unsigned i = 32767; i < 65536; ++i) {
        log2_table[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    VifState *s = fex->priv;

    s->filter1d_8 = filter1d_8;
    s->filter1d_16 = filter1d_16;
    s->filter1d_rd_8 = filter1d_rd_8;
    s->filter1d_rd_16 = filter1d_rd_16;

#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        s->filter1d_8 = vif_filter1d_8_avx2;
        s->filter1d_16 = vif_filter1d_16_avx2;
        s->filter1d_rd_8 = vif_filter1d_rd_8_avx2;
        s->filter1d_rd_16 = vif_filter1d_rd_16_avx2;
    }
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        s->filter1d_8 = vif_filter1d_8_avx512;
        s->filter1d_16 = vif_filter1d_16_avx512;
        s->filter1d_rd_8 = vif_filter1d_rd_8_avx512;
        s->filter1d_rd_16 = vif_filter1d_rd_16_avx512;
    }
#endif
#endif

    log_generate(s->log2_table);

    (void) pix_fmt;
    const bool hbd = bpc > 8;

    s->buf.stride = ALIGN_CEIL(w << hbd);
    s->buf.stride_16 = ALIGN_CEIL(w * sizeof(uint16_t));
    s->buf.stride_32 = ALIGN_CEIL(w * sizeof(uint32_t));
    s->buf.stride_tmp =
        ALIGN_CEIL((MAX_ALIGN + w + MAX_ALIGN) * sizeof(uint32_t));
    const size_t frame_size = s->buf.stride * h;
    const size_t pad_size = s->buf.stride * 8;
    const size_t data_sz =
        2 * (pad_size + frame_size + pad_size) + 2 * (h * s->buf.stride_16) +
        5 * (h * s->buf.stride_32) + 7 * s->buf.stride_tmp;
    void *data = aligned_malloc(data_sz, MAX_ALIGN);
    if (!data) goto fail;

    s->buf.data = data; data += pad_size;
    s->buf.ref = data; data += frame_size + pad_size + pad_size;
    s->buf.dis = data; data += frame_size + pad_size;
    s->buf.mu1 = data; data += h * s->buf.stride_16;
    s->buf.mu2 = data; data += h * s->buf.stride_16;
    s->buf.mu1_32 = data; data += h * s->buf.stride_32;
    s->buf.mu2_32 = data; data += h * s->buf.stride_32;
    s->buf.ref_sq = data; data += h * s->buf.stride_32;
    s->buf.dis_sq = data; data += h * s->buf.stride_32;
    s->buf.ref_dis = data; data += h * s->buf.stride_32;
    s->buf.tmp.mu1 = data; data += s->buf.stride_tmp;
    s->buf.tmp.mu2 = data; data += s->buf.stride_tmp;
    s->buf.tmp.ref = data; data += s->buf.stride_tmp;
    s->buf.tmp.dis = data; data += s->buf.stride_tmp;
    s->buf.tmp.ref_dis = data; data += s->buf.stride_tmp;
    s->buf.tmp.ref_convol = data; data += s->buf.stride_tmp;
    s->buf.tmp.dis_convol = data;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (data) aligned_free(data);
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

typedef struct VifScore {
    struct {
        float num;
        float den;
    } scale[4];
} VifScore;

static int write_scores(VmafFeatureCollector *feature_collector, unsigned index,
                        VifScore vif, VifState *s)
{
    int err = 0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale0_score",
            vif.scale[0].num / vif.scale[0].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale1_score",
            vif.scale[1].num / vif.scale[1].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale2_score",
            vif.scale[2].num / vif.scale[2].den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_vif_scale3_score",
            vif.scale[3].num / vif.scale[3].den, index);

    if (!s->debug) return err;

    const double score_num =
        (double)vif.scale[0].num + (double)vif.scale[1].num +
        (double)vif.scale[2].num + (double)vif.scale[3].num;

    const double score_den =
        (double)vif.scale[0].den + (double)vif.scale[1].den +
        (double)vif.scale[2].den + (double)vif.scale[3].den;

    const double score =
        score_den == 0.0 ? 1.0f : score_num / score_den;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale0", vif.scale[0].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale0", vif.scale[0].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale1", vif.scale[1].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale1", vif.scale[1].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale2", vif.scale[2].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale2", vif.scale[2].den,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_num_scale3", vif.scale[3].num,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_vif_den_scale3", vif.scale[3].den,
            index);

    return err;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifState *s = fex->priv;

    (void) ref_pic_90;
    (void) dist_pic_90;

    unsigned w = ref_pic->w[0];
    unsigned h = dist_pic->h[0];

    void *ref_in = ref_pic->data[0];
    void *dis_in = dist_pic->data[0];
    void *ref_out = s->buf.ref;
    void *dis_out = s->buf.dis;

    for (unsigned i = 0; i < h; i++) {
        memcpy(ref_out, ref_in, ref_pic->stride[0]);
        memcpy(dis_out, dis_in, dist_pic->stride[0]);
        ref_in += ref_pic->stride[0];
        dis_in += dist_pic->stride[0];
        ref_out += s->buf.stride;
        dis_out += s->buf.stride;
    }
    pad_top_and_bottom(s->buf, h, vif_filter1d_width[0]);

    VifScore vif_score;
    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale > 0) {
            if (ref_pic->bpc == 8 && scale == 1)
                s->filter1d_rd_8(s->buf, w, h);
            else
                s->filter1d_rd_16(s->buf, w, h, scale - 1, ref_pic->bpc);

            decimate_and_pad(s->buf, w, h, scale);
            w /= 2; h /= 2;
        }

        if (ref_pic->bpc == 8 && scale == 0)
            s->filter1d_8(s->buf, w, h);
        else
            s->filter1d_16(s->buf, w, h, scale, ref_pic->bpc);

        vif_statistic(s->buf, &vif_score.scale[scale].num,
                      &vif_score.scale[scale].den, w, h, s->log2_table,
                      s->vif_enhn_gain_limit);
    }

    return write_scores(feature_collector, index, vif_score, s);
}

static int close(VmafFeatureExtractor *fex)
{
    VifState *s = fex->priv;
    if (s->buf.data) aligned_free(s->buf.data);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale3_score",
    "integer_vif", "integer_vif_num", "integer_vif_den", "integer_vif_num_scale0",
    "integer_vif_den_scale0", "integer_vif_num_scale1", "integer_vif_den_scale1",
    "integer_vif_num_scale2", "integer_vif_den_scale2", "integer_vif_num_scale3",
    "integer_vif_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif = {
    .name = "vif",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(VifState),
    .provided_features = provided_features,
};
