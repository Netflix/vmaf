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

#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"

#include "picture.h"
#include "vif_buffer.h"

typedef struct VifState {
    VifBuffer buf;
    uint16_t log2_table[65537];
} VifState;

static const uint16_t vif_filter1d_table[4][17] = {
    { 489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935, 489 },
    { 1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244 },
    { 3571, 16004, 26386, 16004, 3571 },
    { 10904, 43728, 10904 }
};

static const int vif_filter1d_width[4] = { 17, 9, 5, 3 };

/**
 * Padding on top is mirrored accrossed first row
 * Padding on bottom is mirrored accrossed imaginary mirror after bottom end row
 */
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
               ref + buf.stride * (h - 1) - buf.stride * (i - 1),
               buf.stride); 
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
               dis + buf.stride * (h - 1) - buf.stride * (i - 1),
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

/**
 * Padding on left is mirrored accrossed first column
 * Padding on right is mirrored accrossed imaginary mirror after right end column
 */
static FORCE_INLINE inline void
PADDING_SQ_DATA(VifBuffer buf, int w, unsigned fwidth_half)
{
    for (unsigned f = 1; f <= fwidth_half; ++f) {
        int left_point = -f;
        int right_point = f;
        buf.tmp.mu1[left_point] = buf.tmp.mu1[right_point];
        buf.tmp.mu2[left_point] = buf.tmp.mu2[right_point];
        buf.tmp.ref[left_point] = buf.tmp.ref[right_point];
        buf.tmp.dis[left_point] = buf.tmp.dis[right_point];
        buf.tmp.ref_dis[left_point] = buf.tmp.ref_dis[right_point];
        
        left_point = w - f;
        right_point = w - 1 + f;
        buf.tmp.mu1[right_point] = buf.tmp.mu1[left_point];
        buf.tmp.mu2[right_point] = buf.tmp.mu2[left_point];
        buf.tmp.ref[right_point] = buf.tmp.ref[left_point];
        buf.tmp.dis[right_point] = buf.tmp.dis[left_point];
        buf.tmp.ref_dis[right_point] = buf.tmp.ref_dis[left_point];
    }
}

static FORCE_INLINE inline void
PADDING_SQ_DATA_2(VifBuffer buf, int w, unsigned fwidth_half)
{
    for (unsigned f = 1; f <= fwidth_half; ++f) {
        int left_point = -f;
        int right_point = f;
        buf.tmp.ref_convol[left_point] = buf.tmp.ref_convol[right_point];
        buf.tmp.dis_convol[left_point] = buf.tmp.dis_convol[right_point];
        
        left_point = w - f;
        right_point = w - 1 + f;
        buf.tmp.ref_convol[right_point] = buf.tmp.ref_convol[left_point];
        buf.tmp.dis_convol[right_point] = buf.tmp.dis_convol[left_point];
    }
}

static void vif_filter1d_8(VifBuffer buf, unsigned w, unsigned h)
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

static void vif_filter1d_16(VifBuffer buf, unsigned w, unsigned h, int scale,
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

static void vif_statistic(VifBuffer buf, float *num, float *den,
                          unsigned w, unsigned h, uint16_t *log2_table)
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

            int32_t sigma1_sq = (int32_t)(xx_filt_val - mu1_sq_val);
            int32_t sigma2_sq = (int32_t)(yy_filt_val - mu2_sq_val);

            if (sigma1_sq >= sigma_nsq) {
                uint32_t xy_filt_val = xy_filt[i * stride + j];
                int32_t sigma12 = (int32_t)(xy_filt_val - mu1_mu2_val);

                uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                int x;

                //Best 16 bits are taken to calculate the log functions
                //return value is between 16384 to 65535
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
                    /**
                    * In floating-point numerator = log2(num_log_num/num_log_den)
                    * and num_log_num = (sigma2_sq+sigma_nsq)*sigma1_sq
                    * num_log_den = num_log_num - sigma12*sigma12
                    *
                    * In Fixed-point the above is converted to
                    * numerator = log2((sigma2_sq+sigma_nsq)/((sigma2_sq+sigma_nsq)-(sigma12*sigma12/sigma1_sq)))
                    * i.e numerator = log2(sigma2_sq + sigma_nsq)*sigma1_sq - log2((sigma2_sq+sigma_nsq)sigma1_sq-(sigma12*sigma12))
                    */
                    int x1, x2;
                    int32_t numer1 = (sigma2_sq + sigma_nsq);
                    int64_t sigma12_sq = (int64_t)sigma12*sigma12;
                    int64_t numer1_tmp = (int64_t)numer1*sigma1_sq; //numerator 
                    uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                    int64_t denom = numer1_tmp - sigma12_sq;
                    if (denom > 0) {
                        uint16_t denlog = get_best16_from64((uint64_t)denom, &x2);
                        accum_x2 += (x2 - x1);
                        num_val = log2_table[numlog] - log2_table[denlog];
                        accum_num_log += num_val;
                        accum_den_log += den_val;
                    } else {
                        den_val = 1;
                        accum_num_non_log += sigma2_sq;
                        accum_den_non_log += den_val;
                    }
                } else {
                    num_val = 0;
                    accum_num_log += num_val;
                    accum_den_log += den_val;
                }

            } else {
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
    num[0] = accum_num_log / 2048.0 + accum_x2 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 - (accum_x + (num_accum_x * 17)) + accum_den_non_log;
}

static void vif_filter1d_rd_8(VifBuffer buf, unsigned w, unsigned h)
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

static void vif_filter1d_rd_16(VifBuffer buf, unsigned w, unsigned h, int scale,
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

static const float log2_poly[9] = { -0.012671635276421, 0.064841182402670,
                                    -0.157048836463065, 0.257167726303123,
                                    -0.353800560300520, 0.480131410397451,
                                    -0.721314327952201, 1.442694803896991, 0 };

static const int log2_poly_width = sizeof(log2_poly) / sizeof(float);

static inline float horner(float x)
{
    float var = 0;
    for (unsigned i = 0; i < log2_poly_width; ++i)
        var = var * x + log2_poly[i];
    return var;
}

static inline float log2f_approx(float x)
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
    exponent = (u32 & exp_expo_mask) >> 23;
    mant = (u32 & exp_mant_mask) >> 0;
    u32remain = mant | exp_zero_const;

    memcpy(&remain, &u32remain, sizeof(float));

    log_base = (int32_t)exponent - 127;
    log_remain = horner(remain - 1.0f);

    return log_base + log_remain;
}

static inline void log_generate(uint16_t *log2_table)
{
    for (unsigned i = 32767; i < 65536; ++i) {
        log2_table[i] = (uint16_t)round(log2f_approx((float)i) * 2048);
    }
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    VifState *s = fex->priv;
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

    return 0;

fail:
    aligned_free(data);
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dis_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifState *s = fex->priv;
    unsigned w = ref_pic->w[0];
    unsigned h = dis_pic->h[0];

    void *ref_in = ref_pic->data[0];
    void *dis_in = dis_pic->data[0];
    void *ref_out = s->buf.ref;
    void *dis_out = s->buf.dis;

    for (unsigned i = 0; i < h; i++) {
        memcpy(ref_out, ref_in, ref_pic->stride[0]);
        memcpy(dis_out, dis_in, dis_pic->stride[0]);
        ref_in += ref_pic->stride[0];
        dis_in += dis_pic->stride[0];
        ref_out += s->buf.stride;
        dis_out += s->buf.stride;
    }
    pad_top_and_bottom(s->buf, h, vif_filter1d_width[0]);

    double scores[8];
    double score_num = 0.0;
    double score_den = 0.0;

    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale > 0) {
            if (ref_pic->bpc == 8 && scale == 1)
                vif_filter1d_rd_8(s->buf, w, h);
            else
                vif_filter1d_rd_16(s->buf, w, h, scale - 1, ref_pic->bpc);

            decimate_and_pad(s->buf, w, h, scale);
            w /= 2; h /= 2;
        }

        if (ref_pic->bpc == 8 && scale == 0)
            vif_filter1d_8(s->buf, w, h);
        else
            vif_filter1d_16(s->buf, w, h, scale, ref_pic->bpc);

        float num, den;
        vif_statistic(s->buf, &num, &den, w, h, s->log2_table);
        scores[2 * scale] = num;
        scores[2 * scale + 1] = den;
        score_num += scores[2 * scale];
        score_den += scores[2 * scale + 1];
    }

    //double score = score_den = 0.0 ? 1.0 : score_num / score_den;

    int err = 0;
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale0_integer_score'",
                                         scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale1_integer_score'",
                                         scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale2_integer_score'",
                                         scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale3_integer_score'",
                                         scores[6] / scores[7], index);
    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    VifState *s = fex->priv;
    if (s->buf.data) aligned_free(s->buf.data);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_vif_scale0_integer_score'",
    "'VMAF_feature_vif_scale1_integer_score'",
    "'VMAF_feature_vif_scale2_integer_score'",
    "'VMAF_feature_vif_scale3_integer_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif = {
    .name = "vif",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(VifState),
    .provided_features = provided_features,
};
