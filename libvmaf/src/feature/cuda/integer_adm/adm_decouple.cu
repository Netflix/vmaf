/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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

#include "feature_collector.h"
#include "cuda/integer_adm_cuda.h"

#include "common.h"

#include "cuda_helper.cuh"

/*
 * This is a 1:1 port of adm_decouple() / adm_decouple_s123() of the reference
 * implementation. Only the division look up table is realized as an integer
 * division instead of a table, both are equivalent for the entire range of
 * int16 (scale 0) and int32 (scales 1..3) inputs:
 *
 *     div_lookup[32768 + i] =   (int32_t)(div_Q_factor / i);
 *     div_lookup[32768 - i] = - (int32_t)(div_Q_factor / i);
 */

#define ADM_DIV_Q_FACTOR 1073741824 /* 2^30 */

__device__ __forceinline__ int32_t adm_div_lookup(int32_t den)
{
    const int32_t recip = ADM_DIV_Q_FACTOR / (den < 0 ? -den : den);
    return den < 0 ? -recip : recip;
}

__device__ __forceinline__ uint16_t get_best15_from32(uint32_t temp, int *x)
{
    int k = __clz(temp); // built in for cuda
    k = 17 - k;
    temp = (temp + (1 << (k - 1))) >> k;
    *x = k;
    return temp;
}

extern "C" {

__global__ void adm_decouple_kernel(AdmBufferCuda buf, int top, int bottom,
        int left, int right, int stride,
        double adm_enhn_gain_limit, float cos_1deg_sq)
{
    const cuda_adm_dwt_band_t *ref = &buf.ref_dwt2;
    const cuda_adm_dwt_band_t *dis = &buf.dis_dwt2;
    const cuda_adm_dwt_band_t *r = &buf.decouple_r;
    const cuda_adm_dwt_band_t *a = &buf.decouple_a;

    int i = top + blockIdx.y * blockDim.y + threadIdx.y;
    int j = left + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bottom && j < right) {
        int64_t ot_dp, o_mag_sq, t_mag_sq;

        int16_t oh = ref->band_h[i * stride + j];
        int16_t ov = ref->band_v[i * stride + j];
        int16_t od = ref->band_d[i * stride + j];
        int16_t th = dis->band_h[i * stride + j];
        int16_t tv = dis->band_v[i * stride + j];
        int16_t td = dis->band_d[i * stride + j];
        int16_t rst_h, rst_v, rst_d;

        ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
        o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
        t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

        /*
         * the reference implementation evaluates this in double precision:
         * (float)ot_dp / 4096.0 is a double division by a power of two, which
         * is exact
         */
        const double d_ot = (double)(float)ot_dp / 4096.0;
        const double d_o_mag_sq = (double)(float)o_mag_sq / 4096.0;
        const double d_t_mag_sq = (double)(float)t_mag_sq / 4096.0;

        int angle_flag = (d_ot >= 0.0) &&
            (d_ot * d_ot >= (double)cos_1deg_sq * d_o_mag_sq * d_t_mag_sq);

        int32_t tmp_kh = (oh == 0) ? 32768 :
            (int32_t)((((int64_t)adm_div_lookup(oh) * th) + 16384) >> 15);
        int32_t tmp_kv = (ov == 0) ? 32768 :
            (int32_t)((((int64_t)adm_div_lookup(ov) * tv) + 16384) >> 15);
        int32_t tmp_kd = (od == 0) ? 32768 :
            (int32_t)((((int64_t)adm_div_lookup(od) * td) + 16384) >> 15);

        int32_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
        int32_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
        int32_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

        rst_h = (int16_t)(((kh * oh) + 16384) >> 15);
        rst_v = (int16_t)(((kv * ov) + 16384) >> 15);
        rst_d = (int16_t)(((kd * od) + 16384) >> 15);

        const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
        const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
        const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

        if (angle_flag && (rst_h_f > 0.f))
            rst_h = (int16_t)(((double)rst_h * adm_enhn_gain_limit) < (double)th ?
                    ((double)rst_h * adm_enhn_gain_limit) : (double)th);
        if (angle_flag && (rst_h_f < 0.f))
            rst_h = (int16_t)(((double)rst_h * adm_enhn_gain_limit) > (double)th ?
                    ((double)rst_h * adm_enhn_gain_limit) : (double)th);

        if (angle_flag && (rst_v_f > 0.f))
            rst_v = (int16_t)(((double)rst_v * adm_enhn_gain_limit) < (double)tv ?
                    ((double)rst_v * adm_enhn_gain_limit) : (double)tv);
        if (angle_flag && (rst_v_f < 0.f))
            rst_v = (int16_t)(((double)rst_v * adm_enhn_gain_limit) > (double)tv ?
                    ((double)rst_v * adm_enhn_gain_limit) : (double)tv);

        if (angle_flag && (rst_d_f > 0.f))
            rst_d = (int16_t)(((double)rst_d * adm_enhn_gain_limit) < (double)td ?
                    ((double)rst_d * adm_enhn_gain_limit) : (double)td);
        if (angle_flag && (rst_d_f < 0.f))
            rst_d = (int16_t)(((double)rst_d * adm_enhn_gain_limit) > (double)td ?
                    ((double)rst_d * adm_enhn_gain_limit) : (double)td);

        r->band_h[i * stride + j] = rst_h;
        r->band_v[i * stride + j] = rst_v;
        r->band_d[i * stride + j] = rst_d;

        a->band_h[i * stride + j] = (int16_t)(th - rst_h);
        a->band_v[i * stride + j] = (int16_t)(tv - rst_v);
        a->band_d[i * stride + j] = (int16_t)(td - rst_d);
    }
}

__global__ void adm_decouple_s123_kernel(AdmBufferCuda buf, int top, int bottom,
        int left, int right, int stride,
        double adm_enhn_gain_limit, float cos_1deg_sq)
{
    const cuda_i4_adm_dwt_band_t *ref = &buf.i4_ref_dwt2;
    const cuda_i4_adm_dwt_band_t *dis = &buf.i4_dis_dwt2;
    const cuda_i4_adm_dwt_band_t *r = &buf.i4_decouple_r;
    const cuda_i4_adm_dwt_band_t *a = &buf.i4_decouple_a;

    int i = top + blockIdx.y * blockDim.y + threadIdx.y;
    int j = left + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bottom && j < right) {
        int64_t ot_dp, o_mag_sq, t_mag_sq;

        int32_t oh = ref->band_h[i * stride + j];
        int32_t ov = ref->band_v[i * stride + j];
        int32_t od = ref->band_d[i * stride + j];
        int32_t th = dis->band_h[i * stride + j];
        int32_t tv = dis->band_v[i * stride + j];
        int32_t td = dis->band_d[i * stride + j];
        int32_t rst_h, rst_v, rst_d;

        ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
        o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
        t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

        const double d_ot = (double)(float)ot_dp / 4096.0;
        const double d_o_mag_sq = (double)(float)o_mag_sq / 4096.0;
        const double d_t_mag_sq = (double)(float)t_mag_sq / 4096.0;

        int angle_flag = (d_ot >= 0.0) &&
            (d_ot * d_ot >= (double)cos_1deg_sq * d_o_mag_sq * d_t_mag_sq);

        int32_t kh_shift = 0;
        int32_t kv_shift = 0;
        int32_t kd_shift = 0;

        uint32_t abs_oh = abs(oh);
        uint32_t abs_ov = abs(ov);
        uint32_t abs_od = abs(od);

        int8_t kh_sign = (oh < 0 ? -1 : 1);
        int8_t kv_sign = (ov < 0 ? -1 : 1);
        int8_t kd_sign = (od < 0 ? -1 : 1);

        uint16_t kh_msb = (abs_oh < 32768 ? (uint16_t)abs_oh : get_best15_from32(abs_oh, &kh_shift));
        uint16_t kv_msb = (abs_ov < 32768 ? (uint16_t)abs_ov : get_best15_from32(abs_ov, &kv_shift));
        uint16_t kd_msb = (abs_od < 32768 ? (uint16_t)abs_od : get_best15_from32(abs_od, &kd_shift));

        int64_t tmp_kh = (oh == 0) ? 32768 :
            ((((int64_t)adm_div_lookup(kh_msb) * th) * kh_sign) +
             (1 << (14 + kh_shift))) >> (15 + kh_shift);
        int64_t tmp_kv = (ov == 0) ? 32768 :
            ((((int64_t)adm_div_lookup(kv_msb) * tv) * kv_sign) +
             (1 << (14 + kv_shift))) >> (15 + kv_shift);
        int64_t tmp_kd = (od == 0) ? 32768 :
            ((((int64_t)adm_div_lookup(kd_msb) * td) * kd_sign) +
             (1 << (14 + kd_shift))) >> (15 + kd_shift);

        int64_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
        int64_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
        int64_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

        rst_h = (int32_t)(((kh * oh) + 16384) >> 15);
        rst_v = (int32_t)(((kv * ov) + 16384) >> 15);
        rst_d = (int32_t)(((kd * od) + 16384) >> 15);

        const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
        const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
        const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

        if (angle_flag && (rst_h_f > 0.f))
            rst_h = (int32_t)(((double)rst_h * adm_enhn_gain_limit) < (double)th ?
                    ((double)rst_h * adm_enhn_gain_limit) : (double)th);
        if (angle_flag && (rst_h_f < 0.f))
            rst_h = (int32_t)(((double)rst_h * adm_enhn_gain_limit) > (double)th ?
                    ((double)rst_h * adm_enhn_gain_limit) : (double)th);

        if (angle_flag && (rst_v_f > 0.f))
            rst_v = (int32_t)(((double)rst_v * adm_enhn_gain_limit) < (double)tv ?
                    ((double)rst_v * adm_enhn_gain_limit) : (double)tv);
        if (angle_flag && (rst_v_f < 0.f))
            rst_v = (int32_t)(((double)rst_v * adm_enhn_gain_limit) > (double)tv ?
                    ((double)rst_v * adm_enhn_gain_limit) : (double)tv);

        if (angle_flag && (rst_d_f > 0.f))
            rst_d = (int32_t)(((double)rst_d * adm_enhn_gain_limit) < (double)td ?
                    ((double)rst_d * adm_enhn_gain_limit) : (double)td);
        if (angle_flag && (rst_d_f < 0.f))
            rst_d = (int32_t)(((double)rst_d * adm_enhn_gain_limit) > (double)td ?
                    ((double)rst_d * adm_enhn_gain_limit) : (double)td);

        r->band_h[i * stride + j] = rst_h;
        r->band_v[i * stride + j] = rst_v;
        r->band_d[i * stride + j] = rst_d;

        a->band_h[i * stride + j] = th - rst_h;
        a->band_v[i * stride + j] = tv - rst_v;
        a->band_d[i * stride + j] = td - rst_d;
    }
}

}
