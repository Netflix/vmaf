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

__device__ __forceinline__ uint16_t get_best15_from32(uint32_t temp, int *x) {
    int k = __clz(temp); // built in for cuda
    k = 17 - k;
    temp = (temp + (1 << (k - 1))) >> k;
    *x = k;
    return temp;
}

extern "C" {

__global__ void adm_decouple_kernel(AdmBufferCuda buf, int top, int bottom,
        int left, int right, int stride,
        double adm_enhn_gain_limit) {
    const float cos_1deg_sq =
        cosf(1.0f * float(M_PI) / 180.0f) * cosf(1.0f * float(M_PI) / 180.0f);
    const float div_Q_factor = 1073741824; // 2^30

    const cuda_adm_dwt_band_t *ref = &buf.ref_dwt2;
    const cuda_adm_dwt_band_t *dis = &buf.dis_dwt2;
    const cuda_adm_dwt_band_t *r = &buf.decouple_r;
    const cuda_adm_dwt_band_t *a = &buf.decouple_a;

    int i = top + blockIdx.y * blockDim.y + threadIdx.y;
    int j = left + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bottom && j < right) {
        int32_t ot_dp, o_mag_sq, t_mag_sq;

        int16_t oh = ref->band_h[i * stride + j];
        int16_t ov = ref->band_v[i * stride + j];
        int16_t od = ref->band_d[i * stride + j];
        int16_t th = dis->band_h[i * stride + j];
        int16_t tv = dis->band_v[i * stride + j];
        int16_t td = dis->band_d[i * stride + j];
        int32_t rst_h, rst_v, rst_d;

        // the result of 2*int16_t needs 31 bits.
        // adding two 31 bit values results in 32 bits.
        // thus there is no need for 64-bit math.
        ot_dp = (int32_t)oh * th + (int32_t)ov * tv;
        o_mag_sq = (int32_t)oh * oh + (int32_t)ov * ov;
        t_mag_sq = (int32_t)th * th + (int32_t)tv * tv;

        int angle_flag =
            (ot_dp >= 0) &&
            double(int64_t(ot_dp) * ot_dp) >=
            double(int64_t(o_mag_sq) * t_mag_sq) * cos_1deg_sq;

        int32_t tmp_kh =
            (oh == 0)
            ? 32768
            : (((int64_t)int32_t(div_Q_factor / float(oh)) * th) + 16384) >> 15;
        int32_t tmp_kv =
            (ov == 0)
            ? 32768
            : (((int64_t)int32_t(div_Q_factor / float(ov)) * tv) + 16384) >> 15;
        int32_t tmp_kd =
            (od == 0)
            ? 32768
            : (((int64_t)int32_t(div_Q_factor / float(od)) * td) + 16384) >> 15;

        int32_t kh = max(0, min(32768, tmp_kh));
        int32_t kv = max(0, min(32768, tmp_kv));
        int32_t kd = max(0, min(32768, tmp_kd));

        if (!angle_flag)
            adm_enhn_gain_limit = 1;

        rst_h = (int32_t)(((kh * oh) + 16384) >> 15) * adm_enhn_gain_limit;
        rst_v = (int32_t)(((kv * ov) + 16384) >> 15) * adm_enhn_gain_limit;
        rst_d = (int32_t)(((kd * od) + 16384) >> 15) * adm_enhn_gain_limit;

        const int32_t rst_h_s = kh * oh;
        const int32_t rst_v_s = kv * ov;
        const int32_t rst_d_s = kd * od;

        if (angle_flag) {
            if (rst_h_s > 0)
                rst_h = min(rst_h, (int32_t)th);
            if (rst_h_s < 0)
                rst_h = max(rst_h, (int32_t)th);

            if (rst_v_s > 0)
                rst_v = min(rst_v, (int32_t)tv);
            if (rst_v_s < 0)
                rst_v = max(rst_v, (int32_t)tv);

            if (rst_d_s > 0)
                rst_d = min(rst_d, (int32_t)td);
            if (rst_d_s < 0)
                rst_d = max(rst_d, (int32_t)td);
        }

        r->band_h[i * stride + j] = rst_h;
        r->band_v[i * stride + j] = rst_v;
        r->band_d[i * stride + j] = rst_d;

        a->band_h[i * stride + j] = th - rst_h;
        a->band_v[i * stride + j] = tv - rst_v;
        a->band_d[i * stride + j] = td - rst_d;
    }
}

__global__ void adm_decouple_s123_kernel(AdmBufferCuda buf, int top, int bottom,
        int left, int right, int stride,
        double adm_enhn_gain_limit) {
    const float cos_1deg_sq = cosf(1.0f * float(M_PI) / 180.0f) * cosf(1.0f * float(M_PI) / 180.0f);
    const int32_t div_Q_factor = 1073741824; // 2^30

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

        int angle_flag =
            (((float)ot_dp / 4096.0) >= 0.0f) &&
            (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
             cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

        int32_t kh_shift = 0;
        int32_t kv_shift = 0;
        int32_t kd_shift = 0;

        uint32_t abs_oh = abs(oh);
        uint32_t abs_ov = abs(ov);
        uint32_t abs_od = abs(od);

        int8_t kh_sign = (oh < 0 ? -1 : 1);
        int8_t kv_sign = (ov < 0 ? -1 : 1);
        int8_t kd_sign = (od < 0 ? -1 : 1);

        int32_t kh_msb =
            (abs_oh < (32768) ? abs_oh : get_best15_from32(abs_oh, &kh_shift));
        int32_t kv_msb =
            (abs_ov < (32768) ? abs_ov : get_best15_from32(abs_ov, &kv_shift));
        int32_t kd_msb =
            (abs_od < (32768) ? abs_od : get_best15_from32(abs_od, &kd_shift));

        // executing the division as float division does not work without altering results very slightly
        int64_t tmp_kh =
            (oh == 0) ? 32768
            : (((int64_t)(div_Q_factor / kh_msb) * th) * (kh_sign) +
                    (1 << (14 + kh_shift))) >>
            (15 + kh_shift);
        int64_t tmp_kv =
            (ov == 0) ? 32768
            : (((int64_t)(div_Q_factor / kv_msb) * tv) * (kv_sign) +
                    (1 << (14 + kv_shift))) >>
            (15 + kv_shift);
        int64_t tmp_kd =
            (od == 0) ? 32768
            : (((int64_t)(div_Q_factor / kd_msb) * td) * (kd_sign) +
                    (1 << (14 + kd_shift))) >>
            (15 + kd_shift);

        int64_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
        int64_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
        int64_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

        if (!angle_flag)
            adm_enhn_gain_limit = 1;

        rst_h = (int32_t)(((kh * oh) + 16384) >> 15) * adm_enhn_gain_limit;
        rst_v = (int32_t)(((kv * ov) + 16384) >> 15) * adm_enhn_gain_limit;
        rst_d = (int32_t)(((kd * od) + 16384) >> 15) * adm_enhn_gain_limit;

        const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
        const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
        const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

        if (angle_flag && (rst_h_f > 0.f))
            rst_h = min(rst_h, th);
        if (angle_flag && (rst_h_f < 0.f))
            rst_h = max(rst_h, th);

        if (angle_flag && (rst_v_f > 0.f))
            rst_v = min(rst_v, tv);
        if (angle_flag && (rst_v_f < 0.f))
            rst_v = max(rst_v, tv);

        if (angle_flag && (rst_d_f > 0.f))
            rst_d = min(rst_d, td);
        if (angle_flag && (rst_d_f < 0.f))
            rst_d = max(rst_d, td);

        r->band_h[i * stride + j] = rst_h;
        r->band_v[i * stride + j] = rst_v;
        r->band_d[i * stride + j] = rst_d;

        a->band_h[i * stride + j] = th - rst_h;
        a->band_v[i * stride + j] = tv - rst_v;
        a->band_d[i * stride + j] = td - rst_d;
    }
}

}
