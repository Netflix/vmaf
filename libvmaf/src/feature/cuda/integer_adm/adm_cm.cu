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
 * One warp is assigned to a single output row of a single band. The neighbors
 * of the (i,j) cell of the 3x3 contrast masking neighborhood are selected with
 * the exact same rules as the reference implementation:
 *
 *   rows: i = 0     -> (1, 0, 1)
 *         i = h - 1 -> (h - 2, h - 1, h - 1)
 *         otherwise -> (i - 1, i, i + 1)
 *   cols: j = 0     -> (1, 0, 1)
 *         j = w - 1 -> (w - 2, w - 1, w - 1)
 *         otherwise -> (j - 1, j, j + 1)
 *
 * and the set of evaluated cells is identical as well, which implies that the
 * border rows/columns are only evaluated when the frame border falls inside
 * (or next to) the frame.
 *
 * All integer values are computed exactly like the reference, and the per row
 * rounding shift is applied once per row before the integer atomicAdd, so that
 * the order in which the rows are processed does not influence the result.
 */

#define ADM_CM_ONE_BY_15 8738
#define ADM_CM_I4_ONE_BY_15 286331153

namespace {

/* selects the source buffer of a pipeline (dlm or aim) */
template <typename T>
__device__ __forceinline__ const T *adm_cm_select(const T *dlm, const T *aim, int measure_aim)
{
    return measure_aim ? aim : dlm;
}

/*
 * the 3x3 contrast masking neighborhood of a cell, the center term is taken
 * from the "angles" array, the 8 direct neighbors from the "flt" array
 */
template <typename T, bool i4>
__device__ __forceinline__ int32_t adm_cm_thr(const T *const *angles, const T *const *flt, int src_stride,
        int r0, int r1, int r2, int c0, int c1, int c2)
{
    int32_t thr = 0;
#pragma unroll
    for (int theta = 0; theta < 3; ++theta) {
        const T *a = angles[theta];
        const T *f = flt[theta];
        int32_t sum = 0;
        sum += (int32_t)f[(int64_t)r0 * src_stride + c0];
        sum += (int32_t)f[(int64_t)r0 * src_stride + c1];
        sum += (int32_t)f[(int64_t)r0 * src_stride + c2];
        sum += (int32_t)f[(int64_t)r1 * src_stride + c0];
        if (i4)
            /* the reference implementation stores 1u << 31 in an int32_t */
            sum += (int32_t)((((int64_t)ADM_CM_I4_ONE_BY_15 * abs((int32_t)a[(int64_t)r1 * src_stride + c1])) -
                        2147483648LL) >> 32);
        else
            sum += (int32_t)(int16_t)(((ADM_CM_ONE_BY_15 * abs((int32_t)a[(int64_t)r1 * src_stride + c1])) + 2048) >> 12);
        sum += (int32_t)f[(int64_t)r1 * src_stride + c2];
        sum += (int32_t)f[(int64_t)r2 * src_stride + c0];
        sum += (int32_t)f[(int64_t)r2 * src_stride + c1];
        sum += (int32_t)f[(int64_t)r2 * src_stride + c2];
        thr += sum;
    }
    return thr;
}

/* the addressable cell range of a row/column as evaluated by the reference */
__device__ __forceinline__ bool adm_cm_row_range(const AdmCmKernelParamsCuda *p, int i, int *r0, int *r1, int *r2)
{
    if (i == 0) {
        if (p->top > 0)
            return false;
        *r0 = 1;
        *r1 = 0;
        *r2 = 1;
    } else if (i == p->h - 1) {
        if (p->bottom <= p->h - 1)
            return false;
        *r0 = p->h - 2;
        *r1 = p->h - 1;
        *r2 = p->h - 1;
    } else {
        if (i < p->start_row || i >= p->end_row)
            return false;
        *r0 = i - 1;
        *r1 = i;
        *r2 = i + 1;
    }
    /* degenerate images would read out of bounds in the reference as well */
    *r0 = *r0 < 0 ? 0 : (*r0 > p->h - 1 ? p->h - 1 : *r0);
    *r1 = *r1 < 0 ? 0 : (*r1 > p->h - 1 ? p->h - 1 : *r1);
    *r2 = *r2 < 0 ? 0 : (*r2 > p->h - 1 ? p->h - 1 : *r2);
    return true;
}

__device__ __forceinline__ bool adm_cm_col_select(const AdmCmKernelParamsCuda *p, int k, int n_range,
        int has_left, int *j, int *c0, int *c1, int *c2)
{
    if (k < n_range)
        *j = p->start_col + k;
    else if (k == n_range && has_left)
        *j = 0;
    else if (has_left + n_range == k)
        *j = p->w - 1;
    else
        return false;

    if (*j == 0) {
        *c0 = 1;
        *c1 = 0;
        *c2 = 1;
    } else if (*j == p->w - 1) {
        *c0 = p->w - 2;
        *c1 = p->w - 1;
        *c2 = p->w - 1;
    } else {
        *c0 = *j - 1;
        *c1 = *j;
        *c2 = *j + 1;
    }
    /* degenerate images would read out of bounds in the reference as well */
    *c0 = *c0 < 0 ? 0 : (*c0 > p->w - 1 ? p->w - 1 : *c0);
    *c1 = *c1 < 0 ? 0 : (*c1 > p->w - 1 ? p->w - 1 : *c1);
    *c2 = *c2 < 0 ? 0 : (*c2 > p->w - 1 ? p->w - 1 : *c2);
    return true;
}

} // namespace

extern "C" {

__global__ void adm_cm_row_kernel(AdmBufferCuda buf, AdmCmKernelParamsCuda p, int64_t *accum)
{
    int r0, r1, r2;
    if (!adm_cm_row_range(&p, blockIdx.y, &r0, &r1, &r2))
        return;

    const int n_range = p.end_col - p.start_col;
    const int has_left = p.left <= 0 ? 1 : 0;
    const int has_right = p.right > p.w - 1 ? 1 : 0;
    const int n_cols = n_range + has_left + has_right;
    if (n_cols <= 0)
        return;

    const cuda_adm_dwt_band_t *src = p.measure_aim ? &buf.decouple_a : &buf.decouple_r;
    const cuda_adm_dwt_band_t *csf_a = p.measure_aim ? &buf.csf_f : &buf.csf_a;
    const cuda_adm_dwt_band_t *csf_f = p.measure_aim ? &buf.csf_a : &buf.csf_f;
    const int16_t *const *angles = csf_a->bands + 1;
    const int16_t *const *flt_angles = csf_f->bands + 1;
    const int16_t *src_bands[3] = { src->band_h, src->band_v, src->band_d };

    int64_t accum_band[3] = { 0, 0, 0 };

    for (int k = threadIdx.x; k < n_cols; k += blockDim.x) {
        int j, c0, c1, c2;
        if (!adm_cm_col_select(&p, k, n_range, has_left, &j, &c0, &c1, &c2))
            continue;

        const int32_t thr = adm_cm_thr<int16_t, false>(angles, flt_angles, p.src_stride, r0, r1, r2, c0, c1, c2);
        const int col = (int64_t)r1 * p.src_stride + c1;

#pragma unroll
        for (int theta = 0; theta < 3; ++theta) {
            int32_t x = (int32_t)src_bands[theta][col] * (int32_t)p.i_rfactor[theta];
            x = abs(x) - (int32_t)((uint32_t)thr << p.shift_sub[theta]);
            x = x < 0 ? 0 : x;
            const int32_t x_sq = (int32_t)((((int64_t)x * x) + p.add_shift_sq[theta]) >> p.shift_sq[theta]);
            accum_band[theta] += (((int64_t)x_sq * x) + p.add_shift_cub[theta]) >> p.shift_cub[theta];
        }
    }

#pragma unroll
    for (int theta = 0; theta < 3; ++theta) {
        accum_band[theta] = warp_reduce(accum_band[theta]);
        if (threadIdx.x == 0)
            atomicAdd_int64(&accum[theta],
                    (accum_band[theta] + p.add_shift_inner_accum) >> p.shift_inner_accum);
    }
}

__global__ void i4_adm_cm_row_kernel(AdmBufferCuda buf, AdmCmKernelParamsCuda p, int64_t *accum)
{
    int r0, r1, r2;
    if (!adm_cm_row_range(&p, blockIdx.y, &r0, &r1, &r2))
        return;

    const int n_range = p.end_col - p.start_col;
    const int has_left = p.left <= 0 ? 1 : 0;
    const int has_right = p.right > p.w - 1 ? 1 : 0;
    const int n_cols = n_range + has_left + has_right;
    if (n_cols <= 0)
        return;

    const cuda_i4_adm_dwt_band_t *src = p.measure_aim ? &buf.i4_decouple_a : &buf.i4_decouple_r;
    const cuda_i4_adm_dwt_band_t *csf_a = p.measure_aim ? &buf.i4_csf_f : &buf.i4_csf_a;
    const cuda_i4_adm_dwt_band_t *csf_f = p.measure_aim ? &buf.i4_csf_a : &buf.i4_csf_f;
    const int32_t *const *angles = csf_a->bands + 1;
    const int32_t *const *flt_angles = csf_f->bands + 1;
    const int32_t *src_bands[3] = { src->band_h, src->band_v, src->band_d };

    int64_t accum_band[3] = { 0, 0, 0 };

    for (int k = threadIdx.x; k < n_cols; k += blockDim.x) {
        int j, c0, c1, c2;
        if (!adm_cm_col_select(&p, k, n_range, has_left, &j, &c0, &c1, &c2))
            continue;

        const int32_t thr = adm_cm_thr<int32_t, true>(angles, flt_angles, p.src_stride, r0, r1, r2, c0, c1, c2);
        const int col = (int64_t)r1 * p.src_stride + c1;

#pragma unroll
        for (int theta = 0; theta < 3; ++theta) {
            int32_t x = (int32_t)((((int64_t)src_bands[theta][col] * (int64_t)(uint32_t)p.i_rfactor[theta]) +
                        134217728LL) >> 28);
            x = abs(x) - (thr >> p.shift_sub[theta]);
            x = x < 0 ? 0 : x;
            const int32_t x_sq = (int32_t)((((int64_t)x * x) + p.add_shift_sq[theta]) >> p.shift_sq[theta]);
            accum_band[theta] += (((int64_t)x_sq * x) + p.add_shift_cub[theta]) >> p.shift_cub[theta];
        }
    }

#pragma unroll
    for (int theta = 0; theta < 3; ++theta) {
        accum_band[theta] = warp_reduce(accum_band[theta]);
        if (threadIdx.x == 0)
            atomicAdd_int64(&accum[theta],
                    (accum_band[theta] + p.add_shift_inner_accum) >> p.shift_inner_accum);
    }
}

} // extern "C"
