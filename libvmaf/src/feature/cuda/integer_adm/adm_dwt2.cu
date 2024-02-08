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

#include <vector>

#include "cuda_helper.cuh"

// Calculates and returns vector with indices for dwt, if upper limit is
// reached, the indices will be mirrored
__device__ __forceinline__ int4 calculate_indices(const int n,
        const int upper_limit) {
    int4 indices = make_int4(2 * n - 1, 2 * n, 2 * n + 1, 2 * n + 2);

    if (!n) {
        indices.x = 1;
        indices.y = 0;
        indices.z = 1;
        indices.w = 2;
    }

    if (indices.x >= upper_limit)
        indices.x = (2 * upper_limit - indices.x - 1);

    if (indices.y >= upper_limit)
        indices.y = (2 * upper_limit - indices.y - 1);

    if (indices.z >= upper_limit)
        indices.z = (2 * upper_limit - indices.z - 1);

    if (indices.w >= upper_limit)
        indices.w = (2 * upper_limit - indices.w - 1);

    return indices;
}

template<int32_t add_shift, int16_t shift, typename T>
__device__ __forceinline__ void dwt_s123_combined_vert_kernel(const T *d_image_scale, int32_t *tmplo_start, int w, int h, int img_stride, AdmFixedParametersCuda params) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = blockIdx.y;
    if (idx >= w)
        return;

    const int32_t *filter_lo = params.dwt2_db2_coeffs_lo;
    const int32_t *filter_hi = params.dwt2_db2_coeffs_hi;

    int64_t accum_lo = 0, accum_hi = 0;
    int32_t * const tmplo = tmplo_start + w * i * 2;
    int32_t * const tmphi = tmplo + w;

    const int4 pixel = calculate_indices(i, h);
    const T s10 = d_image_scale[pixel.x * img_stride + idx];
    const T s11 = d_image_scale[pixel.y * img_stride + idx];
    const T s12 = d_image_scale[pixel.z * img_stride + idx];
    const T s13 = d_image_scale[pixel.w * img_stride + idx];

    accum_lo += (int64_t)filter_lo[0] * s10;
    accum_lo += (int64_t)filter_lo[1] * s11;
    accum_lo += (int64_t)filter_lo[2] * s12;
    accum_lo += (int64_t)filter_lo[3] * s13;
    tmplo[idx] = (int32_t)((accum_lo + add_shift) >> shift);

    accum_hi += (int64_t)filter_hi[0] * s10;
    accum_hi += (int64_t)filter_hi[1] * s11;
    accum_hi += (int64_t)filter_hi[2] * s12;
    accum_hi += (int64_t)filter_hi[3] * s13;
    tmphi[idx] = (int32_t)((accum_hi + add_shift) >> shift);
}

template<int32_t add_shift, int16_t shift>
__device__ __forceinline__ void dwt_s123_combined_hori_kernel(cuda_i4_adm_dwt_band_t i4_dwt2, int32_t *tmplo_start, int w, int h,  int dst_stride, AdmFixedParametersCuda params) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = blockIdx.y;
    if (idx >= (w + 1) / 2)
        return;

    const int32_t *filter_lo = params.dwt2_db2_coeffs_lo;
    const int32_t *filter_hi = params.dwt2_db2_coeffs_hi;

    int64_t accum = 0;

    const int4 pixels = calculate_indices(idx, w);
    int32_t * const tmplo = tmplo_start + w * i * 2;
    int32_t * const tmphi = tmplo + w;

    const int32_t s0_lo = tmplo[pixels.x];
    const int32_t s1_lo = tmplo[pixels.y];
    const int32_t s2_lo = tmplo[pixels.z];
    const int32_t s3_lo = tmplo[pixels.w];

    const int32_t s0_hi = tmphi[pixels.x];
    const int32_t s1_hi = tmphi[pixels.y];
    const int32_t s2_hi = tmphi[pixels.z];
    const int32_t s3_hi = tmphi[pixels.w];

    accum = 0;
    accum += (int64_t)filter_lo[0] * s0_lo;
    accum += (int64_t)filter_lo[1] * s1_lo;
    accum += (int64_t)filter_lo[2] * s2_lo;
    accum += (int64_t)filter_lo[3] * s3_lo;
    i4_dwt2.band_a[i * dst_stride + idx] = (int32_t)((accum + add_shift) >> shift);

    accum = 0;
    accum += (int64_t)filter_hi[0] * s0_lo;
    accum += (int64_t)filter_hi[1] * s1_lo;
    accum += (int64_t)filter_hi[2] * s2_lo;
    accum += (int64_t)filter_hi[3] * s3_lo;
    i4_dwt2.band_v[i * dst_stride + idx] = (int32_t)((accum + add_shift) >> shift);

    accum = 0;
    accum += (int64_t)filter_lo[0] * s0_hi;
    accum += (int64_t)filter_lo[1] * s1_hi;
    accum += (int64_t)filter_lo[2] * s2_hi;
    accum += (int64_t)filter_lo[3] * s3_hi;
    i4_dwt2.band_h[i * dst_stride + idx] = (int32_t)((accum + add_shift) >> shift);

    accum = 0;
    accum += (int64_t)filter_hi[0] * s0_hi;
    accum += (int64_t)filter_hi[1] * s1_hi;
    accum += (int64_t)filter_hi[2] * s2_hi;
    accum += (int64_t)filter_hi[3] * s3_hi;
    i4_dwt2.band_d[i * dst_stride + idx] = (int32_t)((accum + add_shift) >> shift);
}

template<int v_rows_per_thread, int32_t h_shift, int32_t h_add_shift, int32_t tile_width, int tile_height, typename T>
__device__ __forceinline__ void adm_dwt2_8_vert_hori_kernel(const T * d_picture,
        cuda_adm_dwt_band_t dst,
        cuda_i4_adm_dwt_band_t i4_dwt2,
        int w, int h,
        int src_stride, int dst_stride,
        int16_t v_shift, int32_t v_add_shift,
        AdmFixedParametersCuda params) {

    // The fused kernel writes the result of the vertical dwt2 to shared memory for consumption by the horizontal dwt2.
    // The vertical pass computes a tile of size [tile_width, tile_height].
    // The horizontal pass computes a tile of size [tile_width / 2 - 2, tile_height] due to the required overlap of the tiles.

    __shared__ short2 s_tile[tile_height][tile_width];

    const int horz_out_tile_rows = tile_height;
    const int horz_out_tile_cols = tile_width / 2 - 2;

    {
        int x = threadIdx.x + blockIdx.x * 2 * horz_out_tile_cols;
        // For each output at position x of the horizontal pass we have to read the range [2*x-1, 2*x+2]. Thus start one element to the left
        x = max(0, x-1); // TODO when using abs instead we'd already do the x-mirroring here.

        const int y_out = (threadIdx.y + blockIdx.y * horz_out_tile_rows / v_rows_per_thread) * v_rows_per_thread;

        if (x < w) {

            const int32_t *filter_lo = params.dwt2_db2_coeffs_lo;
            const int32_t *filter_hi = params.dwt2_db2_coeffs_hi;

            // we need 2 addition rows for each item processed by a thread.
            const int size_u = 2 + 2 * v_rows_per_thread;
            int32_t u_s[size_u];

            // load size_u rows applying the mirroring on the top & bottom
            const int read_backwards_elements = 1;
            int y_in_start = (2 * y_out) - read_backwards_elements;

#pragma unroll
            for (int i = 0;i < size_u;++i) {
                int y_in = y_in_start + i;

                // mirror bottom
                y_in = y_in - max(0, 2*(y_in - h)+1);

                // mirror top, only required for read_backwards_elements elements.
                if (i < read_backwards_elements) y_in = abs(y_in);

                // no bounds check required due to the mirroring
                u_s[i] = d_picture[y_in * src_stride + x];
            }

            short2 *s_tmp_thread = s_tile[threadIdx.y] + threadIdx.x;

            // accumulate items_per_thread values and store them
#pragma unroll
            for (int item = 0;item < v_rows_per_thread;++item) {
                // stop for the first row outside of the image
                if ((y_out + item) >= h) {
                    break;
                }

                int32_t accum_lo = 0, accum_hi = 0;
#pragma unroll
                for (int i = 0;i< 4;++i) {
                    accum_lo += filter_lo[i] * u_s[i + 2 * item];
                    accum_hi += filter_hi[i] * u_s[i + 2 * item];
                }

                /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
                accum_lo -= params.dwt2_db2_coeffs_lo_sum * v_add_shift;
                accum_hi -= params.dwt2_db2_coeffs_hi_sum * v_add_shift;

                s_tile[threadIdx.y * v_rows_per_thread + item][threadIdx.x] = make_short2((accum_lo + v_add_shift) >> v_shift, (accum_hi + v_add_shift) >> v_shift);
            }
        }
    }

    __syncthreads();

    // only ~50% of the threads in the x direction have work to do
    if (threadIdx.x >= horz_out_tile_cols)
        return;

    // hori
    {
        const int x_out = threadIdx.x + blockIdx.x * horz_out_tile_cols;
        const int x_out_cta = blockIdx.x * horz_out_tile_cols;
        if (x_out >= (w + 1) / 2)
            return;

        const int32_t *filter_lo = params.dwt2_db2_coeffs_lo;
        const int32_t *filter_hi = params.dwt2_db2_coeffs_hi;
        const int4 pixel = calculate_indices(x_out, w);

        for (int y_thread = 0;y_thread < v_rows_per_thread;++y_thread) {
            const int y_out = threadIdx.y * v_rows_per_thread + y_thread + blockIdx.y * horz_out_tile_rows;
            if (y_out >= (h+1)/2)
                return;

            int32_t accum = 0;

            short2 * const tmp = s_tile[threadIdx.y * v_rows_per_thread + y_thread];

            const int32_t s0_lo = tmp[pixel.x - 2 * x_out_cta + 1].x;
            const int32_t s0_hi = tmp[pixel.x - 2 * x_out_cta + 1].y;
            const int32_t s1_lo = tmp[pixel.y - 2 * x_out_cta + 1].x;
            const int32_t s1_hi = tmp[pixel.y - 2 * x_out_cta + 1].y;
            const int32_t s2_lo = tmp[pixel.z - 2 * x_out_cta + 1].x;
            const int32_t s2_hi = tmp[pixel.z - 2 * x_out_cta + 1].y;
            const int32_t s3_lo = tmp[pixel.w - 2 * x_out_cta + 1].x;
            const int32_t s3_hi = tmp[pixel.w - 2 * x_out_cta + 1].y;

            accum = 0;
            accum += filter_lo[0] * s0_lo;
            accum += filter_lo[1] * s1_lo;
            accum += filter_lo[2] * s2_lo;
            accum += filter_lo[3] * s3_lo;
            accum = ((accum + h_add_shift) >> h_shift);
            dst.band_a[y_out * dst_stride + x_out] = accum;
            i4_dwt2.band_a[y_out * dst_stride + x_out] = accum;

            accum = 0;
            accum += filter_hi[0] * s0_lo;
            accum += filter_hi[1] * s1_lo;
            accum += filter_hi[2] * s2_lo;
            accum += filter_hi[3] * s3_lo;
            dst.band_v[y_out * dst_stride + x_out] = (accum + h_add_shift) >> h_shift;


            accum = 0;
            accum += filter_lo[0] * s0_hi;
            accum += filter_lo[1] * s1_hi;
            accum += filter_lo[2] * s2_hi;
            accum += filter_lo[3] * s3_hi;
            dst.band_h[y_out * dst_stride + x_out] = (accum + h_add_shift) >> h_shift;

            accum = 0;
            accum += filter_hi[0] * s0_hi;
            accum += filter_hi[1] * s1_hi;
            accum += filter_hi[2] * s2_hi;
            accum += filter_hi[3] * s3_hi;
            dst.band_d[y_out * dst_stride + x_out] = (accum + h_add_shift) >> h_shift;
        }
    }
}

// C __global__ KERNEL C DEFINES
#pragma region

#define DWT_S123_COMBINED_VERT(add_shift, shift, type)                             \
    __global__ void dwt_s123_combined_vert_kernel_##add_shift##_##shift##_##type ( \
            const type *d_image_scale,                                             \
            int32_t *tmplo_start, int w, int h,                                    \
            int img_stride,                                                        \
            AdmFixedParametersCuda params ) {                                      \
        dwt_s123_combined_vert_kernel<add_shift, shift, type>(d_image_scale,       \
                tmplo_start, w ,h,                                                 \
                img_stride, params);                                               \
    }

#define DWT_S123_COMBINED_HORI(add_shift, shift)                                   \
    __global__ void dwt_s123_combined_hori_kernel_##add_shift##_##shift (          \
            const cuda_i4_adm_dwt_band_t i4_dwt2,                                  \
            int32_t *tmplo_start, int w, int h,                                    \
            int dst_stride,                                                        \
            AdmFixedParametersCuda params ) {                                      \
        dwt_s123_combined_hori_kernel<add_shift, shift>(i4_dwt2,                   \
                tmplo_start, w ,h,                                                 \
                dst_stride, params);                                               \
    }


#define DWT_8_VERT_HORI(v_rows_per_thread, h_shift, h_add_shift, tile_width, tile_height, type)                                                  \
    __global__ void adm_dwt2_8_vert_hori_kernel_##v_rows_per_thread##_##h_shift##_##h_add_shift##_##tile_width##_##tile_height##_##type (  \
            const type * d_picture,                                                                                                                                    \
            cuda_adm_dwt_band_t dst,                                                                                                                                   \
            cuda_i4_adm_dwt_band_t i4_dwt2,                                                                                                                            \
            int w, int h,                                                                                                                                              \
            int src_stride, int dst_stride,                                                                                                                            \
            int16_t v_shift, int32_t v_add_shift,                                                                                                                      \
            AdmFixedParametersCuda params) {                                                                                                                           \
        adm_dwt2_8_vert_hori_kernel<v_rows_per_thread, h_shift, h_add_shift, tile_width, tile_height, type>(                                                           \
                d_picture, dst, i4_dwt2,                                                                                                                               \
                w, h, src_stride, dst_stride, v_shift, v_add_shift, params);                                                                                           \
    }
#pragma endregion

extern "C" {
    DWT_S123_COMBINED_VERT(0, 0, int32_t);             // dwt_s123_combined_vert_kernel_0_0_int32_t
    DWT_S123_COMBINED_VERT(32768, 16, int32_t);        // dwt_s123_combined_vert_kernel_32768_16_int32_t
    DWT_S123_COMBINED_HORI(16384, 15);                 // dwt_s123_combined_hori_kernel_16384_15
    DWT_S123_COMBINED_HORI(32768, 16);                 // dwt_s123_combined_hori_kernel_32768_16
    DWT_8_VERT_HORI(4, 16, 32768, 128, 8, uint8_t);    // adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t
    DWT_8_VERT_HORI(4, 16, 32768, 128, 8, uint16_t);   // adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t
}
