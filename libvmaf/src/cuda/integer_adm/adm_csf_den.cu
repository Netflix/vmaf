/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

 #include "integer_adm_kernels.h"
#include "cuda_helper.cuh"


template <int val_per_thread, int cta_size>
__global__ void adm_csf_den_scale_line_kernel(const cuda_adm_dwt_band_t src, int h,
                                              int top, int bottom, int left,
                                              int right, int src_stride,
                                              uint64_t *accum) {
  const int band = blockIdx.z + 1;
  // this is evaluated to a 4 STL and one LDL therefore we need the switch 
  // const int16_t * src_ptr = src.bands[band] + (top + blockIdx.y) * src_stride;
  int16_t * src_ptr;
  switch (band) {
  case 1:
    src_ptr = src.band_h + (top + blockIdx.y) * src_stride;
    break;
  case 2:
    src_ptr = src.band_v + (top + blockIdx.y) * src_stride;
    break;
  case 3:
    src_ptr = src.band_d + (top + blockIdx.y) * src_stride;
    break;
  }
  int j = left + (cta_size * blockIdx.x) * val_per_thread + threadIdx.x;

  uint64_cu temp_value = 0;
#pragma unroll
  for (int i = 0; i < val_per_thread; ++i) {
    if (j < right) {
      uint16_t t = (uint16_t)abs(src_ptr[j]);
      temp_value += ((uint64_t)t * t) * t;
    }
    j += cta_size;
  }
  temp_value = warp_reduce(temp_value);

  if ((threadIdx.x % threads_per_warp) == 0) {
    uint32_t shift_accum = (uint32_t)__float2uint_ru(
        __log2f((bottom - top) * (right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    int32_t add_shift_accum = shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;
    atomicAdd((uint64_cu *)&accum[band - 1],
              (temp_value + add_shift_accum) >> shift_accum);
  }
}

template <int val_per_thread, int cta_size>
__global__ void adm_csf_den_s123_line_kernel(
    const cuda_i4_adm_dwt_band_t src, int h, int top, int bottom, int left, int right,
    int src_stride, uint32_t add_shift_sq, uint32_t shift_sq, uint64_t *accum) {
  const uint32_t shift_cub = (uint32_t)__float2uint_ru(__log2f((right - left)));
  const uint32_t add_shift_cub = (uint32_t)1 << (shift_cub - 1);

  const int band = blockIdx.z + 1;
  // this is evaluated to a 4 STL and one LDL therefore we need the switch 
  // const int32_t *src_ptr = src.bands[band] + (top + blockIdx.y) * src_stride;
  int32_t * src_ptr;
  switch (band) {
  case 1:
    src_ptr = src.band_h + (top + blockIdx.y) * src_stride;
    break;
  case 2:
    src_ptr = src.band_v + (top + blockIdx.y) * src_stride;
    break;
  case 3:
    src_ptr = src.band_d + (top + blockIdx.y) * src_stride;
    break;
  }
  int j = left + (cta_size * blockIdx.x) * val_per_thread + threadIdx.x;

  uint64_cu temp_value = 0;
#pragma unroll
  for (int i = 0; i < val_per_thread; ++i) {
    if (j < right) {
      uint32_t t = (uint32_t)abs(src_ptr[j]);
      temp_value += (((((((uint64_t)t * t) + add_shift_sq) >> shift_sq) * t) +
                      add_shift_cub) >>
                     shift_cub);
    }
    j += cta_size;
  }
  temp_value = warp_reduce(temp_value);

  if ((threadIdx.x % threads_per_warp) == 0) {
    uint32_t shift_accum = (uint32_t)__float2uint_ru(__log2f((bottom - top)));
    uint32_t add_shift_accum = (uint32_t)1 << (shift_accum - 1);
    atomicAdd((uint64_cu *)&accum[band - 1],
              (temp_value + add_shift_accum) >> shift_accum);
  }
}

extern "C" {

void adm_csf_den_s123_device(AdmBufferCuda *buf, int scale, int w, int h,
                             int src_stride, double adm_norm_view_dist,
                             int adm_ref_display_height, CUstream c_stream) {
  /* The computation of the denominator scales is not required for the regions
   * which lie outside the frame borders
   */

  const int left = w * float(ADM_BORDER_FACTOR) - 0.5f;
  const int top = h * float(ADM_BORDER_FACTOR) - 0.5f;
  const int right = w - left;
  const int bottom = h - top;

  const int buffer_stride = right - left;
  const int buffer_h = bottom - top;

  const int val_per_thread = 8;
  const int warps_per_cta = 4;
  dim3 reduce_block(threads_per_warp * warps_per_cta);
  dim3 reduce_grid(DIV_ROUND_UP(buffer_stride, reduce_block.x * val_per_thread),
                   buffer_h, 3);
  const uint32_t shift_sq[3] = {31, 30, 31};
  const uint32_t add_shift_sq[3] = {1u << shift_sq[0], 1u << shift_sq[1],
                                    1u << shift_sq[2]};

  adm_csf_den_s123_line_kernel<val_per_thread, threads_per_warp * warps_per_cta>
      <<<reduce_grid, reduce_block, 0, c_stream>>>(
          buf->i4_ref_dwt2, h, top, bottom, left, right, src_stride,
          add_shift_sq[scale - 1], shift_sq[scale - 1],
          (uint64_t *)buf->adm_csf_den[scale]);
  CudaCheckError();
}

void adm_csf_den_scale_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                              double adm_norm_view_dist,
                              int adm_ref_display_height, CUstream c_stream) {
  /* The computation of the denominator scales is not required for the regions
   * which lie outside the frame borders
   */
  const int scale = 0;
  const int left = w * float(ADM_BORDER_FACTOR) - 0.5f;
  const int top = h * float(ADM_BORDER_FACTOR) - 0.5f;
  const int right = w - left;
  const int bottom = h - top;

  const int buffer_stride = right - left;
  const int buffer_h = bottom - top;

  const int val_per_thread = 8;
  const int warps_per_cta = 4;
  dim3 reduce_block(threads_per_warp * warps_per_cta);
  dim3 reduce_grid(DIV_ROUND_UP(buffer_stride, reduce_block.x * val_per_thread),
                   buffer_h, 3);

  adm_csf_den_scale_line_kernel<val_per_thread,
                                threads_per_warp * warps_per_cta>
      <<<reduce_grid, reduce_block, 0, c_stream>>>(
          buf->ref_dwt2, h, top, bottom, left, right, src_stride,
          (uint64_t *)buf->adm_csf_den[scale]);
  CudaCheckError();
}

} // extern "C"