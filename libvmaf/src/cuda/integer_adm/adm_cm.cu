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

#include <algorithm>

//#define COMPARE_FUSED_SPLIT
#if defined(COMPARE_FUSED_SPLIT)
#include <iostream>
#endif


__global__ void i4_adm_cm_line_kernel(
    AdmBufferCuda buf, int h, int w, int top, int bottom, int left, int right,
    int start_row, int end_row, int start_col, int end_col, int src_stride,
    int csf_a_stride, int scale, int buffer_h, int buffer_stride,
    int32_t *accum_per_block, AdmFixedParametersCuda params) {
  const cuda_i4_adm_dwt_band_t *src = &buf.i4_decouple_r;
  const cuda_i4_adm_dwt_band_t *csf_f = &buf.i4_csf_f;
  const cuda_i4_adm_dwt_band_t *csf_a = &buf.i4_csf_a;
  const int band = blockIdx.z + 1;
  int32_t *src_band = src->bands[band];
  int32_t *const *angles = csf_a->bands + 1;
  int32_t *const *flt_angles = csf_f->bands + 1;

  // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
  // 1 to 4 (from finest scale to coarsest scale).
  const uint32_t *rfactor = &params.i_rfactor[scale * 3];

  const uint32_t shift_dst = 28;
  const uint32_t shift_flt = 32;
  const int32_t add_bef_shift_dst = (1u << (shift_dst - 1));
  const int32_t add_bef_shift_flt = (1u << (shift_flt - 1));

  uint32_t shift_cub = __float2uint_ru(__log2f(w));

  const int32_t shift_sub = 0;

  int i = start_row + blockIdx.x;
  int j = start_col + blockIdx.y * blockDim.x + threadIdx.x;

  int32_t accum_thread = 0;
  if (i < end_row && j < end_col) {

    int16_t offset_i[2] = {-1, 1};
    if (i == 0 && top <= 0) {
      offset_i[0] = 1;
    } else if (i == (h - 1) && bottom > (h - 1)) {
      offset_i[1] = 0;
    }

    int16_t offset_j[2] = {-1, 1};
    if (j == 0 && left <= 0) {
      offset_j[0] = 1;
    } else if (j == (w - 1) && right > (w - 1)) {
      offset_j[1] = 0;
    }

    int32_t thr = 0;
    for (int theta = 0; theta < 3; ++theta) {
      int32_t sum = 0;
      int32_t src = angles[theta][src_stride * (i + offset_i[0] + 1) + j];
      int32_t *flt_ptr = flt_angles[theta];
      flt_ptr += (src_stride * (i + offset_i[0]));
      sum += flt_ptr[j + offset_j[0]];
      sum += flt_ptr[j];
      sum += flt_ptr[j + offset_j[1]];
      flt_ptr += src_stride;
      sum += flt_ptr[j + offset_j[0]];
      sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t)src)) +
                        add_bef_shift_flt) >>
                       shift_flt);
      sum += flt_ptr[j + offset_j[1]];
      flt_ptr += src_stride * offset_i[1];
      sum += flt_ptr[j + offset_j[0]];
      sum += flt_ptr[j];
      sum += flt_ptr[j + offset_j[1]];
      thr += sum;
    }
    int32_t x = (int32_t)((((int64_t)src_band[i * src_stride + j] *
                            rfactor[blockIdx.z]) +
                           add_bef_shift_dst) >>
                          shift_dst);
    x = abs(x) - (thr >> shift_sub);
    accum_thread = x < 0 ? 0 : x;
  }
  if ((blockIdx.y * blockDim.x + threadIdx.x) < buffer_stride)
    accum_per_block[(blockIdx.z * buffer_h + blockIdx.x) * buffer_stride +
                    blockIdx.y * blockDim.x + threadIdx.x] = accum_thread;
}

__constant__ const int32_t shift_sub[3] = {10, 10, 12};
__constant__ const int fixed_shift[3] = {4, 4, 3};

// accumulation
__constant__ const int32_t shift_xsq[3] = {29, 29, 30};
__constant__ const int32_t add_shift_xsq[3] = {268435456, 268435456, 536870912};

// HACK: the 256 byte alignment is required to ensure that the struct is not moved to lmem
struct alignas(256) WarpShift 
{
  uint32_t shift_cub[3];
  uint32_t add_shift_cub[3];
  uint32_t shift_sq[3];
  uint32_t add_shift_sq[3];
};

template <int fused_accumulator=true, int rows_per_thread>
__global__ void adm_cm_line_kernel(AdmBufferCuda buf, int h, int w, int top,
                                   int bottom, int left, int right,
                                   int start_row, int end_row, int start_col,
                                   int end_col, int src_stride,
                                   int csf_a_stride, int buffer_h,
                                   int buffer_stride, int32_t *accum_per_block,
                                   AdmFixedParametersCuda params,
                                   // reduce
                                   int scale, int64_t* accum_global,

                                   // shift warp
                                   WarpShift ws,
                                   // shift global
                                   const uint32_t shift_inner_accum, const uint32_t add_shift_inner_accum
                                   ) {
  const cuda_adm_dwt_band_t *src = &buf.decouple_r;
  const cuda_adm_dwt_band_t *csf_f = &buf.csf_f;
  const cuda_adm_dwt_band_t *csf_a = &buf.csf_a;
  const int band = blockIdx.z + 1;
  int16_t *src_band = src->bands[band];
  int16_t *const *angles = csf_a->bands + 1;
  int16_t *const *flt_angles = csf_f->bands + 1;

  uint32_t *i_rfactor = params.i_rfactor;

  int cta_y = (blockDim.y * blockIdx.y + threadIdx.y) * rows_per_thread;
  int cta_x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = start_row + cta_y;
  int x = start_col + cta_x;

  const int total_rows = (3 + rows_per_thread - 1);
  int32_t thr[rows_per_thread] = {0};

  if (y < end_row && x < end_col)
  {
    int pos_x[3] = {x - 1, x, x + 1};
    pos_x[0] = abs(pos_x[0]);
    pos_x[2] = pos_x[2] - max(0, 2*(x - w)+1);

    #pragma unroll
    for (int theta = 0; theta < 3; ++theta)
    {
      // the goal of the algorithm below is to read each row only once.
      // to achieve this goal iterate over all rows and then sum up the row
      // to each row accumulator.
      // the sum is accumulated through the following pattern
      // flt[0][0] flt[0][1] flt[0][2]
      // flt[1][0] src[1][1] flt[1][2]
      // flt[2][0] flt[2][1] flt[2][2]
      // for each input row we compute the row overlap with the filter
      // and use the correct summation.
      // since all loops are unrolled the compiler eliminates all ifs
      // and generates just the minimal amount of LDG / IADD instructions.
      #pragma unroll
      for (int row = 0; row < total_rows;++row)
      {
        int pos_y = y - 1 + row;
        pos_y = abs(pos_y);
        pos_y = pos_y - max(0, 2*(y - h)+1);

        int16_t src = angles[theta][pos_y * src_stride + x];
        int16_t *flt_ptr = flt_angles[theta] + pos_y*src_stride;
        int16_t flt_row[3] = {flt_ptr[pos_x[0]], flt_ptr[pos_x[1]], flt_ptr[pos_x[2]]};

        #pragma unroll
        for (int thread_item = 0;thread_item < rows_per_thread;++thread_item)
        {
          uint32_t thread_row = row - thread_item;
          if (thread_row >= 0 && thread_row < 3)
          {
            thr[thread_item] += flt_row[0] + flt_row[2];
            if (thread_row != 1) {
              thr[thread_item] += flt_row[1];
            }
            else {
              thr[thread_item] += (int16_t)(((ONE_BY_15 * abs((int32_t)src)) + 2048) >> 12);;
            }
          }
        }
      }
    }
  }


  int32_t shift_sub_block = shift_sub[blockIdx.z];
  int32_t accum_thread_reg[rows_per_thread] = {0};
  for (int row = 0;row < rows_per_thread;++row) {
    int32_t sb = 0;
    if ((y + row) < end_row && x < end_col) {
      sb = src_band[(y + row) * src_stride + x];
    }
    sb = abs(int32_t(i_rfactor[blockIdx.z] * sb)) - (thr[row] << shift_sub_block);
    accum_thread_reg[row] = max(0, sb);
  }

  if (fused_accumulator)
  {
    const int band2 = blockIdx.z;
    int64_t accum = 0;

    // the compiler does not assume that parameters are constant, move them to local variables to give the compiler
    // a hint that those values have to be loaded only once from constant memory.
    int32_t add_shift_cub = ws.add_shift_cub[band2];
    int32_t shift_cub = ws.shift_cub[band2];
    int32_t add_shift_sq = ws.add_shift_sq[band2];
    int32_t shift_sq = ws.shift_sq[band2];

    // accumulate per thread
    for (int row = 0;row < rows_per_thread;++row) {
      int32_t accum_thread = accum_thread_reg[row];
      const int32_t x_sq = (int32_t)((((int64_t)accum_thread * accum_thread) + add_shift_sq >> shift_sq));
      accum += (((int64_t)x_sq * accum_thread) + add_shift_cub) >> shift_cub;
    }

    // accumulate warp
    accum = warp_reduce(accum);

    if (threadIdx.x % 32 == 0)
    {
      accum = (accum + add_shift_inner_accum) >> shift_inner_accum;
      atomicAdd_int64(&accum_global[band2],
                accum);
    }
  } else {
    for (int row = 0;row < rows_per_thread;++row) {
      if ((y + row) < end_row && x < end_col) {
          accum_per_block[(blockIdx.z * buffer_h + cta_y + row) * buffer_stride +
                          cta_x] = accum_thread_reg[row];
      }
    }
  }
}

template <int val_per_thread, int cta_size>
__global__ void adm_cm_reduce_line_kernel(int h, int w, int scale, int buffer_h,
                                          int buffer_stride,
                                          const int32_t *buffer,
                                          int64_t *accum) {
  const int band = blockIdx.z;
  const int line = blockIdx.y;
  const int off = band * (buffer_h * buffer_stride);
  const int b_off = off + line * buffer_stride;

  uint32_t shift_cub = __float2uint_ru(__log2f(w));
  uint32_t add_shift_cub = 1 << (shift_cub - 1);
  int32_t shift_sq = 30;
  int32_t add_shift_sq = 536870912; // 2^29
  if (scale == 0) {
    shift_cub = __float2uint_ru(__log2f(w) - fixed_shift[band]);
    add_shift_cub = 1 << (shift_cub - 1);
    shift_sq = shift_xsq[band];
    add_shift_sq = add_shift_xsq[band];
  }

  int64_t temp_value = 0;
  const int buffer_col = (cta_size * blockIdx.x) * val_per_thread + threadIdx.x;
  const int32_t *buffer_loc = buffer + b_off + buffer_col;
  for (int i = 0; i < val_per_thread; ++i) {
    if ((buffer_col + i * cta_size) < buffer_stride) {
      const int32_t x = buffer_loc[i * cta_size];
      const int32_t x_sq =
          (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq);
      temp_value += (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub;
    }
  }
  temp_value = warp_reduce(temp_value);

  if ((threadIdx.x % threads_per_warp) == 0) {
    const uint32_t shift_inner_accum = __float2uint_ru(__log2f(h));
    const uint32_t add_shift_inner_accum = 1 << (shift_inner_accum - 1);
    atomicAdd_int64(&accum[band],
              (temp_value + add_shift_inner_accum) >> shift_inner_accum);
  }
}

extern "C" {

void i4_adm_cm_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                      int csf_a_stride, int scale, AdmFixedParametersCuda *p,
                      CUstream c_stream) {

  const int left = w * float(ADM_BORDER_FACTOR) - 0.5f;
  const int top = h * float(ADM_BORDER_FACTOR) - 0.5f;
  const int right = w - left;
  const int bottom = h - top;

  const int start_col = (left > 1) ? left : ((left <= 0) ? 0 : 1);
  const int end_col =
      (right < (w - 1)) ? right : ((right > (w - 1)) ? w : w - 1);
  const int start_row = (top > 1) ? top : ((top <= 0) ? 0 : 1);
  const int end_row =
      (bottom < (h - 1)) ? bottom : ((bottom > (h - 1)) ? h : h - 1);

  const int buffer_stride = end_col - start_col;
  const int buffer_h = end_row - start_row;

  {
    dim3 block(128);
    dim3 grid(buffer_h, DIV_ROUND_UP(buffer_stride, block.x),
              3); // 3 for per band

    i4_adm_cm_line_kernel<<<grid, block, 0, c_stream>>>(
        *buf, h, w, top, bottom, left, right, start_row, end_row, start_col,
        end_col, src_stride, csf_a_stride, scale, buffer_h, buffer_stride,
        (int32_t *)buf->tmp_accum->data, *p);
    CudaCheckError();
  }
  {
    const int val_per_thread = 4;
    const int warps_per_cta = 4;
    dim3 reduce_block(threads_per_warp * warps_per_cta);
    dim3 reduce_grid(
        DIV_ROUND_UP(buffer_stride, reduce_block.x * val_per_thread), buffer_h,
        3);

    adm_cm_reduce_line_kernel<val_per_thread, threads_per_warp * warps_per_cta>
        <<<reduce_grid, reduce_block, 0, c_stream>>>(
            h, w, scale, buffer_h, buffer_stride,
            (int32_t *)buf->tmp_accum->data, (int64_t *)buf->adm_cm[scale]);
    CudaCheckError();
  }
}

void adm_cm_device(AdmBufferCuda *buf, int w, int h, int src_stride,
                   int csf_a_stride, AdmFixedParametersCuda *p,
                   CUstream c_stream) {
  const int scale = 0;
  const int left = w * float(ADM_BORDER_FACTOR) - 0.5f;
  const int top = h * float(ADM_BORDER_FACTOR) - 0.5f;
  const int right = w - left;
  const int bottom = h - top;

  const int start_col = std::max(0, left);
  const int end_col = std::min(right, w);
  const int start_row = std::max(0, top);
  const int end_row = std::min(bottom, h);

  const int buffer_stride = end_col - start_col;
  const int buffer_h = end_row - start_row;

  // precompute warp shift per band
  //const int32_t shift_sub[3] = {10, 10, 12};
  const int fixed_shift[3] = {4, 4, 3};

  // accumulation
  const int32_t shift_xsq[3] = {29, 29, 30};
  const int32_t add_shift_xsq[3] = {268435456, 268435456, 536870912};

  const int NUM_BANDS = 3;
  WarpShift ws;
  for (int band = 0;band < NUM_BANDS;++band) {
    ws.shift_cub[band] = uint32_t(ceil(log2f(w)));
    if (scale == 0) {
      ws.shift_cub[band] -= fixed_shift[band];
      ws.shift_sq[band] = shift_xsq[band];
      ws.add_shift_sq[band] = add_shift_xsq[band];
    } else {
      ws.shift_sq[band] = 30;
      ws.add_shift_sq[band] = (1 << (ws.shift_sq[band]-1));
    }
    ws.add_shift_cub[band] = 1 << (ws.shift_cub[band] - 1);
  }

    // precompute global shift
  const uint32_t shift_inner_accum = uint32_t(ceil(log2f(h)));
  const uint32_t add_shift_inner_accum = 1 << (shift_inner_accum - 1);

//#define COMPARE_FUSED_SPLIT

#if defined(COMPARE_FUSED_SPLIT)
    int64_t adm_cm_fused[4], adm_cm2[4];
  // split
  {
    const int rows_per_thread = 2;
    dim3 block(32,4);
    dim3 grid(DIV_ROUND_UP(buffer_stride, block.x), DIV_ROUND_UP(buffer_h, block.y * rows_per_thread), 3); // 3 for per band

    adm_cm_line_kernel<0, rows_per_thread><<<grid, block, 0, c_stream>>>(
        *buf, h, w, top, bottom, left, right, start_row, end_row, start_col,
        end_col, src_stride, csf_a_stride, buffer_h, buffer_stride,
        (int32_t *)buf->tmp_accum->data, *p,
        scale, (int64_t *)buf->adm_cm[scale],
        ws,
        shift_inner_accum, add_shift_inner_accum
        );
    CudaCheckError();
    const int val_per_thread = 8;
    const int warps_per_cta = 4;
    dim3 reduce_block(threads_per_warp * warps_per_cta);
    dim3 reduce_grid(
        DIV_ROUND_UP(buffer_stride, reduce_block.x * val_per_thread), buffer_h,
        3);

    adm_cm_reduce_line_kernel<val_per_thread, threads_per_warp * warps_per_cta>
        <<<reduce_grid, reduce_block, 0, c_stream>>>(
            h, w, scale, buffer_h, buffer_stride,
            (int32_t *)buf->tmp_accum->data, (int64_t *)buf->adm_cm[scale]);
    CudaCheckError();

    // for verification
    cudaDeviceSynchronize();
    cudaMemcpy(adm_cm2, buf->adm_cm[scale], 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemset(buf->adm_cm[scale], 0, 4 * sizeof(int64_t));
  }

#endif

  // fused
  {
    const int rows_per_thread = 8;
    dim3 block(32,4);
    dim3 grid(DIV_ROUND_UP(buffer_stride, block.x), DIV_ROUND_UP(buffer_h, block.y * rows_per_thread), 3); // 3 for per band

    adm_cm_line_kernel<1, rows_per_thread><<<grid, block, 0, c_stream>>>(
        *buf, h, w, top, bottom, left, right, start_row, end_row, start_col,
        end_col, src_stride, csf_a_stride, buffer_h, buffer_stride,
        (int32_t *)buf->tmp_accum->data, *p,
        scale, (int64_t *)buf->adm_cm[scale],
        ws,
        shift_inner_accum, add_shift_inner_accum);
    CudaCheckError();

#if defined(COMPARE_FUSED_SPLIT)
    // for verification
    cudaDeviceSynchronize();
    cudaMemcpy(adm_cm_fused, buf->adm_cm[scale], 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("\nadm\n");
    for (int i = 0;i < 3;++i) {
      printf("fused[%d]: %16ld  unfused[%d]: %16ld delta[%d]: %16ld\n", i, adm_cm_fused[i], i, adm_cm2[i], i, adm_cm2[i] - adm_cm_fused[i]);
    }
#endif
  }
}
}