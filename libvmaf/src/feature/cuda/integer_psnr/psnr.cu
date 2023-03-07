/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2022 NVIDIA Corporation.
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

#include "cuda/integer_motion_cuda.h"
#include "cuda_helper.cuh"

#include "common.h"

template <typename T, typename LOAD_TYPE>
__device__ void sse_calculation(T *ref, T *dis, unsigned int w, unsigned int h,
                                unsigned int stride, uint64_t *sse) {
  constexpr int val_per_thread = sizeof(LOAD_TYPE) / sizeof(T);
  unsigned int idx_x = (threadIdx.x + blockDim.x * blockIdx.x) * val_per_thread;
  unsigned int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx_y < h && idx_x < w) {
    int idx = idx_y * (stride / sizeof(T)) + idx_x;
    uint64_t thread_sse = 0u;
    union {
      T value_ref[val_per_thread];
      LOAD_TYPE load_value_dis;
    };
    union {
      T value_dis[val_per_thread];
      LOAD_TYPE load_value_ref;
    };
    load_value_ref = *reinterpret_cast<LOAD_TYPE *>(&ref[idx]);
    load_value_dis = *reinterpret_cast<LOAD_TYPE *>(&dis[idx]);
    for (unsigned int i = 0; i < val_per_thread; ++i) {
      if ((idx_x + i) < w) {
        const int e = value_ref[i] - value_dis[i];
        thread_sse += e * e;
      }
    }

    // Warp-reduce abs_dist
    thread_sse += __shfl_down_sync(0xffffffff, thread_sse, 16);
    thread_sse += __shfl_down_sync(0xffffffff, thread_sse, 8);
    thread_sse += __shfl_down_sync(0xffffffff, thread_sse, 4);
    thread_sse += __shfl_down_sync(0xffffffff, thread_sse, 2);
    thread_sse += __shfl_down_sync(0xffffffff, thread_sse, 1);
    // Let threads in lane zero add warp-reduced abs_dist atomically to global
    // sad
    const int lane =
        (threadIdx.y * blockDim.x + threadIdx.x) % VMAF_CUDA_THREADS_PER_WARP;
    if (lane == 0)
      atomicAdd_uint64(sse, static_cast<uint64_t>(thread_sse));
  }
}

template <int chn>
__device__ void psnr8_impl(const VmafPicture ref_pic,
                           const VmafPicture dist_pic,
                           const VmafCudaBuffer sse) {
  unsigned int stride = ref_pic.stride[chn];
  // if second channel is smaller use smaller load
  if (stride <= (ref_pic.stride[0] / 2))
    sse_calculation<uint8_t, uint64_t>(
        reinterpret_cast<uint8_t *>(ref_pic.data[chn]),
        reinterpret_cast<uint8_t *>(dist_pic.data[chn]), ref_pic.w[chn],
        ref_pic.h[chn], stride, reinterpret_cast<uint64_t *>(sse.data) + chn);
  else
    sse_calculation<uint8_t, ushort4>(
        reinterpret_cast<uint8_t *>(ref_pic.data[chn]),
        reinterpret_cast<uint8_t *>(dist_pic.data[chn]), ref_pic.w[chn],
        ref_pic.h[chn], stride, reinterpret_cast<uint64_t *>(sse.data) + chn);
}

template <int chn>
__device__ void psnr16_impl(const VmafPicture ref_pic,
                            const VmafPicture dist_pic,
                            const VmafCudaBuffer sse) {
  unsigned int stride = ref_pic.stride[chn];
  // if second channel is smaller use smaller load
  if (stride <= (ref_pic.stride[0] / 2))
    sse_calculation<uint16_t, ushort4>(
        reinterpret_cast<uint16_t *>(ref_pic.data[chn]),
        reinterpret_cast<uint16_t *>(dist_pic.data[chn]), ref_pic.w[chn],
        ref_pic.h[chn], stride, reinterpret_cast<uint64_t *>(sse.data) + chn);
  else
    sse_calculation<uint16_t, uint4>(
        reinterpret_cast<uint16_t *>(ref_pic.data[chn]),
        reinterpret_cast<uint16_t *>(dist_pic.data[chn]), ref_pic.w[chn],
        ref_pic.h[chn], stride, reinterpret_cast<uint64_t *>(sse.data) + chn);
}

extern "C" {

__global__ void psnr(const VmafPicture ref_pic, const VmafPicture dist_pic,
                     const VmafCudaBuffer sse) {
  // this is needed to not produce local load/store ops when accessing with
  // "dynamic" index although blockIdx.z is not really dynamic
  switch (blockIdx.z) {
  case 0:
    psnr8_impl<0>(ref_pic, dist_pic, sse);
    return;
  case 1:
    psnr8_impl<1>(ref_pic, dist_pic, sse);
    return;
  case 2:
    psnr8_impl<2>(ref_pic, dist_pic, sse);
    return;
  }
}

__global__ void psnr_hbd(const VmafPicture ref_pic, const VmafPicture dist_pic,
                         const VmafCudaBuffer sse) {
  // this is needed to not produce local load/store ops when accessing with
  // "dynamic" index although blockIdx.z is not really dynamic
  switch (blockIdx.z) {
  case 0:
    psnr16_impl<0>(ref_pic, dist_pic, sse);
    return;
  case 1:
    psnr16_impl<1>(ref_pic, dist_pic, sse);
    return;
  case 2:
    psnr16_impl<2>(ref_pic, dist_pic, sse);
    return;
  }
}
}