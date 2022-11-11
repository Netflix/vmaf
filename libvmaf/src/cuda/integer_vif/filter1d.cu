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

 #include "cuda_helper.cuh"
 #include "integer_vif_kernels.h"
 
 #include "vif_statistics.cuh"
 
 
 template <typename alignment_type = uint2, int fwidth_0 = 17, int fwidth_1 = 9>
 __device__ __forceinline__ void filter1d_8_vertical_kernel(VifBufferCuda buf, uint8_t* ref_in, uint8_t* dis_in, 
                                            int w, int h, filter_table_stuct vif_filt_s0) {
   using writeback_type = uint4;
   constexpr int val_per_thread = sizeof(alignment_type);
   static_assert(val_per_thread % 4 == 0 && val_per_thread <= 16,
                 "val per thread bust be divisible by 4 and under 16");
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
   if (x_start < w && y < h) {
     const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
     __align__(sizeof(writeback_type)) uint32_t accum_mu1[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_mu2[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_ref[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_dis[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_ref_dis[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_ref_rd[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_dis_rd[val_per_thread] = {0};
     union {
       uint8_t ref[val_per_thread];
       alignment_type ref_aligned;
     };
     union {
       uint8_t dis[val_per_thread];
       alignment_type dis_aligned;
     };
     for (int fi = 0; fi < fwidth_0; ++fi) {
       const int ii = y - fwidth_0 / 2;
       unsigned int ii_check = abs(ii + fi);
       if (ii_check >= h) {
         ii_check = 2 * h - ii_check - 2;
       }
       ref_aligned = *reinterpret_cast<alignment_type *>(
           &(ref_in[ii_check * buf.stride + x_start]));
       dis_aligned = *reinterpret_cast<alignment_type *>(
           &(dis_in[ii_check * buf.stride + x_start]));
       for (int off = 0; off < val_per_thread; ++off) {
         const int j = x_start + off;
         if (j < w) {
           const uint32_t fcoeff = vif_filt_s0.filter[0][fi];
           const uint32_t ref_val = ref[off];
           const uint32_t dis_val = dis[off];
           const uint32_t img_coeff_ref = fcoeff * (uint32_t)ref_val;
           const uint32_t img_coeff_dis = fcoeff * (uint32_t)dis_val;
           accum_mu1[off] += img_coeff_ref;
           accum_mu2[off] += img_coeff_dis;
           accum_ref[off] += img_coeff_ref * (uint32_t)ref_val;
           accum_dis[off] += img_coeff_dis * (uint32_t)dis_val;
           accum_ref_dis[off] += img_coeff_ref * (uint32_t)dis_val;
           if (fi >= (fwidth_0 - fwidth_1) / 2 &&
               fi < (fwidth_0 - (fwidth_0 - fwidth_1) / 2)) {
             const uint16_t fcoeff_rd =
                 vif_filt_s0.filter[1][fi - ((fwidth_0 - fwidth_1) / 2)];
             accum_ref_rd[off] += fcoeff_rd * ref_val;
             accum_dis_rd[off] += fcoeff_rd * dis_val;
           }
         }
       }
     }
     for (int off = 0; off < val_per_thread; ++off) {
       accum_mu1[off] = (accum_mu1[off] + 128) >> 8;
       accum_mu2[off] = (accum_mu2[off] + 128) >> 8;
       accum_ref_rd[off] = (accum_ref_rd[off] + 128) >> 8;
       accum_dis_rd[off] = (accum_dis_rd[off] + 128) >> 8;
     }
     for (int idx = 0; idx < val_per_thread; idx += sizeof(writeback_type) / sizeof(uint32_t)) {
       const int buffer_idx = y * stride_tmp + x_start + idx;
       if (x_start + idx < w) {
         *reinterpret_cast<writeback_type *>(&buf.tmp.mu1[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_mu1[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.mu2[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_mu2[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.ref[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_ref[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.dis[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_dis[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.ref_dis[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_ref_dis[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.ref_convol[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_ref_rd[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.dis_convol[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_dis_rd[idx]);
       }
     }
   }
 }
 
 template <int val_per_thread = 1, int fwidth_0 = 17, int fwidth_1 = 9>
 __device__ __forceinline__ void filter1d_8_horizontal_kernel(VifBufferCuda buf, int w, int h,
                                              filter_table_stuct vif_filt_s0,
                                              double vif_enhn_gain_limit,
                                              vif_accums *accum) {
   static_assert(val_per_thread % 2 == 0,
                 "val_per_thread must be divisible by 2");
   int y = blockIdx.y;
   int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
   if (y < h && x_start < w) {
 
     union {
       vif_accums thread_accum;
       int64_t thread_accum_i64[7] = {0};
     };
 
     uint32_t accum_mu1[val_per_thread] = {0};
     uint32_t accum_mu2[val_per_thread] = {0};
     uint32_t accum_ref[val_per_thread] = {0};
     uint32_t accum_dis[val_per_thread] = {0};
     uint32_t accum_ref_dis[val_per_thread] = {0};
     uint64_t accum_ref_tmp[val_per_thread] = {0};
     uint64_t accum_dis_tmp[val_per_thread] = {0};
     uint64_t accum_ref_dis_tmp[val_per_thread] = {0};
 
     uint32_t accum_ref_rd[val_per_thread / 2] = {0};
     uint32_t accum_dis_rd[val_per_thread / 2] = {0};
 
     const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
 #pragma unroll
     for (int fj = 0; fj < fwidth_0; ++fj) {
 #pragma unroll
       for (int off = 0; off < val_per_thread; ++off) {
         const int j = x_start + off;
         if (j < w) {
           int jj = j - fwidth_0 / 2;
           int jj_check = abs(jj + fj);
           if (jj_check >= w) {
             jj_check = 2 * w - jj_check - 2;
           }
           const uint16_t fcoeff = vif_filt_s0.filter[0][fj];
           accum_mu1[off] +=
               fcoeff * ((uint32_t)buf.tmp.mu1[y * stride_tmp + jj_check]);
           accum_mu2[off] +=
               fcoeff * ((uint32_t)buf.tmp.mu2[y * stride_tmp + jj_check]);
           accum_ref_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.ref[y * stride_tmp + jj_check]);
           accum_dis_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.dis[y * stride_tmp + jj_check]);
           accum_ref_dis_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.ref_dis[y * stride_tmp + jj_check]);
 
           if (fj >= (fwidth_0 - fwidth_1) / 2 &&
               fj < (fwidth_0 - (fwidth_0 - fwidth_1) / 2) && off % 2 == 0) {
             const uint16_t fcoeff_rd =
                 vif_filt_s0.filter[1][fj - ((fwidth_0 - fwidth_1) / 2)];
             accum_ref_rd[off / 2] +=
                 fcoeff_rd * buf.tmp.ref_convol[y * stride_tmp + jj_check];
             accum_dis_rd[off / 2] +=
                 fcoeff_rd * buf.tmp.dis_convol[y * stride_tmp + jj_check];
           }
         }
       }
     }
     for (int off = 0; off < val_per_thread; ++off) {
       const int x = x_start + off;
       if (x < w) {
         accum_ref[off] = (uint32_t)((accum_ref_tmp[off] + 32768) >> 16);
         accum_dis[off] = (uint32_t)((accum_dis_tmp[off] + 32768) >> 16);
         accum_ref_dis[off] = (uint32_t)((accum_ref_dis_tmp[off] + 32768) >> 16);
         vif_statistic_calculation<uint32_t>(
             accum_mu1[off], accum_mu2[off], accum_ref[off], accum_dis[off],
             accum_ref_dis[off], x, w, h, vif_enhn_gain_limit, thread_accum);
       }
     }
 
     // reduce sums for each warp
     for (int i = 0; i < 7; ++i) {
       thread_accum_i64[i] = warp_reduce(thread_accum_i64[i]);
     }
     const int warp_id = threadIdx.x % threads_per_warp;
     // each warp writes its sum to global mem
     if (warp_id == 0) {
       for (int i = 0; i < 7; ++i) {
         atomicAdd_int64(&reinterpret_cast<int64_t *>(accum)[i],
                         thread_accum_i64[i]);
       }
     }
 
     uint16_t *ref = (uint16_t *)buf.ref;
     uint16_t *dis = (uint16_t *)buf.dis;
     const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
     for (int off = 0; off < val_per_thread; ++off) {
       const int x = x_start + off;
       if (y < h && x < w) {
         if ((y % 2) == 0 && (off % 2) == 0) {
           const ptrdiff_t rd_stride = buf.stride / sizeof(uint16_t);
           ref[(y / 2) * rd_stride + (x / 2)] =
               (uint16_t)((accum_ref_rd[off / 2] + 32768) >> 16);
           dis[(y / 2) * rd_stride + (x / 2)] =
               (uint16_t)((accum_dis_rd[off / 2] + 32768) >> 16);
         }
       }
     }
   }
 }
 
 template <typename alignment_type = uint2, int fwidth, int fwidth_rd, int scale>
 __device__ __forceinline__ void
 filter1d_16_vertical_kernel(VifBufferCuda buf, uint16_t* ref_in, uint16_t* dis_in, int w, int h,
                             int32_t add_shift_round_VP, int32_t shift_VP,
                             int32_t add_shift_round_VP_sq, int32_t shift_VP_sq,
                             filter_table_stuct vif_filt) {
   using writeback_type = uint4;
   constexpr int val_per_thread = sizeof(alignment_type) / sizeof(uint16_t);
   static_assert(val_per_thread % 4 == 0 && val_per_thread <= 8,
                 "val per thread bust be divisible by 4 and under 16");
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
   if (x_start < w && y < h) {
     __align__(sizeof(writeback_type)) uint32_t accum_mu1[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_mu2[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_ref_rd[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_dis_rd[val_per_thread] = {0};
     uint64_t accum_ref[val_per_thread] = {0};
     uint64_t accum_dis[val_per_thread] = {0};
     uint64_t accum_ref_dis[val_per_thread] = {0};
     union {
       uint16_t ref[val_per_thread];
       alignment_type ref_aligned;
     };
     union {
       uint16_t dis[val_per_thread];
       alignment_type dis_aligned;
     };
     const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
     for (int fi = 0; fi < fwidth; ++fi) {
       int ii = y - fwidth / 2;
       int ii_check = abs(ii + fi);
       if (ii_check >= h) {
         ii_check = 2 * h - ii_check - 2;
       }
       ref_aligned = *reinterpret_cast<alignment_type *>(
           &(ref_in)[ii_check * stride + x_start]);
       dis_aligned = *reinterpret_cast<alignment_type *>(
           &(dis_in)[ii_check * stride + x_start]);
       for (int off = 0; off < val_per_thread; ++off) {
         const int j = x_start + off;
         if (j < w) {
           const uint16_t fcoeff = vif_filt.filter[scale][fi];
           uint32_t imgcoeff_ref = ref[off];
           uint32_t imgcoeff_dis = dis[off];
           uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
           uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
           accum_mu1[off] += img_coeff_ref;
           accum_mu2[off] += img_coeff_dis;
           accum_ref[off] += img_coeff_ref * (uint64_t)imgcoeff_ref;
           accum_dis[off] += img_coeff_dis * (uint64_t)imgcoeff_dis;
           accum_ref_dis[off] += img_coeff_ref * (uint64_t)imgcoeff_dis;
           if (fi >= (fwidth - fwidth_rd) / 2 &&
               fi < (fwidth - (fwidth_rd - fwidth_rd) / 2) && fwidth_rd > 0) {
             const uint16_t fcoeff_rd =
                 vif_filt.filter[scale + 1][fi - ((fwidth - fwidth_rd) / 2)];
             accum_ref_rd[off] += fcoeff_rd * imgcoeff_ref;
             accum_dis_rd[off] += fcoeff_rd * imgcoeff_dis;
           }
         }
       }
     }
     const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
 
     __align__(sizeof(writeback_type)) uint32_t accum_ref_32[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_dis_32[val_per_thread] = {0};
     __align__(sizeof(writeback_type)) uint32_t accum_ref_dis_32[val_per_thread] = {0};
     for (int off = 0; off < val_per_thread; ++off) {
       accum_mu1[off] =
           (uint16_t)((accum_mu1[off] + add_shift_round_VP) >> shift_VP);
       accum_mu2[off] =
           (uint16_t)((accum_mu2[off] + add_shift_round_VP) >> shift_VP);
       accum_ref_32[off] =
           (uint32_t)((accum_ref[off] + add_shift_round_VP_sq) >> shift_VP_sq);
       accum_dis_32[off] =
           (uint32_t)((accum_dis[off] + add_shift_round_VP_sq) >> shift_VP_sq);
       accum_ref_dis_32[off] =
           (uint32_t)((accum_ref_dis[off] + add_shift_round_VP_sq) >>
                      shift_VP_sq);
       if (fwidth_rd > 0) {
         accum_ref_rd[off] =
             (uint16_t)((accum_ref_rd[off] + add_shift_round_VP) >> shift_VP);
         accum_dis_rd[off] =
             (uint16_t)((accum_dis_rd[off] + add_shift_round_VP) >> shift_VP);
       }
     }
 
     for (int idx = 0; idx < val_per_thread; idx += sizeof(writeback_type) / sizeof(uint32_t)) {
       const int buffer_idx = y * stride_tmp + x_start + idx;
       if (x_start + idx < w) {
         *reinterpret_cast<writeback_type *>(&buf.tmp.mu1[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_mu1[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.mu2[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_mu2[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.ref[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_ref_32[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.dis[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_dis_32[idx]);
         *reinterpret_cast<writeback_type *>(&buf.tmp.ref_dis[buffer_idx]) =
             *reinterpret_cast<writeback_type *>(&accum_ref_dis_32[idx]);
         if (fwidth_rd > 0) {
           *reinterpret_cast<writeback_type *>(&buf.tmp.ref_convol[buffer_idx]) =
               *reinterpret_cast<writeback_type *>(&accum_ref_rd[idx]);
           *reinterpret_cast<writeback_type *>(&buf.tmp.dis_convol[buffer_idx]) =
               *reinterpret_cast<writeback_type *>(&accum_dis_rd[idx]);
         }
       }
     }
   }
 }
 
 template <int val_per_thread = 2, int fwidth, int fwidth_rd, int scale>
 __device__ __forceinline__ void
 filter1d_16_horizontal_kernel(VifBufferCuda buf, int w, int h,
                               int32_t add_shift_round_HP, int32_t shift_HP,
                               filter_table_stuct vif_filt,
                               double vif_enhn_gain_limit, vif_accums *accum) {
   static_assert(val_per_thread % 2 == 0,
                 "val_per_thread must be divisible by 2");
 
   int y = blockIdx.y;
   int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
   if (x_start < w && y < h) {
     union {
       vif_accums thread_accum;
       int64_t thread_accum_i64[7] = {0};
     };
 
     uint32_t accum_mu1[val_per_thread] = {0};
     uint32_t accum_mu2[val_per_thread] = {0};
     uint32_t accum_ref[val_per_thread] = {0};
     uint32_t accum_dis[val_per_thread] = {0};
     uint32_t accum_ref_dis[val_per_thread] = {0};
     uint64_t accum_ref_tmp[val_per_thread] = {0};
     uint64_t accum_dis_tmp[val_per_thread] = {0};
     uint64_t accum_ref_dis_tmp[val_per_thread] = {0};
 
     uint32_t accum_ref_rd[val_per_thread / 2] = {0};
     uint32_t accum_dis_rd[val_per_thread / 2] = {0};
 
     const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
 #pragma unroll
     for (int fj = 0; fj < fwidth; ++fj) {
 #pragma unroll
       for (int off = 0; off < val_per_thread; ++off) {
         const int j = x_start + off;
         if (j < w) {
           int jj = j - fwidth / 2;
           int jj_check = abs(jj + fj);
           if (jj_check >= w) {
             jj_check = 2 * w - jj_check - 2;
           }
           const uint16_t fcoeff = vif_filt.filter[scale][fj];
           accum_mu1[off] +=
               fcoeff * ((uint32_t)buf.tmp.mu1[y * stride_tmp + jj_check]);
           accum_mu2[off] +=
               fcoeff * ((uint32_t)buf.tmp.mu2[y * stride_tmp + jj_check]);
           accum_ref_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.ref[y * stride_tmp + jj_check]);
           accum_dis_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.dis[y * stride_tmp + jj_check]);
           accum_ref_dis_tmp[off] +=
               fcoeff * ((uint64_t)buf.tmp.ref_dis[y * stride_tmp + jj_check]);
 
           if (fj >= (fwidth - fwidth_rd) / 2 &&
               fj < (fwidth - (fwidth - fwidth_rd) / 2) && fwidth_rd > 0 &&
               off % 2 == 0) {
             const uint16_t fcoeff_rd =
                 vif_filt.filter[scale + 1][fj - ((fwidth - fwidth_rd) / 2)];
             accum_ref_rd[off / 2] +=
                 fcoeff_rd *
                 ((uint32_t)buf.tmp.ref_convol[y * stride_tmp + jj_check]);
             accum_dis_rd[off / 2] +=
                 fcoeff_rd *
                 ((uint32_t)buf.tmp.dis_convol[y * stride_tmp + jj_check]);
           }
         }
       }
     }
     for (int off = 0; off < val_per_thread; ++off) {
       const int x = x_start + off;
       if (x < w) {
         accum_ref[off] =
             (uint32_t)((accum_ref_tmp[off] + add_shift_round_HP) >> shift_HP);
         accum_dis[off] =
             (uint32_t)((accum_dis_tmp[off] + add_shift_round_HP) >> shift_HP);
         accum_ref_dis[off] =
             (uint32_t)((accum_ref_dis_tmp[off] + add_shift_round_HP) >>
                        shift_HP);
         vif_statistic_calculation<uint32_t>(
             accum_mu1[off], accum_mu2[off], accum_ref[off], accum_dis[off],
             accum_ref_dis[off], x, w, h, vif_enhn_gain_limit, thread_accum);
       }
     }
 
     // reduce sums for each warp
     for (int i = 0; i < 7; ++i) {
       thread_accum_i64[i] = warp_reduce(thread_accum_i64[i]);
     }
     const int warp_id = threadIdx.x % threads_per_warp;
     // each warp writes its sum to global mem
     if (warp_id == 0) {
       for (int i = 0; i < 7; ++i) {
         atomicAdd_int64(&reinterpret_cast<int64_t *>(accum)[i],
                         thread_accum_i64[i]);
       }
     }
 
     const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
     for (int off = 0; off < val_per_thread; ++off) {
       const int x = x_start + off;
       if (y < h && x < w) {
         if ((y % 2) == 0 && (off % 2) == 0) {
           uint16_t *ref = (uint16_t *)buf.ref;
           uint16_t *dis = (uint16_t *)buf.dis;
           const ptrdiff_t rd_stride = buf.stride / sizeof(uint16_t);
           ref[(y / 2) * rd_stride + (x / 2)] =
               (uint16_t)((accum_ref_rd[off / 2] + 32768) >> 16);
           dis[(y / 2) * rd_stride + (x / 2)] =
               (uint16_t)((accum_dis_rd[off / 2] + 32768) >> 16);
         }
       }
     }
   }
 }

#define FILTER1D_8_VERT(alignment_type, fwidth_0, fwidth_1)                                \
  __global__ void filter1d_8_vertical_kernel_##alignment_type##_##fwidth_0##_##fwidth_1 (   \
    VifBufferCuda buf, uint8_t* ref_in, uint8_t* dis_in,                                    \
      int w, int h, filter_table_stuct vif_filt_s0)                                         \
    {                                                                                       \
      filter1d_8_vertical_kernel<alignment_type, fwidth_0, fwidth_1>(                       \
        buf, ref_in, dis_in, w, h, vif_filt_s0);                                            \
    }

#define FILTER1D_8_HORI(val_per_thread, fwidth_0, fwidth_1)                                 \
  __global__ void filter1d_8_horizontal_kernel_##val_per_thread##_##fwidth_0##_##fwidth_1 (   \
  VifBufferCuda buf, int w, int h,  filter_table_stuct vif_filt_s0,                         \
    double vif_enhn_gain_limit,  vif_accums *accum)                                         \
  {                                                                                         \
    filter1d_8_horizontal_kernel<val_per_thread, fwidth_0, fwidth_1>(                       \
      buf, w, h, vif_filt_s0, vif_enhn_gain_limit, accum);                                 \
  }

#define FILTER1D_16_VERT(alignment_type, fwidth, fwidth_rd, scale)                                    \
  __global__ void filter1d_16_vertical_kernel_##alignment_type##_##fwidth##_##fwidth_rd##_##scale (   \
    VifBufferCuda buf, uint16_t* ref_in, uint16_t* dis_in, int w, int h,                              \
      int32_t add_shift_round_VP, int32_t shift_VP,                                                   \
      int32_t add_shift_round_VP_sq, int32_t shift_VP_sq,                                             \
      filter_table_stuct vif_filt)                                                                    \
    {                                                                                                 \
      filter1d_16_vertical_kernel<alignment_type, fwidth, fwidth_rd, scale>(                          \
        buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP,                                      \
        add_shift_round_VP_sq, shift_VP_sq, vif_filt);                                                \
    }

#define FILTER1D_16_HORI(val_per_thread, fwidth, fwidth_rd, scale)                                      \
  __global__ void filter1d_16_horizontal_kernel_##val_per_thread##_##fwidth##_##fwidth_rd##_##scale (   \
    VifBufferCuda buf, int w, int h,                                                                    \
      int32_t add_shift_round_HP, int32_t shift_HP,                                                     \
      filter_table_stuct vif_filt,                                                                      \
      double vif_enhn_gain_limit, vif_accums *accum)                                                    \
    {                                                                                                   \
      filter1d_16_horizontal_kernel<val_per_thread, fwidth, fwidth_rd, scale>(                          \
        buf, w, h, add_shift_round_HP, shift_HP,                                                        \
        vif_filt, vif_enhn_gain_limit, accum);                                                          \
    }

extern "C" {
  // constexpr int fwidth[4] = {17, 9, 5, 3};
  FILTER1D_8_VERT(uint32_t, 17, 9);   // filter1d_8_vertical_kernel_uint32_t_17_9
  FILTER1D_8_HORI(2, 17, 9);          // filter1d_8_horizontal_kernel_2_17_9
  FILTER1D_16_VERT(uint2, 17, 9, 0);  // filter1d_16_vertical_kernel_uint2_17_9_0
  FILTER1D_16_VERT(uint2, 9, 5, 1);   // filter1d_16_vertical_kernel_uint2_9_5_1
  FILTER1D_16_VERT(uint2, 5, 3, 2);   // filter1d_16_vertical_kernel_uint2_5_3_2
  FILTER1D_16_VERT(uint2, 3, 0, 3);   // filter1d_16_vertical_kernel_uint2_3_0_3

  FILTER1D_16_HORI(2, 17, 9, 0);      // filter1d_16_horizontal_kernel_2_17_9_0
  FILTER1D_16_HORI(2, 9, 5, 1);       // filter1d_16_horizontal_kernel_2_9_5_1
  FILTER1D_16_HORI(2, 5, 3, 2);       // filter1d_16_horizontal_kernel_2_5_3_2
  FILTER1D_16_HORI(2, 3, 0, 3);       // filter1d_16_horizontal_kernel_2_3_0_3

}
    

// extern "C" {
 
//  void filter1d_8(VifBufferCuda *buf, uint8_t* ref_in, uint8_t* dis_in, int w, int h, double vif_enhn_gain_limit,
//                  CUstream stream) {
//    {
//      using alignment_type = uint32_t;
//      dim3 block(cache_line_size / sizeof(alignment_type),
//                 128 / (cache_line_size / sizeof(alignment_type)));
//      dim3 grid(DIV_ROUND_UP(w, block.x * sizeof(alignment_type)),
//                DIV_ROUND_UP(h, block.y));
 
//       filter1d_8_vertical_kernel_uint32_t_17_9<<<grid, block, 0, stream>>>(
//          *buf, ref_in, dis_in, w, h, *(filter_table_stuct *)vif_filter1d_table);
//      CudaCheckError();
//    }
//    {
//      constexpr int val_per_thread = 2;
//      dim3 block(128);
//      dim3 grid(DIV_ROUND_UP(w, block.x * val_per_thread),
//                DIV_ROUND_UP(h, block.y));
 
//                filter1d_8_horizontal_kernel_2_17_9<<<grid, block, 0, stream>>>(
//          *buf, w, h, *(filter_table_stuct *)vif_filter1d_table,
//          vif_enhn_gain_limit, &((vif_accums *)buf->accum)[0]);
//      CudaCheckError();
//    }
//  }
 
//  void filter1d_16(VifBufferCuda *buf, uint16_t* ref_in, uint16_t* dis_in, int w, int h, int scale, int bpc,
//                   double vif_enhn_gain_limit, CUstream stream) {
 
//    int32_t add_shift_round_HP, shift_HP;
//    int32_t add_shift_round_VP, shift_VP;
//    int32_t add_shift_round_VP_sq, shift_VP_sq;
//    if (scale == 0) {
//      shift_HP = 16;
//      add_shift_round_HP = 32768;
//      shift_VP = bpc;
//      add_shift_round_VP = 1 << (bpc - 1);
//      shift_VP_sq = (bpc - 8) * 2;
//      add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
//    } else {
//      shift_HP = 16;
//      add_shift_round_HP = 32768;
//      shift_VP = 16;
//      add_shift_round_VP = 32768;
//      shift_VP_sq = 16;
//      add_shift_round_VP_sq = 32768;
//    }
 
//    constexpr int fwidth[4] = {17, 9, 5, 3};
 
//    using alignment_type = uint2;
//    dim3 block(128);
//    constexpr int val_per_thread = sizeof(alignment_type) / sizeof(uint16_t);
//    constexpr int val_per_thread_horizontal = 2;
//    dim3 block_vertical(cache_line_size / val_per_thread,
//                        128 / (cache_line_size / val_per_thread));
//    dim3 grid_vertical(DIV_ROUND_UP(w, block.x), h);
//    dim3 grid_horizontal(DIV_ROUND_UP(w, block.x), h);
 
//    switch (scale) {
//    case 0: {
//     filter1d_16_vertical_kernel_uint2_17_9_0
//          <<<grid_vertical, block_vertical, 0, stream>>>(
//              *buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP, add_shift_round_VP_sq,
//              shift_VP_sq, *(filter_table_stuct *)vif_filter1d_table);
 
//      CudaCheckError();
 
//      filter1d_16_horizontal_kernel_2_17_9_0
//          <<<grid_horizontal, block, 0, stream>>>(
//              *buf, w, h, add_shift_round_HP, shift_HP,
//              *(filter_table_stuct *)vif_filter1d_table, vif_enhn_gain_limit,
//              &((vif_accums *)buf->accum)[scale]);
//      CudaCheckError();
//      break;
//    }
//    case 1: {
//     filter1d_16_vertical_kernel_uint2_9_5_1
//          <<<grid_vertical, block_vertical, 0, stream>>>(
//              *buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP, add_shift_round_VP_sq,
//              shift_VP_sq, *(filter_table_stuct *)vif_filter1d_table);
 
//      CudaCheckError();
 
//      filter1d_16_horizontal_kernel_2_9_5_1
//          <<<grid_horizontal, block, 0, stream>>>(
//              *buf, w, h, add_shift_round_HP, shift_HP,
//              *(filter_table_stuct *)vif_filter1d_table, vif_enhn_gain_limit,
//              &((vif_accums *)buf->accum)[scale]);
//      CudaCheckError();
//      break;
//    }
//    case 2: {
//     filter1d_16_vertical_kernel_uint2_5_3_2
//          <<<grid_vertical, block_vertical, 0, stream>>>(
//              *buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP, add_shift_round_VP_sq,
//              shift_VP_sq, *(filter_table_stuct *)vif_filter1d_table);
 
//      CudaCheckError();
 
//      filter1d_16_horizontal_kernel_2_5_3_2
//          <<<grid_horizontal, block, 0, stream>>>(
//              *buf, w, h, add_shift_round_HP, shift_HP,
//              *(filter_table_stuct *)vif_filter1d_table, vif_enhn_gain_limit,
//              &((vif_accums *)buf->accum)[scale]);
//      CudaCheckError();
//      break;
//    }
//    case 3: {
//     filter1d_16_vertical_kernel_uint2_3_0_3
//          <<<grid_vertical, block_vertical, 0, stream>>>(
//              *buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP, add_shift_round_VP_sq,
//              shift_VP_sq, *(filter_table_stuct *)vif_filter1d_table);
 
//      CudaCheckError();
 
//      filter1d_16_horizontal_kernel_2_3_0_3
//          <<<grid_horizontal, block, 0, stream>>>(
//              *buf, w, h, add_shift_round_HP, shift_HP,
//              *(filter_table_stuct *)vif_filter1d_table, vif_enhn_gain_limit,
//              &((vif_accums *)buf->accum)[scale]);
//      CudaCheckError();
//      break;
//    }
//    }
//  }
// }
 