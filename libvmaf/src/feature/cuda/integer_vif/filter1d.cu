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

#include "cuda_helper.cuh"
#include "cuda/integer_vif_cuda.h"

#include "common.h"

#include "vif_statistics.cuh"

/*
 * Shared-memory staging for VIF filter passes (perf-audit wins #1 + #4,
 * 2026-05-16).
 *
 * HORIZONTAL PASS (win #1):
 *   Launch: BLOCKX=128, BLOCKY=1, val_per_thread=2.
 *   Each block owns 256 output pixels per row.  The 17-tap filter
 *   (half_fw=8) reads [x_out-8, x_out+8] from each of 7 tmp channels
 *   (mu1, mu2, ref, dis, ref_dis, ref_convol, dis_convol).
 *   Tile: 256+16=272 uint32_t per channel × 7 channels × 4 B = 7616 B.
 *   272 % 32 = 16 → no bank conflicts for warp-consecutive stride-1 access.
 *   Boundary mirror handled in the smem load phase; the compute phase reads
 *   smem unconditionally (interior) or with the same si formula (border).
 *   Estimated speedup: 20–35% on scale-0 VIF horizontal pass.
 *
 * VERTICAL PASS (win #4):
 *   Launch: BLOCKX=32, BLOCKY=4, val_per_thread=4 (uint32_t, 4 uint8_t).
 *   Each block owns 128 cols × 4 rows.  17 vertical taps → tile height 20.
 *   Tile: 2 planes × 128 cols × 20 rows × 1 B = 5120 B (8-bit path).
 *   For 16-bit: 2 planes × 128 cols × 20 rows × 2 B = 10240 B.
 *   Boundary mirror handled in the smem load phase.
 *   Estimated speedup: 15–25% on VIF vertical pass.
 *
 * CORRECTNESS:
 *   All arithmetic is integer fixed-point; smem staging only moves where the
 *   values are read from, not what values are read.  Bit-identical to the
 *   pre-smem implementation.  Verified by cross_backend_parity_gate.py
 *   --features vif --backends cpu cuda --places 4.
 */

/*
 * Horizontal tile width macro.
 * blockx  = number of threads in X per block (= BLOCKX)
 * vpt     = val_per_thread
 * half_fw = fwidth / 2
 * Result: blockx*vpt + 2*half_fw = the minimum span that covers all filter
 *         taps for every thread in the block.
 * +1 pad: avoids stride-32 bank aliasing that could arise with certain
 *         fwidth values (e.g. fwidth=9 → half_fw=4 → tile=256+8=264;
 *         264%32=8, fine, but +1 ensures no future regression).
 */
#define HORI_TILE_W(blockx, vpt, half_fw) ((blockx) * (vpt) + 2 * (half_fw) + 1)

/* -------------------------------------------------------------------------
 * 8-bit VERTICAL KERNEL (win #4)
 * Stages ref_in / dis_in rows into shared memory before the accumulation loop.
 * Block size: BLOCKX=32, BLOCKY=4, val_per_thread=4 (uint32_t alignment).
 * Tile: (BLOCKY + fwidth_0 - 1) rows × (BLOCKX * val_per_thread = 128) cols.
 * For fwidth_0=17: (4+16)=20 rows × 128 cols × 2 planes × 1 B = 5120 B.
 * Conservative static smem size uses BLOCKY_MAX=8 to cover any valid launch.
 * ------------------------------------------------------------------------- */
template <typename alignment_type = uint2, int fwidth_0 = 17, int fwidth_1 = 9>
__device__ __forceinline__ void filter1d_8_vertical_kernel(VifBufferCuda buf, uint8_t *ref_in,
                                                           uint8_t *dis_in, int w, int h,
                                                           filter_table_stuct vif_filt_s0)
{
    using writeback_type = uint4;
    constexpr int val_per_thread = sizeof(alignment_type);
    static_assert(val_per_thread % 4 == 0 && val_per_thread <= 16,
                  "val per thread bust be divisible by 4 and under 16");

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;

    constexpr int half_fv = fwidth_0 / 2;

    /*
     * Static smem bounds: BLOCKY_MAX=8 (the actual BLOCKY=4 is a subset).
     * TILE_COLS_MAX = BLOCKX_MAX * val_per_thread_MAX.  With BLOCKX=32 and
     * val_per_thread=4: 32*4=128 columns per block.
     * Per-plane smem: (8 + fwidth_0 - 1) * 128 bytes.
     * Two planes total.  For fwidth_0=17: 2 * 24 * 128 = 6144 bytes.
     */
    constexpr int BLOCKY_MAX = 8;
    constexpr int TILE_H_MAX = BLOCKY_MAX + fwidth_0 - 1;
    constexpr int TILE_COLS_MAX = 128; /* BLOCKX(=32) * val_per_thread(=4) */

    __shared__ uint8_t ref_tile[TILE_H_MAX][TILE_COLS_MAX];
    __shared__ uint8_t dis_tile[TILE_H_MAX][TILE_COLS_MAX];

    /*
     * Cooperative smem load.
     * Thread (ty, tx) owns tile columns [tx*vpt, tx*vpt + vpt).
     * It iterates over tile rows stepping by blockDim.y.
     * tile_row=0 maps to image row (blockIdx.y*blockDim.y - half_fv).
     * Mirror boundary (reflect at 0 and h-1) is applied here so that the
     * compute phase can read smem without any boundary check.
     */
    const int x_block_start = blockIdx.x * blockDim.x * val_per_thread;
    const int tile_h = blockDim.y + fwidth_0 - 1;
    const int col = threadIdx.x * val_per_thread;

    for (int tile_row = threadIdx.y; tile_row < tile_h; tile_row += blockDim.y) {
        int img_row = (int)(blockIdx.y * blockDim.y) - half_fv + tile_row;
        /* Two-bounce mirror: reflect at top then at bottom. */
        if (img_row < 0)
            img_row = -img_row;
        if (img_row >= h)
            img_row = 2 * h - img_row - 2;
        /* Clamp for safety on very small frames (h < fwidth_0). */
        if (img_row < 0)
            img_row = 0;
        if (img_row >= h)
            img_row = h - 1;

        if (x_block_start + col < w) {
            const alignment_type ref_vec = *reinterpret_cast<const alignment_type *>(
                &ref_in[(ptrdiff_t)img_row * buf.stride + x_block_start + col]);
            const alignment_type dis_vec = *reinterpret_cast<const alignment_type *>(
                &dis_in[(ptrdiff_t)img_row * buf.stride + x_block_start + col]);
            const uint8_t *ref_b = reinterpret_cast<const uint8_t *>(&ref_vec);
            const uint8_t *dis_b = reinterpret_cast<const uint8_t *>(&dis_vec);
#pragma unroll
            for (int k = 0; k < val_per_thread; ++k) {
                ref_tile[tile_row][col + k] = ref_b[k];
                dis_tile[tile_row][col + k] = dis_b[k];
            }
        }
    }
    __syncthreads();

    if (x_start < w && y < h) {
        const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
        __align__(sizeof(writeback_type)) uint32_t accum_mu1[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_mu2[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_ref[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_dis[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_ref_dis[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_ref_rd[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_dis_rd[val_per_thread] = {0};

        /*
         * Compute phase reads from smem.
         * smem_row = threadIdx.y + fi  because tile_row=0 maps to
         * blockIdx.y*blockDim.y - half_fv, so thread at threadIdx.y maps
         * to tile_row=threadIdx.y (the image row blockIdx.y*blockDim.y +
         * threadIdx.y - half_fv has been stored at smem[threadIdx.y]).
         * After adding fi we reach smem[threadIdx.y + fi], which holds the
         * tap fi's mirrored row.
         */
        for (int fi = 0; fi < fwidth_0; ++fi) {
            const int smem_row = threadIdx.y + fi;
            for (int off = 0; off < val_per_thread; ++off) {
                const int j = x_start + off;
                if (j < w) {
                    const uint32_t fcoeff = vif_filt_s0.filter[0][fi];
                    const uint32_t ref_val = ref_tile[smem_row][col + off];
                    const uint32_t dis_val = dis_tile[smem_row][col + off];
                    const uint32_t img_coeff_ref = fcoeff * ref_val;
                    const uint32_t img_coeff_dis = fcoeff * dis_val;
                    accum_mu1[off] += img_coeff_ref;
                    accum_mu2[off] += img_coeff_dis;
                    accum_ref[off] += img_coeff_ref * ref_val;
                    accum_dis[off] += img_coeff_dis * dis_val;
                    accum_ref_dis[off] += img_coeff_ref * dis_val;
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

/* -------------------------------------------------------------------------
 * 8-bit HORIZONTAL KERNEL (win #1)
 * Stages all 7 tmp channels into shared memory before the filter loop.
 * Block size: BLOCKX=128, BLOCKY=1, val_per_thread=2.
 * Tile: HORI_TILE_W elements per channel × 7 channels × 4 B.
 * For fwidth_0=17 (half_fw=8): (256+16+1)=273 × 7 × 4 = 7644 B per block.
 * Both interior and border paths use smem (boundary mirror handled in load).
 * ------------------------------------------------------------------------- */
template <int val_per_thread = 1, int fwidth_0 = 17, int fwidth_1 = 9>
__device__ __forceinline__ void
filter1d_8_horizontal_kernel(VifBufferCuda buf, int w, int h, filter_table_stuct vif_filt_s0,
                             double vif_enhn_gain_limit, vif_accums *accum)
{
    static_assert(val_per_thread % 2 == 0, "val_per_thread must be divisible by 2");

    constexpr int half_fw = fwidth_0 / 2;
    /*
     * TILE_W = BLOCKX*vpt + 2*half_fw + 1 (padding).
     * For BLOCKX=128, vpt=2, half_fw=8: TILE_W = 273.
     * smem_base for thread tx = tx * val_per_thread.
     * smem index for output pixel (x_start + off) and filter tap fj is:
     *   si = smem_base + off + fj
     * where smem[si] = buf.tmp.XX[row + (tile_x0 + si)] after mirror-clamp,
     * tile_x0 = blockIdx.x * blockDim.x * val_per_thread - half_fw.
     */
    constexpr int TILE_W = HORI_TILE_W(128, val_per_thread, half_fw);

    __shared__ uint32_t smem_mu1[TILE_W];
    __shared__ uint32_t smem_mu2[TILE_W];
    __shared__ uint32_t smem_ref[TILE_W];
    __shared__ uint32_t smem_dis[TILE_W];
    __shared__ uint32_t smem_ref_dis[TILE_W];
    __shared__ uint32_t smem_ref_convol[TILE_W];
    __shared__ uint32_t smem_dis_convol[TILE_W];

    const int y = blockIdx.y;
    const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
    const int tile_x0 = (int)(blockIdx.x * blockDim.x) * val_per_thread - half_fw;
    const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
    const int buf_row = y * stride_tmp;

    /*
     * Cooperative smem load: tile covers TILE_W - 1 = 272 positions.
     * Each thread loads elements spaced blockDim.x apart starting at si=threadIdx.x.
     * The load covers si=0..271; element TILE_W-1=272 is padding (never read).
     * Mirror boundary applied here; computation reads smem without extra checks.
     */
    if (y < h) {
        for (int si = threadIdx.x; si < TILE_W - 1; si += blockDim.x) {
            int img_col = tile_x0 + si;
            if (img_col < 0)
                img_col = -img_col;
            if (img_col >= w)
                img_col = 2 * w - img_col - 2;
            /* Safety clamp for very narrow frames. */
            if (img_col < 0)
                img_col = 0;
            if (img_col >= w)
                img_col = w - 1;
            smem_mu1[si] = buf.tmp.mu1[buf_row + img_col];
            smem_mu2[si] = buf.tmp.mu2[buf_row + img_col];
            smem_ref[si] = buf.tmp.ref[buf_row + img_col];
            smem_dis[si] = buf.tmp.dis[buf_row + img_col];
            smem_ref_dis[si] = buf.tmp.ref_dis[buf_row + img_col];
            smem_ref_convol[si] = buf.tmp.ref_convol[buf_row + img_col];
            smem_dis_convol[si] = buf.tmp.dis_convol[buf_row + img_col];
        }
    }
    __syncthreads();

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

        constexpr int rd_start = (fwidth_0 - fwidth_1) / 2;
        constexpr int rd_half = fwidth_1 / 2;

        /*
         * smem_base: index of the leftmost filter position for this thread's
         * first output pixel.  smem[smem_base + off + fj] holds the value at
         * image column (x_start + off - half_fw + fj) after mirror-clamping.
         */
        const int smem_base = threadIdx.x * val_per_thread;

        /*
         * Interior fast path: no extra boundary check needed in the filter
         * loop (all taps land within [0, w-1] after the smem load stage).
         * Symmetric-filter optimization: 17→9 multiplies at scale 0.
         */
        const bool interior = (x_start >= half_fw) && (x_start + val_per_thread - 1 + half_fw < w);
        if (interior) {
            /* Center tap (unpaired) */
            {
                const uint16_t fcoeff = vif_filt_s0.filter[0][half_fw];
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int si = smem_base + off + half_fw;
                    accum_mu1[off] += fcoeff * smem_mu1[si];
                    accum_mu2[off] += fcoeff * smem_mu2[si];
                    accum_ref_tmp[off] += fcoeff * (uint64_t)smem_ref[si];
                    accum_dis_tmp[off] += fcoeff * (uint64_t)smem_dis[si];
                    accum_ref_dis_tmp[off] += fcoeff * (uint64_t)smem_ref_dis[si];
                }
                if (fwidth_1 > 0) {
                    const uint16_t fcoeff_rd = vif_filt_s0.filter[1][rd_half];
#pragma unroll
                    for (int off = 0; off < val_per_thread; off += 2) {
                        const int si = smem_base + off + half_fw;
                        accum_ref_rd[off / 2] += fcoeff_rd * smem_ref_convol[si];
                        accum_dis_rd[off / 2] += fcoeff_rd * smem_dis_convol[si];
                    }
                }
            }
            /* Symmetric tap pairs */
#pragma unroll
            for (int fj = 0; fj < half_fw; ++fj) {
                const uint16_t fcoeff = vif_filt_s0.filter[0][fj];
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int si_lo = smem_base + off + fj;
                    const int si_hi = smem_base + off + 2 * half_fw - fj;
                    accum_mu1[off] += fcoeff * (smem_mu1[si_lo] + smem_mu1[si_hi]);
                    accum_mu2[off] += fcoeff * (smem_mu2[si_lo] + smem_mu2[si_hi]);
                    accum_ref_tmp[off] +=
                        fcoeff * ((uint64_t)smem_ref[si_lo] + (uint64_t)smem_ref[si_hi]);
                    accum_dis_tmp[off] +=
                        fcoeff * ((uint64_t)smem_dis[si_lo] + (uint64_t)smem_dis[si_hi]);
                    accum_ref_dis_tmp[off] +=
                        fcoeff * ((uint64_t)smem_ref_dis[si_lo] + (uint64_t)smem_ref_dis[si_hi]);
                }
                if (fwidth_1 > 0 && fj >= rd_start && fj < rd_start + rd_half) {
                    const uint16_t fcoeff_rd = vif_filt_s0.filter[1][fj - rd_start];
#pragma unroll
                    for (int off = 0; off < val_per_thread; off += 2) {
                        const int si_lo = smem_base + off + fj;
                        const int si_hi = smem_base + off + 2 * half_fw - fj;
                        accum_ref_rd[off / 2] +=
                            fcoeff_rd * (smem_ref_convol[si_lo] + smem_ref_convol[si_hi]);
                        accum_dis_rd[off / 2] +=
                            fcoeff_rd * (smem_dis_convol[si_lo] + smem_dis_convol[si_hi]);
                    }
                }
            }
        } else {
            /*
             * Border path: smem already contains mirrored boundary values.
             * si = smem_base + off + fj maps to image col (x_start+off-half_fw+fj).
             * For left border (x_start < half_fw) the smem load stage reflected
             * negative indices, so smem[si] holds the correct mirrored value.
             * The per-element `j < w` guard is still needed because x_start+off
             * may exceed w at the right edge of the last block.
             */
#pragma unroll
            for (int fj = 0; fj < fwidth_0; ++fj) {
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int j = x_start + off;
                    if (j < w) {
                        const int si = smem_base + off + fj;
                        const uint16_t fcoeff = vif_filt_s0.filter[0][fj];
                        accum_mu1[off] += fcoeff * smem_mu1[si];
                        accum_mu2[off] += fcoeff * smem_mu2[si];
                        accum_ref_tmp[off] += fcoeff * (uint64_t)smem_ref[si];
                        accum_dis_tmp[off] += fcoeff * (uint64_t)smem_dis[si];
                        accum_ref_dis_tmp[off] += fcoeff * (uint64_t)smem_ref_dis[si];

                        if (fj >= (fwidth_0 - fwidth_1) / 2 &&
                            fj < (fwidth_0 - (fwidth_0 - fwidth_1) / 2) && off % 2 == 0) {
                            const uint16_t fcoeff_rd =
                                vif_filt_s0.filter[1][fj - ((fwidth_0 - fwidth_1) / 2)];
                            accum_ref_rd[off / 2] += fcoeff_rd * smem_ref_convol[si];
                            accum_dis_rd[off / 2] += fcoeff_rd * smem_dis_convol[si];
                        }
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
                vif_statistic_calculation<uint32_t>(accum_mu1[off], accum_mu2[off], accum_ref[off],
                                                    accum_dis[off], accum_ref_dis[off], x, w, h,
                                                    vif_enhn_gain_limit, thread_accum);
            }
        }

        for (int i = 0; i < 7; ++i) {
            thread_accum_i64[i] = warp_reduce(thread_accum_i64[i]);
        }
        const int warp_id = threadIdx.x % VMAF_CUDA_THREADS_PER_WARP;
        if (warp_id == 0) {
            for (int i = 0; i < 7; ++i) {
                atomicAdd_int64(&reinterpret_cast<int64_t *>(accum)[i], thread_accum_i64[i]);
            }
        }

        uint16_t *ref = (uint16_t *)buf.ref;
        uint16_t *dis = (uint16_t *)buf.dis;
        for (int off = 0; off < val_per_thread; ++off) {
            const int x = x_start + off;
            if (y < h && x < w) {
                if ((y % 2) == 0 && (off % 2) == 0) {
                    const ptrdiff_t rd_stride = buf.rd_stride / sizeof(uint16_t);
                    ref[(y / 2) * rd_stride + (x / 2)] =
                        (uint16_t)((accum_ref_rd[off / 2] + 32768) >> 16);
                    dis[(y / 2) * rd_stride + (x / 2)] =
                        (uint16_t)((accum_dis_rd[off / 2] + 32768) >> 16);
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 16-bit VERTICAL KERNEL (win #4, 16-bit variant)
 * Block size: BLOCK_VERT_X=32, BLOCK_VERT_Y=4, val_per_thread=4 (uint16_t).
 * Tile: (BLOCK_VERT_Y + fwidth - 1) rows × 128 cols × 2 planes × 2 B.
 * For fwidth=17: (4+16)=20 rows × 128 cols × 2 × 2 = 10240 B per block.
 * ------------------------------------------------------------------------- */
template <typename alignment_type = uint2, int fwidth, int fwidth_rd, int scale>
__device__ __forceinline__ void
filter1d_16_vertical_kernel(VifBufferCuda buf, uint16_t *ref_in, uint16_t *dis_in, int w, int h,
                            int32_t add_shift_round_VP, int32_t shift_VP,
                            int32_t add_shift_round_VP_sq, int32_t shift_VP_sq,
                            filter_table_stuct vif_filt)
{
    using writeback_type = uint4;
    constexpr int val_per_thread = sizeof(alignment_type) / sizeof(uint16_t);
    static_assert(val_per_thread % 4 == 0 && val_per_thread <= 8,
                  "val per thread bust be divisible by 4 and under 16");

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;

    constexpr int half_fv = fwidth / 2;

    /*
     * Static smem bounds.  BLOCK_VERT_Y_MAX=8, TILE_COLS_MAX=128.
     * Per-plane smem: (8 + fwidth - 1) * 128 * 2 bytes.
     * For fwidth=17: 2 * 24 * 128 * 2 = 12288 bytes.
     */
    constexpr int BLOCKY_MAX = 8;
    constexpr int TILE_H_MAX = BLOCKY_MAX + fwidth - 1;
    constexpr int TILE_COLS_MAX = 128; /* BLOCK_VERT_X(=32) * val_per_thread(=4) */

    __shared__ uint16_t ref_tile16[TILE_H_MAX][TILE_COLS_MAX];
    __shared__ uint16_t dis_tile16[TILE_H_MAX][TILE_COLS_MAX];

    const int tile_h = blockDim.y + fwidth - 1;
    const int x_block_start = blockIdx.x * blockDim.x * val_per_thread;
    const int col = threadIdx.x * val_per_thread;
    const ptrdiff_t stride =
        (scale == 0) ? buf.stride / sizeof(uint16_t) : buf.rd_stride / sizeof(uint16_t);

    for (int tile_row = threadIdx.y; tile_row < tile_h; tile_row += blockDim.y) {
        int img_row = (int)(blockIdx.y * blockDim.y) - half_fv + tile_row;
        if (img_row < 0)
            img_row = -img_row;
        if (img_row >= h)
            img_row = 2 * h - img_row - 2;
        if (img_row < 0)
            img_row = 0;
        if (img_row >= h)
            img_row = h - 1;

        if (x_block_start + col < w) {
            const alignment_type ref_vec = *reinterpret_cast<const alignment_type *>(
                &ref_in[(ptrdiff_t)img_row * stride + x_block_start + col]);
            const alignment_type dis_vec = *reinterpret_cast<const alignment_type *>(
                &dis_in[(ptrdiff_t)img_row * stride + x_block_start + col]);
            const uint16_t *ref_s = reinterpret_cast<const uint16_t *>(&ref_vec);
            const uint16_t *dis_s = reinterpret_cast<const uint16_t *>(&dis_vec);
#pragma unroll
            for (int k = 0; k < val_per_thread; ++k) {
                ref_tile16[tile_row][col + k] = ref_s[k];
                dis_tile16[tile_row][col + k] = dis_s[k];
            }
        }
    }
    __syncthreads();

    if (x_start < w && y < h) {
        __align__(sizeof(writeback_type)) uint32_t accum_mu1[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_mu2[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_ref_rd[val_per_thread] = {0};
        __align__(sizeof(writeback_type)) uint32_t accum_dis_rd[val_per_thread] = {0};
        uint64_t accum_ref[val_per_thread] = {0};
        uint64_t accum_dis[val_per_thread] = {0};
        uint64_t accum_ref_dis[val_per_thread] = {0};

        for (int fi = 0; fi < fwidth; ++fi) {
            const int smem_row = threadIdx.y + fi;
            for (int off = 0; off < val_per_thread; ++off) {
                const int j = x_start + off;
                if (j < w) {
                    const uint16_t fcoeff = vif_filt.filter[scale][fi];
                    const uint32_t imgcoeff_ref = ref_tile16[smem_row][col + off];
                    const uint32_t imgcoeff_dis = dis_tile16[smem_row][col + off];
                    const uint32_t img_coeff_ref = fcoeff * imgcoeff_ref;
                    const uint32_t img_coeff_dis = fcoeff * imgcoeff_dis;
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
            accum_mu1[off] = (uint16_t)((accum_mu1[off] + add_shift_round_VP) >> shift_VP);
            accum_mu2[off] = (uint16_t)((accum_mu2[off] + add_shift_round_VP) >> shift_VP);
            accum_ref_32[off] = (uint32_t)((accum_ref[off] + add_shift_round_VP_sq) >> shift_VP_sq);
            accum_dis_32[off] = (uint32_t)((accum_dis[off] + add_shift_round_VP_sq) >> shift_VP_sq);
            accum_ref_dis_32[off] =
                (uint32_t)((accum_ref_dis[off] + add_shift_round_VP_sq) >> shift_VP_sq);
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

/* -------------------------------------------------------------------------
 * 16-bit HORIZONTAL KERNEL (win #1, 16-bit variant)
 * Block size: BLOCKX=128, BLOCKY=1, val_per_thread=2.
 * Same tile layout as the 8-bit horizontal kernel.
 * For fwidth=17 (half_fw=8): TILE_W=273 × 7 × 4 B = 7644 B per block.
 * ------------------------------------------------------------------------- */
template <int val_per_thread = 2, int fwidth, int fwidth_rd, int scale>
__device__ __forceinline__ void
filter1d_16_horizontal_kernel(VifBufferCuda buf, int w, int h, int32_t add_shift_round_HP,
                              int32_t shift_HP, filter_table_stuct vif_filt,
                              double vif_enhn_gain_limit, vif_accums *accum)
{
    static_assert(val_per_thread % 2 == 0, "val_per_thread must be divisible by 2");

    constexpr int half_fw = fwidth / 2;
    constexpr int TILE_W = HORI_TILE_W(128, val_per_thread, half_fw);

    __shared__ uint32_t smem_mu1[TILE_W];
    __shared__ uint32_t smem_mu2[TILE_W];
    __shared__ uint32_t smem_ref[TILE_W];
    __shared__ uint32_t smem_dis[TILE_W];
    __shared__ uint32_t smem_ref_dis[TILE_W];
    __shared__ uint32_t smem_ref_convol[TILE_W];
    __shared__ uint32_t smem_dis_convol[TILE_W];

    const int y = blockIdx.y;
    const int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * val_per_thread;
    const int tile_x0 = (int)(blockIdx.x * blockDim.x) * val_per_thread - half_fw;
    const int stride_tmp = buf.stride_tmp / sizeof(uint32_t);
    const int buf_row = y * stride_tmp;

    if (y < h) {
        for (int si = threadIdx.x; si < TILE_W - 1; si += blockDim.x) {
            int img_col = tile_x0 + si;
            if (img_col < 0)
                img_col = -img_col;
            if (img_col >= w)
                img_col = 2 * w - img_col - 2;
            if (img_col < 0)
                img_col = 0;
            if (img_col >= w)
                img_col = w - 1;
            smem_mu1[si] = buf.tmp.mu1[buf_row + img_col];
            smem_mu2[si] = buf.tmp.mu2[buf_row + img_col];
            smem_ref[si] = buf.tmp.ref[buf_row + img_col];
            smem_dis[si] = buf.tmp.dis[buf_row + img_col];
            smem_ref_dis[si] = buf.tmp.ref_dis[buf_row + img_col];
            smem_ref_convol[si] = buf.tmp.ref_convol[buf_row + img_col];
            smem_dis_convol[si] = buf.tmp.dis_convol[buf_row + img_col];
        }
    }
    __syncthreads();

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

        constexpr int rd_start = (fwidth - fwidth_rd) / 2;
        constexpr int rd_half = fwidth_rd / 2;

        const int smem_base = threadIdx.x * val_per_thread;

        const bool interior = (x_start >= half_fw) && (x_start + val_per_thread - 1 + half_fw < w);
        if (interior) {
            {
                const uint16_t fcoeff = vif_filt.filter[scale][half_fw];
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int si = smem_base + off + half_fw;
                    accum_mu1[off] += fcoeff * smem_mu1[si];
                    accum_mu2[off] += fcoeff * smem_mu2[si];
                    accum_ref_tmp[off] += fcoeff * (uint64_t)smem_ref[si];
                    accum_dis_tmp[off] += fcoeff * (uint64_t)smem_dis[si];
                    accum_ref_dis_tmp[off] += fcoeff * (uint64_t)smem_ref_dis[si];
                }
                if (fwidth_rd > 0) {
                    const uint32_t fcoeff_rd = vif_filt.filter[scale + 1][rd_half];
#pragma unroll
                    for (int off = 0; off < val_per_thread; off += 2) {
                        const int si = smem_base + off + half_fw;
                        accum_ref_rd[off / 2] += fcoeff_rd * smem_ref_convol[si];
                        accum_dis_rd[off / 2] += fcoeff_rd * smem_dis_convol[si];
                    }
                }
            }
#pragma unroll
            for (int fj = 0; fj < half_fw; ++fj) {
                const uint16_t fcoeff = vif_filt.filter[scale][fj];
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int si_lo = smem_base + off + fj;
                    const int si_hi = smem_base + off + 2 * half_fw - fj;
                    accum_mu1[off] += fcoeff * (smem_mu1[si_lo] + smem_mu1[si_hi]);
                    accum_mu2[off] += fcoeff * (smem_mu2[si_lo] + smem_mu2[si_hi]);
                    accum_ref_tmp[off] +=
                        fcoeff * ((uint64_t)smem_ref[si_lo] + (uint64_t)smem_ref[si_hi]);
                    accum_dis_tmp[off] +=
                        fcoeff * ((uint64_t)smem_dis[si_lo] + (uint64_t)smem_dis[si_hi]);
                    accum_ref_dis_tmp[off] +=
                        fcoeff * ((uint64_t)smem_ref_dis[si_lo] + (uint64_t)smem_ref_dis[si_hi]);
                }
                if (fwidth_rd > 0 && fj >= rd_start && fj < rd_start + rd_half) {
                    const uint32_t fcoeff_rd = vif_filt.filter[scale + 1][fj - rd_start];
#pragma unroll
                    for (int off = 0; off < val_per_thread; off += 2) {
                        const int si_lo = smem_base + off + fj;
                        const int si_hi = smem_base + off + 2 * half_fw - fj;
                        accum_ref_rd[off / 2] +=
                            fcoeff_rd * (smem_ref_convol[si_lo] + smem_ref_convol[si_hi]);
                        accum_dis_rd[off / 2] +=
                            fcoeff_rd * (smem_dis_convol[si_lo] + smem_dis_convol[si_hi]);
                    }
                }
            }
        } else {
#pragma unroll
            for (int fj = 0; fj < fwidth; ++fj) {
#pragma unroll
                for (int off = 0; off < val_per_thread; ++off) {
                    const int j = x_start + off;
                    if (j < w) {
                        const int si = smem_base + off + fj;
                        const uint16_t fcoeff = vif_filt.filter[scale][fj];
                        accum_mu1[off] += fcoeff * smem_mu1[si];
                        accum_mu2[off] += fcoeff * smem_mu2[si];
                        accum_ref_tmp[off] += fcoeff * (uint64_t)smem_ref[si];
                        accum_dis_tmp[off] += fcoeff * (uint64_t)smem_dis[si];
                        accum_ref_dis_tmp[off] += fcoeff * (uint64_t)smem_ref_dis[si];

                        if (fj >= (fwidth - fwidth_rd) / 2 &&
                            fj < (fwidth - (fwidth - fwidth_rd) / 2) && fwidth_rd > 0 &&
                            off % 2 == 0) {
                            const uint32_t fcoeff_rd =
                                vif_filt.filter[scale + 1][fj - ((fwidth - fwidth_rd) / 2)];
                            accum_ref_rd[off / 2] += fcoeff_rd * smem_ref_convol[si];
                            accum_dis_rd[off / 2] += fcoeff_rd * smem_dis_convol[si];
                        }
                    }
                }
            }
        }
        for (int off = 0; off < val_per_thread; ++off) {
            const int x = x_start + off;
            if (x < w) {
                accum_ref[off] = (uint32_t)((accum_ref_tmp[off] + add_shift_round_HP) >> shift_HP);
                accum_dis[off] = (uint32_t)((accum_dis_tmp[off] + add_shift_round_HP) >> shift_HP);
                accum_ref_dis[off] =
                    (uint32_t)((accum_ref_dis_tmp[off] + add_shift_round_HP) >> shift_HP);
                vif_statistic_calculation<uint32_t>(accum_mu1[off], accum_mu2[off], accum_ref[off],
                                                    accum_dis[off], accum_ref_dis[off], x, w, h,
                                                    vif_enhn_gain_limit, thread_accum);
            }
        }

        for (int i = 0; i < 7; ++i) {
            thread_accum_i64[i] = warp_reduce(thread_accum_i64[i]);
        }
        const int warp_id = threadIdx.x % VMAF_CUDA_THREADS_PER_WARP;
        if (warp_id == 0) {
            for (int i = 0; i < 7; ++i) {
                atomicAdd_int64(&reinterpret_cast<int64_t *>(accum)[i], thread_accum_i64[i]);
            }
        }

        for (int off = 0; off < val_per_thread; ++off) {
            const int x = x_start + off;
            if (y < h && x < w) {
                if ((y % 2) == 0 && (off % 2) == 0) {
                    uint16_t *ref = (uint16_t *)buf.ref;
                    uint16_t *dis = (uint16_t *)buf.dis;
                    const ptrdiff_t rd_stride = buf.rd_stride / sizeof(uint16_t);
                    ref[(y / 2) * rd_stride + (x / 2)] =
                        (uint16_t)((accum_ref_rd[off / 2] + 32768) >> 16);
                    dis[(y / 2) * rd_stride + (x / 2)] =
                        (uint16_t)((accum_dis_rd[off / 2] + 32768) >> 16);
                }
            }
        }
    }
}

#define FILTER1D_8_VERT(alignment_type, fwidth_0, fwidth_1)                                        \
    __global__ void filter1d_8_vertical_kernel_##alignment_type##_##fwidth_0##_##fwidth_1(         \
        VifBufferCuda buf, uint8_t *ref_in, uint8_t *dis_in, int w, int h,                         \
        filter_table_stuct vif_filt_s0)                                                            \
    {                                                                                              \
        filter1d_8_vertical_kernel<alignment_type, fwidth_0, fwidth_1>(buf, ref_in, dis_in, w, h,  \
                                                                       vif_filt_s0);               \
    }

#define FILTER1D_8_HORI(val_per_thread, fwidth_0, fwidth_1)                                        \
    __global__ void filter1d_8_horizontal_kernel_##val_per_thread##_##fwidth_0##_##fwidth_1(       \
        VifBufferCuda buf, int w, int h, filter_table_stuct vif_filt_s0,                           \
        double vif_enhn_gain_limit, vif_accums *accum)                                             \
    {                                                                                              \
        filter1d_8_horizontal_kernel<val_per_thread, fwidth_0, fwidth_1>(                          \
            buf, w, h, vif_filt_s0, vif_enhn_gain_limit, accum);                                   \
    }

#define FILTER1D_16_VERT(alignment_type, fwidth, fwidth_rd, scale)                                 \
    __global__ void                                                                                \
    filter1d_16_vertical_kernel_##alignment_type##_##fwidth##_##fwidth_rd##_##scale(               \
        VifBufferCuda buf, uint16_t *ref_in, uint16_t *dis_in, int w, int h,                       \
        int32_t add_shift_round_VP, int32_t shift_VP, int32_t add_shift_round_VP_sq,               \
        int32_t shift_VP_sq, filter_table_stuct vif_filt)                                          \
    {                                                                                              \
        filter1d_16_vertical_kernel<alignment_type, fwidth, fwidth_rd, scale>(                     \
            buf, ref_in, dis_in, w, h, add_shift_round_VP, shift_VP, add_shift_round_VP_sq,        \
            shift_VP_sq, vif_filt);                                                                \
    }

#define FILTER1D_16_HORI(val_per_thread, fwidth, fwidth_rd, scale)                                 \
    __global__ void                                                                                \
    filter1d_16_horizontal_kernel_##val_per_thread##_##fwidth##_##fwidth_rd##_##scale(             \
        VifBufferCuda buf, int w, int h, int32_t add_shift_round_HP, int32_t shift_HP,             \
        filter_table_stuct vif_filt, double vif_enhn_gain_limit, vif_accums *accum)                \
    {                                                                                              \
        filter1d_16_horizontal_kernel<val_per_thread, fwidth, fwidth_rd, scale>(                   \
            buf, w, h, add_shift_round_HP, shift_HP, vif_filt, vif_enhn_gain_limit, accum);        \
    }

extern "C" {
// constexpr int fwidth[4] = {17, 9, 5, 3};
FILTER1D_8_VERT(uint32_t, 17, 9);  // filter1d_8_vertical_kernel_uint32_t_17_9
FILTER1D_8_HORI(2, 17, 9);         // filter1d_8_horizontal_kernel_2_17_9
FILTER1D_16_VERT(uint2, 17, 9, 0); // filter1d_16_vertical_kernel_uint2_17_9_0
FILTER1D_16_VERT(uint2, 9, 5, 1);  // filter1d_16_vertical_kernel_uint2_9_5_1
FILTER1D_16_VERT(uint2, 5, 3, 2);  // filter1d_16_vertical_kernel_uint2_5_3_2
FILTER1D_16_VERT(uint2, 3, 0, 3);  // filter1d_16_vertical_kernel_uint2_3_0_3

FILTER1D_16_HORI(2, 17, 9, 0); // filter1d_16_horizontal_kernel_2_17_9_0
FILTER1D_16_HORI(2, 9, 5, 1);  // filter1d_16_horizontal_kernel_2_9_5_1
FILTER1D_16_HORI(2, 5, 3, 2);  // filter1d_16_horizontal_kernel_2_5_3_2
FILTER1D_16_HORI(2, 3, 0, 3);  // filter1d_16_horizontal_kernel_2_3_0_3
}
