/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP compute kernel for the integer_motion feature extractor.
 *  Direct port of the CUDA twin at
 *  `libvmaf/src/feature/cuda/integer_motion/motion_score.cu`.
 *
 *  Algorithm: 5x5 separable Gaussian blur on the input Y-plane (blur
 *  written into a ping-pong uint16 device buffer), then per-pixel
 *  absolute difference against the previous blurred frame, warp-reduced
 *  and atomic-added into a single uint64 SAD accumulator.
 *
 *  The host post-processes:
 *    motion_score  = SAD / 256.0 / (W * H)
 *    motion2_score = min(cur, prev) * fps_weight, clamped at motion_max_val
 *    motion3_score = blend(motion2_score)  [host-only]
 *
 *  Mirror padding: skip-boundary reflect-101
 *  (`2 * size - idx - 2` for idx >= size), SAME as the CUDA twin.
 *  This differs from motion_v2's mirror (which uses idx - 1 reflection)
 *  and matches the CPU integer_motion.c edge_8/edge_16 functions.
 *
 *  HIP adaptation from CUDA:
 *  - Warp size 64 on GCN/RDNA; warp reduction uses `__shfl_down` (no mask).
 *  - Blurred buffer is a plain `uint16_t *` device pointer (not VmafCudaBuffer).
 *  - SAD accumulator is a plain `uint64_t *` device pointer.
 *  - TILE_PITCH = TILE_W + 1 retained (bank-conflict padding, same as CUDA).
 *
 *  Kernel entry-point names (must match hipModuleGetFunction calls in host):
 *    calculate_motion_score_kernel_8bpc
 *    calculate_motion_score_kernel_16bpc
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

__constant__ uint16_t ms_filter_d[5] = {3571, 16004, 26386, 16004, 3571};

/* 5-tap filter radius */
#define MS_RADIUS  2
#define MS_BLOCK_X 16
#define MS_BLOCK_Y 16
#define MS_TILE_W  (MS_BLOCK_X + 2 * MS_RADIUS)  /* 20 */
#define MS_TILE_H  (MS_BLOCK_Y + 2 * MS_RADIUS)  /* 20 */
/* Pad inner stride to break shared-memory bank conflicts (same as CUDA twin). */
#define MS_TILE_PITCH (MS_TILE_W + 1)              /* 21 */

/* HIP warp size on GCN/RDNA. */
#define MS_WARP_SIZE 64

/*
 * Skip-boundary mirror (reflect-101): matches CPU edge_8/edge_16.
 *   idx = -1  -> 1   (skips boundary 0)
 *   idx = sup -> sup - 2  (skips boundary sup-1)
 * The previous simpler +1 form repeated the boundary row and produced
 * ~2.6e-3 motion drift vs CPU (T7-15, fixed in CUDA PR #120).
 */
__device__ __forceinline__ int ms_mirror(int idx, int sup)
{
    int out = (idx < 0) ? -idx : idx;
    return (out < sup) ? out : (sup - (out - sup + 2));
}

extern "C" {

__launch_bounds__(MS_BLOCK_X *MS_BLOCK_Y, 8) __global__
    void calculate_motion_score_kernel_8bpc(
        const uint8_t *__restrict__ src,
        uint16_t *__restrict__       src_blurred,
        const uint16_t *__restrict__ prev_blurred,
        uint64_t *__restrict__       sad,
        unsigned width,
        unsigned height,
        ptrdiff_t src_stride,
        ptrdiff_t blurred_stride_bytes)
{
    /*
     * Shared memory tile holds source pixels (uint32) to avoid narrowing
     * during the separable Gaussian.  Inner dim is MS_TILE_PITCH for
     * bank-conflict padding.
     */
    __shared__ uint32_t s_tile[MS_TILE_H][MS_TILE_PITCH];

    constexpr unsigned shift_var_y     = 8u;
    constexpr unsigned add_before_y    = 128u;
    constexpr unsigned shift_var_x     = 16u;
    constexpr unsigned add_before_x    = 32768u;

    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned lid = threadIdx.y * blockDim.x + threadIdx.x;

    /* --- Phase 1: cooperative tile load --- */
    const int tile_origin_x = blockIdx.x * MS_BLOCK_X - MS_RADIUS;
    const int tile_origin_y = blockIdx.y * MS_BLOCK_Y - MS_RADIUS;
    const unsigned tile_elems = MS_TILE_W * MS_TILE_H;  /* 400 */
    const unsigned wg_size    = MS_BLOCK_X * MS_BLOCK_Y; /* 256 */

    for (unsigned i = lid; i < tile_elems; i += wg_size) {
        const unsigned ty = i / MS_TILE_W;
        const unsigned tx = i % MS_TILE_W;
        const int gx = ms_mirror(tile_origin_x + (int)tx, (int)width);
        const int gy = ms_mirror(tile_origin_y + (int)ty, (int)height);
        s_tile[ty][tx] = (uint32_t)src[gy * src_stride + gx];
    }
    __syncthreads();

    /* --- Phase 2: 5x5 Gaussian blur + SAD --- */
    uint32_t abs_dist = 0u;
    if (x < (int)width && y < (int)height) {
        const unsigned lx = threadIdx.x + MS_RADIUS;
        const unsigned ly = threadIdx.y + MS_RADIUS;

        uint32_t blurred = 0u;
#pragma unroll
        for (int xf = 0; xf < 5; ++xf) {
            uint32_t blurred_y = 0u;
#pragma unroll
            for (int yf = 0; yf < 5; ++yf) {
                blurred_y += (uint32_t)ms_filter_d[yf] *
                             s_tile[ly - MS_RADIUS + yf][lx - MS_RADIUS + xf];
            }
            blurred += (uint32_t)ms_filter_d[xf] *
                       ((blurred_y + add_before_y) >> shift_var_y);
        }
        blurred = (blurred + add_before_x) >> shift_var_x;

        /* Write blurred pixel into ping-pong slot. */
        const ptrdiff_t blurred_pitch = blurred_stride_bytes / (ptrdiff_t)sizeof(uint16_t);
        src_blurred[y * blurred_pitch + x] = (uint16_t)blurred;

        /* Absolute difference against the previous blurred frame. */
        abs_dist = (uint32_t)abs((int)blurred -
                                  (int)prev_blurred[y * blurred_pitch + x]);
    }

    /* --- Phase 3: warp reduction (64 lanes on GCN/RDNA) --- */
    for (int off = MS_WARP_SIZE / 2; off > 0; off >>= 1)
        abs_dist += __shfl_down(abs_dist, off);

    const unsigned lane = lid % MS_WARP_SIZE;
    if (lane == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(sad),
                  (unsigned long long)abs_dist);
}

__launch_bounds__(MS_BLOCK_X *MS_BLOCK_Y, 8) __global__
    void calculate_motion_score_kernel_16bpc(
        const uint8_t *__restrict__ src,
        uint16_t *__restrict__       src_blurred,
        const uint16_t *__restrict__ prev_blurred,
        uint64_t *__restrict__       sad,
        unsigned width,
        unsigned height,
        ptrdiff_t src_stride,
        ptrdiff_t blurred_stride_bytes,
        unsigned  bpc)
{
    __shared__ uint32_t s_tile[MS_TILE_H][MS_TILE_PITCH];

    const unsigned shift_var_y  = bpc;
    const unsigned add_before_y = 1u << (bpc - 1u);
    constexpr unsigned shift_var_x  = 16u;
    constexpr unsigned add_before_x = 32768u;

    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned lid = threadIdx.y * blockDim.x + threadIdx.x;

    const int tile_origin_x = blockIdx.x * MS_BLOCK_X - MS_RADIUS;
    const int tile_origin_y = blockIdx.y * MS_BLOCK_Y - MS_RADIUS;
    const unsigned tile_elems = MS_TILE_W * MS_TILE_H;
    const unsigned wg_size    = MS_BLOCK_X * MS_BLOCK_Y;

    for (unsigned i = lid; i < tile_elems; i += wg_size) {
        const unsigned ty = i / MS_TILE_W;
        const unsigned tx = i % MS_TILE_W;
        const int gx = ms_mirror(tile_origin_x + (int)tx, (int)width);
        const int gy = ms_mirror(tile_origin_y + (int)ty, (int)height);
        s_tile[ty][tx] = (uint32_t)(
            reinterpret_cast<const uint16_t *>(src + gy * src_stride)[gx]);
    }
    __syncthreads();

    uint32_t abs_dist = 0u;
    if (x < (int)width && y < (int)height) {
        const unsigned lx = threadIdx.x + MS_RADIUS;
        const unsigned ly = threadIdx.y + MS_RADIUS;

        uint32_t blurred = 0u;
#pragma unroll
        for (int xf = 0; xf < 5; ++xf) {
            uint32_t blurred_y = 0u;
#pragma unroll
            for (int yf = 0; yf < 5; ++yf) {
                blurred_y += (uint32_t)ms_filter_d[yf] *
                             s_tile[ly - MS_RADIUS + yf][lx - MS_RADIUS + xf];
            }
            blurred += (uint32_t)ms_filter_d[xf] *
                       ((blurred_y + add_before_y) >> shift_var_y);
        }
        blurred = (blurred + add_before_x) >> shift_var_x;

        const ptrdiff_t blurred_pitch = blurred_stride_bytes / (ptrdiff_t)sizeof(uint16_t);
        src_blurred[y * blurred_pitch + x] = (uint16_t)blurred;
        abs_dist = (uint32_t)abs((int)blurred -
                                  (int)prev_blurred[y * blurred_pitch + x]);
    }

    for (int off = MS_WARP_SIZE / 2; off > 0; off >>= 1)
        abs_dist += __shfl_down(abs_dist, off);

    const unsigned lane = lid % MS_WARP_SIZE;
    if (lane == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(sad),
                  (unsigned long long)abs_dist);
}

} /* extern "C" */
