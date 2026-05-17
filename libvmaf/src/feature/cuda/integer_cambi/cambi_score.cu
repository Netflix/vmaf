/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA compute kernels for the CAMBI banding-detection feature extractor
 *  (T3-15 / ADR-0360). CUDA twin of cambi_vulkan.c (ADR-0210).
 *
 *  Strategy II hybrid (mirrors the Vulkan precedent from ADR-0205 and
 *  ADR-0210): three GPU kernels handle the embarrassingly parallel stages
 *  (spatial mask, 2x decimate, 3-tap separable mode filter), while the
 *  precision-sensitive sliding-histogram calculate_c_values pass and
 *  top-K spatial pooling run on the host CPU via the shared wrappers in
 *  cambi_internal.h. This keeps the CUDA port bit-exact at places=4
 *  w.r.t. the CPU extractor (ULP=0 on the emitted score) at no extra
 *  development risk from a fully-on-GPU histogram pass.
 *
 *  GPU kernel inventory:
 *
 *    cambi_spatial_mask_kernel -- derivative (pixel == right AND == below)
 *        + 7x7 box sum from a shared-memory zero_deriv tile, then threshold
 *        compare.  A 22x22 uint8 tile (ZD_TILE) is populated cooperatively
 *        by the 16x16 block (2-pass, 256 threads, 484 elements = 3x484 = 1452
 *        global reads total per block) and then each thread sums its 7x7
 *        window from SLM.  Global reads fall from 49*3 = 147 per thread to
 *        ~1.9 average (1452/768 effective lanes).  Bit-exact with
 *        cambi.c::get_spatial_mask_for_index.  (ADR-0464 / perf-audit win 3)
 *
 *    cambi_decimate_kernel -- strict 2x stride-2 subsample of a uint16
 *        luma buffer. One thread per output pixel. Bit-exact with
 *        cambi.c::decimate.
 *
 *    cambi_filter_mode_kernel -- separable 3-tap mode filter, horizontal
 *        pass first then vertical, each in a separate kernel launch
 *        (axis == 0 -> H, axis == 1 -> V). One thread per output pixel.
 *        Bit-exact with cambi.c::filter_mode.
 *
 *  All buffers are flat uint16_t device arrays; the host glue converts
 *  from/to VmafPicture layout via DtoH / HtoD memcpy.
 *
 *  Precision contract: places=4 (ULP=0 on host-emitted score).
 *  The three GPU phases are integer + bit-exact w.r.t. CPU scalar.
 *  The host residual runs the exact CPU code path from cambi_internal.h,
 *  so the final CAMBI score is bit-for-bit identical to vmaf_fex_cambi.
 */

#include "cuda_helper.cuh"
#include "cuda/integer_cambi_cuda.h"
#include "common.h"

/* ------------------------------------------------------------------
 * Shared-memory tile constants for cambi_spatial_mask_kernel.
 *
 * The 7x7 zero_deriv box sum for each output pixel in a 16x16 block
 * touches a (16+6) x (16+6) = 22x22 = 484-element footprint of
 * zero_deriv values.  We pre-compute those 484 values cooperatively
 * from global memory (3 reads per element = 1452 total reads per block)
 * and store them in a __shared__ tile.  Then each thread sums its 7x7
 * window from the tile (49 SLM reads, no global traffic).
 *
 * ZD_TILE_STRIDE is padded to 32 to keep each row on a 32-byte boundary
 * and reduce false sharing at row edges; 4-way shared-memory bank
 * conflicts on uint8 within a row are accepted (SLM is still >10x faster
 * than L2 for this access pattern).  Total smem per block: 22x32 = 704 B.
 * ------------------------------------------------------------------ */
#define SMEM_HALF 3u       /* (MASK_FILTER_SIZE=7) >> 1               */
#define ZD_TILE_H 22u      /* BLOCK_Y + 2*SMEM_HALF = 16 + 6          */
#define ZD_TILE_W 22u      /* BLOCK_X + 2*SMEM_HALF = 16 + 6          */
#define ZD_TILE_STRIDE 32u /* padded row stride (uint8 cols)           */

/* ------------------------------------------------------------------
 * Kernel 1: Spatial mask  (ADR-0464 SLM-tile optimisation)
 *
 * Input:  image  -- flat uint16 array, stride_words columns per row.
 * Output: mask   -- flat uint16 array, same layout; 1 = edge, 0 = flat.
 *
 * Algorithm (matches cambi.c::get_spatial_mask_for_index):
 *   Phase A (cooperative, all threads): populate zd_tile[22][32].
 *     For each tile position (i,j) in [0,22) x [0,22):
 *       gx = clamp(bx*16 - 3 + j, 0, width-1)
 *       gy = clamp(by*16 - 3 + i, 0, height-1)
 *       p  = image[gy][gx]
 *       r  = image[gy][gx == width-1  ? gx : gx+1]
 *       b  = image[gy == height-1 ? gy : gy+1][gx]
 *       zd = (gx==width-1  || p==r) && (gy==height-1 || p==b)
 *     This is identical to the inline computation in the original kernel.
 *   __syncthreads()
 *
 *   Phase B (per-thread): 7x7 box sum from zd_tile.
 *     For dy in [-3,3], dx in [-3,3]:
 *       box_sum += zd_tile[ly+3+dy][lx+3+dx]
 *     No bounds check needed: (ly+3+dy) in [0,21] and (lx+3+dx) in [0,21].
 *     mask[y][x] = (box_sum > mask_index) ? 1 : 0.
 *
 * Bit-exactness: Phase A produces the same zero_deriv values as the
 * original per-thread computation (same clamping, same integer arithmetic).
 * Phase B sums the same values in the same order (row-major, dy inner).
 * ULP=0 guaranteed (integer-only path).
 * ------------------------------------------------------------------ */
extern "C" {

__global__ __launch_bounds__(256) void cambi_spatial_mask_kernel(const uint16_t *image,
                                                                 uint16_t *mask, unsigned width,
                                                                 unsigned height,
                                                                 unsigned stride_words,
                                                                 unsigned mask_index)
{
    /* Shared-memory zero_deriv tile: 22 rows x 32 padded uint8 cols = 704 B. */
    __shared__ uint8_t zd_tile[ZD_TILE_H][ZD_TILE_STRIDE];

    const int bx = (int)(blockIdx.x * blockDim.x);
    const int by = (int)(blockIdx.y * blockDim.y);
    const int lx = (int)threadIdx.x;
    const int ly = (int)threadIdx.y;
    const int tid = ly * (int)blockDim.x + lx; /* 0..255 */

    /* ------------------------------------------------------------------
     * Phase A: populate zd_tile cooperatively (2 passes, 256 threads,
     * 484 elements).  Thread tid loads element tid in pass 0, and element
     * tid+256 in pass 1 (only the 228 threads with tid < 228 do pass 1).
     * ------------------------------------------------------------------ */
    /* Pass 0 */
    {
        const int k = tid; /* k in [0, 255] -- all < ZD_TILE_H*ZD_TILE_W=484 */
        const int ti = k / (int)ZD_TILE_W;
        const int tj = k % (int)ZD_TILE_W;
        const int raw_gy = by - (int)SMEM_HALF + ti;
        const int raw_gx = bx - (int)SMEM_HALF + tj;
        const int gy = (raw_gy < 0) ? 0 : ((raw_gy >= (int)height) ? (int)height - 1 : raw_gy);
        const int gx = (raw_gx < 0) ? 0 : ((raw_gx >= (int)width) ? (int)width - 1 : raw_gx);
        const uint16_t p = image[(size_t)gy * stride_words + (unsigned)gx];
        const unsigned r_gx = (unsigned)((gx == (int)width - 1) ? gx : gx + 1);
        const unsigned b_gy = (unsigned)((gy == (int)height - 1) ? gy : gy + 1);
        const uint16_t r = image[(size_t)gy * stride_words + r_gx];
        const uint16_t b = image[(size_t)b_gy * stride_words + (unsigned)gx];
        const int eq_r = (gx == (int)width - 1) || (p == r);
        const int eq_b = (gy == (int)height - 1) || (p == b);
        zd_tile[ti][tj] = (uint8_t)(eq_r & eq_b);
    }
    /* Pass 1: elements 256..483 (228 threads active). */
    if (tid < (int)(ZD_TILE_H * ZD_TILE_W) - 256) {
        const int k = tid + 256;
        const int ti = k / (int)ZD_TILE_W;
        const int tj = k % (int)ZD_TILE_W;
        const int raw_gy = by - (int)SMEM_HALF + ti;
        const int raw_gx = bx - (int)SMEM_HALF + tj;
        const int gy = (raw_gy < 0) ? 0 : ((raw_gy >= (int)height) ? (int)height - 1 : raw_gy);
        const int gx = (raw_gx < 0) ? 0 : ((raw_gx >= (int)width) ? (int)width - 1 : raw_gx);
        const uint16_t p = image[(size_t)gy * stride_words + (unsigned)gx];
        const unsigned r_gx = (unsigned)((gx == (int)width - 1) ? gx : gx + 1);
        const unsigned b_gy = (unsigned)((gy == (int)height - 1) ? gy : gy + 1);
        const uint16_t r = image[(size_t)gy * stride_words + r_gx];
        const uint16_t b = image[(size_t)b_gy * stride_words + (unsigned)gx];
        const int eq_r = (gx == (int)width - 1) || (p == r);
        const int eq_b = (gy == (int)height - 1) || (p == b);
        zd_tile[ti][tj] = (uint8_t)(eq_r & eq_b);
    }
    __syncthreads();

    /* ------------------------------------------------------------------
     * Phase B: per-thread 7x7 box sum from SLM.  Guard against threads
     * outside the image boundary writing to mask (they still participated
     * in Phase A to fill the full tile cooperatively).
     * ------------------------------------------------------------------ */
    const int x = bx + lx;
    const int y = by + ly;
    if (x >= (int)width || y >= (int)height)
        return;

    /* zd_tile[ly+3+dy][lx+3+dx] accesses rows [0,21] and cols [0,21]
     * -- always in-bounds by construction of ZD_TILE_H / ZD_TILE_W. */
    unsigned box_sum = 0u;
    {
        /* Unrolled 7x7 loop: rows first to maximise SLM row reuse. */
        const int base_row = ly + (int)SMEM_HALF; /* 3..18 */
        const int base_col = lx + (int)SMEM_HALF; /* 3..18 */
#pragma unroll
        for (int dy = -(int)SMEM_HALF; dy <= (int)SMEM_HALF; dy++) {
#pragma unroll
            for (int dx = -(int)SMEM_HALF; dx <= (int)SMEM_HALF; dx++) {
                box_sum += (unsigned)zd_tile[base_row + dy][base_col + dx];
            }
        }
    }
    mask[(size_t)(unsigned)y * stride_words + (unsigned)x] =
        (uint16_t)(box_sum > mask_index ? 1u : 0u);
}

/* ------------------------------------------------------------------
 * Kernel 2: 2x decimate
 *
 * Output pixel (x,y) samples input pixel (2x, 2y). Matches cambi.c::decimate
 * (strict even-pixel subsample, no filtering). Input and output are in
 * separate flat buffers (src and dst); both use stride_words columns
 * (the output width is (width+1)/2, (height+1)/2).
 *
 * src_stride_words is the stride of the source (larger) buffer.
 * dst_stride_words is the stride of the destination (smaller) buffer.
 * Both strides are in uint16_t words.
 * ------------------------------------------------------------------ */
__global__ void cambi_decimate_kernel(const uint16_t *src, uint16_t *dst, unsigned out_width,
                                      unsigned out_height, unsigned src_stride_words,
                                      unsigned dst_stride_words)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height)
        return;

    /* Sample at stride-2 -- exact match of cambi.c::decimate:
     *   data[i * stride + j] = data[(i<<1) * stride + (j<<1)]; */
    dst[(size_t)y * dst_stride_words + x] = src[(size_t)(y * 2u) * src_stride_words + x * 2u];
}

/* ------------------------------------------------------------------
 * Kernel 3: Separable 3-tap mode filter
 *
 * One kernel, two launches: axis=0 (horizontal), axis=1 (vertical).
 * The mode of three equal-length 1-D triplets is the value that appears
 * at least twice, or the minimum if all three are distinct -- matching
 * cambi.c::mode3.
 *
 * For axis=0 (H pass): each thread (x,y) writes
 *     out[y][x] = mode3(in[y][x-1], in[y][x], in[y][x+1])
 * except at the left/right border where in[y][-1] = in[y][0] etc.
 *
 * For axis=1 (V pass): same but over rows:
 *     out[y][x] = mode3(in[y-1][x], in[y][x], in[y+1][x])
 * with clamped borders.
 *
 * Matches cambi.c::filter_mode's row-by-row logic; the only difference
 * is that the GPU computes all rows in parallel (no rolling 3-row buffer
 * trick needed -- we have enough global memory).
 * ------------------------------------------------------------------ */
__device__ static inline uint16_t mode3_dev(uint16_t a, uint16_t b, uint16_t c)
{
    /* Two equal -> that value. All distinct -> min of the three. */
    if (a == b || a == c)
        return a;
    if (b == c)
        return b;
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

__global__ void cambi_filter_mode_kernel(const uint16_t *in, uint16_t *out, unsigned width,
                                         unsigned height, unsigned stride_words, int axis)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (int)width || y >= (int)height)
        return;

    uint16_t a, b, c;
    if (axis == 0) {
        /* Horizontal: neighbours in x. */
        const int xl = (x > 0) ? x - 1 : 0;
        const int xr = (x < (int)width - 1) ? x + 1 : (int)width - 1;
        a = in[(size_t)y * stride_words + (unsigned)xl];
        b = in[(size_t)y * stride_words + (unsigned)x];
        c = in[(size_t)y * stride_words + (unsigned)xr];
    } else {
        /* Vertical: neighbours in y. */
        const int yu = (y > 0) ? y - 1 : 0;
        const int yd = (y < (int)height - 1) ? y + 1 : (int)height - 1;
        a = in[(size_t)(unsigned)yu * stride_words + (unsigned)x];
        b = in[(size_t)(unsigned)y * stride_words + (unsigned)x];
        c = in[(size_t)(unsigned)yd * stride_words + (unsigned)x];
    }
    out[(size_t)(unsigned)y * stride_words + (unsigned)x] = mode3_dev(a, b, c);
}

} /* extern "C" */
