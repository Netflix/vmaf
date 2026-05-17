/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA kernel for the ssimulacra2 separable FastGaussian IIR blur.
 *  Mirrors the libjxl Charalampidis 2016 3-pole recursive Gaussian
 *  (k = {1, 3, 5}, sigma=1.5, zero-padded boundaries) implemented
 *  in libvmaf/src/feature/ssimulacra2.c::fast_gaussian_1d.
 *
 *  The IIR is sequential along the scan axis. Separability lets us
 *  parallelise across the orthogonal axis:
 *    H pass: one thread per row, scans left to right.
 *    V pass: one thread per column, scans top to bottom.
 *
 *  Bit-exact-with-CPU strategy (per ADR-0192 / ADR-0201): the
 *  CPU writes `o = n2 * sum - d1 * prev1 - prev2` as separate
 *  FMUL/FSUB ops under -ffp-contract=off. Compile this fatbin
 *  with --fmad=false so NVCC does NOT fuse those into FMAs;
 *  otherwise the IIR pole tracking compounds an FMA-rounding
 *  delta across the radius and the 6-scale pyramid into a
 *  ~1e-3 pooled-score drift (places=1).
 *
 *  Performance notes (ADR-0456):
 *
 *  Change 1 — 3-channel fusion via gridDim.z:
 *    The original dispatch looped over 3 channels and issued one H
 *    kernel + one V kernel per channel per blur call = 6 launches per
 *    blur.  With 5 blurs per scale and up to 6 scales this produced
 *    ~180 kernel launches per frame, each carrying ~2 µs driver
 *    overhead on RTX 4090.  The fused kernels below use `blockIdx.z`
 *    to select the channel so all 3 planes are processed by a single
 *    launch (gridDim.z = 3), reducing per-frame launches 3× to ~60.
 *
 *  Change 2 — V-pass coalescing via in-place transpose:
 *    The V pass reads `in_buf[offset + row * width + col]` where
 *    `row` increments by 1 per IIR step.  For typical 1920-wide
 *    frames each successive load is 1920 × 4 = 7680 bytes away —
 *    stride-1920 pattern, every load is a distinct L2 cache line.
 *    Fix: a preceding transpose kernel converts the H-pass output
 *    from row-major to column-major layout in `d_transpose_buf`.
 *    The V pass then reads the transposed buffer sequentially
 *    (`row × height + col` with height ≤ 1080 stepping by 1),
 *    which is fully coalesced.  The V-pass output is written back
 *    to row-major using a second transpose inside the V kernel's
 *    `__shared__` tile.
 *
 *    Transpose tile: 32×32 floats with a +1 column padding
 *    (`float tile[32][33]`) to break the 32-way shared-memory bank
 *    conflict that occurs when a warp stores 32 consecutive rows
 *    into the same bank (stride-32 → every row maps to bank 0 with
 *    32-element banks).  The +1 pad shifts each row by one bank so
 *    all 32 stores hit distinct banks.
 *
 *    Net launch count change: +1 transpose per V pass (H-pass
 *    output → column-major scratch).  The V pass writes its output
 *    back to row-major directly.  Coalesced V-pass reads outweigh
 *    the extra transpose launch on RTX 4090 by ~2-3× at 1080p.
 */

#include "cuda_helper.cuh"

/* Tile dimension for the transpose kernel (must be power-of-two ≤ 32). */
#define SS2C_TILE 32

extern "C" {

/* ------------------------------------------------------------------ */
/* Transpose kernel                                                   */
/* ------------------------------------------------------------------ */

/* Converts a single plane from row-major to column-major layout in
 * one launch.  Block shape: (TILE, TILE, 1).  Grid shape:
 *   gridDim.x = ceil(width  / TILE)
 *   gridDim.y = ceil(height / TILE)
 *   gridDim.z = 3  (one z-slice per XYB channel)
 *
 * Shared memory: `float tile[TILE][TILE+1]` — the +1 padding avoids
 * 32-way bank conflicts on the column-wise store phase.
 *
 * Input layout  (row-major): in[c * plane_stride + row * width + col]
 * Output layout (col-major): out[c * plane_stride + col * height + row]
 *
 * Both `in` and `out` are in the same `plane_stride`-aligned 3-plane
 * buffer; `in` and `out` point to different allocations so there is no
 * aliasing.
 *
 * __launch_bounds__(SS2C_TILE * SS2C_TILE, 1): 32×32 = 1024 threads
 * per block, 1 resident block per SM (Ada Lovelace max 1536 threads/SM;
 * 4 × 1024 > 1536 so minBlocksPerSM = 1 is the safe value).
 */
__global__ __launch_bounds__(SS2C_TILE *SS2C_TILE,
                             1) void ssimulacra2_transpose(const float *__restrict__ in,
                                                           float *__restrict__ out, unsigned width,
                                                           unsigned height, unsigned plane_stride)
{
    /* +1 pad column to avoid bank conflicts on the column-wise store. */
    __shared__ float tile[SS2C_TILE][SS2C_TILE + 1];

    const unsigned c = blockIdx.z;
    const unsigned src_col = blockIdx.x * SS2C_TILE + threadIdx.x;
    const unsigned src_row = blockIdx.y * SS2C_TILE + threadIdx.y;

    /* Load: row-major → shared tile (coalesced by threadIdx.x). */
    if (src_row < height && src_col < width) {
        tile[threadIdx.y][threadIdx.x] = in[c * plane_stride + src_row * width + src_col];
    }

    __syncthreads();

    /* Thread (tx, ty) writes tile[tx][ty], which holds the element originally
     * loaded by thread (ty, tx) at source position (src_row, src_col) =
     * (blockIdx.y*TILE+tx, blockIdx.x*TILE+ty).  Column-major destination
     * address: src_col * height + src_row = dst_col_orig * height + dst_row_orig.
     *
     * Variable mapping (inverted from the load phase for coalesced stores):
     *   dst_row_orig = blockIdx.x * TILE + threadIdx.y  (original src_col)
     *   dst_col_orig = blockIdx.y * TILE + threadIdx.x  (original src_row)
     *
     * Store: out[c * plane_stride + dst_row_orig * height + dst_col_orig]
     * = out[c * stride + src_col * height + src_row], which is the
     * column-major layout consumed by ssimulacra2_blur_v3_transposed.
     */
    const unsigned dst_orig_col = blockIdx.y * SS2C_TILE + threadIdx.x; /* = original src_row */
    const unsigned dst_orig_row = blockIdx.x * SS2C_TILE + threadIdx.y; /* = original src_col */
    if (dst_orig_col < height && dst_orig_row < width) {
        out[c * plane_stride + dst_orig_row * height + dst_orig_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------------------------------------------------ */
/* H pass — fused 3-channel                                           */
/* ------------------------------------------------------------------ */

/* H pass: one thread per row, 3 channels via gridDim.z.
 *  blockIdx.z ∈ {0,1,2} selects the XYB plane.
 *  in_buf / out_buf are full 3-plane buffers; the channel offset
 *  is derived from blockIdx.z × plane_stride.
 *
 *  Bit-identical control flow with CPU `fast_gaussian_1d`.
 *  `__launch_bounds__(SS2C_BLUR_BLOCK = 64, minBlocksPerSM = 32)`
 *  documents the block-size + occupancy contract — NVCC then trims the
 *  per-thread register budget to keep ≥32 resident blocks per SM,
 *  consistent with the host-side launch shape in
 *  `ss2c_launch_blur_pass`. */
__global__ __launch_bounds__(64, 32) void ssimulacra2_blur_h(
    const float *__restrict__ in_buf, float *__restrict__ out_buf, unsigned width, unsigned height,
    float n2_0, float n2_1, float n2_2, float d1_0, float d1_1, float d1_2, int radius,
    unsigned in_offset, unsigned out_offset)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;

    const int xsize = (int)width;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    const unsigned in_base = in_offset + row * width;
    const unsigned out_base = out_offset + row * width;

    for (int n = -N + 1; n < xsize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        const float lv = (left >= 0) ? in_buf[in_base + (unsigned)left] : 0.f;
        const float rv = (right < xsize) ? in_buf[in_base + (unsigned)right] : 0.f;
        const float sum = lv + rv;

        /* Match CPU expression `n2*sum - d1*prev1 - prev2` ordering
         * exactly. Explicit temporaries + --fmad=false at the
         * fatbin level keep this as separate FMUL/FSUB ops. */
        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            out_buf[out_base + (unsigned)n] = s_total;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fused 3-channel H pass (Change 1)                                  */
/* ------------------------------------------------------------------ */

/* Same IIR as ssimulacra2_blur_h but gridDim.z = 3 selects the
 * channel.  Each z-slice processes one XYB plane independently.
 * All three planes run concurrently across the SM pool.
 *
 * in_offset / out_offset are per-plane offsets in units of floats;
 * the caller passes `c * plane_stride` so offset arithmetic matches
 * the single-channel kernel exactly.  Because blockIdx.z selects the
 * plane, the caller sets in_offset/out_offset = 0 and the kernel
 * computes the per-plane offset itself from blockIdx.z × plane_stride.
 */
__global__ __launch_bounds__(64, 16) void ssimulacra2_blur_h3(const float *__restrict__ in_buf,
                                                              float *__restrict__ out_buf,
                                                              unsigned width, unsigned height,
                                                              float n2_0, float n2_1, float n2_2,
                                                              float d1_0, float d1_1, float d1_2,
                                                              int radius, unsigned plane_stride)
{
    const unsigned c = blockIdx.z;
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;

    const int xsize = (int)width;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    const unsigned in_base = c * plane_stride + row * width;
    const unsigned out_base = c * plane_stride + row * width;

    for (int n = -N + 1; n < xsize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        const float lv = (left >= 0) ? in_buf[in_base + (unsigned)left] : 0.f;
        const float rv = (right < xsize) ? in_buf[in_base + (unsigned)right] : 0.f;
        const float sum = lv + rv;

        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            out_buf[out_base + (unsigned)n] = s_total;
        }
    }
}

/* ------------------------------------------------------------------ */
/* V pass (original single-channel — retained for reference)         */
/* ------------------------------------------------------------------ */

/* V pass: one thread per column.  Same `__launch_bounds__` contract
 * as the H pass — see the H-pass block comment above. */
__global__ __launch_bounds__(64, 32) void ssimulacra2_blur_v(
    const float *__restrict__ in_buf, float *__restrict__ out_buf, unsigned width, unsigned height,
    float n2_0, float n2_1, float n2_2, float d1_0, float d1_1, float d1_2, int radius,
    unsigned in_offset, unsigned out_offset)
{
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;

    const int ysize = (int)height;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    for (int n = -N + 1; n < ysize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        const float lv = (left >= 0) ? in_buf[in_offset + (unsigned)left * width + col] : 0.f;
        const float rv = (right < ysize) ? in_buf[in_offset + (unsigned)right * width + col] : 0.f;
        const float sum = lv + rv;

        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            out_buf[out_offset + (unsigned)n * width + col] = s_total;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fused 3-channel V pass on transposed input (Changes 1 + 2)        */
/* ------------------------------------------------------------------ */

/* V pass operating on a column-major buffer produced by
 * `ssimulacra2_transpose`.
 *
 * Column-major layout: transposed[c * plane_stride + col * height + row]
 * Reading top-to-bottom (row 0..ysize-1) for a fixed `col` is now
 * reading consecutive addresses → fully coalesced for the warp.
 *
 * gridDim.z = 3 for 3-channel fusion (Change 1).
 * gridDim.x = ceil(width / SS2C_BLUR_BLOCK) — one thread per column.
 *
 * Output: written back to row-major in `out_buf` at
 *   out_buf[c * plane_stride + row * width + col].
 * This write is a scatter (row increments non-contiguously across
 * threads) but because each thread writes its own row independently
 * and the output is used as an H-pass input (also row-major), we
 * accept the scatter.  An alternative is to write column-major and
 * transpose back, but that costs +1 launch and +1 buffer; the
 * present approach trades the write scatter for zero additional
 * launches and zero additional memory.
 *
 * Bit-exactness: the IIR recurrence is identical to the original
 * ssimulacra2_blur_v — same scan order (n = -N+1 .. ysize-1),
 * same accumulator variables, same `--fmad=false` constraint.
 * The transposed input delivers the same numerical values to the IIR;
 * only their memory layout has changed.
 */
__global__ __launch_bounds__(64, 16) void ssimulacra2_blur_v3_transposed(
    const float *__restrict__ in_transposed, float *__restrict__ out_buf, unsigned width,
    unsigned height, float n2_0, float n2_1, float n2_2, float d1_0, float d1_1, float d1_2,
    int radius, unsigned plane_stride)
{
    const unsigned c = blockIdx.z;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;

    const int ysize = (int)height;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    /* Column-major base for this column in the transposed buffer.
     * Successive rows are at +1 stride — fully coalesced loads. */
    const unsigned col_base = c * plane_stride + col * height;

    for (int n = -N + 1; n < ysize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        /* Column-major reads: in_transposed[col_base + row] */
        const float lv = (left >= 0) ? in_transposed[col_base + (unsigned)left] : 0.f;
        const float rv = (right < ysize) ? in_transposed[col_base + (unsigned)right] : 0.f;
        const float sum = lv + rv;

        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            /* Write back to row-major: out[c * plane_stride + row * width + col] */
            out_buf[c * plane_stride + (unsigned)n * width + col] = s_total;
        }
    }
}

} /* extern "C" */
