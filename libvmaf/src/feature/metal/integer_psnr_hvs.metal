/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernel for the psnr_hvs feature extractor.
 *  Port of `libvmaf/src/feature/cuda/integer_psnr_hvs/psnr_hvs_score.cu`
 *  to MSL — same integer 8-point DCT, same CSF tables, same masking
 *  and float accumulation logic.
 *
 *  One kernel function `psnr_hvs` dispatched per plane per frame by
 *  `integer_psnr_hvs_metal.mm`.  Grid: (num_blocks_x, num_blocks_y, 1),
 *  threads (8, 8, 1) — one threadgroup per output 8×8 image block
 *  (step = 7).
 *
 *  Numeric design notes:
 *   - CSF tables are identical to the CUDA and Vulkan twins (sourced
 *     from third_party/xiph/psnr_hvs.c csf_y / csf_cb420 / csf_cr420).
 *   - od_dct_rshift: round-toward-zero shift, same as OD_UNBIASED_RSHIFT32
 *     in the xiph reference and the CUDA twin.
 *   - Two integer DCT passes are row/column-parallel across the first
 *     eight threads in the threadgroup; mean/variance, masking, and
 *     final accumulation stay on thread 0 to preserve the established
 *     CPU/CUDA/Vulkan numeric contract.
 *   - Per-block partial is written to `partials_out[blk_y * num_blocks_x
 *     + blk_x]`; host sums all partials and applies the samplemax²
 *     normalisation before the 10·log10 conversion.
 *
 *  Buffer bindings:
 *   [[buffer(0)]] ref_in       — const float * packed row-major (width × height)
 *   [[buffer(1)]] dist_in      — const float * packed row-major (width × height)
 *   [[buffer(2)]] partials_out — float * num_blocks_x × num_blocks_y
 *   [[buffer(3)]] params       — uint4 (.x=width, .y=height,
 *                                        .z=num_blocks_x, .w=num_blocks_y)
 *   [[buffer(4)]] bpc_plane    — uint2 (.x=bpc, .y=plane index 0/1/2)
 *
 *  Grid: (num_blocks_x, num_blocks_y, 1), threads (8, 8, 1).
 */

#include <metal_stdlib>
using namespace metal;

/* Per-plane CSF tables — same constants as csf_y / csf_cb420 / csf_cr420
 * in third_party/xiph/psnr_hvs.c. */
constant float CSF_Y[64] = {
    1.6193873005f,   2.2901594831f,   2.08509755623f,  1.48366094411f,  1.00227514334f,
    0.678296995242f, 0.466224900598f, 0.3265091542f,   2.2901594831f,   1.94321815382f,
    2.04793073064f,  1.68731108984f,  1.2305666963f,   0.868920337363f, 0.61280991668f,
    0.436405793551f, 2.08509755623f,  2.04793073064f,  1.34329019223f,  1.09205635862f,
    0.875748795257f, 0.670882927016f, 0.501731932449f, 0.372504254596f, 1.48366094411f,
    1.68731108984f,  1.09205635862f,  0.772819797575f, 0.605636379554f, 0.48309405692f,
    0.380429446972f, 0.295774038565f, 1.00227514334f,  1.2305666963f,   0.875748795257f,
    0.605636379554f, 0.448996256676f, 0.352889268808f, 0.283006984131f, 0.226951348204f,
    0.678296995242f, 0.868920337363f, 0.670882927016f, 0.48309405692f,  0.352889268808f,
    0.27032073436f,  0.215017739696f, 0.17408067321f,  0.466224900598f, 0.61280991668f,
    0.501731932449f, 0.380429446972f, 0.283006984131f, 0.215017739696f, 0.168869545842f,
    0.136153931001f, 0.3265091542f,   0.436405793551f, 0.372504254596f, 0.295774038565f,
    0.226951348204f, 0.17408067321f,  0.136153931001f, 0.109083846276f,
};

constant float CSF_CB[64] = {
    1.91113096927f,  2.46074210438f,  1.18284184739f,  1.14982565193f,  1.05017074788f,
    0.898018824055f, 0.74725392039f,  0.615105596242f, 2.46074210438f,  1.58529308355f,
    1.21363250036f,  1.38190029285f,  1.33100189972f,  1.17428548929f,  0.996404342439f,
    0.830890433625f, 1.18284184739f,  1.21363250036f,  0.978712413627f, 1.02624506078f,
    1.03145147362f,  0.960060382087f, 0.849823426169f, 0.731221236837f, 1.14982565193f,
    1.38190029285f,  1.02624506078f,  0.861317501629f, 0.801821139099f, 0.751437590932f,
    0.685398513368f, 0.608694761374f, 1.05017074788f,  1.33100189972f,  1.03145147362f,
    0.801821139099f, 0.676555426187f, 0.605503172737f, 0.55002013668f,  0.495804539034f,
    0.898018824055f, 1.17428548929f,  0.960060382087f, 0.751437590932f, 0.605503172737f,
    0.514674450957f, 0.454353482512f, 0.407050308965f, 0.74725392039f,  0.996404342439f,
    0.849823426169f, 0.685398513368f, 0.55002013668f,  0.454353482512f, 0.389234902883f,
    0.342353999733f, 0.615105596242f, 0.830890433625f, 0.731221236837f, 0.608694761374f,
    0.495804539034f, 0.407050308965f, 0.342353999733f, 0.295530605237f,
};

constant float CSF_CR[64] = {
    2.03871978502f,  2.62502345193f,  1.26180942886f,  1.11019789803f,  1.01397751469f,
    0.867069376285f, 0.721500455585f, 0.593906509971f, 2.62502345193f,  1.69112867013f,
    1.17180569821f,  1.3342742857f,   1.28513006198f,  1.13381474809f,  0.962064122248f,
    0.802254508198f, 1.26180942886f,  1.17180569821f,  0.944981930573f, 0.990876405848f,
    0.995903384143f, 0.926972725286f, 0.820534991409f, 0.706020324706f, 1.11019789803f,
    1.3342742857f,   0.990876405848f, 0.831632933426f, 0.77418706195f,  0.725539939514f,
    0.661776842059f, 0.587716619023f, 1.01397751469f,  1.28513006198f,  0.995903384143f,
    0.77418706195f,  0.653238524286f, 0.584635025748f, 0.531064164893f, 0.478717061273f,
    0.867069376285f, 1.13381474809f,  0.926972725286f, 0.725539939514f, 0.584635025748f,
    0.496936637883f, 0.438694579826f, 0.393021669543f, 0.721500455585f, 0.962064122248f,
    0.820534991409f, 0.661776842059f, 0.531064164893f, 0.438694579826f, 0.375820256136f,
    0.330555063063f, 0.593906509971f, 0.802254508198f, 0.706020324706f, 0.587716619023f,
    0.478717061273f, 0.393021669543f, 0.330555063063f, 0.285345396658f,
};

/* Round-toward-zero right shift — matches OD_UNBIASED_RSHIFT32 in
 * xiph/psnr_hvs.c and the CUDA twin's od_dct_rshift().
 * C signed >> of negatives is implementation-defined (rounds toward -inf).
 * Adding the sign bit shifted to position 0 biases negatives toward zero. */
static inline int od_dct_rshift(int a, int b)
{
    return (int)(((uint)a >> (uint)(32 - b)) + (uint)a) >> b;
}

/* Forward 8-point integer DCT — port of od_bin_fdct8 from
 * libvmaf/src/feature/third_party/xiph/psnr_hvs.c:72.
 * All inputs/outputs passed by reference. */
static void od_bin_fdct8(
    thread int &y0, thread int &y1, thread int &y2, thread int &y3,
    thread int &y4, thread int &y5, thread int &y6, thread int &y7,
    int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7)
{
    int t0 = x0, t4 = x1, t2 = x2, t6 = x3;
    int t7 = x4, t3 = x5, t5 = x6, t1 = x7;
    int t1h, t4h, t6h;

    t1 = t0 - t1;
    t1h = od_dct_rshift(t1, 1);
    t0 -= t1h;
    t4 += t5;
    t4h = od_dct_rshift(t4, 1);
    t5 -= t4h;
    t3 = t2 - t3;
    t2 -= od_dct_rshift(t3, 1);
    t6 += t7;
    t6h = od_dct_rshift(t6, 1);
    t7 = t6h - t7;
    t0 += t6h;
    t6 = t0 - t6;
    t2 = t4h - t2;
    t4 = t2 - t4;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t4 += (t0 * 11585 + 8192)  >> 14;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t2 += (t6 * 15137 + 8192)  >> 14;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t3 += (t5 * 19195 + 16384) >> 15;
    t5 += (t3 * 11585 + 8192)  >> 14;
    t3 -= (t5 * 7489  + 4096)  >> 13;
    t7  = od_dct_rshift(t5, 1) - t7;
    t5 -= t7;
    t3  = t1h - t3;
    t1 -= t3;
    t7 += (t1 * 3227  + 16384) >> 15;
    t1 -= (t7 * 6393  + 16384) >> 15;
    t7 += (t1 * 3227  + 16384) >> 15;
    t5 += (t3 * 2485  + 4096)  >> 13;
    t3 -= (t5 * 18205 + 16384) >> 15;
    t5 += (t3 * 2485  + 4096)  >> 13;

    y0 = t0; y1 = t1; y2 = t2; y3 = t3;
    y4 = t4; y5 = t5; y6 = t6; y7 = t7;
}

/* Cooperative 2-pass integer DCT8×8 in threadgroup memory.
 * Pass 1: lanes 0..7 read column i of blk, write transposed column DCT
 *         to z[i*8 + 0..7].
 * Pass 2: lanes 0..7 read column i of z, write blk[i*8 + 0..7].
 * Barrier separates the two passes. */
static void od_bin_fdct8x8_parallel(
    threadgroup int blk[64],
    threadgroup int z[64],
    uint lane)
{
    /* Pass 1 */
    if (lane < 8u) {
        int y0, y1, y2, y3, y4, y5, y6, y7;
        od_bin_fdct8(y0, y1, y2, y3, y4, y5, y6, y7,
                     blk[0*8 + lane], blk[1*8 + lane], blk[2*8 + lane], blk[3*8 + lane],
                     blk[4*8 + lane], blk[5*8 + lane], blk[6*8 + lane], blk[7*8 + lane]);
        z[lane*8+0] = y0; z[lane*8+1] = y1; z[lane*8+2] = y2; z[lane*8+3] = y3;
        z[lane*8+4] = y4; z[lane*8+5] = y5; z[lane*8+6] = y6; z[lane*8+7] = y7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Pass 2 */
    if (lane < 8u) {
        int y0, y1, y2, y3, y4, y5, y6, y7;
        od_bin_fdct8(y0, y1, y2, y3, y4, y5, y6, y7,
                     z[0*8 + lane], z[1*8 + lane], z[2*8 + lane], z[3*8 + lane],
                     z[4*8 + lane], z[5*8 + lane], z[6*8 + lane], z[7*8 + lane]);
        blk[lane*8+0] = y0; blk[lane*8+1] = y1; blk[lane*8+2] = y2; blk[lane*8+3] = y3;
        blk[lane*8+4] = y4; blk[lane*8+5] = y5; blk[lane*8+6] = y6; blk[lane*8+7] = y7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* Reverse the picture_copy float normalisation to recover the integer
 * sample value.  8-bit: scale=1; 10-bit: /4 → ×4; 12-bit: /16 → ×16. */
static inline int sample_to_int(float v, uint bpc)
{
    if (bpc == 8u)  { return (int)(v + 0.5f); }
    if (bpc == 10u) { return (int)(v * 4.0f + 0.5f); }
    return (int)(v * 16.0f + 0.5f);
}

/* -----------------------------------------------------------------------
 *  Kernel: psnr_hvs
 *
 *  One threadgroup per output 8×8 image block (step = 7 pixels).
 *  8×8 = 64 threads per threadgroup.
 *
 *  Thread layout:
 *   - All 64 threads cooperatively load ref/dist samples into shared mem
 *     and fill the DCT input buffers (dct_s / dct_d).
 *   - Thread 0 serially computes means and variances in the CPU's exact
 *     i,j summation order (matches CUDA/Vulkan twins).
 *   - Lanes 0..7 run the two integer DCT passes in parallel.
 *   - Thread 0 resumes for mask computation and final float reduction.
 *   - Thread 0 writes one float partial to partials_out.
 * ----------------------------------------------------------------------- */
kernel void psnr_hvs(
    const device float  *ref_in       [[buffer(0)]],
    const device float  *dist_in      [[buffer(1)]],
    device       float  *partials_out [[buffer(2)]],
    constant     uint4  &params       [[buffer(3)]],
    constant     uint2  &bpc_plane    [[buffer(4)]],
    uint2  tgid  [[threadgroup_position_in_grid]],
    uint2  tid   [[thread_position_in_threadgroup]],
    uint   lid   [[thread_index_in_threadgroup]])
{
    const uint width        = params.x;
    const uint height       = params.y;
    const uint num_blocks_x = params.z;
    const uint num_blocks_y = params.w;
    const uint bpc          = bpc_plane.x;
    const uint plane        = bpc_plane.y;

    const uint blk_x = tgid.x;
    const uint blk_y = tgid.y;
    const uint lx    = tid.x;
    const uint ly    = tid.y;

    const uint x0 = blk_x * 7u;
    const uint y0 = blk_y * 7u;

    const bool valid_block =
        (blk_x < num_blocks_x && blk_y < num_blocks_y &&
         x0 + 7u < width && y0 + 7u < height);

    threadgroup int s_ref[64];
    threadgroup int s_dist[64];
    threadgroup int dct_s[64];
    threadgroup int dct_d[64];
    threadgroup int z_s[64];
    threadgroup int z_d[64];

    int my_ref  = 0;
    int my_dist = 0;
    if (valid_block) {
        const uint sx      = x0 + lx;
        const uint sy      = y0 + ly;
        const uint src_idx = sy * width + sx;
        my_ref  = sample_to_int(ref_in[src_idx],  bpc);
        my_dist = sample_to_int(dist_in[src_idx], bpc);
    }
    s_ref[lid]  = my_ref;
    s_dist[lid] = my_dist;
    dct_s[lid]  = my_ref;
    dct_d[lid]  = my_dist;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s_means[4]  = {0.f, 0.f, 0.f, 0.f};
    float d_means[4]  = {0.f, 0.f, 0.f, 0.f};
    float s_vars[4]   = {0.f, 0.f, 0.f, 0.f};
    float d_vars[4]   = {0.f, 0.f, 0.f, 0.f};
    float s_gmean     = 0.f;
    float d_gmean     = 0.f;
    float s_gvar      = 0.f;
    float d_gvar      = 0.f;
    float s_mc        = 0.f;
    float d_mc        = 0.f;

    if (lid == 0u) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                s_gmean += (float)s_ref[i * 8 + j];
                d_gmean += (float)s_dist[i * 8 + j];
                s_means[sub] += (float)s_ref[i * 8 + j];
                d_means[sub] += (float)s_dist[i * 8 + j];
            }
        }
        s_gmean /= 64.f;
        d_gmean /= 64.f;
        for (int i = 0; i < 4; ++i) {
            s_means[i] /= 16.f;
            d_means[i] /= 16.f;
        }
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                const float ds = (float)s_ref[i*8+j]  - s_gmean;
                const float dd = (float)s_dist[i*8+j] - d_gmean;
                s_gvar += ds * ds;
                d_gvar += dd * dd;
                const float qs = (float)s_ref[i*8+j]  - s_means[sub];
                const float qd = (float)s_dist[i*8+j] - d_means[sub];
                s_vars[sub] += qs * qs;
                d_vars[sub] += qd * qd;
            }
        }
        s_gvar *= 1.f / 63.f * 64.f;
        d_gvar *= 1.f / 63.f * 64.f;
        for (int i = 0; i < 4; ++i) {
            s_vars[i] *= 1.f / 15.f * 16.f;
            d_vars[i] *= 1.f / 15.f * 16.f;
        }
        if (s_gvar > 0.f) {
            s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
        }
        if (d_gvar > 0.f) {
            d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;
        }
    }

    od_bin_fdct8x8_parallel(dct_s, z_s, lid);
    od_bin_fdct8x8_parallel(dct_d, z_d, lid);

    if (lid != 0u) { return; }

    constant float *csf = (plane == 0u) ? CSF_Y :
                          (plane == 1u) ? CSF_CB : CSF_CR;

    float mask[64];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const float c = csf[i * 8 + j];
            const float m = c * 0.3885746225901003f;
            mask[i * 8 + j] = m * m;
        }
    }
    for (int i = 0; i < 8; ++i) {
        const int j0 = (i == 0) ? 1 : 0;
        for (int j = j0; j < 8; ++j) {
            const int sq = dct_s[i*8+j] * dct_s[i*8+j];
            s_mc += (float)sq * mask[i*8+j];
        }
    }
    for (int i = 0; i < 8; ++i) {
        const int j0 = (i == 0) ? 1 : 0;
        for (int j = j0; j < 8; ++j) {
            const int sq = dct_d[i*8+j] * dct_d[i*8+j];
            d_mc += (float)sq * mask[i*8+j];
        }
    }
    float sm = sqrt(s_mc * s_gvar) / 32.f;
    const float dm = sqrt(d_mc * d_gvar) / 32.f;
    if (dm > sm) { sm = dm; }
    const float thresh = sm;

    float ret = 0.f;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const float c = csf[i * 8 + j];
            float err = fabs((float)dct_s[i*8+j] - (float)dct_d[i*8+j]);
            if (i != 0 || j != 0) {
                const float t = thresh / mask[i*8+j];
                err = (err < t) ? 0.f : err - t;
            }
            ret += (err * c) * (err * c);
        }
    }
    if (!valid_block) { ret = 0.f; }

    const uint slot = blk_y * num_blocks_x + blk_x;
    partials_out[slot] = ret;
}
