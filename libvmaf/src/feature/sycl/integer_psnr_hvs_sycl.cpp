/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  psnr_hvs feature extractor on the SYCL backend
 *  (T7-23 / ADR-0188 / ADR-0191, GPU long-tail batch 2 part 3c).
 *  SYCL twin of psnr_hvs_vulkan (PR #143) and psnr_hvs_cuda
 *  (this PR's batch 2 part 3b).
 *
 *  Self-contained submit/collect — does NOT register with
 *  vmaf_sycl_graph_register because shared_frame is luma-only
 *  packed at uint width and psnr_hvs needs picture_copy-
 *  normalised float planes for all three planes (Y, Cb, Cr).
 *  Same pattern as ssim_sycl / ms_ssim_sycl.
 *
 *  Per-plane single-dispatch design — one work-group per output
 *  8×8 image block (step=7), 64 threads/WG. Cooperative load +
 *  thread-0-serial reductions matching CPU's exact i,j summation
 *  order (locks float bit-order to CPU's calc_psnrhvs).
 *
 *  fp64-free (Intel Arc A380 lacks native fp64 — same constraint
 *  as ssim_sycl / ms_ssim_sycl).
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "../picture_copy.h"
#include "sycl/common.h"
}

namespace
{

static constexpr int PSNR_HVS_BLOCK = 8;
static constexpr int PSNR_HVS_STEP = 7;
static constexpr int PSNR_HVS_NUM_PLANES = 3;
static constexpr size_t WG_DIM = 8;

static constexpr float CSF_TABLES[3][64] = {
    /* Y */
    {1.6193873005f,   2.2901594831f,   2.08509755623f,  1.48366094411f,  1.00227514334f,
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
     0.226951348204f, 0.17408067321f,  0.136153931001f, 0.109083846276f},
    /* Cb */
    {1.91113096927f,  2.46074210438f,  1.18284184739f,  1.14982565193f,  1.05017074788f,
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
     0.495804539034f, 0.407050308965f, 0.342353999733f, 0.295530605237f},
    /* Cr */
    {2.03871978502f,  2.62502345193f,  1.26180942886f,  1.11019789803f,  1.01397751469f,
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
     0.478717061273f, 0.393021669543f, 0.330555063063f, 0.285345396658f}};

struct PsnrHvsStateSycl {
    unsigned width[PSNR_HVS_NUM_PLANES];
    unsigned height[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks[PSNR_HVS_NUM_PLANES];
    unsigned bpc;
    int32_t samplemax_sq;
    /* enable_chroma: when false, only the luma (Y) plane is dispatched.
     * Default true mirrors CPU integer_psnr_hvs — see ADR-0453. */
    bool enable_chroma;
    /* n_active_planes: 1 when enable_chroma=false or YUV400P, else 3. */
    unsigned n_active_planes;

    VmafSyclState *sycl_state;

    /* Host pinned float planes for picture_copy upload. */
    float *h_ref[PSNR_HVS_NUM_PLANES];
    float *h_dist[PSNR_HVS_NUM_PLANES];
    /* Device USM ref / dist / partials × 3 planes. */
    float *d_ref[PSNR_HVS_NUM_PLANES];
    float *d_dist[PSNR_HVS_NUM_PLANES];
    float *d_partials[PSNR_HVS_NUM_PLANES];
    float *h_partials[PSNR_HVS_NUM_PLANES];

    bool has_pending;
    unsigned pending_index;
    VmafDictionary *feature_name_dict;
};

/* OD_UNBIASED_RSHIFT32 — round-to-zero right shift. */
static inline int od_dct_rshift(int a, int b)
{
    return (int)(((unsigned int)a >> (32 - b)) + (unsigned int)a) >> b;
}

static void od_bin_fdct8(int &y0, int &y1, int &y2, int &y3, int &y4, int &y5, int &y6, int &y7,
                         int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7)
{
    int t0 = x0;
    int t4 = x1;
    int t2 = x2;
    int t6 = x3;
    int t7 = x4;
    int t3 = x5;
    int t5 = x6;
    int t1 = x7;
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
    t4 += (t0 * 11585 + 8192) >> 14;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t2 += (t6 * 15137 + 8192) >> 14;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t3 += (t5 * 19195 + 16384) >> 15;
    t5 += (t3 * 11585 + 8192) >> 14;
    t3 -= (t5 * 7489 + 4096) >> 13;
    t7 = od_dct_rshift(t5, 1) - t7;
    t5 -= t7;
    t3 = t1h - t3;
    t1 -= t3;
    t7 += (t1 * 3227 + 16384) >> 15;
    t1 -= (t7 * 6393 + 16384) >> 15;
    t7 += (t1 * 3227 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    t3 -= (t5 * 18205 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    y0 = t0;
    y1 = t1;
    y2 = t2;
    y3 = t3;
    y4 = t4;
    y5 = t5;
    y6 = t6;
    y7 = t7;
}

static void od_bin_fdct8x8(int blk[64])
{
    int z[64];
    for (int i = 0; i < 8; i++) {
        int y0, y1, y2, y3, y4, y5, y6, y7;
        od_bin_fdct8(y0, y1, y2, y3, y4, y5, y6, y7, blk[0 * 8 + i], blk[1 * 8 + i], blk[2 * 8 + i],
                     blk[3 * 8 + i], blk[4 * 8 + i], blk[5 * 8 + i], blk[6 * 8 + i],
                     blk[7 * 8 + i]);
        z[i * 8 + 0] = y0;
        z[i * 8 + 1] = y1;
        z[i * 8 + 2] = y2;
        z[i * 8 + 3] = y3;
        z[i * 8 + 4] = y4;
        z[i * 8 + 5] = y5;
        z[i * 8 + 6] = y6;
        z[i * 8 + 7] = y7;
    }
    for (int i = 0; i < 8; i++) {
        int y0, y1, y2, y3, y4, y5, y6, y7;
        od_bin_fdct8(y0, y1, y2, y3, y4, y5, y6, y7, z[0 * 8 + i], z[1 * 8 + i], z[2 * 8 + i],
                     z[3 * 8 + i], z[4 * 8 + i], z[5 * 8 + i], z[6 * 8 + i], z[7 * 8 + i]);
        blk[i * 8 + 0] = y0;
        blk[i * 8 + 1] = y1;
        blk[i * 8 + 2] = y2;
        blk[i * 8 + 3] = y3;
        blk[i * 8 + 4] = y4;
        blk[i * 8 + 5] = y5;
        blk[i * 8 + 6] = y6;
        blk[i * 8 + 7] = y7;
    }
}

static inline int sample_to_int(float v, int bpc)
{
    if (bpc == 8)
        return (int)(v + 0.5f);
    if (bpc == 10)
        return (int)(v * 4.0f + 0.5f);
    return (int)(v * 16.0f + 0.5f);
}

static void launch_psnr_hvs(sycl::queue &q, const float *ref, const float *dist, float *partials,
                            unsigned width, unsigned height, unsigned num_blocks_x,
                            unsigned num_blocks_y, int plane, int bpc)
{
    sycl::nd_range<2> ndr{
        sycl::range<2>{(size_t)num_blocks_y * WG_DIM, (size_t)num_blocks_x * WG_DIM},
        sycl::range<2>{WG_DIM, WG_DIM}};
    const unsigned e_w = width;
    const unsigned e_h = height;
    const unsigned e_nbx = num_blocks_x;
    const unsigned e_nby = num_blocks_y;
    const int e_plane = plane;
    const int e_bpc = bpc;
    const float *e_ref = ref;
    const float *e_dist = dist;
    float *e_partials = partials;

    q.submit([=](sycl::handler &h_) {
        sycl::local_accessor<int, 1> s_ref(sycl::range<1>(64), h_);
        sycl::local_accessor<int, 1> s_dist(sycl::range<1>(64), h_);

        h_.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const size_t blk_y = it.get_group(0);
            const size_t blk_x = it.get_group(1);
            const size_t ly = it.get_local_id(0);
            const size_t lx = it.get_local_id(1);
            const size_t local_idx = ly * 8u + lx;

            const size_t x0 = blk_x * 7u;
            const size_t y0 = blk_y * 7u;
            const bool valid_block = (blk_x < (size_t)e_nbx && blk_y < (size_t)e_nby &&
                                      x0 + 7u < (size_t)e_w && y0 + 7u < (size_t)e_h);

            int my_ref = 0;
            int my_dist = 0;
            if (valid_block) {
                const size_t sx = x0 + lx;
                const size_t sy = y0 + ly;
                const size_t src_idx = sy * (size_t)e_w + sx;
                my_ref = sample_to_int(e_ref[src_idx], e_bpc);
                my_dist = sample_to_int(e_dist[src_idx], e_bpc);
            }
            s_ref[local_idx] = my_ref;
            s_dist[local_idx] = my_dist;
            it.barrier(sycl::access::fence_space::local_space);

            if (local_idx != 0u)
                return;

            int dct_s[64];
            int dct_d[64];
            for (int i = 0; i < 64; i++)
                dct_s[i] = s_ref[i];
            for (int i = 0; i < 64; i++)
                dct_d[i] = s_dist[i];

            float s_means[4] = {0.f, 0.f, 0.f, 0.f};
            float d_means[4] = {0.f, 0.f, 0.f, 0.f};
            float s_vars[4] = {0.f, 0.f, 0.f, 0.f};
            float d_vars[4] = {0.f, 0.f, 0.f, 0.f};
            float s_gmean = 0.f, d_gmean = 0.f;
            float s_gvar = 0.f, d_gvar = 0.f;
            float s_mc = 0.f, d_mc = 0.f;

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    s_gmean += (float)dct_s[i * 8 + j];
                    d_gmean += (float)dct_d[i * 8 + j];
                    s_means[sub] += (float)dct_s[i * 8 + j];
                    d_means[sub] += (float)dct_d[i * 8 + j];
                }
            }
            s_gmean /= 64.f;
            d_gmean /= 64.f;
            for (int i = 0; i < 4; i++)
                s_means[i] /= 16.f;
            for (int i = 0; i < 4; i++)
                d_means[i] /= 16.f;

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    const float ds = (float)dct_s[i * 8 + j] - s_gmean;
                    const float dd = (float)dct_d[i * 8 + j] - d_gmean;
                    s_gvar += ds * ds;
                    d_gvar += dd * dd;
                    const float qs = (float)dct_s[i * 8 + j] - s_means[sub];
                    const float qd = (float)dct_d[i * 8 + j] - d_means[sub];
                    s_vars[sub] += qs * qs;
                    d_vars[sub] += qd * qd;
                }
            }
            s_gvar *= 1.f / 63.f * 64.f;
            d_gvar *= 1.f / 63.f * 64.f;
            for (int i = 0; i < 4; i++)
                s_vars[i] *= 1.f / 15.f * 16.f;
            for (int i = 0; i < 4; i++)
                d_vars[i] *= 1.f / 15.f * 16.f;
            if (s_gvar > 0.f)
                s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
            if (d_gvar > 0.f)
                d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;

            od_bin_fdct8x8(dct_s);
            od_bin_fdct8x8(dct_d);

            float mask[64];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const float c = CSF_TABLES[e_plane][i * 8 + j];
                    const float m = c * 0.3885746225901003f;
                    mask[i * 8 + j] = m * m;
                }
            }
            for (int i = 0; i < 8; i++) {
                const int j0 = (i == 0) ? 1 : 0;
                for (int j = j0; j < 8; j++) {
                    const int sq = dct_s[i * 8 + j] * dct_s[i * 8 + j];
                    s_mc += (float)sq * mask[i * 8 + j];
                }
            }
            for (int i = 0; i < 8; i++) {
                const int j0 = (i == 0) ? 1 : 0;
                for (int j = j0; j < 8; j++) {
                    const int sq = dct_d[i * 8 + j] * dct_d[i * 8 + j];
                    d_mc += (float)sq * mask[i * 8 + j];
                }
            }
            float sm = sycl::sqrt(s_mc * s_gvar) / 32.f;
            const float dm = sycl::sqrt(d_mc * d_gvar) / 32.f;
            if (dm > sm)
                sm = dm;
            const float thresh = sm;

            float ret = 0.f;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const float c = CSF_TABLES[e_plane][i * 8 + j];
                    float err = sycl::fabs((float)dct_s[i * 8 + j] - (float)dct_d[i * 8 + j]);
                    if (i != 0 || j != 0) {
                        const float t = thresh / mask[i * 8 + j];
                        err = err < t ? 0.f : err - t;
                    }
                    ret += (err * c) * (err * c);
                }
            }
            if (!valid_block)
                ret = 0.f;

            const size_t slot = blk_y * (size_t)e_nbx + blk_x;
            e_partials[slot] = ret;
        });
    });
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_psnr_hvs_sycl[] = {
    {
        .name = "enable_chroma",
        .help = "enable psnr_hvs calculation for chroma channels (Cb and Cr); "
                "when false only the luma plane is scored and psnr_hvs equals "
                "psnr_hvs_y (mirrors CPU PR #946 / ADR-0453)",
        .offset = offsetof(PsnrHvsStateSycl, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {0},
};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    auto *s = static_cast<PsnrHvsStateSycl *>(fex->priv);

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_sycl: invalid bitdepth (%u); bpc must be ≤ 12\n",
                 bpc);
        return -EINVAL;
    }
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "psnr_hvs_sycl: YUV400P unsupported (psnr_hvs needs all 3 planes)\n");
        return -EINVAL;
    }
    if (w < (unsigned)PSNR_HVS_BLOCK || h < (unsigned)PSNR_HVS_BLOCK) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_sycl: input %ux%u smaller than 8×8 block\n", w, h);
        return -EINVAL;
    }

    s->bpc = bpc;
    const int32_t samplemax = (1 << bpc) - 1;
    s->samplemax_sq = samplemax * samplemax;

    s->width[0] = w;
    s->height[0] = h;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h >> 1;
        break;
    case VMAF_PIX_FMT_YUV422P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h;
        break;
    case VMAF_PIX_FMT_YUV444P:
        s->width[1] = s->width[2] = w;
        s->height[1] = s->height[2] = h;
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_sycl: unsupported pix_fmt\n");
        return -EINVAL;
    }

    /* Mirror CPU PR #946 / ADR-0453: clamp to luma-only when caller
     * passes enable_chroma=false.  YUV400P is already rejected above,
     * so this covers 4:2:0 / 4:2:2 / 4:4:4 callers that opt out. */
    s->n_active_planes = s->enable_chroma ? (unsigned)PSNR_HVS_NUM_PLANES : 1U;

    for (int p = 0; p < (int)s->n_active_planes; p++) {
        if (s->width[p] < (unsigned)PSNR_HVS_BLOCK || s->height[p] < (unsigned)PSNR_HVS_BLOCK) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "psnr_hvs_sycl: plane %d dims %ux%u smaller than 8×8 block\n", p, s->width[p],
                     s->height[p]);
            return -EINVAL;
        }
        s->num_blocks_x[p] = (s->width[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks_y[p] = (s->height[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks[p] = s->num_blocks_x[p] * s->num_blocks_y[p];
    }

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_sycl: no SYCL state\n");
        return -EINVAL;
    }
    s->sycl_state = fex->sycl_state;

    for (int p = 0; p < (int)s->n_active_planes; p++) {
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * sizeof(float);
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        s->h_ref[p] = (float *)vmaf_sycl_malloc_host(s->sycl_state, plane_bytes);
        s->h_dist[p] = (float *)vmaf_sycl_malloc_host(s->sycl_state, plane_bytes);
        s->d_ref[p] = (float *)vmaf_sycl_malloc_device(s->sycl_state, plane_bytes);
        s->d_dist[p] = (float *)vmaf_sycl_malloc_device(s->sycl_state, plane_bytes);
        s->d_partials[p] = (float *)vmaf_sycl_malloc_device(s->sycl_state, partials_bytes);
        s->h_partials[p] = (float *)vmaf_sycl_malloc_host(s->sycl_state, partials_bytes);
        if (!s->h_ref[p] || !s->h_dist[p] || !s->d_ref[p] || !s->d_dist[p] || !s->d_partials[p] ||
            !s->h_partials[p]) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_sycl: USM allocation failed\n");
            return -ENOMEM;
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    s->has_pending = false;
    return 0;
}

/* Per-plane picture_copy clone (since libvmaf's picture_copy
 * hardcodes plane 0). */
static void picture_copy_plane(float *dst, VmafPicture *pic, int plane, unsigned width,
                               unsigned height)
{
    if (pic->bpc <= 8) {
        const uint8_t *src = (const uint8_t *)pic->data[plane];
        const size_t src_stride = (size_t)pic->stride[plane];
        for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x++) {
                dst[y * width + x] = (float)src[y * src_stride + x];
            }
        }
    } else {
        const float scaler = (pic->bpc == 10) ? 4.0f :
                             (pic->bpc == 12) ? 16.0f :
                             (pic->bpc == 16) ? 256.0f :
                                                1.0f;
        const uint16_t *src = (const uint16_t *)pic->data[plane];
        const size_t src_stride_words = (size_t)pic->stride[plane] / sizeof(uint16_t);
        for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x++) {
                dst[y * width + x] = (float)src[y * src_stride_words + x] / scaler;
            }
        }
    }
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    auto *s = static_cast<PsnrHvsStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    for (int p = 0; p < (int)s->n_active_planes; p++) {
        picture_copy_plane(s->h_ref[p], ref_pic, p, s->width[p], s->height[p]);
        picture_copy_plane(s->h_dist[p], dist_pic, p, s->width[p], s->height[p]);
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * sizeof(float);
        q.memcpy(s->d_ref[p], s->h_ref[p], plane_bytes);
        q.memcpy(s->d_dist[p], s->h_dist[p], plane_bytes);
    }

    for (int p = 0; p < (int)s->n_active_planes; p++) {
        launch_psnr_hvs(q, s->d_ref[p], s->d_dist[p], s->d_partials[p], s->width[p], s->height[p],
                        s->num_blocks_x[p], s->num_blocks_y[p], p, (int)s->bpc);
    }

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<PsnrHvsStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    for (int p = 0; p < (int)s->n_active_planes; p++) {
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        q.memcpy(s->h_partials[p], s->d_partials[p], partials_bytes);
    }
    q.wait();

    /* Per-plane reduction matching CPU's float `ret` register. */
    double plane_score[PSNR_HVS_NUM_PLANES];
    for (int p = 0; p < (int)s->n_active_planes; p++) {
        float ret = 0.0f;
        for (unsigned i = 0; i < s->num_blocks[p]; i++)
            ret += s->h_partials[p][i];
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    int err = 0;
    static const char *plane_features[PSNR_HVS_NUM_PLANES] = {"psnr_hvs_y", "psnr_hvs_cb",
                                                              "psnr_hvs_cr"};
    for (int p = 0; p < (int)s->n_active_planes; p++) {
        const double db = 10.0 * (-1.0 * std::log10(plane_score[p]));
        err |= vmaf_feature_collector_append(feature_collector, plane_features[p], db, index);
    }
    /* When enable_chroma=false, psnr_hvs == psnr_hvs_y (luma-only).
     * When enable_chroma=true, use the standard 0.8·Y + 0.1·(Cb+Cr)
     * weighted combination (matches CPU integer_psnr_hvs). */
    const double combined = (s->n_active_planes == 1U) ?
                                plane_score[0] :
                                0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
    const double db_combined = 10.0 * (-1.0 * std::log10(combined));
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs", db_combined, index);
    return err;
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<PsnrHvsStateSycl *>(fex->priv);
    if (s->sycl_state) {
        for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
            if (s->h_ref[p])
                vmaf_sycl_free(s->sycl_state, s->h_ref[p]);
            if (s->h_dist[p])
                vmaf_sycl_free(s->sycl_state, s->h_dist[p]);
            if (s->d_ref[p])
                vmaf_sycl_free(s->sycl_state, s->d_ref[p]);
            if (s->d_dist[p])
                vmaf_sycl_free(s->sycl_state, s->d_dist[p]);
            if (s->d_partials[p])
                vmaf_sycl_free(s->sycl_state, s->d_partials[p]);
            if (s->h_partials[p])
                vmaf_sycl_free(s->sycl_state, s->h_partials[p]);
        }
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_psnr_hvs_sycl[] = {"psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr",
                                                        "psnr_hvs", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_psnr_hvs_sycl = {
    .name = "psnr_hvs_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_psnr_hvs_sycl,
    .priv_size = sizeof(PsnrHvsStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_psnr_hvs_sycl,
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};

} /* extern "C" */
