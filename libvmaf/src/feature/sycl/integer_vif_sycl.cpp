/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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

/**
 * SYCL/DPC++ VIF (Visual Information Fidelity) feature extractor.
 *
 * Implements 4-scale separable Gaussian filtering using SYCL kernels.
 * Based on separable VIF algorithm (filter1d_vert + filter1d_hori).
 *
 * Algorithm:
 *   - For each of 4 scales, apply a separable symmetric Gaussian filter:
 *     1. Vertical pass: filter ref/dis, compute mu, sigma^2, cross-products
 *        Simultaneously compute reduction-filtered data for next scale
 *     2. Horizontal pass: complete the 2D convolution, compute VIF statistics
 *        Downsample result into rd_ref/rd_dis for next scale
 *   - Accumulate 7 int64 statistics per scale on the GPU
 *   - Download and compute final VIF scores on the CPU
 *
 * Pattern: init -> submit (non-blocking GPU work) -> collect (wait + scores)
 * Uses shared frame buffers (ref/dis Y planes uploaded once per frame).
 */

#include <sycl/sycl.hpp>

#include "sycl_compat.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "sycl/common.h"
#include "log.h"
}

// NOLINTBEGIN(misc-use-anonymous-namespace, misc-use-internal-linkage): see
// integer_motion_sycl.cpp for the rationale — C-style `static` is required
// because the entry-point function addresses are consumed via the
// `extern "C" VmafFeatureExtractor` struct at the bottom of this TU; the
// C-API boundary is the load-bearing invariant per CLAUDE.md §12 r12.

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

static constexpr int VIF_NUM_SCALES = 4;
static constexpr int VIF_FILTER_MAX_WIDTH = 17;
static constexpr int VIF_FILTER_TABLE_PAD = 18;

// Filter coefficients per scale (sum to 65536 = 2^16)
static constexpr uint32_t vif_filter1d_table[VIF_NUM_SCALES][VIF_FILTER_TABLE_PAD] = {
    {489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935,
     489, 0},
    {1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244, 0},
    {3571, 16004, 26386, 16004, 3571, 0},
    {10904, 43728, 10904, 0},
};

static constexpr int vif_fwidth[VIF_NUM_SCALES] = {17, 9, 5, 3};
static constexpr int vif_fwidth_rd[VIF_NUM_SCALES] = {9, 5, 3, 0};

static constexpr int64_t SIGMA_NSQ = 131072; // 2 * 65536

static constexpr int LOG2_LUT_SIZE = 32768;

/* ------------------------------------------------------------------ */
/* Per-scale accumulator struct (7 x int64_t = 56 bytes)              */
/* ------------------------------------------------------------------ */

struct vif_accums {
    int64_t x;
    int64_t x2;
    int64_t num_x;
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
};
static constexpr int ACCUM_FIELDS = 7;

/* ------------------------------------------------------------------ */
/* Extractor private state                                             */
/* ------------------------------------------------------------------ */

struct VifStateSycl {
    unsigned width, height;
    unsigned bpc;

    bool debug;
    double vif_enhn_gain_limit;

    VmafDictionary *feature_name_dict;

    // SYCL device buffers
    uint32_t *d_tmp_mu1;
    uint32_t *d_tmp_mu2;
    uint32_t *d_tmp_ref;
    uint32_t *d_tmp_dis;
    uint32_t *d_tmp_ref_dis;
    uint32_t *d_tmp_ref_convol;
    uint32_t *d_tmp_dis_convol;
    uint32_t *d_rd_ref;
    uint32_t *d_rd_dis;
    int64_t *d_accum;     // 4 scales x 7 int64 = 224 bytes
    uint32_t *d_log2_lut; // 32768 entries

    // Host-side accumulator download buffer
    int64_t *h_accum;

    // Subgroup size selection (auto-detected at init)
    bool use_simd16;

    // Fused V+H kernel mode: uses SLM intermediates, skips tmp buffers.
    // Saves ~70 MB VRAM at 4K but may be slower on some GPUs due to
    // SLM pressure and reduced occupancy.
    bool use_fused;

    // Deferred submit/collect state
    unsigned pending_index;
    bool has_pending;
};

/* ------------------------------------------------------------------ */
/* Options                                                             */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifStateSycl, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = true},
    },
    {
        .name = "vif_enhn_gain_limit",
        .help = "enhancement gain imposed on VIF, must be >= 1.0, "
                "where 1.0 means the gain is unrestricted",
        .offset = offsetof(VifStateSycl, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 100.0},
        .min = 1.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_fused",
        .help = "use fused V+H kernel (saves VRAM, may be slower on some GPUs)",
        .offset = offsetof(VifStateSycl, use_fused),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {nullptr}};

/* ------------------------------------------------------------------ */
/* Device-side helpers (used inside kernels)                           */
/* ------------------------------------------------------------------ */

static inline int dev_mirror(int idx, int sup)
{
    if (idx < 0)
        return -idx;
    if (idx >= sup)
        return 2 * (sup - 1) - idx;
    return idx;
}

static inline uint32_t dev_get_best16_from32(uint32_t val, int &exp_out)
{
    if (val == 0) {
        exp_out = 0;
        return 0;
    }
    int msb = 31;
    while (msb >= 0 && !((val >> msb) & 1))
        msb--;
    int k = msb - 15;
    if (k < 0)
        k = 0;
    val >>= k;
    exp_out = -k;
    return val & 0xFFFF;
}

static inline uint32_t dev_get_best16_from64(uint64_t val, int &exp_out)
{
    if (val == 0) {
        exp_out = 0;
        return 0;
    }
    int clz = 0;
    uint64_t tmp = val;
    if (!(tmp >> 32)) {
        clz += 32;
        tmp <<= 32;
    }
    if (!(tmp >> 48)) {
        clz += 16;
        tmp <<= 16;
    }
    if (!(tmp >> 56)) {
        clz += 8;
        tmp <<= 8;
    }
    if (!(tmp >> 60)) {
        clz += 4;
        tmp <<= 4;
    }
    if (!(tmp >> 62)) {
        clz += 2;
        tmp <<= 2;
    }
    if (!(tmp >> 63)) {
        clz += 1;
    }

    int const k = clz;
    if (k > 48) {
        val <<= (k - 48);
        exp_out = k - 48;
    } else if (k < 47) {
        val >>= (48 - k);
        exp_out = -(48 - k);
    } else {
        exp_out = 0;
        if (val >> 16) {
            val >>= 1;
            exp_out = -1;
        }
    }
    return static_cast<uint32_t>(val) & 0xFFFF;
}

/* ------------------------------------------------------------------ */ /* Profiling helper                                                    */
/* ------------------------------------------------------------------ */

static inline void sycl_profile_event(VmafSyclState *state, const char *name, sycl::event ev)
{
    if (vmaf_sycl_profiling_is_enabled(state)) {
        ev.wait();
        uint64_t t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        vmaf_sycl_profiling_record(state, name, t1 - t0);
    }
}

/* ------------------------------------------------------------------ */ /* SYCL Kernel: Vertical Pass                                         */
/* ------------------------------------------------------------------ */

/**
 * Vertical 1D convolution for one scale.
 *
 * Uses shared local memory (SLM) tiling for data reuse: cooperatively loads
 * a tile of (WG_Y + fw - 1) × WG_X pixels into SLM, then each thread reads
 * from SLM instead of global memory. Adjacent output rows share fw-1 input
 * rows, so this reduces global memory reads by ~fw× vs naive per-thread loads.
 *
 * Interior workgroups (no mirror padding needed) use an optimized fast path
 * that skips boundary checks during tile load.
 */
template <int SCALE>
// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch / lifecycle entry — body is dominated by accessor declarations + a single `parallel_for` lambda. Splitting either inlines via macro (no readability win) or introduces a free function the compiler cannot inline back into the device kernel. Keeping it large is the pattern shared across every SYCL TU in this fork (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278).
static sycl::event launch_vif_vert_impl(sycl::queue &q, const void *ref_data, const void *dis_data,
                                        unsigned width, unsigned height, unsigned src_stride,
                                        unsigned bpc, uint32_t *tmp_mu1, uint32_t *tmp_mu2,
                                        uint32_t *tmp_ref, uint32_t *tmp_dis, uint32_t *tmp_ref_dis,
                                        uint32_t *tmp_ref_convol, uint32_t *tmp_dis_convol)
{
    constexpr int FW = vif_fwidth[SCALE];
    constexpr int FW_RD = vif_fwidth_rd[SCALE];
    constexpr int HALF_FW = FW / 2;
    constexpr int RD_START = (FW - FW_RD) / 2;
    constexpr int TILE_H = 16 + FW - 1;
    const unsigned stride_tmp = width;

    unsigned shift_vp;
    unsigned add_shift_round_vp;
    unsigned shift_vp_sq;
    unsigned add_shift_round_vp_sq;
    if constexpr (SCALE == 0) {
        shift_vp = bpc;
        add_shift_round_vp = 1u << (bpc - 1);
        shift_vp_sq = (bpc - 8) * 2;
        add_shift_round_vp_sq = (bpc == 8) ? 0 : (1u << (shift_vp_sq - 1));
    } else {
        shift_vp = 16;
        add_shift_round_vp = 32768;
        shift_vp_sq = 16;
        add_shift_round_vp_sq = 32768;
    }

    uint32_t fcoeff[VIF_FILTER_MAX_WIDTH];
    for (int i = 0; i < FW; i++)
        fcoeff[i] = vif_filter1d_table[SCALE][i];

    uint32_t fcoeff_rd[VIF_FILTER_MAX_WIDTH] = {};
    if constexpr (FW_RD > 0) {
        for (int i = 0; i < FW_RD; i++)
            fcoeff_rd[i] = vif_filter1d_table[SCALE + 1][i];
    }

    constexpr int WG_X = 16;
    constexpr int WG_Y = 16;
    sycl::range<2> global(((height + WG_Y - 1) / WG_Y) * WG_Y, ((width + WG_X - 1) / WG_X) * WG_X);
    sycl::range<2> local(WG_Y, WG_X);

    auto p_ref = ref_data;
    auto p_dis = dis_data;
    auto e_bpc = bpc;
    auto e_w = width;
    auto e_h = height;
    auto e_src_stride = src_stride;

    return q.submit([&](sycl::handler &cgh) {
        // SLM for ref and dis tiles
        sycl::local_accessor<uint32_t, 2> s_ref(sycl::range<2>(TILE_H, WG_X), cgh);
        sycl::local_accessor<uint32_t, 2> s_dis(sycl::range<2>(TILE_H, WG_X), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            const int gx = item.get_global_id(1);
            const int gy = item.get_global_id(0);
            const unsigned lid = item.get_local_linear_id();
            const unsigned lx = item.get_local_id(1);
            const unsigned ly = item.get_local_id(0);

            // --- Phase 1: Cooperative tile load into SLM ---
            int tile_origin_y = (int)(item.get_group(0) * WG_Y) - HALF_FW;
            int tile_col_x = (int)(item.get_group(1) * WG_X);

            constexpr unsigned tile_elems = TILE_H * WG_X;
            constexpr unsigned wg_size = WG_X * WG_Y;

            // Read a pixel from global memory
            auto read_global = [&](const void *src, int y, int x) -> uint32_t {
                if constexpr (SCALE == 0) {
                    if (e_bpc <= 8) {
                        return static_cast<const uint8_t *>(src)[y * e_src_stride + x];
                    } else {
                        return static_cast<const uint16_t *>(src)[y * (e_src_stride / 2) + x];
                    }
                } else {
                    return static_cast<const uint32_t *>(src)[y * e_src_stride + x] & 0xFFFF;
                }
            };

            // Interior workgroups: skip mirror() (~97% of WGs at 4K)
            bool interior_wg = (tile_origin_y >= 0) && (tile_origin_y + TILE_H <= (int)e_h) &&
                               (tile_col_x + WG_X <= (int)e_w);

            if (interior_wg) {
                for (unsigned i = lid; i < tile_elems; i += wg_size) {
                    unsigned const tr = i / WG_X;
                    unsigned const tc = i % WG_X;
                    int const px = tile_col_x + (int)tc;
                    int const py = tile_origin_y + (int)tr;
                    s_ref[tr][tc] = read_global(p_ref, py, px);
                    s_dis[tr][tc] = read_global(p_dis, py, px);
                }
            } else {
                for (unsigned i = lid; i < tile_elems; i += wg_size) {
                    unsigned const tr = i / WG_X;
                    unsigned const tc = i % WG_X;
                    int px = tile_col_x + (int)tc;
                    int const py = dev_mirror(tile_origin_y + (int)tr, (int)e_h);
                    if (px < (int)e_w) {
                        px = dev_mirror(px, (int)e_w);
                        s_ref[tr][tc] = read_global(p_ref, py, px);
                        s_dis[tr][tc] = read_global(p_dis, py, px);
                    } else {
                        s_ref[tr][tc] = 0;
                        s_dis[tr][tc] = 0;
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            // --- Phase 2: Vertical convolution from SLM ---
            if (gx >= (int)e_w || gy >= (int)e_h)
                return;

            // mu and rd accumulators fit uint32 (filter sums to 65536,
            // max pixel 65535 → 65536*65535 < 2^32).
            uint32_t acc_mu1 = 0;
            uint32_t acc_mu2 = 0;
            uint64_t acc_ref = 0;
            uint64_t acc_dis = 0;
            uint64_t acc_ref_dis = 0;
            // NOLINTNEXTLINE(readability-isolate-declaration): SYCL chained zero-init; splitting hides the symmetry of the parallel reduction state.
            uint32_t acc_ref_rd = 0, acc_dis_rd = 0;

#pragma unroll
            for (int fi = 0; fi < FW; fi++) {
                // Thread's local Y + fi maps to the correct tile row
                uint32_t rv = s_ref[ly + fi][lx];
                uint32_t dv = s_dis[ly + fi][lx];
                uint32_t const fc = fcoeff[fi];

                // Keep intermediate as uint32 — widen only for squared terms
                uint32_t const img_coeff_ref = fc * rv;
                uint32_t const img_coeff_dis = fc * dv;

                acc_mu1 += img_coeff_ref;
                acc_mu2 += img_coeff_dis;
                acc_ref += (uint64_t)img_coeff_ref * rv;
                acc_dis += (uint64_t)img_coeff_dis * dv;
                acc_ref_dis += (uint64_t)img_coeff_ref * dv;

                if constexpr (FW_RD > 0) {
                    if (fi >= RD_START && fi < RD_START + FW_RD) {
                        uint32_t fc_rd = fcoeff_rd[fi - RD_START];
                        acc_ref_rd += fc_rd * rv;
                        acc_dis_rd += fc_rd * dv;
                    }
                }
            }

            // Quantize mu
            uint32_t const mu1_out = (uint32_t)((acc_mu1 + add_shift_round_vp) >> shift_vp);
            uint32_t const mu2_out = (uint32_t)((acc_mu2 + add_shift_round_vp) >> shift_vp);

            // Quantize squared terms
            uint32_t ref_out;
            uint32_t dis_out;
            uint32_t ref_dis_out;
            if (shift_vp_sq > 0) {
                ref_out = (uint32_t)((acc_ref + add_shift_round_vp_sq) >> shift_vp_sq);
                dis_out = (uint32_t)((acc_dis + add_shift_round_vp_sq) >> shift_vp_sq);
                ref_dis_out = (uint32_t)((acc_ref_dis + add_shift_round_vp_sq) >> shift_vp_sq);
            } else {
                ref_out = (uint32_t)acc_ref;
                dis_out = (uint32_t)acc_dis;
                ref_dis_out = (uint32_t)acc_ref_dis;
            }

            // Reduction filter output
            uint32_t ref_rd_out = 0;
            uint32_t dis_rd_out = 0;
            if constexpr (FW_RD > 0) {
                ref_rd_out = (uint32_t)((acc_ref_rd + add_shift_round_vp) >> shift_vp);
                dis_rd_out = (uint32_t)((acc_dis_rd + add_shift_round_vp) >> shift_vp);
            }

            unsigned idx = gy * stride_tmp + gx;
            tmp_mu1[idx] = mu1_out;
            tmp_mu2[idx] = mu2_out;
            tmp_ref[idx] = ref_out;
            tmp_dis[idx] = dis_out;
            tmp_ref_dis[idx] = ref_dis_out;
            tmp_ref_convol[idx] = ref_rd_out;
            tmp_dis_convol[idx] = dis_rd_out;
        });
    });
}

static sycl::event launch_vif_vert(sycl::queue &q, const void *ref_data, const void *dis_data,
                                   int scale, unsigned width, unsigned height, unsigned src_stride,
                                   unsigned bpc, uint32_t *tmp_mu1, uint32_t *tmp_mu2,
                                   uint32_t *tmp_ref, uint32_t *tmp_dis, uint32_t *tmp_ref_dis,
                                   uint32_t *tmp_ref_convol, uint32_t *tmp_dis_convol)
{
    switch (scale) {
    case 0:
        return launch_vif_vert_impl<0>(q, ref_data, dis_data, width, height, src_stride, bpc,
                                       tmp_mu1, tmp_mu2, tmp_ref, tmp_dis, tmp_ref_dis,
                                       tmp_ref_convol, tmp_dis_convol);
    case 1:
        return launch_vif_vert_impl<1>(q, ref_data, dis_data, width, height, src_stride, bpc,
                                       tmp_mu1, tmp_mu2, tmp_ref, tmp_dis, tmp_ref_dis,
                                       tmp_ref_convol, tmp_dis_convol);
    case 2:
        return launch_vif_vert_impl<2>(q, ref_data, dis_data, width, height, src_stride, bpc,
                                       tmp_mu1, tmp_mu2, tmp_ref, tmp_dis, tmp_ref_dis,
                                       tmp_ref_convol, tmp_dis_convol);
    default:
        return launch_vif_vert_impl<3>(q, ref_data, dis_data, width, height, src_stride, bpc,
                                       tmp_mu1, tmp_mu2, tmp_ref, tmp_dis, tmp_ref_dis,
                                       tmp_ref_convol, tmp_dis_convol);
    }
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: Horizontal Pass + VIF (subgroup-optimized, v2)        */
/* ------------------------------------------------------------------ */

/**
 * Optimized VIF horizontal kernel using two-phase reduction:
 *   Phase 1: Subgroup shuffle reduction (hardware-level, no barriers)
 *   Phase 2: Small local-memory tree across subgroup leaders
 *
 * With SIMD-32 subgroups (Arc Xe-HPG), a 256-thread WG has 8 subgroups.
 * Phase 1 reduces 32→1 per subgroup (5 shuffle steps, no barriers).
 * Phase 2 reduces 8→1 across leaders (3 barrier steps, only 8 threads active).
 * Total barriers: 3 (vs 8 in the original tree reduction).
 *
 * Additional optimization: uses reqd_sub_group_size(32) on Xe-HPG
 * to ensure SIMD-32 allocation, maximizing EU utilization.
 */
template <int SCALE, int SG_SIZE>
static sycl::event
// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch lambda body, see comment block above (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278)
launch_vif_hori_impl(sycl::queue &q, unsigned width, unsigned height, float vif_enhn_gain_limit,
                     const uint32_t *tmp_mu1, const uint32_t *tmp_mu2, const uint32_t *tmp_ref,
                     const uint32_t *tmp_dis, const uint32_t *tmp_ref_dis,
                     const uint32_t *tmp_ref_convol, const uint32_t *tmp_dis_convol, int64_t *accum,
                     uint32_t *rd_ref, uint32_t *rd_dis, const uint32_t *log2_lut)
{
    constexpr int FW = vif_fwidth[SCALE];
    constexpr int FW_RD = vif_fwidth_rd[SCALE];
    constexpr int HALF_FW = FW / 2;
    constexpr int RD_START = (FW - FW_RD) / 2;
    const unsigned stride_tmp = width;

    uint32_t fcoeff[VIF_FILTER_MAX_WIDTH];
    for (int i = 0; i < FW; i++)
        fcoeff[i] = vif_filter1d_table[SCALE][i];

    uint32_t fcoeff_rd[VIF_FILTER_MAX_WIDTH] = {};
    if constexpr (FW_RD > 0) {
        for (int i = 0; i < FW_RD; i++)
            fcoeff_rd[i] = vif_filter1d_table[SCALE + 1][i];
    }

    constexpr int WG_X = 16;
    constexpr int WG_Y = 16;
    // Max subgroups: 256 / min_sg_size(8) = 32
    constexpr int MAX_SUBGROUPS = 32;
    sycl::range<2> global(((height + WG_Y - 1) / WG_Y) * WG_Y, ((width + WG_X - 1) / WG_X) * WG_X);
    sycl::range<2> local(WG_Y, WG_X);

    auto e_w = width;
    auto e_h = height;

    return q.submit([&](sycl::handler &cgh) {
        // Local memory only for subgroup leader partial sums
        sycl::local_accessor<int64_t, 1> lmem(sycl::range<1>(ACCUM_FIELDS * MAX_SUBGROUPS), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) VMAF_SYCL_REQD_SG_SIZE(SG_SIZE) {
                const int gx = item.get_global_id(1);
                const int gy = item.get_global_id(0);
                const bool valid = (gx < (int)e_w && gy < (int)e_h);

                sycl::sub_group sg = item.get_sub_group();
                const uint32_t sg_id = sg.get_group_linear_id();
                const uint32_t sg_lid = sg.get_local_linear_id();
                const uint32_t n_subgroups = sg.get_group_linear_range();

                // Accumulators (0 for out-of-bounds threads)
                int64_t t_x = 0;
                int64_t t_x2 = 0;
                int64_t t_num_x = 0;
                int64_t t_num_log = 0;
                int64_t t_den_log = 0;
                int64_t t_num_non_log = 0;
                int64_t t_den_non_log = 0;
                // rd accumulators: uint32 safe (max = 65536 * 65535 < 2^32)
                uint32_t h_ref_rd = 0;
                uint32_t h_dis_rd = 0;

                if (valid) {

                    auto mx = [&](int x) -> int {
                        if (x < 0)
                            return -x;
                        if (x >= (int)e_w)
                            return 2 * ((int)e_w - 1) - x;
                        return x;
                    };

                    // Horizontal convolution with symmetric filter optimization.
                    // All VIF filter kernels are symmetric: fcoeff[i] == fcoeff[fw-1-i].
                    // Sum symmetric tap pairs before multiplying, halving the
                    // multiply count (17→9 muls at scale 0, 9→5, 5→3, 3→2).
                    //
                    // mu accumulators: uint32 safe because max = 65536 * 65535 < 2^32
                    // (vert output ≤ 65535 regardless of bpc; filter sums to 65536).
                    // rd accumulators: uint32 safe, same analysis as mu.
                    // ref/dis/ref_dis: need uint64 (vert output can be up to 2^32-1).
                    uint32_t h_mu1 = 0;
                    uint32_t h_mu2 = 0;
                    uint64_t h_ref = 0;
                    uint64_t h_dis = 0;
                    uint64_t h_ref_dis = 0;
                    constexpr bool DO_RD = (FW_RD > 0);
                    unsigned buf_row = gy * stride_tmp;

                    // Interior fast path: no mirror needed (~97% of pixels at 4K)
                    if (gx >= HALF_FW && gx < (int)e_w - HALF_FW) {
                        unsigned buf_base = buf_row + (unsigned)(gx - HALF_FW);

                        // Center tap (unpaired)
                        unsigned ci = buf_base + (unsigned)HALF_FW;
                        uint32_t fcc = fcoeff[HALF_FW];
                        h_mu1 += fcc * tmp_mu1[ci];
                        h_mu2 += fcc * tmp_mu2[ci];
                        h_ref += (uint64_t)fcc * tmp_ref[ci];
                        h_dis += (uint64_t)fcc * tmp_dis[ci];
                        h_ref_dis += (uint64_t)fcc * tmp_ref_dis[ci];
                        if constexpr (DO_RD) {
                            constexpr int RD_HALF = FW_RD / 2;
                            uint32_t fcc_rd = fcoeff_rd[RD_HALF];
                            h_ref_rd += fcc_rd * tmp_ref_convol[ci];
                            h_dis_rd += fcc_rd * tmp_dis_convol[ci];
                        }

// Symmetric pairs
#pragma unroll
                        for (int fj = 0; fj < HALF_FW; fj++) {
                            unsigned const idx_lo = buf_base + (unsigned)fj;
                            unsigned idx_hi = buf_base + (unsigned)(FW - 1 - fj);
                            uint32_t const fc = fcoeff[fj];

                            uint32_t const sum_mu1 = tmp_mu1[idx_lo] + tmp_mu1[idx_hi];
                            uint32_t const sum_mu2 = tmp_mu2[idx_lo] + tmp_mu2[idx_hi];
                            h_mu1 += fc * sum_mu1;
                            h_mu2 += fc * sum_mu2;

                            // uint64 for pair sum: each tmp value can reach ~4.26e9
                            uint64_t const sum_ref = (uint64_t)tmp_ref[idx_lo] + tmp_ref[idx_hi];
                            uint64_t const sum_dis = (uint64_t)tmp_dis[idx_lo] + tmp_dis[idx_hi];
                            uint64_t const sum_rd =
                                (uint64_t)tmp_ref_dis[idx_lo] + tmp_ref_dis[idx_hi];
                            h_ref += (uint64_t)fc * sum_ref;
                            h_dis += (uint64_t)fc * sum_dis;
                            h_ref_dis += (uint64_t)fc * sum_rd;

                            if constexpr (DO_RD) {
                                if (fj >= RD_START && (int)fj < RD_START + FW_RD / 2) {
                                    uint32_t fc_rd = fcoeff_rd[fj - RD_START];
                                    h_ref_rd +=
                                        fc_rd * (tmp_ref_convol[idx_lo] + tmp_ref_convol[idx_hi]);
                                    h_dis_rd +=
                                        fc_rd * (tmp_dis_convol[idx_lo] + tmp_dis_convol[idx_hi]);
                                }
                            }
                        }
                    } else {
// Border path: mirror needed
#pragma unroll
                        for (int fi = 0; fi < FW; fi++) {
                            int sx = mx(gx - HALF_FW + fi);
                            unsigned const sidx = buf_row + sx;
                            uint32_t const fc = fcoeff[fi];

                            h_mu1 += fc * tmp_mu1[sidx];
                            h_mu2 += fc * tmp_mu2[sidx];
                            h_ref += (uint64_t)fc * tmp_ref[sidx];
                            h_dis += (uint64_t)fc * tmp_dis[sidx];
                            h_ref_dis += (uint64_t)fc * tmp_ref_dis[sidx];

                            if constexpr (DO_RD) {
                                if (fi >= RD_START && fi < RD_START + FW_RD) {
                                    uint32_t fc_rd = fcoeff_rd[fi - RD_START];
                                    h_ref_rd += fc_rd * tmp_ref_convol[sidx];
                                    h_dis_rd += fc_rd * tmp_dis_convol[sidx];
                                }
                            }
                        }
                    }

                    // Horizontal quantization (shift by 16)
                    uint32_t const mu1_val = (uint32_t)h_mu1;
                    uint32_t const mu2_val = (uint32_t)h_mu2;
                    uint32_t const xx_filt = (uint32_t)((h_ref + 32768) >> 16);
                    uint32_t const yy_filt = (uint32_t)((h_dis + 32768) >> 16);
                    uint32_t const xy_filt = (uint32_t)((h_ref_dis + 32768) >> 16);

                    uint32_t const mu1_sq =
                        (uint32_t)(((uint64_t)mu1_val * mu1_val + 2147483648ULL) >> 32);
                    uint32_t const mu2_sq =
                        (uint32_t)(((uint64_t)mu2_val * mu2_val + 2147483648ULL) >> 32);
                    uint32_t const mu1_mu2 =
                        (uint32_t)(((uint64_t)mu1_val * mu2_val + 2147483648ULL) >> 32);

                    int32_t sigma1_sq = (int32_t)(xx_filt - mu1_sq);
                    int32_t sigma2_sq = (int32_t)(yy_filt - mu2_sq);
                    int32_t const sigma12 = (int32_t)(xy_filt - mu1_mu2);
                    if (sigma1_sq < 0)
                        sigma1_sq = 0;
                    if (sigma2_sq < 0)
                        sigma2_sq = 0;

                    if (sigma1_sq >= (int32_t)SIGMA_NSQ) {
                        float g = 0.0f;
                        float sv_sq = 0.0f;
                        float gg_sigma_f = 0.0f;

                        if (sigma12 > 0 && sigma1_sq != 0 && sigma2_sq != 0) {
                            g = (float)sigma12 / (float)sigma1_sq;
                            sv_sq = (float)sigma2_sq - g * (float)sigma12;
                            if (sv_sq < 0.0f)
                                sv_sq = 0.0f;
                            g = sycl::fmin(g, vif_enhn_gain_limit);
                            gg_sigma_f = g * g * (float)sigma1_sq;
                        }

                        uint32_t const log_den_stage1 = (uint32_t)((int64_t)SIGMA_NSQ + sigma1_sq);
                        int x_exp = 0;
                        uint32_t const log_den1 = dev_get_best16_from32(log_den_stage1, x_exp);

                        t_num_x += 1;
                        t_x += x_exp;

                        uint32_t const den_val = log2_lut[log_den1 - 32768];

                        if (sigma12 >= 0) {
                            uint32_t const numer1 = (uint32_t)(sv_sq) + (uint32_t)SIGMA_NSQ;
                            uint64_t const numer1_tmp =
                                (uint64_t)(int64_t)(gg_sigma_f) + (uint64_t)numer1;

                            int x1 = 0;
                            int x2_val = 0;
                            uint32_t const numlog = dev_get_best16_from64(numer1_tmp, x1);
                            uint32_t const denlog = dev_get_best16_from64((uint64_t)numer1, x2_val);

                            t_x2 += (x2_val - x1);

                            int32_t const num_val = (int32_t)log2_lut[numlog - 32768] -
                                                    (int32_t)log2_lut[denlog - 32768];
                            t_num_log += (int64_t)num_val;
                        }

                        t_den_log += (int64_t)den_val;
                    } else {
                        t_num_non_log += sigma2_sq;
                        t_den_non_log += 1;
                    }

                } // end if (valid)

                // ---- Phase 1: Subgroup shuffle reduction ----
                // Uses hardware shuffle instructions (no barriers, no local mem)
                int64_t sg_x = sycl::reduce_over_group(sg, t_x, sycl::plus<int64_t>());
                int64_t sg_x2 = sycl::reduce_over_group(sg, t_x2, sycl::plus<int64_t>());
                int64_t sg_num_x = sycl::reduce_over_group(sg, t_num_x, sycl::plus<int64_t>());
                int64_t sg_num_log = sycl::reduce_over_group(sg, t_num_log, sycl::plus<int64_t>());
                int64_t sg_den_log = sycl::reduce_over_group(sg, t_den_log, sycl::plus<int64_t>());
                int64_t sg_num_nlog =
                    sycl::reduce_over_group(sg, t_num_non_log, sycl::plus<int64_t>());
                int64_t sg_den_nlog =
                    sycl::reduce_over_group(sg, t_den_non_log, sycl::plus<int64_t>());

                // ---- Phase 2: Cross-subgroup reduction via local memory ----
                // Only subgroup leaders write and participate
                if (sg_lid == 0) {
                    lmem[0 * MAX_SUBGROUPS + sg_id] = sg_x;
                    lmem[1 * MAX_SUBGROUPS + sg_id] = sg_x2;
                    lmem[2 * MAX_SUBGROUPS + sg_id] = sg_num_x;
                    lmem[3 * MAX_SUBGROUPS + sg_id] = sg_num_log;
                    lmem[4 * MAX_SUBGROUPS + sg_id] = sg_den_log;
                    lmem[5 * MAX_SUBGROUPS + sg_id] = sg_num_nlog;
                    lmem[6 * MAX_SUBGROUPS + sg_id] = sg_den_nlog;
                }

                item.barrier(sycl::access::fence_space::local_space);

                // Thread 0 of the entire workgroup does final reduction
                // Only n_subgroups entries to sum (typically 8 for SIMD-32)
                const int lid = item.get_local_linear_id();
                if (lid == 0) {
                    // NOLINTNEXTLINE(misc-const-correctness): atomic_ref / reduction-loop target — clang-tidy cannot see the writes through SYCL atomic_ref or sub-group reductions, but the variable is mutated and must not be const
                    int64_t final_vals[ACCUM_FIELDS] = {};
                    for (uint32_t s = 0; s < n_subgroups; s++) {
                        final_vals[0] += lmem[0 * MAX_SUBGROUPS + s];
                        final_vals[1] += lmem[1 * MAX_SUBGROUPS + s];
                        final_vals[2] += lmem[2 * MAX_SUBGROUPS + s];
                        final_vals[3] += lmem[3 * MAX_SUBGROUPS + s];
                        final_vals[4] += lmem[4 * MAX_SUBGROUPS + s];
                        final_vals[5] += lmem[5 * MAX_SUBGROUPS + s];
                        final_vals[6] += lmem[6 * MAX_SUBGROUPS + s];
                    }
                    for (int f = 0; f < ACCUM_FIELDS; f++) {
                        sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref(accum[f]);
                        ref.fetch_add(final_vals[f]);
                    }
                }

                // Downsample for next scale (even coords, valid threads only)
                if constexpr (FW_RD > 0) {
                    if (valid && (gx % 2 == 0) && (gy % 2 == 0)) {
                        uint32_t const ref_rd_val = (uint32_t)((h_ref_rd + 32768) >> 16);
                        uint32_t const dis_rd_val = (uint32_t)((h_dis_rd + 32768) >> 16);
                        unsigned rd_x = gx / 2;
                        unsigned rd_y = gy / 2;
                        unsigned const rd_stride = e_w / 2;
                        rd_ref[rd_y * rd_stride + rd_x] = ref_rd_val & 0xFFFF;
                        rd_dis[rd_y * rd_stride + rd_x] = dis_rd_val & 0xFFFF;
                    }
                } // if constexpr (FW_RD > 0)
            });
    });
}

static sycl::event launch_vif_hori_v2(sycl::queue &q, int scale, unsigned width, unsigned height,
                                      float vif_enhn_gain_limit, uint32_t *tmp_mu1,
                                      uint32_t *tmp_mu2, uint32_t *tmp_ref, uint32_t *tmp_dis,
                                      uint32_t *tmp_ref_dis, uint32_t *tmp_ref_convol,
                                      uint32_t *tmp_dis_convol, int64_t *accum, uint32_t *rd_ref,
                                      uint32_t *rd_dis, const uint32_t *log2_lut)
{
    switch (scale) {
    case 0:
        return launch_vif_hori_impl<0, 32>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    case 1:
        return launch_vif_hori_impl<1, 32>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    case 2:
        return launch_vif_hori_impl<2, 32>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    default:
        return launch_vif_hori_impl<3, 32>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    }
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: Horizontal Pass + VIF (subgroup-optimized, SIMD-16)   */
/* ------------------------------------------------------------------ */

/**
 * Same as launch_vif_hori_v2 but with reqd_sub_group_size(16).
 * On Xe-LP (UHD 770), SIMD-16 may be optimal.
 * On Xe-HPG (Arc), SIMD-32 is typically better.
 */
static sycl::event launch_vif_hori_v2_sg16(sycl::queue &q, int scale, unsigned width,
                                           unsigned height, float vif_enhn_gain_limit,
                                           uint32_t *tmp_mu1, uint32_t *tmp_mu2, uint32_t *tmp_ref,
                                           uint32_t *tmp_dis, uint32_t *tmp_ref_dis,
                                           uint32_t *tmp_ref_convol, uint32_t *tmp_dis_convol,
                                           int64_t *accum, uint32_t *rd_ref, uint32_t *rd_dis,
                                           const uint32_t *log2_lut)
{
    switch (scale) {
    case 0:
        return launch_vif_hori_impl<0, 16>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    case 1:
        return launch_vif_hori_impl<1, 16>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    case 2:
        return launch_vif_hori_impl<2, 16>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    default:
        return launch_vif_hori_impl<3, 16>(q, width, height, vif_enhn_gain_limit, tmp_mu1, tmp_mu2,
                                           tmp_ref, tmp_dis, tmp_ref_dis, tmp_ref_convol,
                                           tmp_dis_convol, accum, rd_ref, rd_dis, log2_lut);
    }
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: Fused Vertical + Horizontal Pass (single dispatch)    */
/* ------------------------------------------------------------------ */

/**
 * Fused VIF kernel V3: SLM intermediate with WG_Y=8 for better occupancy.
 *
 * Eliminates 7 intermediate global memory buffers and halves VIF kernel
 * launches (8→4).  Uses WG_Y=8 (instead of 16) to keep s_vert SLM at
 * ~7KB (scale 0) → total SLM ~15KB → 4 WGs/DSS (vs 2 with WG_Y=16).
 *
 * Phases:
 *   1. Cooperative load of input tile with V+H halos into s_ref/s_dis
 *   2. Cooperative vertical convolution → s_vert (7 channels in SLM)
 *   3. Horizontal convolution from s_vert + VIF statistics + downsample
 *   4. Subgroup + cross-subgroup reduction → atomic accumulation
 *
 * SLM budget (scale 0, FW=17, WG_Y=8, WG_X=16):
 *   s_ref:  (8+16) × (16+16) × 4 =  3,072 B
 *   s_dis:  same                   =  3,072 B
 *   s_vert: 7 × 8 × 32 × 4       =  7,168 B
 *   lmem:   7 × 16 × 8            =    896 B
 *   Total:                         ≈ 14.2 KB → 4 WGs/DSS
 */
template <int SCALE, int SG_SIZE>
static sycl::event
// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch lambda body, see comment block above (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278)
launch_vif_fused_impl(sycl::queue &q, const void *ref_data, const void *dis_data, unsigned width,
                      unsigned height, unsigned src_stride, unsigned bpc, float vif_enhn_gain_limit,
                      int64_t *accum, uint32_t *rd_ref, uint32_t *rd_dis, const uint32_t *log2_lut)
{
    constexpr int FW_V = vif_fwidth[SCALE];
    constexpr int HALF_FW_V = FW_V / 2;
    constexpr int FW_H = FW_V;
    constexpr int HALF_FW_H = FW_H / 2;
    constexpr int FW_RD = vif_fwidth_rd[SCALE];
    constexpr int RD_START = (FW_V - FW_RD) / 2;
    constexpr bool DO_RD = (FW_RD > 0);

    // WG_Y=8 keeps s_vert small enough for good occupancy
    constexpr int WG_X = 16;
    constexpr int WG_Y = 8;
    constexpr int WG_SIZE = WG_X * WG_Y; // 128
    constexpr int MAX_SUBGROUPS = 16;    // ≥ WG_SIZE/min_sg_size

    constexpr int TILE_H = WG_Y + FW_V - 1;        // input rows (V halo)
    constexpr int TILE_W = WG_X + FW_H - 1;        // input cols (H halo)
    constexpr unsigned VERT_TOTAL = WG_Y * TILE_W; // vert outputs per ch
    constexpr int N_CH = DO_RD ? 7 : 5;            // channels in s_vert

    uint32_t fcoeff[VIF_FILTER_MAX_WIDTH];
    for (int i = 0; i < FW_V; i++)
        fcoeff[i] = vif_filter1d_table[SCALE][i];

    uint32_t fcoeff_rd[VIF_FILTER_MAX_WIDTH] = {};
    if constexpr (FW_RD > 0) {
        for (int i = 0; i < FW_RD; i++)
            fcoeff_rd[i] = vif_filter1d_table[SCALE + 1][i];
    }

    unsigned shift_vp;
    unsigned add_shift_round_vp;
    unsigned shift_vp_sq;
    unsigned add_shift_round_vp_sq;
    if constexpr (SCALE == 0) {
        shift_vp = bpc;
        add_shift_round_vp = 1u << (bpc - 1);
        shift_vp_sq = (bpc - 8) * 2;
        add_shift_round_vp_sq = (bpc == 8) ? 0 : (1u << (shift_vp_sq - 1));
    } else {
        shift_vp = 16;
        add_shift_round_vp = 32768;
        shift_vp_sq = 16;
        add_shift_round_vp_sq = 32768;
    }

    sycl::range<2> global(((height + WG_Y - 1) / WG_Y) * WG_Y, ((width + WG_X - 1) / WG_X) * WG_X);
    sycl::range<2> local(WG_Y, WG_X);

    auto p_ref = ref_data;
    auto p_dis = dis_data;
    auto e_bpc = bpc;
    auto e_w = width;
    auto e_h = height;
    auto e_src_stride = src_stride;

    return q.submit([&](sycl::handler &cgh) {
        // SLM: input tiles
        sycl::local_accessor<uint32_t, 1> s_ref(sycl::range<1>(TILE_H * TILE_W), cgh);
        sycl::local_accessor<uint32_t, 1> s_dis(sycl::range<1>(TILE_H * TILE_W), cgh);
        // SLM: intermediate vertical convolution results
        sycl::local_accessor<uint32_t, 1> s_vert(sycl::range<1>(N_CH * VERT_TOTAL), cgh);
        // SLM: reduction across subgroups
        sycl::local_accessor<int64_t, 1> lmem(sycl::range<1>(ACCUM_FIELDS * MAX_SUBGROUPS), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) VMAF_SYCL_REQD_SG_SIZE(SG_SIZE) {
                const int gx = item.get_global_id(1);
                const int gy = item.get_global_id(0);
                const unsigned lid = item.get_local_linear_id();
                const unsigned lx = item.get_local_id(1);
                const unsigned ly = item.get_local_id(0);
                const bool valid = (gx < (int)e_w && gy < (int)e_h);

                // ============================================================
                // Phase 1: Cooperative load of input tile into SLM
                // ============================================================
                int tile_origin_y = (int)(item.get_group(0) * WG_Y) - HALF_FW_V;
                int tile_origin_x = (int)(item.get_group(1) * WG_X) - HALF_FW_H;

                constexpr unsigned tile_elems = TILE_H * TILE_W;

                auto read_global = [&](const void *src, int y, int x) -> uint32_t {
                    if constexpr (SCALE == 0) {
                        if (e_bpc <= 8) {
                            return static_cast<const uint8_t *>(src)[y * e_src_stride + x];
                        } else {
                            return static_cast<const uint16_t *>(src)[y * (e_src_stride / 2) + x];
                        }
                    } else {
                        return static_cast<const uint32_t *>(src)[y * e_src_stride + x] & 0xFFFF;
                    }
                };

                bool interior_wg = (tile_origin_y >= 0) && (tile_origin_y + TILE_H <= (int)e_h) &&
                                   (tile_origin_x >= 0) && (tile_origin_x + TILE_W <= (int)e_w);

                if (interior_wg) {
                    for (unsigned i = lid; i < tile_elems; i += WG_SIZE) {
                        unsigned tr = i / TILE_W;
                        unsigned tc = i % TILE_W;
                        int const py = tile_origin_y + (int)tr;
                        int const px = tile_origin_x + (int)tc;
                        s_ref[tr * TILE_W + tc] = read_global(p_ref, py, px);
                        s_dis[tr * TILE_W + tc] = read_global(p_dis, py, px);
                    }
                } else {
                    for (unsigned i = lid; i < tile_elems; i += WG_SIZE) {
                        unsigned tr = i / TILE_W;
                        unsigned tc = i % TILE_W;
                        int const py = dev_mirror(tile_origin_y + (int)tr, (int)e_h);
                        int const px = dev_mirror(tile_origin_x + (int)tc, (int)e_w);
                        s_ref[tr * TILE_W + tc] = read_global(p_ref, py, px);
                        s_dis[tr * TILE_W + tc] = read_global(p_dis, py, px);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                // ============================================================
                // Phase 2: Cooperative vertical convolution → s_vert
                // Each thread processes VERT_TOTAL/WG_SIZE output positions,
                // computing all 5-7 channels per position from shared input.
                // ============================================================
                for (unsigned i = lid; i < VERT_TOTAL; i += WG_SIZE) {
                    unsigned r = i / TILE_W;
                    unsigned c = i % TILE_W;

                    uint32_t a_mu1 = 0;
                    uint32_t a_mu2 = 0;
                    uint64_t a_ref = 0;
                    uint64_t a_dis = 0;
                    uint64_t a_ref_dis = 0;
                    // NOLINTNEXTLINE(readability-isolate-declaration): SYCL chained zero-init; splitting hides the symmetry of the parallel reduction state.
                    uint32_t a_ref_rd = 0, a_dis_rd = 0;

#pragma unroll
                    for (int fi = 0; fi < FW_V; fi++) {
                        unsigned sidx = (r + (unsigned)fi) * TILE_W + c;
                        uint32_t rv = s_ref[sidx];
                        uint32_t dv = s_dis[sidx];
                        uint32_t const fc = fcoeff[fi];

                        uint32_t const icr = fc * rv;
                        uint32_t const icd = fc * dv;
                        a_mu1 += icr;
                        a_mu2 += icd;
                        a_ref += (uint64_t)icr * rv;
                        a_dis += (uint64_t)icd * dv;
                        a_ref_dis += (uint64_t)icr * dv;

                        if constexpr (FW_RD > 0) {
                            if (fi >= RD_START && fi < RD_START + FW_RD) {
                                uint32_t fc_rd = fcoeff_rd[fi - RD_START];
                                a_ref_rd += fc_rd * rv;
                                a_dis_rd += fc_rd * dv;
                            }
                        }
                    }

                    // Quantize (identical to separate vert kernel)
                    unsigned base = r * TILE_W + c;
                    s_vert[0 * VERT_TOTAL + base] =
                        (uint32_t)((a_mu1 + add_shift_round_vp) >> shift_vp);
                    s_vert[1 * VERT_TOTAL + base] =
                        (uint32_t)((a_mu2 + add_shift_round_vp) >> shift_vp);
                    // NOLINTNEXTLINE(bugprone-branch-clone): SYCL template specialization branch — both arms reach the same code today but the conditional pins the bit-exactness contract for future SCALE / SG_SIZE divergence.
                    if (shift_vp_sq > 0) {
                        s_vert[2 * VERT_TOTAL + base] =
                            (uint32_t)((a_ref + add_shift_round_vp_sq) >> shift_vp_sq);
                        s_vert[3 * VERT_TOTAL + base] =
                            (uint32_t)((a_dis + add_shift_round_vp_sq) >> shift_vp_sq);
                        s_vert[4 * VERT_TOTAL + base] =
                            (uint32_t)((a_ref_dis + add_shift_round_vp_sq) >> shift_vp_sq);
                    } else {
                        s_vert[2 * VERT_TOTAL + base] = (uint32_t)a_ref;
                        s_vert[3 * VERT_TOTAL + base] = (uint32_t)a_dis;
                        s_vert[4 * VERT_TOTAL + base] = (uint32_t)a_ref_dis;
                    }
                    if constexpr (FW_RD > 0) {
                        s_vert[5 * VERT_TOTAL + base] =
                            (uint32_t)((a_ref_rd + add_shift_round_vp) >> shift_vp);
                        s_vert[6 * VERT_TOTAL + base] =
                            (uint32_t)((a_dis_rd + add_shift_round_vp) >> shift_vp);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

// ============================================================
// Phase 3: Horizontal conv from s_vert + VIF stats + downsample
// No border handling needed — SLM tile already has mirrored halos.
// ============================================================
// Helper macro: read s_vert channel ch at (row, col)
#define SV(ch, r, c) s_vert[(ch) * VERT_TOTAL + (r) * TILE_W + (c)]

                int64_t t_x = 0;
                int64_t t_x2 = 0;
                int64_t t_num_x = 0;
                int64_t t_num_log = 0;
                int64_t t_den_log = 0;
                int64_t t_num_non_log = 0;
                int64_t t_den_non_log = 0;
                // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                uint32_t h_ref_rd = 0;
                // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                uint32_t h_dis_rd = 0;

                if (valid) {
                    // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                    uint32_t h_mu1 = 0;
                    // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                    uint32_t h_mu2 = 0;
                    // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                    uint64_t h_ref = 0;
                    // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                    uint64_t h_dis = 0;
                    // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
                    uint64_t h_ref_dis = 0;

                    // Center tap (unpaired)
                    {
                        uint32_t fcc = fcoeff[HALF_FW_H];
                        unsigned cc = lx + (unsigned)HALF_FW_H;
                        h_mu1 += fcc * SV(0, ly, cc);
                        h_mu2 += fcc * SV(1, ly, cc);
                        h_ref += (uint64_t)fcc * SV(2, ly, cc);
                        h_dis += (uint64_t)fcc * SV(3, ly, cc);
                        h_ref_dis += (uint64_t)fcc * SV(4, ly, cc);
                        if constexpr (DO_RD) {
                            constexpr int RD_HALF = FW_RD / 2;
                            h_ref_rd += fcoeff_rd[RD_HALF] * SV(5, ly, cc);
                            h_dis_rd += fcoeff_rd[RD_HALF] * SV(6, ly, cc);
                        }
                    }

// Symmetric pairs
#pragma unroll
                    for (int fj = 0; fj < HALF_FW_H; fj++) {
                        uint32_t const fc = fcoeff[fj];
                        unsigned lo_c = lx + (unsigned)fj;
                        unsigned hi_c = lx + (unsigned)(FW_H - 1 - fj);

                        h_mu1 += fc * (SV(0, ly, lo_c) + SV(0, ly, hi_c));
                        h_mu2 += fc * (SV(1, ly, lo_c) + SV(1, ly, hi_c));
                        h_ref += (uint64_t)fc * ((uint64_t)SV(2, ly, lo_c) + SV(2, ly, hi_c));
                        h_dis += (uint64_t)fc * ((uint64_t)SV(3, ly, lo_c) + SV(3, ly, hi_c));
                        h_ref_dis += (uint64_t)fc * ((uint64_t)SV(4, ly, lo_c) + SV(4, ly, hi_c));

                        if constexpr (DO_RD) {
                            if (fj >= RD_START && (int)fj < RD_START + FW_RD / 2) {
                                uint32_t fc_rd = fcoeff_rd[fj - RD_START];
                                h_ref_rd += fc_rd * (SV(5, ly, lo_c) + SV(5, ly, hi_c));
                                h_dis_rd += fc_rd * (SV(6, ly, lo_c) + SV(6, ly, hi_c));
                            }
                        }
                    }

#undef SV

                    // Horizontal quantization (shift by 16)
                    uint32_t const mu1_val = (uint32_t)h_mu1;
                    uint32_t const mu2_val = (uint32_t)h_mu2;
                    uint32_t const xx_filt = (uint32_t)((h_ref + 32768) >> 16);
                    uint32_t const yy_filt = (uint32_t)((h_dis + 32768) >> 16);
                    uint32_t const xy_filt = (uint32_t)((h_ref_dis + 32768) >> 16);

                    uint32_t const mu1_sq =
                        (uint32_t)(((uint64_t)mu1_val * mu1_val + 2147483648ULL) >> 32);
                    uint32_t const mu2_sq =
                        (uint32_t)(((uint64_t)mu2_val * mu2_val + 2147483648ULL) >> 32);
                    uint32_t const mu1_mu2 =
                        (uint32_t)(((uint64_t)mu1_val * mu2_val + 2147483648ULL) >> 32);

                    int32_t sigma1_sq = (int32_t)(xx_filt - mu1_sq);
                    int32_t sigma2_sq = (int32_t)(yy_filt - mu2_sq);
                    int32_t const sigma12 = (int32_t)(xy_filt - mu1_mu2);
                    if (sigma1_sq < 0)
                        sigma1_sq = 0;
                    if (sigma2_sq < 0)
                        sigma2_sq = 0;

                    if (sigma1_sq >= (int32_t)SIGMA_NSQ) {
                        float g = 0.0f;
                        float sv_sq = 0.0f;
                        float gg_sigma_f = 0.0f;

                        if (sigma12 > 0 && sigma1_sq != 0 && sigma2_sq != 0) {
                            g = (float)sigma12 / (float)sigma1_sq;
                            sv_sq = (float)sigma2_sq - g * (float)sigma12;
                            if (sv_sq < 0.0f)
                                sv_sq = 0.0f;
                            g = sycl::fmin(g, vif_enhn_gain_limit);
                            gg_sigma_f = g * g * (float)sigma1_sq;
                        }

                        uint32_t const log_den_stage1 = (uint32_t)((int64_t)SIGMA_NSQ + sigma1_sq);
                        int x_exp = 0;
                        uint32_t const log_den1 = dev_get_best16_from32(log_den_stage1, x_exp);

                        t_num_x += 1;
                        t_x += x_exp;

                        uint32_t const den_val = log2_lut[log_den1 - 32768];

                        if (sigma12 >= 0) {
                            uint32_t const numer1 = (uint32_t)(sv_sq) + (uint32_t)SIGMA_NSQ;
                            uint64_t const numer1_tmp =
                                (uint64_t)(int64_t)(gg_sigma_f) + (uint64_t)numer1;

                            int x1 = 0;
                            int x2_val = 0;
                            uint32_t const numlog = dev_get_best16_from64(numer1_tmp, x1);
                            uint32_t const denlog = dev_get_best16_from64((uint64_t)numer1, x2_val);

                            t_x2 += (x2_val - x1);

                            int32_t const num_val = (int32_t)log2_lut[numlog - 32768] -
                                                    (int32_t)log2_lut[denlog - 32768];
                            t_num_log += (int64_t)num_val;
                        }

                        t_den_log += (int64_t)den_val;
                    } else {
                        t_num_non_log += sigma2_sq;
                        t_den_non_log += 1;
                    }
                } // end if (valid)

                // ============================================================
                // Phase 4: Subgroup + cross-subgroup reduction
                // ============================================================
                sycl::sub_group sg = item.get_sub_group();
                const uint32_t sg_id = sg.get_group_linear_id();
                const uint32_t sg_lid = sg.get_local_linear_id();
                const uint32_t n_subgroups = sg.get_group_linear_range();

                int64_t sg_x = sycl::reduce_over_group(sg, t_x, sycl::plus<int64_t>());
                int64_t sg_x2 = sycl::reduce_over_group(sg, t_x2, sycl::plus<int64_t>());
                int64_t sg_num_x = sycl::reduce_over_group(sg, t_num_x, sycl::plus<int64_t>());
                int64_t sg_num_log = sycl::reduce_over_group(sg, t_num_log, sycl::plus<int64_t>());
                int64_t sg_den_log = sycl::reduce_over_group(sg, t_den_log, sycl::plus<int64_t>());
                int64_t sg_num_nlog =
                    sycl::reduce_over_group(sg, t_num_non_log, sycl::plus<int64_t>());
                int64_t sg_den_nlog =
                    sycl::reduce_over_group(sg, t_den_non_log, sycl::plus<int64_t>());

                if (sg_lid == 0) {
                    lmem[0 * MAX_SUBGROUPS + sg_id] = sg_x;
                    lmem[1 * MAX_SUBGROUPS + sg_id] = sg_x2;
                    lmem[2 * MAX_SUBGROUPS + sg_id] = sg_num_x;
                    lmem[3 * MAX_SUBGROUPS + sg_id] = sg_num_log;
                    lmem[4 * MAX_SUBGROUPS + sg_id] = sg_den_log;
                    lmem[5 * MAX_SUBGROUPS + sg_id] = sg_num_nlog;
                    lmem[6 * MAX_SUBGROUPS + sg_id] = sg_den_nlog;
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (lid == 0) {
                    // NOLINTNEXTLINE(misc-const-correctness): atomic_ref / reduction-loop target — clang-tidy cannot see the writes through SYCL atomic_ref or sub-group reductions, but the variable is mutated and must not be const
                    int64_t final_vals[ACCUM_FIELDS] = {};
                    for (uint32_t s = 0; s < n_subgroups; s++) {
                        final_vals[0] += lmem[0 * MAX_SUBGROUPS + s];
                        final_vals[1] += lmem[1 * MAX_SUBGROUPS + s];
                        final_vals[2] += lmem[2 * MAX_SUBGROUPS + s];
                        final_vals[3] += lmem[3 * MAX_SUBGROUPS + s];
                        final_vals[4] += lmem[4 * MAX_SUBGROUPS + s];
                        final_vals[5] += lmem[5 * MAX_SUBGROUPS + s];
                        final_vals[6] += lmem[6 * MAX_SUBGROUPS + s];
                    }
                    for (int f = 0; f < ACCUM_FIELDS; f++) {
                        sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref(accum[f]);
                        ref.fetch_add(final_vals[f]);
                    }
                }

                // Downsample for next scale
                if constexpr (FW_RD > 0) {
                    if (valid && (gx % 2 == 0) && (gy % 2 == 0)) {
                        uint32_t const ref_rd_val = (uint32_t)((h_ref_rd + 32768) >> 16);
                        uint32_t const dis_rd_val = (uint32_t)((h_dis_rd + 32768) >> 16);
                        unsigned rd_x = gx / 2;
                        unsigned rd_y = gy / 2;
                        rd_ref[rd_y * (e_w / 2) + rd_x] = ref_rd_val & 0xFFFF;
                        rd_dis[rd_y * (e_w / 2) + rd_x] = dis_rd_val & 0xFFFF;
                    }
                }
            });
    });
}

static sycl::event launch_vif_fused(sycl::queue &q, const void *ref_data, const void *dis_data,
                                    int scale, unsigned width, unsigned height, unsigned src_stride,
                                    unsigned bpc, bool use_simd16, float vif_enhn_gain_limit,
                                    int64_t *accum, uint32_t *rd_ref, uint32_t *rd_dis,
                                    const uint32_t *log2_lut)
{
    if (use_simd16) {
        switch (scale) {
        case 0:
            return launch_vif_fused_impl<0, 16>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        case 1:
            return launch_vif_fused_impl<1, 16>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        case 2:
            return launch_vif_fused_impl<2, 16>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        default:
            return launch_vif_fused_impl<3, 16>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        }
    } else {
        switch (scale) {
        case 0:
            return launch_vif_fused_impl<0, 32>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        case 1:
            return launch_vif_fused_impl<1, 32>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        case 2:
            return launch_vif_fused_impl<2, 32>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        default:
            return launch_vif_fused_impl<3, 32>(q, ref_data, dis_data, width, height, src_stride,
                                                bpc, vif_enhn_gain_limit, accum, rd_ref, rd_dis,
                                                log2_lut);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Feature extractor callbacks                                         */
/* ------------------------------------------------------------------ */

// Forward declarations for combined graph callbacks (defined after enqueue_vif_work_impl)
static void enqueue_vif_work(void *queue_ptr, void *priv, void *shared_ref, void *shared_dis);
static void vif_pre_graph(void *queue_ptr, void *priv);
static void vif_post_graph(void *queue_ptr, void *priv);

// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch / lifecycle entry — body is dominated by accessor declarations + a single `parallel_for` lambda. Splitting either inlines via macro (no readability win) or introduces a free function the compiler cannot inline back into the device kernel. Keeping it large is the pattern shared across every SYCL TU in this fork (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278).
static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<VifStateSycl *>(fex->priv);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->has_pending = false;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vif_sycl: no SYCL state\n");
        return -EINVAL;
    }

    VmafSyclState *state = fex->sycl_state;

    auto *q_ptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(state));
    if (!q_ptr)
        return -EINVAL;
    sycl::queue &q = *q_ptr;

    // Initialize shared frame buffers (idempotent, first extractor wins)
    int err = vmaf_sycl_shared_frame_init(state, w, h, bpc);
    if (err)
        return err;

    // Intermediate buffers: only needed for separate V+H pipeline
    size_t const tmp_size = (size_t)w * h * sizeof(uint32_t);

    if (!s->use_fused) {
        s->d_tmp_mu1 = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_mu2 = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_ref = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_dis = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_ref_dis = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_ref_convol = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
        s->d_tmp_dis_convol = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, tmp_size));
    }
    // Downsampled buffers only need quarter-frame: (w/2)*(h/2) elements
    size_t const rd_size = (size_t)(w / 2) * (h / 2) * sizeof(uint32_t);
    s->d_rd_ref = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, rd_size));
    s->d_rd_dis = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, rd_size));

    // Accumulators: 4 scales x 7 fields x 8 bytes = 224 bytes
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): SYCL stride / global-id arithmetic; operands are bounded by the kernel `nd_range` and the widening to ptrdiff_t / size_t is the intended index calc.
    size_t const accum_size = VIF_NUM_SCALES * ACCUM_FIELDS * sizeof(int64_t);
    s->d_accum = static_cast<int64_t *>(vmaf_sycl_malloc_device(state, accum_size));
    s->h_accum = static_cast<int64_t *>(vmaf_sycl_malloc_host(state, accum_size));

    // Log2 LUT: 32768 entries of uint32
    size_t const lut_size = LOG2_LUT_SIZE * sizeof(uint32_t);
    s->d_log2_lut = static_cast<uint32_t *>(vmaf_sycl_malloc_device(state, lut_size));

    if (!s->use_fused && (!s->d_tmp_mu1 || !s->d_tmp_mu2 || !s->d_tmp_ref || !s->d_tmp_dis ||
                          !s->d_tmp_ref_dis || !s->d_tmp_ref_convol || !s->d_tmp_dis_convol)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vif_sycl: tmp buffer allocation failed\n");
        return -ENOMEM;
    }
    if (!s->d_rd_ref || !s->d_rd_dis || !s->d_accum || !s->h_accum || !s->d_log2_lut) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vif_sycl: device memory allocation failed\n");
        return -ENOMEM;
    }

    // Generate and upload log2 LUT
    {
        uint32_t *lut_host = static_cast<uint32_t *>(std::malloc(lut_size));
        if (!lut_host)
            return -ENOMEM;
        for (int j = 0; j < LOG2_LUT_SIZE; j++) {
            lut_host[j] = (uint32_t)std::roundf(std::log2f((float)(j + 32768)) * 2048.0f);
        }
        vmaf_sycl_memcpy_h2d(state, s->d_log2_lut, lut_host, lut_size);
        std::free(lut_host);
    }

    // Auto-detect optimal subgroup size for VIF hori kernel.
    // SIMD-16 is faster on Xe-HPG (Arc) and Xe-LP due to lower per-thread
    // latency; SIMD-32 only benefits Xe-HPC (Data Center GPU Max).
    {
        auto dev = q.get_device();
        auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        // NOLINTNEXTLINE(misc-const-correctness): SYCL kernel-local — variable is mutated via atomic_ref / sub-group reduction the analyzer cannot trace.
        bool has_sg16 = false;
        for (auto sz : sg_sizes) {
            if (sz == 16) {
                has_sg16 = true;
                break;
            }
        }
        s->use_simd16 = has_sg16;
        vmaf_log(VMAF_LOG_LEVEL_DEBUG, "vif_sycl: auto-selected SIMD-%d subgroup size\n",
                 s->use_simd16 ? 16 : 32);
        vmaf_log(VMAF_LOG_LEVEL_DEBUG, "vif_sycl: kernel mode = %s\n",
                 s->use_fused ? "fused V+H (saves VRAM)" : "separate V+H");
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    // Register with combined command graph
    err = vmaf_sycl_graph_register(state, enqueue_vif_work, vif_pre_graph, vif_post_graph, nullptr,
                                   s, "VIF");
    if (err)
        return err;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Enqueue all VIF compute work (used for both recording and direct)    */
/* ------------------------------------------------------------------ */

// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch / lifecycle entry — body is dominated by accessor declarations + a single `parallel_for` lambda. Splitting either inlines via macro (no readability win) or introduces a free function the compiler cannot inline back into the device kernel. Keeping it large is the pattern shared across every SYCL TU in this fork (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278).
static void enqueue_vif_work_impl(sycl::queue &q, VifStateSycl *s, void *shared_ref,
                                  void *shared_dis)
{
    unsigned cur_w = s->width;
    unsigned cur_h = s->height;

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        const void *ref_src;
        const void *dis_src;
        unsigned src_stride;

        if (scale == 0) {
            ref_src = shared_ref;
            dis_src = shared_dis;
            if (s->bpc <= 8) {
                src_stride = cur_w;
            } else {
                src_stride = cur_w * 2;
            }
        } else {
            ref_src = s->d_rd_ref;
            dis_src = s->d_rd_dis;
            src_stride = cur_w;
        }

        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): SYCL stride / global-id arithmetic; operands are bounded by the kernel `nd_range` and the widening to ptrdiff_t / size_t is the intended index calc.
        int64_t *scale_accum = s->d_accum + scale * ACCUM_FIELDS;

        if (s->use_fused) {
            // Fused V+H: single dispatch per scale, no tmp buffers needed.
            // Uses SLM intermediates. Saves ~70 MB VRAM at 4K.
            launch_vif_fused(q, ref_src, dis_src, scale, cur_w, cur_h, src_stride, s->bpc,
                             s->use_simd16, (float)s->vif_enhn_gain_limit, scale_accum, s->d_rd_ref,
                             s->d_rd_dis, s->d_log2_lut);
        } else {
            // Separate vert + hori pipeline (8 kernels = 4 vert + 4 hori).
            // Profiled faster than fused V+H on Arc A380 with graph mode:
            //   separate = 34.8 FPS, fused = 31.8 FPS at 4K
            // because graph mode already amortizes launch overhead, and the
            // fused kernel's extra SLM barrier + lower occupancy hurts.

            // Vertical pass: writes to tmp buffers
            launch_vif_vert(q, ref_src, dis_src, scale, cur_w, cur_h, src_stride, s->bpc,
                            s->d_tmp_mu1, s->d_tmp_mu2, s->d_tmp_ref, s->d_tmp_dis,
                            s->d_tmp_ref_dis, s->d_tmp_ref_convol, s->d_tmp_dis_convol);

            // Horizontal pass: reads from tmp buffers, computes VIF stats
            if (s->use_simd16) {
                launch_vif_hori_v2_sg16(q, scale, cur_w, cur_h, (float)s->vif_enhn_gain_limit,
                                        s->d_tmp_mu1, s->d_tmp_mu2, s->d_tmp_ref, s->d_tmp_dis,
                                        s->d_tmp_ref_dis, s->d_tmp_ref_convol, s->d_tmp_dis_convol,
                                        scale_accum, s->d_rd_ref, s->d_rd_dis, s->d_log2_lut);
            } else {
                launch_vif_hori_v2(q, scale, cur_w, cur_h, (float)s->vif_enhn_gain_limit,
                                   s->d_tmp_mu1, s->d_tmp_mu2, s->d_tmp_ref, s->d_tmp_dis,
                                   s->d_tmp_ref_dis, s->d_tmp_ref_convol, s->d_tmp_dis_convol,
                                   scale_accum, s->d_rd_ref, s->d_rd_dis, s->d_log2_lut);
            }
        }

        // Next scale: half dimensions
        cur_w /= 2;
        cur_h /= 2;
    }
}

/* ------------------------------------------------------------------ */
/* Command graph recording (one graph per double-buffer slot)          */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* C-compatible callbacks for combined command graph                   */
/* ------------------------------------------------------------------ */

// Pre-graph: zero accumulators (direct enqueue, outside graph)
static void vif_pre_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<VifStateSycl *>(priv);
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): SYCL stride / global-id arithmetic; operands are bounded by the kernel `nd_range` and the widening to ptrdiff_t / size_t is the intended index calc.
    size_t accum_size = VIF_NUM_SCALES * ACCUM_FIELDS * sizeof(int64_t);
    q.memset(s->d_accum, 0, accum_size);
}

// Graph-recorded: compute kernels only
static void enqueue_vif_work(void *queue_ptr, void *priv, void *shared_ref, void *shared_dis)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<VifStateSycl *>(priv);
    enqueue_vif_work_impl(q, s, shared_ref, shared_dis);
}

// Post-graph: D2H accumulator download (direct enqueue, outside graph)
static void vif_post_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<VifStateSycl *>(priv);
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): SYCL stride / global-id arithmetic; operands are bounded by the kernel `nd_range` and the widening to ptrdiff_t / size_t is the intended index calc.
    size_t accum_size = VIF_NUM_SCALES * ACCUM_FIELDS * sizeof(int64_t);
    q.memcpy(s->h_accum, s->d_accum, accum_size);
}

/* ------------------------------------------------------------------ */
/* Submit / Collect / Extract                                          */
/* ------------------------------------------------------------------ */

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;

    auto *s = static_cast<VifStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph submit (idempotent per frame — first extractor wins)
    int const err = vmaf_sycl_graph_submit(state);
    if (err)
        return err;

    s->pending_index = index;
    s->has_pending = true;

    return 0;
}

// NOLINTNEXTLINE(readability-function-size): SYCL kernel-launch / lifecycle entry — body is dominated by accessor declarations + a single `parallel_for` lambda. Splitting either inlines via macro (no readability win) or introduces a free function the compiler cannot inline back into the device kernel. Keeping it large is the pattern shared across every SYCL TU in this fork (ADR-0141 §2 load-bearing invariant; T7-5 sweep closeout — ADR-0278).
static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<VifStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph wait (idempotent per frame — first extractor wins)
    vmaf_sycl_graph_wait(state);

    // Read back per-scale accumulators
    struct vif_accums accums[VIF_NUM_SCALES];
    std::memcpy(accums, s->h_accum, sizeof(accums));

    // Compute VIF scores per scale (CPU)
    double score_num = 0.0;
    double score_den = 0.0;
    double vif_scale_num[VIF_NUM_SCALES];
    double vif_scale_den[VIF_NUM_SCALES];

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        double const num =
            accums[scale].num_log / 2048.0 + accums[scale].x2 +
            (accums[scale].den_non_log - (accums[scale].num_non_log / 16384.0) / 65025.0);

        double const den = accums[scale].den_log / 2048.0 -
                           (accums[scale].x + accums[scale].num_x * 17) + accums[scale].den_non_log;

        vif_scale_num[scale] = num;
        vif_scale_den[scale] = den;
        score_num += num;
        score_den += den;
    }

    // Write primary per-scale features
    for (int i = 0; i < VIF_NUM_SCALES; i++) {
        double const score = (vif_scale_den[i] > 0.0) ? vif_scale_num[i] / vif_scale_den[i] : 1.0;

        static const char *const key_names[] = {
            "VMAF_integer_feature_vif_scale0_score",
            "VMAF_integer_feature_vif_scale1_score",
            "VMAF_integer_feature_vif_scale2_score",
            "VMAF_integer_feature_vif_scale3_score",
        };

        int const err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, key_names[i], score, index);
        if (err)
            return err;
    }

    // Debug features
    if (s->debug) {
        double const vif = (score_den > 0.0) ? score_num / score_den : 1.0;

        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_vif", vif, index);
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_vif_num", score_num, index);
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_vif_den", score_den, index);

        for (int i = 0; i < VIF_NUM_SCALES; i++) {
            char name[64];
            (void)std::snprintf(name, sizeof(name), "integer_vif_num_scale%d", i);
            vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict, name,
                                                    vif_scale_num[i], index);
            (void)std::snprintf(name, sizeof(name), "integer_vif_den_scale%d", i);
            vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict, name,
                                                    vif_scale_den[i], index);
        }
    }

    s->has_pending = false;
    return 0;
}

static int extract_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    int const err = submit_fex_sycl(fex, ref_pic, ref_pic_90, dist_pic, dist_pic_90, index);
    if (err)
        return err;
    return collect_fex_sycl(fex, index, feature_collector);
}

static int flush_fex_sycl(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    if (!fex)
        return -EINVAL;
    VmafSyclState *state = fex->sycl_state;
    if (state)
        vmaf_sycl_queue_wait(state);
    return 1; // done — collect already consumed pending work
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<VifStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    if (state) {
        vmaf_sycl_queue_wait(state);

        if (s->d_tmp_mu1)
            vmaf_sycl_free(state, s->d_tmp_mu1);
        if (s->d_tmp_mu2)
            vmaf_sycl_free(state, s->d_tmp_mu2);
        if (s->d_tmp_ref)
            vmaf_sycl_free(state, s->d_tmp_ref);
        if (s->d_tmp_dis)
            vmaf_sycl_free(state, s->d_tmp_dis);
        if (s->d_tmp_ref_dis)
            vmaf_sycl_free(state, s->d_tmp_ref_dis);
        if (s->d_tmp_ref_convol)
            vmaf_sycl_free(state, s->d_tmp_ref_convol);
        if (s->d_tmp_dis_convol)
            vmaf_sycl_free(state, s->d_tmp_dis_convol);
        if (s->d_rd_ref)
            vmaf_sycl_free(state, s->d_rd_ref);
        if (s->d_rd_dis)
            vmaf_sycl_free(state, s->d_rd_dis);
        if (s->d_accum)
            vmaf_sycl_free(state, s->d_accum);
        if (s->h_accum)
            vmaf_sycl_free(state, s->h_accum);
        if (s->d_log2_lut)
            vmaf_sycl_free(state, s->d_log2_lut);
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Feature extractor definition                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"VMAF_integer_feature_vif_scale0_score",
                                          "VMAF_integer_feature_vif_scale1_score",
                                          "VMAF_integer_feature_vif_scale2_score",
                                          "VMAF_integer_feature_vif_scale3_score",
                                          "integer_vif",
                                          "integer_vif_num",
                                          "integer_vif_den",
                                          "integer_vif_num_scale0",
                                          "integer_vif_den_scale0",
                                          "integer_vif_num_scale1",
                                          "integer_vif_den_scale1",
                                          "integer_vif_num_scale2",
                                          "integer_vif_den_scale2",
                                          "integer_vif_num_scale3",
                                          "integer_vif_den_scale3",
                                          nullptr};

// NOLINTEND(misc-use-anonymous-namespace, misc-use-internal-linkage)

extern "C" VmafFeatureExtractor vmaf_fex_integer_vif_sycl = {
    .name = "vif_sycl",
    .init = init_fex_sycl,
    .extract = extract_fex_sycl,
    .flush = flush_fex_sycl,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options,
    .priv_size = sizeof(VifStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features,
};
