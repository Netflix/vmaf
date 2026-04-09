/**
 *
 *  Copyright 2016-2024 Netflix, Inc.
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
 * SYCL/DPC++ ADM (Adaptive Detail Model) feature extractor.
 *
 * Implements 4-scale CDF 9/7 (DB2) Discrete Wavelet Transform, decoupling,
 * CSF weighting, and contrast masking using SYCL kernels.
 *
 * Pipeline per scale:
 *   1. DWT vertical pass -> tmp (lo/hi interleaved)
 *   2. DWT horizontal pass -> 4 sub-bands (LL, LH, HL, HH)
 *   3. Decouple+CSF fused pass -> r, csf_a, csf_f
 *   4. CSF denominator reduction -> csf_den_accum
 *   5. Contrast measure reduction -> cm_accum
 *
 * Pattern: init -> submit (non-blocking) -> collect (wait + scores)
 */

#include <sycl/sycl.hpp>

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

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

static constexpr int ADM_NUM_SCALES  = 4;
static constexpr int ADM_NUM_BANDS   = 3; // h, v, d (skip band_a for scoring)
static constexpr double ADM_BORDER_FACTOR = 0.1;

// DWT filter coefficients (DB2, 4-tap)
static constexpr int32_t dwt_lo[4] = { 15826, 27411, 7345, -4240 };
static constexpr int32_t dwt_hi[4] = { -4240, -7345, 27411, -15826 };
static constexpr int32_t dwt_lo_sum = 46342;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Noise floor model parameters (Watson et al. 1997, Y channel)
struct DwtModelParams {
    float a, k, f0;
    float g[4];
};

static const DwtModelParams dwt_model_Y = {
    0.495f, 0.466f, 0.401f, { 1.501f, 1.0f, 0.534f, 1.0f }
};

// Basis function amplitudes (Watson 1997, Table V, transposed)
static const float dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},
    {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},
    {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f},
    {0.023013f, 0.030018f, 0.039156f, 0.030018f}
};

static constexpr int32_t ONE_BY_15   = 8738;
static constexpr int32_t I4_ONE_BY_15 = 286331153;

/* ------------------------------------------------------------------ */
/* Extractor private state                                             */
/* ------------------------------------------------------------------ */

struct AdmStateSycl {
    unsigned width, height, bpc;
    unsigned buf_stride; // aligned stride for DWT bands

    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;

    VmafDictionary *feature_name_dict;

    // rfactors: 3 bands x 4 scales = 12
    float rfactor[12];
    uint32_t i_rfactor[12];

    // DWT intermediate buffers
    int32_t *d_dwt_tmp_ref;   // vertical DWT output for ref
    int32_t *d_dwt_tmp_dis;   // vertical DWT output for dis

    // DWT band outputs: 4 bands x 2 (ref+dis)
    int32_t *d_ref_band[4];   // [0]=a(LL), [1]=h(HL), [2]=v(LH), [3]=d(HH)
    int32_t *d_dis_band[4];

    // Decouple outputs
    int32_t *d_decouple_r[3]; // h, v, d bands

    // CSF outputs
    int32_t *d_csf_a[3];     // CSF-scaled 'a' component
    int32_t *d_csf_f[3];     // |csf_a| / 30

    // Integer division LUT
    int32_t *d_div_lookup;    // 65537 entries

    // Accumulators (device + host)
    int64_t *d_cm_accum;     // 4 scales x 3 bands = 12 int64
    int64_t *d_csf_den_accum; // 4 scales x 3 bands = 12 int64
    int64_t *h_cm_accum;
    int64_t *h_csf_den_accum;

    // Deferred state
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
        .offset = offsetof(AdmStateSycl, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = true},
    },
    {
        .name = "adm_enhn_gain_limit",
        .help = "enhancement gain imposed on ADM, must be >= 1.0",
        .offset = offsetof(AdmStateSycl, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 100.0},
        .min = 1.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .help = "normalized viewing distance",
        .offset = offsetof(AdmStateSycl, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 3.0},
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateSycl, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val = {.i = 1080},
        .min = 480.0,
        .max = 4320.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/* ------------------------------------------------------------------ */
/* Helper: DWT quantization step (visibility threshold model)         */
/* ------------------------------------------------------------------ */

static float dwt_quant_step(int lambda, int theta,
                             double view_dist, int display_h)
{
    float r = (float)(view_dist * display_h * M_PI / 180.0);
    float temp = std::log10f(
        std::powf(2.0f, lambda + 1) * dwt_model_Y.f0 *
        dwt_model_Y.g[theta] / r);
    return 2.0f * dwt_model_Y.a *
           std::powf(10.0f, dwt_model_Y.k * temp * temp) /
           dwt_basis_amp[lambda][theta];
}

/* ------------------------------------------------------------------ */
/* Device-side helpers                                                 */
/* ------------------------------------------------------------------ */

static inline int dev_mirror_adm(int idx, int sup) {
    if (idx < 0)   return -idx;
    if (idx >= sup) return 2 * sup - idx - 1;
    return idx;
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: DWT Vertical Pass (ref+dis fused)                     */
/* ------------------------------------------------------------------ */

static sycl::event launch_dwt_vert_pair(sycl::queue &q,
                              const void *input_ref, int32_t *dwt_tmp_ref,
                              const void *input_dis, int32_t *dwt_tmp_dis,
                              int scale, unsigned width, unsigned height,
                              unsigned in_stride, unsigned bpc,
                              unsigned v_shift, unsigned v_add)
{
    // Output: dwt_tmp has interleaved lo/hi rows:
    //   row i contains lo(i) and hi(i) for half_h rows
    unsigned half_h = (height + 1) / 2;
    auto e_scale = scale;
    auto e_bpc = bpc;
    auto e_w = width;
    auto e_h = height;
    auto e_in_stride = in_stride;
    auto e_v_shift = v_shift;
    auto e_v_add = v_add;
    auto p_ref_in = input_ref;
    auto p_dis_in = input_dis;

    constexpr int WG_X = 32, WG_Y = 8;
    // Each output row n needs input rows 2n-1..2n+2 (4 taps).
    // For WG_Y outputs: 2*WG_Y + 2 input rows in the tile.
    constexpr int TILE_H = 2 * WG_Y + 2;  // 18
    // Z=2: ref(0) + dis(1)
    sycl::range<3> global(2,
                          ((half_h + WG_Y - 1) / WG_Y) * WG_Y,
                          ((width  + WG_X - 1) / WG_X) * WG_X);
    sycl::range<3> local(1, WG_Y, WG_X);

    return q.submit([&](sycl::handler &cgh) {
        // SLM tile: TILE_H rows × WG_X columns
        sycl::local_accessor<int32_t, 2> tile(
            sycl::range<2>(TILE_H, WG_X), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> item) {
                // dim 0 = ref(0) or dis(1), dim 1 = row, dim 2 = col
                const int is_dis = item.get_global_id(0);
                const int gx = item.get_global_id(2);
                const int gy = item.get_global_id(1);
                const int lx = item.get_local_id(2);
                const int ly = item.get_local_id(1);
                const int lid = ly * WG_X + lx;
                constexpr int WG_SIZE = WG_X * WG_Y;

                const void *p_in = (is_dis == 0) ? p_ref_in : p_dis_in;
                int32_t *dwt_out = (is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;

                // Tile origin in input space
                int tile_col = (int)(item.get_group(2) * WG_X);
                int n_start  = (int)(item.get_group(1) * WG_Y);
                int row_start = 2 * n_start - 1;  // first input row

                // Read pixel from source at (x, y) with mirroring
                auto read_px = [&](int x, int y) -> int32_t {
                    y = dev_mirror_adm(y, (int)e_h);
                    if (x >= (int)e_w) return 0;
                    if (e_scale == 0) {
                        if (e_bpc <= 8) {
                            return static_cast<const uint8_t *>(p_in)
                                       [y * e_in_stride + x];
                        } else {
                            return static_cast<const uint16_t *>(p_in)
                                       [y * (e_in_stride / 2) + x];
                        }
                    } else {
                        return static_cast<const int32_t *>(p_in)
                                   [y * e_in_stride + x];
                    }
                };

                // Cooperative tile load: TILE_H * WG_X elements
                constexpr int TILE_ELEMS = TILE_H * WG_X;
                bool interior = (row_start >= 0) &&
                    (row_start + TILE_H <= (int)e_h) &&
                    (tile_col + WG_X <= (int)e_w);

                if (interior) {
                    // Fast path: no boundary checks
                    for (int i = lid; i < TILE_ELEMS; i += WG_SIZE) {
                        int tr = i / WG_X;
                        int tc = i % WG_X;
                        int x = tile_col + tc;
                        int y = row_start + tr;
                        if (e_scale == 0) {
                            if (e_bpc <= 8) {
                                tile[tr][tc] = static_cast<const uint8_t *>(p_in)
                                                   [y * e_in_stride + x];
                            } else {
                                tile[tr][tc] = static_cast<const uint16_t *>(p_in)
                                                   [y * (e_in_stride / 2) + x];
                            }
                        } else {
                            tile[tr][tc] = static_cast<const int32_t *>(p_in)
                                               [y * e_in_stride + x];
                        }
                    }
                } else {
                    // Boundary path: mirror + bounds check
                    for (int i = lid; i < TILE_ELEMS; i += WG_SIZE) {
                        int tr = i / WG_X;
                        int tc = i % WG_X;
                        tile[tr][tc] = read_px(tile_col + tc,
                                               row_start + tr);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (gx >= (int)e_w || gy >= (int)half_h) return;

                // Read 4 filter taps from shared memory
                int base = 2 * ly;  // tile row offset
                int32_t s0 = tile[base][lx];
                int32_t s1 = tile[base + 1][lx];
                int32_t s2 = tile[base + 2][lx];
                int32_t s3 = tile[base + 3][lx];

                // Lo-pass: coeffs = {15826, 27411, 7345, -4240}
                int64_t lo_val = (int64_t)dwt_lo[0] * s0 +
                                 (int64_t)dwt_lo[1] * s1 +
                                 (int64_t)dwt_lo[2] * s2 +
                                 (int64_t)dwt_lo[3] * s3;

                // Hi-pass: coeffs = {-4240, -7345, 27411, -15826}
                int64_t hi_val = (int64_t)dwt_hi[0] * s0 +
                                 (int64_t)dwt_hi[1] * s1 +
                                 (int64_t)dwt_hi[2] * s2 +
                                 (int64_t)dwt_hi[3] * s3;

                // Scale 0: subtract DC offset from lo
                if (e_scale == 0) {
                    lo_val -= (int64_t)dwt_lo_sum * e_v_add;
                }

                // Quantize
                int32_t lo_out, hi_out;
                if (e_v_shift > 0) {
                    lo_out = (int32_t)((lo_val +
                        ((int64_t)1 << (e_v_shift - 1))) >> e_v_shift);
                    hi_out = (int32_t)((hi_val +
                        ((int64_t)1 << (e_v_shift - 1))) >> e_v_shift);
                } else {
                    lo_out = (int32_t)lo_val;
                    hi_out = (int32_t)hi_val;
                }

                // Interleaved output: lo followed by hi
                unsigned out_stride = e_w * 2;
                dwt_out[gy * out_stride + gx]     = lo_out;
                dwt_out[gy * out_stride + e_w + gx] = hi_out;
            });
    });
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: DWT Horizontal Pass (ref+dis fused)                   */
/* ------------------------------------------------------------------ */

static sycl::event launch_dwt_hori_pair(sycl::queue &q,
                              const int32_t *dwt_tmp_ref,
                              int32_t *ref_band_a, int32_t *ref_band_h,
                              int32_t *ref_band_v, int32_t *ref_band_d,
                              const int32_t *dwt_tmp_dis,
                              int32_t *dis_band_a, int32_t *dis_band_h,
                              int32_t *dis_band_v, int32_t *dis_band_d,
                              unsigned width, unsigned height,
                              unsigned buf_stride,
                              unsigned h_shift, unsigned h_add)
{
    unsigned half_w = (width + 1) / 2;
    unsigned half_h = (height + 1) / 2;
    auto e_w = width;
    auto e_half_w = half_w;
    auto e_half_h = half_h;
    auto e_buf_stride = buf_stride;
    auto e_h_shift = h_shift;

    constexpr int WG_X = 32, WG_Y = 8;
    // Z=2: ref(0) + dis(1)
    sycl::range<3> global(2,
                          ((half_h + WG_Y - 1) / WG_Y) * WG_Y,
                          ((half_w + WG_X - 1) / WG_X) * WG_X);
    sycl::range<3> local(1, WG_Y, WG_X);

    return q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> item) {
                const int is_dis = item.get_global_id(0);
                const int gx = item.get_global_id(2);
                const int gy = item.get_global_id(1);
                if (gx >= (int)e_half_w || gy >= (int)e_half_h) return;

                const int32_t *dwt_tmp = (is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;
                int32_t *band_a = (is_dis == 0) ? ref_band_a : dis_band_a;
                int32_t *band_h = (is_dis == 0) ? ref_band_h : dis_band_h;
                int32_t *band_v = (is_dis == 0) ? ref_band_v : dis_band_v;
                int32_t *band_d = (is_dis == 0) ? ref_band_d : dis_band_d;

                unsigned tmp_stride = e_w * 2;

                // Read from lo row (first half of tmp)
                auto read_lo = [&](int x) -> int32_t {
                    x = dev_mirror_adm(x, (int)e_w);
                    return dwt_tmp[gy * tmp_stride + x];
                };

                // Read from hi row (second half of tmp)
                auto read_hi = [&](int x) -> int32_t {
                    x = dev_mirror_adm(x, (int)e_w);
                    return dwt_tmp[gy * tmp_stride + e_w + x];
                };

                int base_x = 2 * gx;

                // Horizontal filter on lo row -> band_a, band_h
                int32_t l0 = read_lo(base_x - 1);
                int32_t l1 = read_lo(base_x);
                int32_t l2 = read_lo(base_x + 1);
                int32_t l3 = read_lo(base_x + 2);

                int64_t a_val = (int64_t)dwt_lo[0] * l0 +
                                (int64_t)dwt_lo[1] * l1 +
                                (int64_t)dwt_lo[2] * l2 +
                                (int64_t)dwt_lo[3] * l3;
                int64_t h_val = (int64_t)dwt_hi[0] * l0 +
                                (int64_t)dwt_hi[1] * l1 +
                                (int64_t)dwt_hi[2] * l2 +
                                (int64_t)dwt_hi[3] * l3;

                // Horizontal filter on hi row -> band_v, band_d
                int32_t h0 = read_hi(base_x - 1);
                int32_t h1 = read_hi(base_x);
                int32_t h2 = read_hi(base_x + 1);
                int32_t h3 = read_hi(base_x + 2);

                int64_t v_val = (int64_t)dwt_lo[0] * h0 +
                                (int64_t)dwt_lo[1] * h1 +
                                (int64_t)dwt_lo[2] * h2 +
                                (int64_t)dwt_lo[3] * h3;
                int64_t d_val = (int64_t)dwt_hi[0] * h0 +
                                (int64_t)dwt_hi[1] * h1 +
                                (int64_t)dwt_hi[2] * h2 +
                                (int64_t)dwt_hi[3] * h3;

                // Quantize
                int32_t a_out, h_out, v_out, d_out;
                if (e_h_shift > 0) {
                    int64_t rnd = (int64_t)1 << (e_h_shift - 1);
                    a_out = (int32_t)((a_val + rnd) >> e_h_shift);
                    h_out = (int32_t)((h_val + rnd) >> e_h_shift);
                    v_out = (int32_t)((v_val + rnd) >> e_h_shift);
                    d_out = (int32_t)((d_val + rnd) >> e_h_shift);
                } else {
                    a_out = (int32_t)a_val;
                    h_out = (int32_t)h_val;
                    v_out = (int32_t)v_val;
                    d_out = (int32_t)d_val;
                }

                unsigned idx = gy * e_buf_stride + gx;
                band_a[idx] = a_out;
                band_h[idx] = h_out;
                band_v[idx] = v_out;
                band_d[idx] = d_out;
            });
    });
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: Decouple + CSF (Fused)                                */
/* ------------------------------------------------------------------ */

/*
 * Enhancement gain limiting: emulate double-precision multiply using int64.
 *
 * The CPU reference does: rst = (int)(r_val * gain_limit_double), then
 * clamps via min/max against th.  gain_limit is in [1.0, 100.0].
 *
 * For production models (gain = 1.0 or 100.0), the Q31 fixed-point
 * representation is exact.  For non-integer gain values (e.g. 1.2 in
 * unit tests), the result may differ by at most ±1 from double precision
 * for extreme DWT coefficient magnitudes (|r_val| near 2^31).
 *
 * The split-multiply avoids int64 overflow:
 *   gain_q31 = round(gain * 2^31)      -- up to 38 bits
 *   gain_hi  = gain_q31 >> 16           -- up to 22 bits
 *   gain_lo  = gain_q31 & 0xFFFF        -- 16 bits
 *   product  = r_val * gain_hi << 16 + r_val * gain_lo
 *   result   = product >> 31
 */
struct GainLimitQ31 {
    int32_t gain_hi;  // upper 22 bits of gain_q31
    int32_t gain_lo;  // lower 16 bits of gain_q31
};

static inline GainLimitQ31 gain_limit_to_q31(double gain_limit)
{
    int64_t gain_q31 = (int64_t)llround(gain_limit * (1LL << 31));
    return {
        .gain_hi = (int32_t)(gain_q31 >> 16),
        .gain_lo = (int32_t)(gain_q31 & 0xFFFF),
    };
}

template <bool UseFP64>
static sycl::event launch_decouple_csf(sycl::queue &q,
                                  int scale, unsigned half_w, unsigned half_h,
                                  unsigned buf_stride,
                                  double adm_enhn_gain_limit,
                                  uint32_t i_rfactor_h,
                                  uint32_t i_rfactor_v,
                                  uint32_t i_rfactor_d,
                                  const int32_t *ref_h, const int32_t *ref_v,
                                  const int32_t *ref_d,
                                  const int32_t *dis_h, const int32_t *dis_v,
                                  const int32_t *dis_d,
                                  int32_t *out_r_h, int32_t *out_r_v,
                                  int32_t *out_r_d,
                                  int32_t *csf_a_h, int32_t *csf_a_v,
                                  int32_t *csf_a_d,
                                  int32_t *csf_f_h, int32_t *csf_f_v,
                                  int32_t *csf_f_d,
                                  const int32_t *div_lookup)
{
    auto e_scale = scale;
    auto e_w = half_w;
    auto e_h = half_h;
    auto e_stride = buf_stride;

    // Always use int64 Q31 fixed-point for gain limiting.
    // This avoids requiring fp64 capability (not available on Intel Arc A-series,
    // Intel iGPUs, mobile GPUs, etc.) while being exact for all production
    // gain values (1.0 and 100.0).  At most ±1 LSB for exotic fractional gains.
    //
    // NOTE: We must NOT instantiate a kernel lambda that uses 'double' operations
    // because in SYCL, all kernels from a translation unit share a SPIR-V module.
    // If the module contains fp64 instructions, the runtime rejects the entire
    // module on non-fp64 hardware — even if the fp64 kernel is never submitted.
    GainLimitQ31 e_gain_q31 = gain_limit_to_q31(adm_enhn_gain_limit);

    constexpr float cos_1deg_sq = 0.99969541789740297f; // cos(pi/180)^2

    constexpr int WG_X = 16, WG_Y = 16;
    sycl::range<2> global(((half_h + WG_Y - 1) / WG_Y) * WG_Y,
                          ((half_w + WG_X - 1) / WG_X) * WG_X);
    sycl::range<2> local(WG_Y, WG_X);

    return q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                const int gx = item.get_global_id(1);
                const int gy = item.get_global_id(0);
                if (gx >= (int)e_w || gy >= (int)e_h) return;

                unsigned idx = gy * e_stride + gx;

                // Load all 3 bands for ref and dis
                int32_t o[3] = { ref_h[idx], ref_v[idx], ref_d[idx] };
                int32_t t[3] = { dis_h[idx], dis_v[idx], dis_d[idx] };
                int32_t *r_ptr[3]   = { out_r_h, out_r_v, out_r_d };
                int32_t *ca_ptr[3]  = { csf_a_h, csf_a_v, csf_a_d };
                int32_t *cf_ptr[3]  = { csf_f_h, csf_f_v, csf_f_d };
                uint32_t irf[3] = { i_rfactor_h, i_rfactor_v, i_rfactor_d };

                // --- 2D Angle test using (H, V) bands only ---
                // Matches CPU: angle between (oh,ov) and (th,tv)
                int64_t ot_dp = (int64_t)o[0] * t[0] +
                                (int64_t)o[1] * t[1];
                int64_t o_mag_sq = (int64_t)o[0] * o[0] +
                                   (int64_t)o[1] * o[1];
                int64_t t_mag_sq = (int64_t)t[0] * t[0] +
                                   (int64_t)t[1] * t[1];

                // CPU divides by 4096 before float conversion to
                // avoid float precision issues with large int64 values
                float ot_f  = (float)ot_dp  / 4096.0f;
                float om_f  = (float)o_mag_sq / 4096.0f;
                float tm_f  = (float)t_mag_sq / 4096.0f;

                bool angle_flag = (ot_f >= 0.0f) &&
                    (ot_f * ot_f >= cos_1deg_sq * om_f * tm_f);

                // Process each band
                for (int band = 0; band < 3; band++) {
                    int32_t oh = o[band];
                    int32_t th = t[band];

                    // --- Decouple: project dis onto ref ---
                    int32_t r_val = 0;
                    int32_t k = 0;

                    if (oh == 0) {
                        // When ref is 0, k = 32768 (Q15 = 1.0)
                        k = 32768;
                        // r = (32768 * 0 + 16384) >> 15 = 0
                        r_val = 0;
                    } else {
                        int32_t abs_oh = oh < 0 ? -oh : oh;
                        int32_t sign_oh = oh < 0 ? -1 : 1;

                        int32_t div_val;
                        int kh_shift = 0;

                        if (e_scale == 0) {
                            // Scale 0: 16-bit values, direct LUT
                            // CPU: div_lookup[oh + 32768] (oh is int16)
                            // Equivalent: div_lookup[32768 + abs_oh] * sign
                            div_val = div_lookup[32768 + abs_oh] *
                                      sign_oh;
                        } else {
                            // Scale 1-3: reduce to ~15 bits
                            // Match CPU's get_best15_from32 with rounding
                            if (abs_oh < 32768) {
                                div_val = div_lookup[32768 + abs_oh] *
                                          sign_oh;
                            } else {
                                // Count how many bits to shift
                                uint32_t tmp = (uint32_t)abs_oh;
                                // Find MSB position
                                int n = 0;
                                uint32_t v = tmp;
                                // Count leading zeros manually
                                // (no __builtin_clz in SYCL device)
                                if (v >= (1u << 16)) { n += 16; v >>= 16; }
                                if (v >= (1u << 8))  { n += 8;  v >>= 8;  }
                                if (v >= (1u << 4))  { n += 4;  v >>= 4;  }
                                if (v >= (1u << 2))  { n += 2;  v >>= 2;  }
                                if (v >= (1u << 1))  { n += 1;  v >>= 1;  }
                                // n is floor(log2(tmp)), so bit width is n+1
                                // clz = 31 - n for 32-bit
                                int clz = 31 - n;
                                int ks = 17 - clz;
                                uint32_t rounded = (tmp +
                                    (1u << (ks - 1))) >> ks;
                                kh_shift = ks;
                                div_val = div_lookup[32768 + rounded] *
                                          sign_oh;
                            }
                        }

                        // k = clamp((div_val * th) >> shift, 0, 32768)
                        int64_t k64 = (int64_t)div_val * th;
                        if (e_scale == 0) {
                            k = (int32_t)((k64 + (1 << 14)) >> 15);
                        } else {
                            int shift = 15 + kh_shift;
                            k = (int32_t)((k64 +
                                 ((int64_t)1 << (shift - 1))) >> shift);
                        }

                        if (k < 0) k = 0;
                        if (k > 32768) k = 32768;

                        r_val = (int32_t)(((int64_t)k * oh + 16384) >> 15);
                    }

                    // --- Enhancement gain limiting ---
                    // Match CPU: uses floating-point rst_f for sign check,
                    // clamps against distorted signal (th), not reference
                    if (angle_flag) {
                        // rst_f = (k/32768) * (oh/64) — sign depends on
                        // k (>=0) and oh, so rst_f > 0 iff oh > 0
                        float rst_f = ((float)k / 32768.0f) *
                                      ((float)oh / 64.0f);

                        // int64 Q31 split-multiply gain limiting
                        // gained = (r_val * gain_q31) >> 31
                        //        = (r_val*hi << 16 + r_val*lo) >> 31
                        //        = r_val*hi >> 15 + remainder >> 31
                        int64_t prod_hi = (int64_t)r_val * e_gain_q31.gain_hi;
                        int64_t prod_lo = (int64_t)r_val * e_gain_q31.gain_lo;
                        int64_t main_part = prod_hi >> 15;
                        int64_t rem = ((prod_hi - (main_part << 15)) << 16) + prod_lo;
                        // Keep as int64 — gained can exceed int32 at scales 1-3
                        // where r_val is 32-bit and gain=100 → gained ≈ r*100
                        int64_t gained = main_part + (rem >> 31);

                        if (rst_f > 0.0f) {
                            int64_t c = (gained < (int64_t)th) ? gained : (int64_t)th;
                            r_val = (int32_t)c;
                        } else if (rst_f < 0.0f) {
                            int64_t c = (gained > (int64_t)th) ? gained : (int64_t)th;
                            r_val = (int32_t)c;
                        }
                    }

                    r_ptr[band][idx] = r_val;

                    // a = dis - r (residual)
                    int32_t a_val = th - r_val;

                    // --- Fused CSF ---
                    int32_t csf_a_val, csf_f_val;
                    if (e_scale == 0) {
                        // Scale 0: rfactor * 2^21 (h,v) or 2^23 (d)
                        int shift = (band < 2) ? 15 : 17;
                        int64_t rnd_csf = ((int64_t)1 << (shift - 1));
                        csf_a_val = (int32_t)(
                            ((int64_t)irf[band] * a_val + rnd_csf) >> shift);
                        // csf_f = (4369 * |csf_a| + 2048) >> 12
                        int32_t abs_csf = csf_a_val < 0 ?
                                          -csf_a_val : csf_a_val;
                        csf_f_val = (int32_t)(
                            ((int64_t)4369 * abs_csf + 2048) >> 12);
                    } else {
                        // Scales 1-3: rfactor * 2^32
                        csf_a_val = (int32_t)(
                            ((int64_t)irf[band] * a_val +
                             (1LL << 27)) >> 28);
                        int32_t abs_csf = csf_a_val < 0 ?
                                          -csf_a_val : csf_a_val;
                        csf_f_val = (int32_t)(
                            ((int64_t)143165577 * abs_csf +
                             (1LL << 31)) >> 32);
                    }

                    ca_ptr[band][idx] = csf_a_val;
                    cf_ptr[band][idx] = csf_f_val;
                }
            });
    });
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: CSF Denominator Reduction — 3 bands fused             */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* SYCL Kernel: CSF Denominator + Contrast Measure — 3 bands fused    */
/*                                                                     */
/* Combines csf_den_3band and cm_3band into a single dispatch to save */
/* kernel launch overhead and improve data locality.  Each work-group  */
/* processes one row of one band, computing both csf_den (|ref|³) and  */
/* cm (contrast masking) values, then does two independent reductions. */
/* ------------------------------------------------------------------ */

static sycl::event launch_csf_den_cm_3band(sycl::queue &q,
                        int scale,
                        unsigned half_w, unsigned half_h,
                        unsigned buf_stride,
                        // csf_den inputs
                        const int32_t *ref_band_h,
                        const int32_t *ref_band_v,
                        const int32_t *ref_band_d,
                        int64_t *csf_accum_h, int64_t *csf_accum_v,
                        int64_t *csf_accum_d,
                        // cm inputs
                        uint32_t i_rfactor_h, uint32_t i_rfactor_v,
                        uint32_t i_rfactor_d,
                        const int32_t *decouple_r_h,
                        const int32_t *decouple_r_v,
                        const int32_t *decouple_r_d,
                        const int32_t *csf_f_h, const int32_t *csf_f_v,
                        const int32_t *csf_f_d,
                        const int32_t *csf_a_h, const int32_t *csf_a_v,
                        const int32_t *csf_a_d,
                        int64_t *cm_accum_h, int64_t *cm_accum_v,
                        int64_t *cm_accum_d)
{
    int left  = (int)(half_w * ADM_BORDER_FACTOR - 0.5);
    int top   = (int)(half_h * ADM_BORDER_FACTOR - 0.5);
    int right = (int)half_w - left;
    int bottom = (int)half_h - top;
    if (left < 0) left = 0;
    if (top < 0) top = 0;

    int active_w = right - left;
    int active_h = bottom - top;
    if (active_w <= 0 || active_h <= 0) return sycl::event{};

    // csf_den shift params
    uint32_t csf_shift_sq, csf_shift_cub, csf_shift_accum;
    if (scale == 0) {
        csf_shift_sq = 0;
        csf_shift_cub = 0;
        int area = active_w * active_h;
        int s = (int)std::ceil(std::log2(area) - 20);
        csf_shift_accum = s > 0 ? (uint32_t)s : 0;
    } else {
        csf_shift_sq = (scale == 2) ? 30 : 31;
        csf_shift_cub = (uint32_t)std::ceil(std::log2(active_w));
        csf_shift_accum = (uint32_t)std::ceil(std::log2(active_h));
    }

    // cm shift params
    uint32_t cm_shift_inner_accum = (uint32_t)std::ceil(std::log2((double)half_h));
    uint32_t cm_shift_sub[3], cm_shift_xsq[3], cm_shift_xcub[3];
    for (int b = 0; b < 3; b++) {
        if (scale == 0) {
            cm_shift_sub[b] = (b < 2) ? 10 : 12;
            cm_shift_xsq[b] = (b < 2) ? 29 : 30;
            cm_shift_xcub[b] = (uint32_t)(std::ceil(std::log2((double)half_w)) -
                                          ((b < 2) ? 4 : 3));
        } else {
            cm_shift_sub[b] = 15;
            cm_shift_xsq[b] = 30;
            cm_shift_xcub[b] = (uint32_t)std::ceil(std::log2((double)half_w));
        }
    }

    auto e_left = left;
    auto e_top = top;
    auto e_right = right;
    auto e_stride = buf_stride;
    auto e_scale = scale;
    auto e_w = half_w;
    auto e_h = half_h;
    // csf_den
    auto e_csf_shift_sq = csf_shift_sq;
    auto e_csf_shift_cub = csf_shift_cub;
    auto e_csf_shift_accum = csf_shift_accum;
    // cm
    auto e_cm_shift_inner = cm_shift_inner_accum;
    auto e_cm_shift_sub0 = cm_shift_sub[0], e_cm_shift_sub2 = cm_shift_sub[2];
    auto e_cm_shift_xsq0 = cm_shift_xsq[0], e_cm_shift_xsq2 = cm_shift_xsq[2];
    auto e_cm_shift_xcub0 = cm_shift_xcub[0], e_cm_shift_xcub2 = cm_shift_xcub[2];

    constexpr int WG_SIZE = 256;
    constexpr int MAX_SUBGROUPS = 32;
    int num_rows = bottom - top;

    // Dispatch 3 × num_rows workgroups: band = wg / num_rows
    return q.submit([&](sycl::handler &cgh) {
        // 2 reduction slots: [0..MAX_SUBGROUPS) for csf_den,
        //                    [MAX_SUBGROUPS..2*MAX_SUBGROUPS) for cm
        sycl::local_accessor<int64_t, 1> lmem(
            sycl::range<1>(2 * MAX_SUBGROUPS), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(3 * num_rows * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] {
                int wg = item.get_group(0);
                int band_idx = wg / num_rows;
                int row_idx = wg % num_rows;
                int lid = item.get_local_id(0);
                int row = e_top + row_idx;

                // Select band-specific pointers
                const int32_t *ref_band = (band_idx == 0) ? ref_band_h :
                                          (band_idx == 1) ? ref_band_v :
                                                            ref_band_d;
                int64_t *csf_accum_ptr = (band_idx == 0) ? csf_accum_h :
                                         (band_idx == 1) ? csf_accum_v :
                                                           csf_accum_d;
                const int32_t *decouple_r = (band_idx == 0) ? decouple_r_h :
                                            (band_idx == 1) ? decouple_r_v :
                                                              decouple_r_d;
                int64_t *cm_accum_ptr = (band_idx == 0) ? cm_accum_h :
                                        (band_idx == 1) ? cm_accum_v :
                                                          cm_accum_d;
                uint32_t i_rfactor = (band_idx == 0) ? i_rfactor_h :
                                     (band_idx == 1) ? i_rfactor_v :
                                                       i_rfactor_d;
                auto e_cm_shift_sub = (band_idx < 2) ? e_cm_shift_sub0 : e_cm_shift_sub2;
                auto e_cm_shift_xsq = (band_idx < 2) ? e_cm_shift_xsq0 : e_cm_shift_xsq2;
                auto e_cm_shift_xcub = (band_idx < 2) ? e_cm_shift_xcub0 : e_cm_shift_xcub2;

                const int32_t *cf_ptrs[3] = { csf_f_h, csf_f_v, csf_f_d };
                const int32_t *ca_ptrs[3] = { csf_a_h, csf_a_v, csf_a_d };

                // Per-thread accumulators for both reductions
                int64_t local_csf_sum = 0;
                int64_t local_cm_sum = 0;

                for (int col = e_left + lid; col < e_right;
                     col += WG_SIZE) {
                    unsigned idx = row * e_stride + col;

                    // ── CSF denominator: |ref|³ ──
                    int32_t t = ref_band[idx];
                    int32_t abs_t = t < 0 ? -t : t;
                    int64_t t_cub;
                    if (e_scale == 0) {
                        t_cub = (int64_t)abs_t * abs_t * abs_t;
                    } else {
                        // CPU uses 1<<N (biased rounding), not 1<<(N-1)
                        int64_t rnd_sq = ((int64_t)1 << e_csf_shift_sq);
                        int64_t t_sq = ((int64_t)abs_t * abs_t + rnd_sq) >>
                                       e_csf_shift_sq;
                        int64_t rnd_cub = e_csf_shift_cub > 0 ?
                            ((int64_t)1 << (e_csf_shift_cub - 1)) : 0;
                        t_cub = (t_sq * abs_t + rnd_cub) >> e_csf_shift_cub;
                    }
                    local_csf_sum += t_cub;

                    // ── Contrast measure ──
                    int64_t thr = 0;
                    for (int b = 0; b < 3; b++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue;
                                int ny = row + dy;
                                int nx = col + dx;
                                if (ny < 0) ny = 0;
                                if (ny >= (int)e_h) ny = (int)e_h - 1;
                                if (nx < 0) nx = 0;
                                if (nx >= (int)e_w) nx = (int)e_w - 1;
                                thr += cf_ptrs[b][ny * e_stride + nx];
                            }
                        }
                        int32_t ca = ca_ptrs[b][idx];
                        int32_t abs_ca = ca < 0 ? -ca : ca;
                        if (e_scale == 0) {
                            thr += ((int64_t)ONE_BY_15 * abs_ca + 2048) >> 12;
                        } else {
                            thr += ((int64_t)I4_ONE_BY_15 * abs_ca +
                                    (1LL << 31)) >> 32;
                        }
                    }

                    int32_t r_val = decouple_r[idx];
                    int64_t cm;
                    if (e_scale == 0) {
                        cm = (int64_t)i_rfactor * r_val;
                        cm = cm < 0 ? -cm : cm;
                        cm -= (thr << e_cm_shift_sub);
                    } else {
                        int64_t scaled_r = ((int64_t)i_rfactor * r_val +
                                           (1LL << 27)) >> 28;
                        cm = scaled_r < 0 ? -scaled_r : scaled_r;
                        cm -= thr;
                    }
                    if (cm < 0) cm = 0;

                    int64_t rnd_sq2 = e_cm_shift_xsq > 0 ?
                        ((int64_t)1 << (e_cm_shift_xsq - 1)) : 0;
                    int32_t x_sq = (int32_t)(((cm * cm) + rnd_sq2) >>
                                             e_cm_shift_xsq);
                    int64_t rnd_cub2 = e_cm_shift_xcub > 0 ?
                        ((int64_t)1 << (e_cm_shift_xcub - 1)) : 0;
                    int64_t x_cub = (((int64_t)x_sq * cm) + rnd_cub2) >>
                                    e_cm_shift_xcub;
                    local_cm_sum += x_cub;
                }

                // ── Two-phase reduction for both values ──
                sycl::sub_group sg = item.get_sub_group();
                int64_t sg_csf = sycl::reduce_over_group(
                    sg, local_csf_sum, sycl::plus<int64_t>{});
                int64_t sg_cm = sycl::reduce_over_group(
                    sg, local_cm_sum, sycl::plus<int64_t>{});

                const uint32_t sg_id = sg.get_group_linear_id();
                const uint32_t sg_lid = sg.get_local_linear_id();
                const uint32_t n_subgroups = sg.get_group_linear_range();

                if (sg_lid == 0) {
                    lmem[sg_id] = sg_csf;
                    lmem[MAX_SUBGROUPS + sg_id] = sg_cm;
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (lid == 0) {
                    int64_t total_csf = 0, total_cm = 0;
                    for (uint32_t s = 0; s < n_subgroups; s++) {
                        total_csf += lmem[s];
                        total_cm  += lmem[MAX_SUBGROUPS + s];
                    }

                    // csf_den: shift and atomic add
                    int64_t rnd_csf = e_csf_shift_accum > 0 ?
                        ((int64_t)1 << (e_csf_shift_accum - 1)) : 0;
                    int64_t shifted_csf = (total_csf + rnd_csf) >>
                                          e_csf_shift_accum;
                    sycl::atomic_ref<int64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        csf_ref(*csf_accum_ptr);
                    csf_ref.fetch_add(shifted_csf);

                    // cm: shift and atomic add
                    int64_t rnd_cm = e_cm_shift_inner > 0 ?
                        ((int64_t)1 << (e_cm_shift_inner - 1)) : 0;
                    int64_t shifted_cm = (total_cm + rnd_cm) >>
                                         e_cm_shift_inner;
                    sycl::atomic_ref<int64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        cm_ref(*cm_accum_ptr);
                    cm_ref.fetch_add(shifted_cm);
                }
            });
    });
}

/* ------------------------------------------------------------------ */
/* CPU scoring functions                                               */
/* ------------------------------------------------------------------ */

static void conclude_adm_cm(int64_t *accum, int h, int w, int scale,
                              float *result)
{
    int left  = (int)(w * ADM_BORDER_FACTOR - 0.5);
    int top   = (int)(h * ADM_BORDER_FACTOR - 0.5);
    int right  = w - left;
    int bottom = h - top;

    const uint32_t shift_inner_accum = (uint32_t)std::ceil(std::log2(h));

    float powf_add = std::powf(
        (float)(bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; i++) {
        float f_accum;
        if (scale == 0) {
            // CPU uses w (full band width) for shift_xcub, not active_w
            const uint32_t shift_xcub[3] = {
                (uint32_t)(std::ceil(std::log2((double)w)) - 4),
                (uint32_t)(std::ceil(std::log2((double)w)) - 4),
                (uint32_t)(std::ceil(std::log2((double)w)) - 3)
            };
            int constant_offset[3] = { 52, 52, 57 };
            f_accum = (float)(accum[i] /
                std::pow(2.0, constant_offset[i] -
                         shift_xcub[i] - shift_inner_accum));
        } else {
            // CPU uses w (full band width) for shift_cub, not active_w
            uint32_t shift_cub = (uint32_t)std::ceil(
                std::log2((double)w));
            float final_shift[3] = {
                std::powf(2.0f, 45.0f - shift_cub - shift_inner_accum),
                std::powf(2.0f, 39.0f - shift_cub - shift_inner_accum),
                std::powf(2.0f, 36.0f - shift_cub - shift_inner_accum)
            };
            f_accum = (float)(accum[i] / final_shift[scale - 1]);
        }
        *result += std::powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

static void conclude_adm_csf_den(uint64_t *accum, int h, int w, int scale,
                                   float *result, float view_dist,
                                   float display_h)
{
    int left  = (int)(w * ADM_BORDER_FACTOR - 0.5);
    int top   = (int)(h * ADM_BORDER_FACTOR - 0.5);
    int right  = w - left;
    int bottom = h - top;

    float factor1 = dwt_quant_step(scale, 1, view_dist, (int)display_h);
    float factor2 = dwt_quant_step(scale, 2, view_dist, (int)display_h);
    float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    const uint32_t accum_convert[4] = { 18, 32, 27, 23 };

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)std::ceil(
            std::log2((bottom - top) * (right - left)) - 20);
        if (shift_accum < 0) shift_accum = 0;
        shift_csf = std::pow(2.0, accum_convert[scale] - shift_accum);
    } else {
        shift_accum = (int32_t)std::ceil(std::log2(bottom - top));
        uint32_t shift_cub = (uint32_t)std::ceil(std::log2(right - left));
        shift_csf = std::pow(2.0, accum_convert[scale] - shift_accum -
                             shift_cub);
    }

    float powf_add = std::powf(
        (float)(bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; i++) {
        double csf = (double)(accum[i] / shift_csf) *
                     std::pow(rfactor[i], 3);
        *result += std::powf((float)csf, 1.0f / 3.0f) + powf_add;
    }
}

/* ------------------------------------------------------------------ */
/* Feature extractor callbacks                                         */
/* ------------------------------------------------------------------ */

// Forward declarations for combined graph callbacks (defined after enqueue_adm_work_impl)
static void enqueue_adm_work(void *queue_ptr, void *priv,
                              void *shared_ref, void *shared_dis);
static void adm_pre_graph(void *queue_ptr, void *priv);
static void adm_post_graph(void *queue_ptr, void *priv);

static int init_fex_sycl(VmafFeatureExtractor *fex,
                          enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<AdmStateSycl *>(fex->priv);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->has_pending = false;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "adm_sycl: no SYCL state\n");
        return -EINVAL;
    }

    VmafSyclState *state = fex->sycl_state;

    // Initialize shared frame buffers (idempotent, first extractor wins)
    int err = vmaf_sycl_shared_frame_init(state, w, h, bpc);
    if (err) return err;

    unsigned half_w = (w + 1) / 2;
    unsigned half_h = (h + 1) / 2;
    s->buf_stride = (half_w + 3) & ~3u; // align to 4

    // Compute rfactors
    for (unsigned scale = 0; scale < 4; scale++) {
        float f1 = dwt_quant_step(scale, 1,
                                   s->adm_norm_view_dist,
                                   s->adm_ref_display_height);
        float f2 = dwt_quant_step(scale, 2,
                                   s->adm_norm_view_dist,
                                   s->adm_ref_display_height);
        s->rfactor[scale * 3 + 0] = 1.0f / f1;
        s->rfactor[scale * 3 + 1] = 1.0f / f1;
        s->rfactor[scale * 3 + 2] = 1.0f / f2;

        double pow2_32 = std::pow(2.0, 32);
        double pow2_21 = std::pow(2.0, 21);
        double pow2_23 = std::pow(2.0, 23);

        if (scale == 0) {
            double default_check = 3.0 * 1080;
            double actual = s->adm_norm_view_dist *
                           s->adm_ref_display_height;
            if (std::fabs(actual - default_check) < 1e-8) {
                s->i_rfactor[0] = 36453;
                s->i_rfactor[1] = 36453;
                s->i_rfactor[2] = 49417;
            } else {
                s->i_rfactor[0] = (uint32_t)(s->rfactor[0] * pow2_21);
                s->i_rfactor[1] = (uint32_t)(s->rfactor[1] * pow2_21);
                s->i_rfactor[2] = (uint32_t)(s->rfactor[2] * pow2_23);
            }
        } else {
            s->i_rfactor[scale * 3 + 0] =
                (uint32_t)(s->rfactor[scale * 3 + 0] * pow2_32);
            s->i_rfactor[scale * 3 + 1] =
                (uint32_t)(s->rfactor[scale * 3 + 1] * pow2_32);
            s->i_rfactor[scale * 3 + 2] =
                (uint32_t)(s->rfactor[scale * 3 + 2] * pow2_32);
        }
    }

    // Allocate device buffers
    size_t dwt_tmp_size = (size_t)w * 2 * half_h * sizeof(int32_t);
    s->d_dwt_tmp_ref = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, dwt_tmp_size));
    s->d_dwt_tmp_dis = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, dwt_tmp_size));

    size_t band_size = (size_t)s->buf_stride * half_h * sizeof(int32_t);
    for (int i = 0; i < 4; i++) {
        s->d_ref_band[i] = static_cast<int32_t *>(
            vmaf_sycl_malloc_device(state, band_size));
        s->d_dis_band[i] = static_cast<int32_t *>(
            vmaf_sycl_malloc_device(state, band_size));
    }

    for (int i = 0; i < 3; i++) {
        s->d_decouple_r[i] = static_cast<int32_t *>(
            vmaf_sycl_malloc_device(state, band_size));
        s->d_csf_a[i] = static_cast<int32_t *>(
            vmaf_sycl_malloc_device(state, band_size));
        s->d_csf_f[i] = static_cast<int32_t *>(
            vmaf_sycl_malloc_device(state, band_size));
    }

    // Division LUT: 65537 entries
    size_t div_size = 65537 * sizeof(int32_t);
    s->d_div_lookup = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, div_size));

    // Accumulators: 4 scales x 3 bands = 12 int64 each
    size_t accum_size = ADM_NUM_SCALES * ADM_NUM_BANDS * sizeof(int64_t);
    s->d_cm_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_device(state, accum_size));
    s->d_csf_den_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_device(state, accum_size));
    s->h_cm_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_host(state, accum_size));
    s->h_csf_den_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_host(state, accum_size));

    // Check critical allocations
    if (!s->d_dwt_tmp_ref || !s->d_dwt_tmp_dis ||
        !s->d_div_lookup || !s->d_cm_accum || !s->d_csf_den_accum ||
        !s->h_cm_accum || !s->h_csf_den_accum) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "adm_sycl: device memory allocation failed\n");
        return -ENOMEM;
    }

    // Generate and upload div_lookup
    {
        int32_t *lut = static_cast<int32_t *>(std::malloc(div_size));
        if (!lut) return -ENOMEM;
        std::memset(lut, 0, div_size);
        static const int32_t Q_factor = 1073741824; // 2^30
        for (int i = 1; i <= 32768; i++) {
            int32_t recip = (int32_t)(Q_factor / i);
            lut[32768 + i] = recip;
            lut[32768 - i] = -recip;
        }
        vmaf_sycl_memcpy_h2d(state, s->d_div_lookup, lut, div_size);
        std::free(lut);
    }

    s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
        fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    // Register with combined command graph
    err = vmaf_sycl_graph_register(state, enqueue_adm_work,
                                   adm_pre_graph, adm_post_graph,
                                   nullptr, s, "ADM");
    if (err) return err;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Enqueue all ADM compute work (used for both recording and direct)   */
/* ------------------------------------------------------------------ */

static void enqueue_adm_work_impl(sycl::queue &q, AdmStateSycl *s,
                                   void *shared_ref, void *shared_dis)
{
    // DWT shift parameters per scale
    struct DwtShifts {
        unsigned v_shift, v_add, h_shift, h_add;
    };
    DwtShifts dwt_shifts[4];
    dwt_shifts[0] = { s->bpc, 1u << (s->bpc - 1), 16u, 32768u };
    dwt_shifts[1] = { 0u, 0u, 15u, 16384u };
    dwt_shifts[2] = { 16u, 32768u, 16u, 32768u };
    dwt_shifts[3] = { 16u, 32768u, 15u, 16384u };

    unsigned cur_w = s->width;
    unsigned cur_h = s->height;
    unsigned cur_stride = s->buf_stride;

    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        unsigned half_w = (cur_w + 1) / 2;
        unsigned half_h = (cur_h + 1) / 2;

        // Input source: scale 0 reads from shared frame, others from LL band
        const void *ref_src = (scale == 0) ? shared_ref :
                              (const void *)s->d_ref_band[0];
        const void *dis_src = (scale == 0) ? shared_dis :
                              (const void *)s->d_dis_band[0];

        unsigned in_stride;
        if (scale == 0) {
            in_stride = (s->bpc <= 8) ? cur_w : cur_w * 2;
        } else {
            in_stride = cur_stride;
        }

        // DWT vertical pass: ref + dis fused
        launch_dwt_vert_pair(q,
                        ref_src, s->d_dwt_tmp_ref,
                        dis_src, s->d_dwt_tmp_dis,
                        scale, cur_w, cur_h, in_stride, s->bpc,
                        dwt_shifts[scale].v_shift,
                        dwt_shifts[scale].v_add);

        // DWT horizontal pass: ref + dis fused
        launch_dwt_hori_pair(q,
                        s->d_dwt_tmp_ref,
                        s->d_ref_band[0], s->d_ref_band[1],
                        s->d_ref_band[2], s->d_ref_band[3],
                        s->d_dwt_tmp_dis,
                        s->d_dis_band[0], s->d_dis_band[1],
                        s->d_dis_band[2], s->d_dis_band[3],
                        cur_w, cur_h, cur_stride,
                        dwt_shifts[scale].h_shift,
                        dwt_shifts[scale].h_add);

        // Decouple + CSF (always uses int64 Q31 gain limiting — no fp64 needed)
        launch_decouple_csf<false>(q,
                        scale, half_w, half_h, cur_stride,
                        s->adm_enhn_gain_limit,
                        s->i_rfactor[scale * 3 + 0],
                        s->i_rfactor[scale * 3 + 1],
                        s->i_rfactor[scale * 3 + 2],
                        s->d_ref_band[1], s->d_ref_band[2],
                        s->d_ref_band[3],
                        s->d_dis_band[1], s->d_dis_band[2],
                        s->d_dis_band[3],
                        s->d_decouple_r[0], s->d_decouple_r[1],
                        s->d_decouple_r[2],
                        s->d_csf_a[0], s->d_csf_a[1], s->d_csf_a[2],
                        s->d_csf_f[0], s->d_csf_f[1], s->d_csf_f[2],
                        s->d_div_lookup);

        // CSF denominator + Contrast measure: fused 3-band kernel
        launch_csf_den_cm_3band(q, scale, half_w, half_h, cur_stride,
                      // csf_den inputs
                      s->d_ref_band[1], s->d_ref_band[2],
                      s->d_ref_band[3],
                      s->d_csf_den_accum + scale * ADM_NUM_BANDS + 0,
                      s->d_csf_den_accum + scale * ADM_NUM_BANDS + 1,
                      s->d_csf_den_accum + scale * ADM_NUM_BANDS + 2,
                      // cm inputs
                      s->i_rfactor[scale * 3 + 0],
                      s->i_rfactor[scale * 3 + 1],
                      s->i_rfactor[scale * 3 + 2],
                      s->d_decouple_r[0], s->d_decouple_r[1],
                      s->d_decouple_r[2],
                      s->d_csf_f[0], s->d_csf_f[1], s->d_csf_f[2],
                      s->d_csf_a[0], s->d_csf_a[1], s->d_csf_a[2],
                      s->d_cm_accum + scale * ADM_NUM_BANDS + 0,
                      s->d_cm_accum + scale * ADM_NUM_BANDS + 1,
                      s->d_cm_accum + scale * ADM_NUM_BANDS + 2);

        // Next scale dimensions
        cur_w = half_w;
        cur_h = half_h;
    }
}

/* ------------------------------------------------------------------ */
/* C-compatible callbacks for combined command graph                   */
/* ------------------------------------------------------------------ */

// Pre-graph: zero accumulators (direct enqueue, outside graph)
static void adm_pre_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<AdmStateSycl *>(priv);
    size_t adm_accum_size = ADM_NUM_SCALES * ADM_NUM_BANDS * sizeof(int64_t);
    q.memset(s->d_cm_accum, 0, adm_accum_size);
    q.memset(s->d_csf_den_accum, 0, adm_accum_size);
}

// Graph-recorded: compute kernels only
static void enqueue_adm_work(void *queue_ptr, void *priv,
                              void *shared_ref, void *shared_dis)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<AdmStateSycl *>(priv);
    enqueue_adm_work_impl(q, s, shared_ref, shared_dis);
}

// Post-graph: D2H accumulator download (direct enqueue, outside graph)
static void adm_post_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<AdmStateSycl *>(priv);
    size_t accum_size = ADM_NUM_SCALES * ADM_NUM_BANDS * sizeof(int64_t);
    q.memcpy(s->h_cm_accum, s->d_cm_accum, accum_size);
    q.memcpy(s->h_csf_den_accum, s->d_csf_den_accum, accum_size);
}

/* ------------------------------------------------------------------ */
/* Submit / Collect / Extract                                          */
/* ------------------------------------------------------------------ */

static int submit_fex_sycl(VmafFeatureExtractor *fex,
                            VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                            VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                            unsigned index)
{
    (void)ref_pic; (void)ref_pic_90;
    (void)dist_pic; (void)dist_pic_90;

    auto *s = static_cast<AdmStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph submit (idempotent per frame — first extractor wins)
    int err = vmaf_sycl_graph_submit(state);
    if (err) return err;

    s->pending_index = index;
    s->has_pending = true;

    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex,
                             unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<AdmStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph wait (idempotent per frame — first extractor wins)
    vmaf_sycl_graph_wait(state);

    // Read back accumulators
    int64_t cm_results[ADM_NUM_SCALES][ADM_NUM_BANDS];
    int64_t csf_den_results[ADM_NUM_SCALES][ADM_NUM_BANDS];
    std::memcpy(cm_results, s->h_cm_accum, sizeof(cm_results));
    std::memcpy(csf_den_results, s->h_csf_den_accum,
                sizeof(csf_den_results));

    // Compute scores
    double num = 0.0, den = 0.0;
    double scores_num[ADM_NUM_SCALES], scores_den[ADM_NUM_SCALES];
    unsigned score_w = s->width, score_h = s->height;

    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        score_w = (score_w + 1) / 2;
        score_h = (score_h + 1) / 2;

        float num_scale, den_scale;
        conclude_adm_cm(cm_results[scale], score_h, score_w,
                        scale, &num_scale);
        conclude_adm_csf_den((uint64_t *)csf_den_results[scale],
                             score_h, score_w, scale, &den_scale,
                             (float)s->adm_norm_view_dist,
                             (float)s->adm_ref_display_height);

        num += num_scale;
        den += den_scale;
        scores_num[scale] = num_scale;
        scores_den[scale] = den_scale;
    }

    // Apply numden_limit
    double numden_limit = 1e-10 * (score_w * score_h) / (1920.0 * 1080.0);
    if (num < numden_limit) num = 0.0;
    if (den < numden_limit) den = 0.0;

    double score = (den == 0.0) ? 1.0 : num / den;

    // Write primary feature
    {
        int err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "VMAF_integer_feature_adm2_score", score, index);
        if (err) return err;
    }

    // Per-scale features
    for (int i = 0; i < ADM_NUM_SCALES; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "integer_adm_scale%d", i);
        double scale_score = (scores_den[i] == 0.0) ?
                             1.0 : scores_num[i] / scores_den[i];
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, name, scale_score, index);
    }

    // Debug features
    if (s->debug) {
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm", score, index);
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num", num, index);
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den", den, index);

        for (int i = 0; i < ADM_NUM_SCALES; i++) {
            char name[64];
            std::snprintf(name, sizeof(name),
                          "integer_adm_num_scale%d", i);
            vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict, name, scores_num[i], index);
            std::snprintf(name, sizeof(name),
                          "integer_adm_den_scale%d", i);
            vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict, name, scores_den[i], index);
        }
    }

    s->has_pending = false;
    return 0;
}

static int extract_fex_sycl(VmafFeatureExtractor *fex,
                              VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                              VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                              unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    int err = submit_fex_sycl(fex, ref_pic, ref_pic_90,
                               dist_pic, dist_pic_90, index);
    if (err) return err;
    return collect_fex_sycl(fex, index, feature_collector);
}

static int flush_fex_sycl(VmafFeatureExtractor *fex,
                            VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    if (!fex) return -EINVAL;
    VmafSyclState *state = fex->sycl_state;
    if (state) vmaf_sycl_queue_wait(state);
    return 1; // done — collect already consumed pending work
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<AdmStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    if (state) {
        vmaf_sycl_queue_wait(state);

        if (s->d_dwt_tmp_ref) vmaf_sycl_free(state, s->d_dwt_tmp_ref);
        if (s->d_dwt_tmp_dis) vmaf_sycl_free(state, s->d_dwt_tmp_dis);

        for (int i = 0; i < 4; i++) {
            if (s->d_ref_band[i]) vmaf_sycl_free(state, s->d_ref_band[i]);
            if (s->d_dis_band[i]) vmaf_sycl_free(state, s->d_dis_band[i]);
        }
        for (int i = 0; i < 3; i++) {
            if (s->d_decouple_r[i])
                vmaf_sycl_free(state, s->d_decouple_r[i]);
            if (s->d_csf_a[i]) vmaf_sycl_free(state, s->d_csf_a[i]);
            if (s->d_csf_f[i]) vmaf_sycl_free(state, s->d_csf_f[i]);
        }

        if (s->d_div_lookup) vmaf_sycl_free(state, s->d_div_lookup);
        if (s->d_cm_accum) vmaf_sycl_free(state, s->d_cm_accum);
        if (s->d_csf_den_accum)
            vmaf_sycl_free(state, s->d_csf_den_accum);
        if (s->h_cm_accum) vmaf_sycl_free(state, s->h_cm_accum);
        if (s->h_csf_den_accum)
            vmaf_sycl_free(state, s->h_csf_den_accum);
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Feature extractor definition                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {
    "VMAF_integer_feature_adm2_score",
    "integer_adm_scale0", "integer_adm_scale1",
    "integer_adm_scale2", "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0",
    "integer_adm_num_scale1", "integer_adm_den_scale1",
    "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL
};

extern "C"
VmafFeatureExtractor vmaf_fex_integer_adm_sycl = {
    .name = "adm_sycl",
    .init = init_fex_sycl,
    .extract = extract_fex_sycl,
    .flush = flush_fex_sycl,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options,
    .priv_size = sizeof(AdmStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features,
};
