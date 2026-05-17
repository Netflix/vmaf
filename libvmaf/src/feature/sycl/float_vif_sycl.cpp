/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature kernel on the SYCL backend (T7-23 / batch 3
 *  part 5c — ADR-0192 / ADR-0197). SYCL twin of float_vif_vulkan +
 *  float_vif_cuda.
 *
 *  v1: kernelscale=1.0 only. CPU's VIF_OPT_HANDLE_BORDERS branch:
 *  per-scale dims = prev/2 (no border crop); decimate samples at
 *  (2*gx, 2*gy) with mirror padding on input filter taps.
 *
 *  Per-frame flow: 4 compute + 3 decimate launches. Self-contained
 *  submit/collect — does NOT register with vmaf_sycl_graph_register
 *  (the multi-scale layout doesn't fit the shared_frame model).
 */

#include <sycl/sycl.hpp>

#include "sycl_compat.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "sycl/common.h"
}

namespace
{

constexpr int FVIF_BX = 16;
constexpr int FVIF_BY = 16;
constexpr int FVIF_MAX_FW = 17;
constexpr int FVIF_MAX_HFW = 8;

struct FloatVifStateSycl {
    bool debug;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[4];
    unsigned scale_h[4];

    VmafSyclState *sycl_state;

    /* Pinned host raw uploads. */
    void *h_ref_raw;
    void *h_dis_raw;
    /* Device raw + ping-pong float buffers. */
    void *d_ref_raw;
    void *d_dis_raw;
    float *d_ref_buf[2];
    float *d_dis_buf[2];

    /* Per-scale (num, den) partials. */
    float *d_num[4];
    float *d_den[4];
    float *h_num[4];
    float *h_den[4];
    unsigned wg_count[4];

    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

static const float FVIF_COEFF_S0[FVIF_MAX_FW] = {
    0.00745626912f, 0.0142655009f, 0.0250313189f, 0.0402820669f, 0.0594526194f, 0.0804751068f,
    0.0999041125f,  0.113746084f,  0.118773937f,  0.113746084f,  0.0999041125f, 0.0804751068f,
    0.0594526194f,  0.0402820669f, 0.0250313189f, 0.0142655009f, 0.00745626912f};
static const float FVIF_COEFF_S1[FVIF_MAX_FW] = {
    0.0189780835f, 0.0558981746f, 0.120920904f,  0.192116052f, 0.224173605f, 0.192116052f,
    0.120920904f,  0.0558981746f, 0.0189780835f, 0.0f,         0.0f,         0.0f,
    0.0f,          0.0f,          0.0f,          0.0f,         0.0f};
static const float FVIF_COEFF_S2[FVIF_MAX_FW] = {
    0.054488685f, 0.244201347f, 0.402619958f, 0.244201347f, 0.054488685f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,         0.0f,         0.0f,         0.0f,         0.0f,         0.0f, 0.0f, 0.0f};
static const float FVIF_COEFF_S3[FVIF_MAX_FW] = {
    0.166378498f, 0.667243004f, 0.166378498f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,         0.0f,         0.0f,         0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

static inline int fvif_fw_for(int scale)
{
    static const int fw[4] = {17, 9, 5, 3};
    return fw[scale];
}

template <int SCALE>
static sycl::event launch_compute(sycl::queue &q, const void *ref_raw, const void *dis_raw,
                                  unsigned raw_stride_bytes, const float *ref_f, const float *dis_f,
                                  unsigned f_stride_floats, float *num_partials,
                                  float *den_partials, unsigned width, unsigned height,
                                  unsigned bpc, unsigned grid_x_count)
{
    constexpr int FW = (SCALE == 0) ? 17 : (SCALE == 1) ? 9 : (SCALE == 2) ? 5 : 3;
    constexpr int HFW = FW / 2;
    constexpr int MAX_TILE_W = FVIF_BX + 2 * FVIF_MAX_HFW;
    const float *coeff = (SCALE == 0) ? FVIF_COEFF_S0 :
                         (SCALE == 1) ? FVIF_COEFF_S1 :
                         (SCALE == 2) ? FVIF_COEFF_S2 :
                                        FVIF_COEFF_S3;

    const size_t global_x = ((width + FVIF_BX - 1) / FVIF_BX) * FVIF_BX;
    const size_t global_y = ((height + FVIF_BY - 1) / FVIF_BY) * FVIF_BY;
    const unsigned e_w = width;
    const unsigned e_h = height;
    const unsigned e_bpc = bpc;
    const unsigned e_raw_stride = raw_stride_bytes;
    const unsigned e_f_stride = f_stride_floats;
    const unsigned e_grid_x = grid_x_count;
    const void *e_ref_raw = ref_raw;
    const void *e_dis_raw = dis_raw;
    const float *e_ref_f = ref_f;
    const float *e_dis_f = dis_f;
    float *e_num = num_partials;
    float *e_den = den_partials;

    /* Local copy of filter coeffs into a plain array for kernel
     * capture (sycl can't capture pointers to host-only static data). */
    float local_coeff[FVIF_MAX_FW];
    for (int i = 0; i < FVIF_MAX_FW; i++)
        local_coeff[i] = coeff[i];

    return q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> s_ref(sycl::range<1>(MAX_TILE_W * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_dis(sycl::range<1>(MAX_TILE_W * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_v_mu1(sycl::range<1>(FVIF_BY * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_v_mu2(sycl::range<1>(FVIF_BY * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_v_xx(sycl::range<1>(FVIF_BY * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_v_yy(sycl::range<1>(FVIF_BY * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_v_xy(sycl::range<1>(FVIF_BY * MAX_TILE_W), cgh);
        sycl::local_accessor<float, 1> s_num_warps(sycl::range<1>(FVIF_BX * FVIF_BY / 32), cgh);
        sycl::local_accessor<float, 1> s_den_warps(sycl::range<1>(FVIF_BX * FVIF_BY / 32), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_y, global_x), sycl::range<2>(FVIF_BY, FVIF_BX)),
            [=](sycl::nd_item<2> item) VMAF_SYCL_REQD_SG_SIZE(32) {
                const int gx = (int)item.get_global_id(1);
                const int gy = (int)item.get_global_id(0);
                const int lx = (int)item.get_local_id(1);
                const int ly = (int)item.get_local_id(0);
                const unsigned lid = (unsigned)(ly * FVIF_BX + lx);
                const bool valid = (gx < (int)e_w && gy < (int)e_h);
                const int tile_w = FVIF_BX + 2 * HFW;
                const int tile_h = FVIF_BY + 2 * HFW;
                const int tile_oy = (int)(item.get_group(0) * FVIF_BY) - HFW;
                const int tile_ox = (int)(item.get_group(1) * FVIF_BX) - HFW;
                const int tile_elems = tile_h * tile_w;
                const bool is_raw = (SCALE == 0);

                auto mirror_v = [](int idx, int sup) -> int {
                    if (idx < 0)
                        return -idx;
                    if (idx >= sup)
                        return 2 * sup - idx - 2;
                    return idx;
                };
                auto mirror_h = [](int idx, int sup) -> int {
                    if (idx < 0)
                        return -idx;
                    if (idx >= sup)
                        return 2 * sup - idx - 2; /* AVX2 border-path mirror */
                    return idx;
                };

                auto read_raw = [&](const void *plane, int y, int x) -> float {
                    if (e_bpc <= 8u) {
                        return (float)static_cast<const uint8_t *>(plane)[y * e_raw_stride + x] -
                               128.0f;
                    }
                    const uint16_t v = reinterpret_cast<const uint16_t *>(
                        static_cast<const uint8_t *>(plane) + y * e_raw_stride)[x];
                    float scaler = 1.0f;
                    if (e_bpc == 10u)
                        scaler = 4.0f;
                    else if (e_bpc == 12u)
                        scaler = 16.0f;
                    else if (e_bpc == 16u)
                        scaler = 256.0f;
                    return (float)v / scaler - 128.0f;
                };

                /* Phase 1: tile load. */
                for (int i = (int)lid; i < tile_elems; i += FVIF_BX * FVIF_BY) {
                    const int tr = i / tile_w;
                    const int tc = i - tr * tile_w;
                    const int py = mirror_v(tile_oy + tr, (int)e_h);
                    const int px = mirror_h(tile_ox + tc, (int)e_w);
                    float r, d;
                    if (is_raw) {
                        r = read_raw(e_ref_raw, py, px);
                        d = read_raw(e_dis_raw, py, px);
                    } else {
                        r = e_ref_f[py * e_f_stride + px];
                        d = e_dis_f[py * e_f_stride + px];
                    }
                    s_ref[tr * MAX_TILE_W + tc] = r;
                    s_dis[tr * MAX_TILE_W + tc] = d;
                }
                item.barrier(sycl::access::fence_space::local_space);

                /* Phase 2: vertical filter for WG_Y rows × tile_w cols. */
                const int vert_total = FVIF_BY * tile_w;
                for (int i = (int)lid; i < vert_total; i += FVIF_BX * FVIF_BY) {
                    const int r = i / tile_w;
                    const int c = i - r * tile_w;
                    float a_mu1 = 0.0f, a_mu2 = 0.0f, a_xx = 0.0f, a_yy = 0.0f, a_xy = 0.0f;
#pragma unroll
                    for (int k = 0; k < FW; k++) {
                        const float c_k = local_coeff[k];
                        const float ref_v = s_ref[(r + k) * MAX_TILE_W + c];
                        const float dis_v = s_dis[(r + k) * MAX_TILE_W + c];
                        a_mu1 += c_k * ref_v;
                        a_mu2 += c_k * dis_v;
                        a_xx += c_k * (ref_v * ref_v);
                        a_yy += c_k * (dis_v * dis_v);
                        a_xy += c_k * (ref_v * dis_v);
                    }
                    s_v_mu1[r * MAX_TILE_W + c] = a_mu1;
                    s_v_mu2[r * MAX_TILE_W + c] = a_mu2;
                    s_v_xx[r * MAX_TILE_W + c] = a_xx;
                    s_v_yy[r * MAX_TILE_W + c] = a_yy;
                    s_v_xy[r * MAX_TILE_W + c] = a_xy;
                }
                item.barrier(sycl::access::fence_space::local_space);

                /* Phase 3: horizontal filter + vif_stat. */
                float my_num = 0.0f, my_den = 0.0f;
                if (valid) {
                    float mu1 = 0.0f, mu2 = 0.0f, xx = 0.0f, yy = 0.0f, xy = 0.0f;
#pragma unroll
                    for (int k = 0; k < FW; k++) {
                        const float c_k = local_coeff[k];
                        mu1 += c_k * s_v_mu1[ly * MAX_TILE_W + (lx + k)];
                        mu2 += c_k * s_v_mu2[ly * MAX_TILE_W + (lx + k)];
                        xx += c_k * s_v_xx[ly * MAX_TILE_W + (lx + k)];
                        yy += c_k * s_v_yy[ly * MAX_TILE_W + (lx + k)];
                        xy += c_k * s_v_xy[ly * MAX_TILE_W + (lx + k)];
                    }
                    const float eps = 1.0e-10f;
                    const float vif_sigma_nsq = 2.0f;
                    const float vif_egl = 100.0f;
                    const float sigma_max_inv = (2.0f * 2.0f) / (255.0f * 255.0f);

                    float sigma1_sq = xx - mu1 * mu1;
                    float sigma2_sq = yy - mu2 * mu2;
                    float sigma12 = xy - mu1 * mu2;
                    sigma1_sq = sycl::fmax(sigma1_sq, 0.0f);
                    sigma2_sq = sycl::fmax(sigma2_sq, 0.0f);
                    float g = sigma12 / (sigma1_sq + eps);
                    float sv_sq = sigma2_sq - g * sigma12;
                    if (sigma1_sq < eps) {
                        g = 0.0f;
                        sv_sq = sigma2_sq;
                        sigma1_sq = 0.0f;
                    }
                    if (sigma2_sq < eps) {
                        g = 0.0f;
                        sv_sq = 0.0f;
                    }
                    if (g < 0.0f) {
                        sv_sq = sigma2_sq;
                        g = 0.0f;
                    }
                    sv_sq = sycl::fmax(sv_sq, eps);
                    g = sycl::fmin(g, vif_egl);

                    float num_val =
                        sycl::log2(1.0f + (g * g * sigma1_sq) / (sv_sq + vif_sigma_nsq));
                    float den_val = sycl::log2(1.0f + sigma1_sq / vif_sigma_nsq);
                    if (sigma12 < 0.0f)
                        num_val = 0.0f;
                    if (sigma1_sq < vif_sigma_nsq) {
                        num_val = 1.0f - sigma2_sq * sigma_max_inv;
                        den_val = 1.0f;
                    }
                    my_num = num_val;
                    my_den = den_val;
                }

                /* Phase 4: subgroup + cross-subgroup reduction. */
                sycl::sub_group sg = item.get_sub_group();
                const float wn = sycl::reduce_over_group(sg, my_num, sycl::plus<float>{});
                const float wd = sycl::reduce_over_group(sg, my_den, sycl::plus<float>{});
                const uint32_t sg_id = sg.get_group_linear_id();
                const uint32_t sg_lid = sg.get_local_linear_id();
                const uint32_t n_sg = sg.get_group_linear_range();
                if (sg_lid == 0) {
                    s_num_warps[sg_id] = wn;
                    s_den_warps[sg_id] = wd;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    float total_n = 0.0f, total_d = 0.0f;
                    for (uint32_t i = 0; i < n_sg; i++) {
                        total_n += s_num_warps[i];
                        total_d += s_den_warps[i];
                    }
                    const size_t wg_idx = item.get_group(0) * e_grid_x + item.get_group(1);
                    e_num[wg_idx] = total_n;
                    e_den[wg_idx] = total_d;
                }
            });
    });
}

template <int SCALE>
static sycl::event launch_decimate(sycl::queue &q, const void *ref_raw, const void *dis_raw,
                                   unsigned raw_stride_bytes, const float *ref_f,
                                   const float *dis_f, unsigned f_stride_floats, float *ref_out,
                                   float *dis_out, unsigned out_stride_floats, unsigned out_w,
                                   unsigned out_h, unsigned in_w, unsigned in_h, unsigned bpc)
{
    constexpr int FW = (SCALE == 1) ? 9 : (SCALE == 2) ? 5 : 3;
    constexpr int HFW = FW / 2;
    const float *coeff = (SCALE == 1) ? FVIF_COEFF_S1 :
                         (SCALE == 2) ? FVIF_COEFF_S2 :
                                        FVIF_COEFF_S3;

    const size_t global_x = ((out_w + FVIF_BX - 1) / FVIF_BX) * FVIF_BX;
    const size_t global_y = ((out_h + FVIF_BY - 1) / FVIF_BY) * FVIF_BY;
    const unsigned e_out_w = out_w;
    const unsigned e_out_h = out_h;
    const unsigned e_in_w = in_w;
    const unsigned e_in_h = in_h;
    const unsigned e_bpc = bpc;
    const unsigned e_raw_stride = raw_stride_bytes;
    const unsigned e_f_stride = f_stride_floats;
    const unsigned e_out_stride = out_stride_floats;
    const void *e_ref_raw = ref_raw;
    const void *e_dis_raw = dis_raw;
    const float *e_ref_f = ref_f;
    const float *e_dis_f = dis_f;
    float *e_ref_out = ref_out;
    float *e_dis_out = dis_out;

    float local_coeff[FVIF_MAX_FW];
    for (int i = 0; i < FVIF_MAX_FW; i++)
        local_coeff[i] = coeff[i];

    return q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_y, global_x), sycl::range<2>(FVIF_BY, FVIF_BX)),
            [=](sycl::nd_item<2> item) {
                const int gx = (int)item.get_global_id(1);
                const int gy = (int)item.get_global_id(0);
                if (gx >= (int)e_out_w || gy >= (int)e_out_h)
                    return;
                const int in_x = 2 * gx;
                const int in_y = 2 * gy;
                const bool is_raw = (SCALE == 1);

                auto mirror_v = [](int idx, int sup) -> int {
                    if (idx < 0)
                        return -idx;
                    if (idx >= sup)
                        return 2 * sup - idx - 2;
                    return idx;
                };
                auto mirror_h = [](int idx, int sup) -> int {
                    if (idx < 0)
                        return -idx;
                    if (idx >= sup)
                        return 2 * sup - idx - 2; /* AVX2 border-path mirror */
                    return idx;
                };
                auto read_raw = [&](const void *plane, int y, int x) -> float {
                    if (e_bpc <= 8u) {
                        return (float)static_cast<const uint8_t *>(plane)[y * e_raw_stride + x] -
                               128.0f;
                    }
                    const uint16_t v = reinterpret_cast<const uint16_t *>(
                        static_cast<const uint8_t *>(plane) + y * e_raw_stride)[x];
                    float scaler = 1.0f;
                    if (e_bpc == 10u)
                        scaler = 4.0f;
                    else if (e_bpc == 12u)
                        scaler = 16.0f;
                    else if (e_bpc == 16u)
                        scaler = 256.0f;
                    return (float)v / scaler - 128.0f;
                };

                float acc_ref = 0.0f, acc_dis = 0.0f;
#pragma unroll
                for (int kj = 0; kj < FW; kj++) {
                    const float c_j = local_coeff[kj];
                    const int px = mirror_h(in_x - HFW + kj, (int)e_in_w);
                    float v_ref = 0.0f, v_dis = 0.0f;
#pragma unroll
                    for (int ki = 0; ki < FW; ki++) {
                        const float c_i = local_coeff[ki];
                        const int py = mirror_v(in_y - HFW + ki, (int)e_in_h);
                        float r, d;
                        if (is_raw) {
                            r = read_raw(e_ref_raw, py, px);
                            d = read_raw(e_dis_raw, py, px);
                        } else {
                            r = e_ref_f[py * e_f_stride + px];
                            d = e_dis_f[py * e_f_stride + px];
                        }
                        v_ref += c_i * r;
                        v_dis += c_i * d;
                    }
                    acc_ref += c_j * v_ref;
                    acc_dis += c_j * v_dis;
                }
                e_ref_out[gy * e_out_stride + gx] = acc_ref;
                e_dis_out[gy * e_out_stride + gx] = acc_dis;
            });
    });
}

template <typename T>
static void copy_y_plane(const VmafPicture *pic, void *dst, unsigned w, unsigned h)
{
    const T *src = static_cast<const T *>(pic->data[0]);
    T *out = static_cast<T *>(dst);
    const ptrdiff_t src_stride_t = pic->stride[0] / static_cast<ptrdiff_t>(sizeof(T));
    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++)
            out[j] = src[j];
        src += src_stride_t;
        out += w;
    }
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_float_vif_sycl[] = {
    {.name = "debug",
     .help = "debug mode",
     .offset = offsetof(FloatVifStateSycl, debug),
     .type = VMAF_OPT_TYPE_BOOL,
     .default_val = {.b = false}},
    {.name = "vif_enhn_gain_limit",
     .alias = "egl",
     .help = "enhancement gain (>=1.0)",
     .offset = offsetof(FloatVifStateSycl, vif_enhn_gain_limit),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val = {.d = 100.0},
     .min = 1.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "vif_kernelscale",
     .help = "kernel scale",
     .offset = offsetof(FloatVifStateSycl, vif_kernelscale),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val = {.d = 1.0},
     .min = 0.1,
     .max = 4.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "vif_sigma_nsq",
     .alias = "snsq",
     .help = "neural noise variance",
     .offset = offsetof(FloatVifStateSycl, vif_sigma_nsq),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val = {.d = 2.0},
     .min = 0.0,
     .max = 5.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {0}};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<FloatVifStateSycl *>(fex->priv);
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->has_pending = false;

    if (s->vif_kernelscale != 1.0)
        return -EINVAL;

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < 4; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }

    if (!fex->sycl_state)
        return -EINVAL;
    s->sycl_state = fex->sycl_state;

    const size_t bpp = (bpc <= 8) ? 1u : 2u;
    const size_t raw_bytes = (size_t)w * h * bpp;
    s->h_ref_raw = vmaf_sycl_malloc_host(s->sycl_state, raw_bytes);
    s->h_dis_raw = vmaf_sycl_malloc_host(s->sycl_state, raw_bytes);
    s->d_ref_raw = vmaf_sycl_malloc_device(s->sycl_state, raw_bytes);
    s->d_dis_raw = vmaf_sycl_malloc_device(s->sycl_state, raw_bytes);

    const size_t fbytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
    s->d_ref_buf[0] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, fbytes));
    s->d_dis_buf[0] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, fbytes));
    s->d_ref_buf[1] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, fbytes));
    s->d_dis_buf[1] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, fbytes));

    for (int i = 0; i < 4; i++) {
        const unsigned gx = (s->scale_w[i] + FVIF_BX - 1u) / FVIF_BX;
        const unsigned gy = (s->scale_h[i] + FVIF_BY - 1u) / FVIF_BY;
        s->wg_count[i] = gx * gy;
        const size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        s->d_num[i] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, pbytes));
        s->d_den[i] = static_cast<float *>(vmaf_sycl_malloc_device(s->sycl_state, pbytes));
        s->h_num[i] = static_cast<float *>(vmaf_sycl_malloc_host(s->sycl_state, pbytes));
        s->h_den[i] = static_cast<float *>(vmaf_sycl_malloc_host(s->sycl_state, pbytes));
    }

    if (!s->h_ref_raw || !s->h_dis_raw || !s->d_ref_raw || !s->d_dis_raw || !s->d_ref_buf[0] ||
        !s->d_dis_buf[0] || !s->d_ref_buf[1] || !s->d_dis_buf[1])
        return -ENOMEM;
    for (int i = 0; i < 4; i++) {
        if (!s->d_num[i] || !s->d_den[i] || !s->h_num[i] || !s->h_den[i])
            return -ENOMEM;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    auto *s = static_cast<FloatVifStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    const size_t bpp = (s->bpc <= 8) ? 1u : 2u;
    const unsigned raw_stride_bytes = (unsigned)(s->width * bpp);
    if (s->bpc <= 8) {
        copy_y_plane<uint8_t>(ref_pic, s->h_ref_raw, s->width, s->height);
        copy_y_plane<uint8_t>(dist_pic, s->h_dis_raw, s->width, s->height);
    } else {
        copy_y_plane<uint16_t>(ref_pic, s->h_ref_raw, s->width, s->height);
        copy_y_plane<uint16_t>(dist_pic, s->h_dis_raw, s->width, s->height);
    }
    const size_t raw_bytes = (size_t)s->width * s->height * bpp;
    q.memcpy(s->d_ref_raw, s->h_ref_raw, raw_bytes);
    q.memcpy(s->d_dis_raw, s->h_dis_raw, raw_bytes);

    /* Reset partials. */
    for (int i = 0; i < 4; i++) {
        q.memset(s->d_num[i], 0, (size_t)s->wg_count[i] * sizeof(float));
        q.memset(s->d_den[i], 0, (size_t)s->wg_count[i] * sizeof(float));
    }

    /* Scale 0 compute. */
    {
        const unsigned grid_x = (s->scale_w[0] + FVIF_BX - 1u) / FVIF_BX;
        launch_compute<0>(q, s->d_ref_raw, s->d_dis_raw, raw_stride_bytes, nullptr, nullptr,
                          s->scale_w[0], s->d_num[0], s->d_den[0], s->scale_w[0], s->scale_h[0],
                          s->bpc, grid_x);
    }

    /* Scales 1, 2, 3: decimate then compute. */
    for (int n = 1; n < 4; n++) {
        const unsigned dst_idx = (n - 1) % 2u;
        const bool prev_is_raw = (n == 1);
        const float *ref_in_f =
            prev_is_raw ? nullptr : (dst_idx == 0 ? s->d_ref_buf[1] : s->d_ref_buf[0]);
        const float *dis_in_f =
            prev_is_raw ? nullptr : (dst_idx == 0 ? s->d_dis_buf[1] : s->d_dis_buf[0]);
        const unsigned in_f_stride = prev_is_raw ? 0 : s->scale_w[n - 1];
        float *ref_out = (dst_idx == 0) ? s->d_ref_buf[0] : s->d_ref_buf[1];
        float *dis_out = (dst_idx == 0) ? s->d_dis_buf[0] : s->d_dis_buf[1];
        const unsigned out_stride = s->scale_w[n];
        if (n == 1)
            launch_decimate<1>(q, s->d_ref_raw, s->d_dis_raw, raw_stride_bytes, ref_in_f, dis_in_f,
                               in_f_stride, ref_out, dis_out, out_stride, s->scale_w[n],
                               s->scale_h[n], s->scale_w[n - 1], s->scale_h[n - 1], s->bpc);
        else if (n == 2)
            launch_decimate<2>(q, s->d_ref_raw, s->d_dis_raw, raw_stride_bytes, ref_in_f, dis_in_f,
                               in_f_stride, ref_out, dis_out, out_stride, s->scale_w[n],
                               s->scale_h[n], s->scale_w[n - 1], s->scale_h[n - 1], s->bpc);
        else
            launch_decimate<3>(q, s->d_ref_raw, s->d_dis_raw, raw_stride_bytes, ref_in_f, dis_in_f,
                               in_f_stride, ref_out, dis_out, out_stride, s->scale_w[n],
                               s->scale_h[n], s->scale_w[n - 1], s->scale_h[n - 1], s->bpc);

        const unsigned grid_x = (s->scale_w[n] + FVIF_BX - 1u) / FVIF_BX;
        if (n == 1)
            launch_compute<1>(q, nullptr, nullptr, 0, ref_out, dis_out, s->scale_w[n], s->d_num[n],
                              s->d_den[n], s->scale_w[n], s->scale_h[n], s->bpc, grid_x);
        else if (n == 2)
            launch_compute<2>(q, nullptr, nullptr, 0, ref_out, dis_out, s->scale_w[n], s->d_num[n],
                              s->d_den[n], s->scale_w[n], s->scale_h[n], s->bpc, grid_x);
        else
            launch_compute<3>(q, nullptr, nullptr, 0, ref_out, dis_out, s->scale_w[n], s->d_num[n],
                              s->d_den[n], s->scale_w[n], s->scale_h[n], s->bpc, grid_x);
    }

    for (int i = 0; i < 4; i++) {
        q.memcpy(s->h_num[i], s->d_num[i], (size_t)s->wg_count[i] * sizeof(float));
        q.memcpy(s->h_den[i], s->d_den[i], (size_t)s->wg_count[i] * sizeof(float));
    }

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<FloatVifStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    qptr->wait();

    double scores[8];
    for (int i = 0; i < 4; i++) {
        double n = 0.0, d = 0.0;
        for (unsigned j = 0; j < s->wg_count[i]; j++) {
            n += (double)s->h_num[i][j];
            d += (double)s->h_den[i][j];
        }
        scores[2 * i + 0] = n;
        scores[2 * i + 1] = d;
    }

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale0_score",
                                                   scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale1_score",
                                                   scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale2_score",
                                                   scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale3_score",
                                                   scores[6] / scores[7], index);

    if (s->debug && !err) {
        double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        double score = score_den == 0.0 ? 1.0 : score_num / score_den;
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif", score, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_den", score_den, index);
        const char *names[8] = {"vif_num_scale0", "vif_den_scale0", "vif_num_scale1",
                                "vif_den_scale1", "vif_num_scale2", "vif_den_scale2",
                                "vif_num_scale3", "vif_den_scale3"};
        for (int i = 0; i < 8; i++) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                           names[i], scores[i], index);
        }
    }
    return err;
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<FloatVifStateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->h_ref_raw)
            vmaf_sycl_free(s->sycl_state, s->h_ref_raw);
        if (s->h_dis_raw)
            vmaf_sycl_free(s->sycl_state, s->h_dis_raw);
        if (s->d_ref_raw)
            vmaf_sycl_free(s->sycl_state, s->d_ref_raw);
        if (s->d_dis_raw)
            vmaf_sycl_free(s->sycl_state, s->d_dis_raw);
        for (int i = 0; i < 2; i++) {
            if (s->d_ref_buf[i])
                vmaf_sycl_free(s->sycl_state, s->d_ref_buf[i]);
            if (s->d_dis_buf[i])
                vmaf_sycl_free(s->sycl_state, s->d_dis_buf[i]);
        }
        for (int i = 0; i < 4; i++) {
            if (s->d_num[i])
                vmaf_sycl_free(s->sycl_state, s->d_num[i]);
            if (s->d_den[i])
                vmaf_sycl_free(s->sycl_state, s->d_den[i]);
            if (s->h_num[i])
                vmaf_sycl_free(s->sycl_state, s->h_num[i]);
            if (s->h_den[i])
                vmaf_sycl_free(s->sycl_state, s->h_den[i]);
        }
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_float_vif_sycl[] = {"VMAF_feature_vif_scale0_score",
                                                         "VMAF_feature_vif_scale1_score",
                                                         "VMAF_feature_vif_scale2_score",
                                                         "VMAF_feature_vif_scale3_score",
                                                         "vif",
                                                         "vif_num",
                                                         "vif_den",
                                                         "vif_num_scale0",
                                                         "vif_den_scale0",
                                                         "vif_num_scale1",
                                                         "vif_den_scale1",
                                                         "vif_num_scale2",
                                                         "vif_den_scale2",
                                                         "vif_num_scale3",
                                                         "vif_den_scale3",
                                                         NULL};

extern "C" VmafFeatureExtractor vmaf_fex_float_vif_sycl = {
    .name = "float_vif_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_float_vif_sycl,
    .priv_size = sizeof(FloatVifStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_float_vif_sycl,
};

} /* extern "C" */
