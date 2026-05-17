/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP compute kernels for the float_adm feature extractor
 *  (T7-10b batch-2 / ADR-0468). Direct port of the CUDA twin at
 *  `libvmaf/src/feature/cuda/float_adm/float_adm_score.cu`.
 *
 *  Stages (selected via a runtime kernel argument):
 *    0 — DWT vertical pass (ref+dis fused, z-dim)
 *    1 — DWT horizontal pass (ref+dis fused) → 4 sub-bands
 *    2 — Decouple + CSF (writes csf_a + csf_f for stage 3)
 *    3 — CSF denominator + CM fused; emits 6 float partials per WG
 *
 *  Per-frame flow: 16 launches (4 stages × 4 scales). Launches
 *  serialise on the picture stream; D2H copy on the secondary stream
 *  behind an event fence. Reduction across WGs runs on the host in
 *  double precision — same trick as the Vulkan host wrapper, matches
 *  CPU `adm_csf_den_scale_s` / `adm_cm_s` row-by-row order to hold
 *  the places=4 contract.
 *
 *  HIP adaptation notes vs. the CUDA twin:
 *  - Warp size is 64 on GCN/RDNA. `__shfl_down` used without mask
 *    (HIP convention; CUDA uses `__shfl_down_sync(0xffffffff, ...)`).
 *  - Shared-memory warp-partial slots are sized for 64-lane warps
 *    (FADM_BX * FADM_BY / FADM_WARP_SIZE = 256/64 = 4 slots).
 *  - The `precise{}` decouple semantics (CUDA `--fmad=false` parenthesis
 *    trick) translate 1-for-1; HIP `hipcc` does not auto-enable FMA
 *    across parentheses by default, so the explicit parenthesisation
 *    in the decouple stage preserves places=4.
 *  - `__forceinline__` → `__forceinline__` (identical in HIP).
 *  - `extern "C"` wrappers kept for `hipModuleGetFunction` name lookup.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define FADM_BX 16
#define FADM_BY 16
#define FADM_NUM_BANDS 3
#define FADM_ACCUM_SLOTS 6

/* HIP warp size is 64 on GCN/RDNA (vs. 32 on CUDA). */
#define FADM_WARP_SIZE 64
#define FADM_WARPS_PER_BLOCK (FADM_BX * FADM_BY / FADM_WARP_SIZE)

#define FADM_LO0 (0.482962913144690f)
#define FADM_LO1 (0.836516303737469f)
#define FADM_LO2 (0.224143868041857f)
#define FADM_LO3 (-0.129409522550921f)
#define FADM_HI0 (-0.129409522550921f)
#define FADM_HI1 (-0.224143868041857f)
#define FADM_HI2 (0.836516303737469f)
#define FADM_HI3 (-0.482962913144690f)

#define FADM_ONE_BY_30 (0.0333333351f)
#define FADM_ONE_BY_15 (0.0666666701f)
#define FADM_COS_1DEG_SQ (0.99969541789740297f)
#define FADM_EPS (1e-30f)

__device__ static __forceinline__ int fadm_mirror(int idx, int sup)
{
    /* Both axes use `2*sup - idx - 1` — matches CPU
     * dwt2_src_indices_filt_s in adm_tools.c. */
    if (idx < 0)
        return -idx;
    if (idx >= sup)
        return 2 * sup - idx - 1;
    return idx;
}

__device__ static __forceinline__ float fadm_read_src_pixel(const uint8_t *plane,
                                                            ptrdiff_t stride_bytes, int y, int x,
                                                            int w, int h, unsigned bpc,
                                                            float scaler, float pixel_offset)
{
    y = fadm_mirror(y, h);
    if (x < 0)
        x = 0;
    if (x >= w)
        x = w - 1;
    if (bpc <= 8u) {
        return (float)plane[y * stride_bytes + x] + pixel_offset;
    }
    const uint16_t v = reinterpret_cast<const uint16_t *>(plane + y * stride_bytes)[x];
    return (float)v / scaler + pixel_offset;
}

__device__ static __forceinline__ float fadm_read_band_a(const float *band_buf, int buf_stride,
                                                         int half_h, int parent_w, int parent_h,
                                                         int y, int x)
{
    y = fadm_mirror(y, parent_h);
    if (x < 0)
        x = 0;
    if (x >= parent_w)
        x = parent_w - 1;
    (void)half_h;
    return band_buf[y * buf_stride + x];
}

/* HIP warp-level reduction using __shfl_down (no mask needed on HIP). */
__device__ static __forceinline__ float fadm_warp_reduce(float v)
{
    for (int off = FADM_WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

extern "C" {

/* ------------------------------------------------------------------
 * Stage 0 — DWT vertical pass.
 *
 * Output row n consumes input rows (2n - 1 .. 2n + 2). Layout of
 * dwt_tmp:  [gy * (cur_w * 2) + gx]              = lo
 *           [gy * (cur_w * 2) + cur_w + gx]      = hi
 * Z dimension fuses ref+dis (z=0 → ref plane, z=1 → dis plane).
 * ------------------------------------------------------------------ */
__global__ void float_adm_dwt_vert(int scale, const uint8_t *ref_raw, const uint8_t *dis_raw,
                                   ptrdiff_t raw_stride, const float *parent_ref_band,
                                   const float *parent_dis_band, int parent_buf_stride,
                                   int parent_half_h, int parent_w, int parent_h,
                                   float *dwt_tmp_ref, float *dwt_tmp_dis, int cur_w, int cur_h,
                                   int half_h, unsigned bpc, float scaler, float pixel_offset)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int plane_is_dis = (int)blockIdx.z;
    if (gx >= cur_w || gy >= half_h)
        return;
    (void)half_h;

    const int row_start = 2 * gy - 1;
    float s[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (scale == 0) {
            const uint8_t *plane = (plane_is_dis == 0) ? ref_raw : dis_raw;
            s[k] = fadm_read_src_pixel(plane, raw_stride, row_start + k, gx, cur_w, cur_h, bpc,
                                       scaler, pixel_offset);
        } else {
            const float *band = (plane_is_dis == 0) ? parent_ref_band : parent_dis_band;
            s[k] = fadm_read_band_a(band, parent_buf_stride, parent_half_h, parent_w, parent_h,
                                    row_start + k, gx);
        }
    }

    const float lo = FADM_LO0 * s[0] + FADM_LO1 * s[1] + FADM_LO2 * s[2] + FADM_LO3 * s[3];
    const float hi = FADM_HI0 * s[0] + FADM_HI1 * s[1] + FADM_HI2 * s[2] + FADM_HI3 * s[3];

    const int out_stride = cur_w * 2;
    float *dst = (plane_is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;
    dst[gy * out_stride + gx] = lo;
    dst[gy * out_stride + cur_w + gx] = hi;
}

/* ------------------------------------------------------------------
 * Stage 1 — DWT horizontal pass.
 *
 * Reads lo / hi sub-rows from stage 0; emits 4 bands (a/h/v/d).
 *   band_a = lo · lo (LL); band_h = hi · lo (HL high-V);
 *   band_v = lo · hi (LH high-H); band_d = hi · hi (HH).
 * Same a/h/v/d order as integer ADM convention.
 * ------------------------------------------------------------------ */
__device__ static __forceinline__ float fadm_read_dwt_tmp(const float *dwt_tmp, int gy, int x_sub,
                                                          int cur_w, int half_offset)
{
    x_sub = fadm_mirror(x_sub, cur_w);
    const int stride = cur_w * 2;
    return dwt_tmp[gy * stride + half_offset + x_sub];
}

__global__ void float_adm_dwt_hori(int scale, const float *dwt_tmp_ref, const float *dwt_tmp_dis,
                                   float *ref_band, float *dis_band, int cur_w, int half_w,
                                   int half_h, int buf_stride)
{
    (void)scale;
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int plane_is_dis = (int)blockIdx.z;
    if (gx >= half_w || gy >= half_h)
        return;

    const float *src = (plane_is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;
    float *dst = (plane_is_dis == 0) ? ref_band : dis_band;

    const int base_x = 2 * gx;
    /* lo sub-row taps. */
    const float l0 = fadm_read_dwt_tmp(src, gy, base_x - 1, cur_w, 0);
    const float l1 = fadm_read_dwt_tmp(src, gy, base_x + 0, cur_w, 0);
    const float l2 = fadm_read_dwt_tmp(src, gy, base_x + 1, cur_w, 0);
    const float l3 = fadm_read_dwt_tmp(src, gy, base_x + 2, cur_w, 0);
    const float a_val = FADM_LO0 * l0 + FADM_LO1 * l1 + FADM_LO2 * l2 + FADM_LO3 * l3;
    const float v_val = FADM_HI0 * l0 + FADM_HI1 * l1 + FADM_HI2 * l2 + FADM_HI3 * l3;

    /* hi sub-row taps. */
    const float h0 = fadm_read_dwt_tmp(src, gy, base_x - 1, cur_w, cur_w);
    const float h1 = fadm_read_dwt_tmp(src, gy, base_x + 0, cur_w, cur_w);
    const float h2 = fadm_read_dwt_tmp(src, gy, base_x + 1, cur_w, cur_w);
    const float h3 = fadm_read_dwt_tmp(src, gy, base_x + 2, cur_w, cur_w);
    const float h_val = FADM_LO0 * h0 + FADM_LO1 * h1 + FADM_LO2 * h2 + FADM_LO3 * h3;
    const float d_val = FADM_HI0 * h0 + FADM_HI1 * h1 + FADM_HI2 * h2 + FADM_HI3 * h3;

    const int slice = buf_stride * half_h;
    dst[0 * slice + gy * buf_stride + gx] = a_val;
    dst[1 * slice + gy * buf_stride + gx] = h_val;
    dst[2 * slice + gy * buf_stride + gx] = v_val;
    dst[3 * slice + gy * buf_stride + gx] = d_val;
}

/* ------------------------------------------------------------------
 * Stage 2 — Decouple + CSF.
 *
 * The angle-flag computation uses explicit parentheses to suppress
 * FMA fusion — same load-bearing ordering as the CUDA twin and the
 * Vulkan `precise{}` block.
 * ------------------------------------------------------------------ */
__device__ static __forceinline__ float fadm_read_band_at(const float *band_buf, int band, int y,
                                                          int x, int buf_stride, int half_h)
{
    const int slice = buf_stride * half_h;
    return band_buf[band * slice + y * buf_stride + x];
}

__device__ static __forceinline__ void fadm_write_csf(float *csf_buf, int band, int y, int x,
                                                      int buf_stride, int half_h, float val)
{
    const int slice = buf_stride * half_h;
    csf_buf[band * slice + y * buf_stride + x] = val;
}

__global__ void float_adm_decouple_csf(const float *ref_band, const float *dis_band, float *csf_a,
                                       float *csf_f, int half_w, int half_h, int buf_stride,
                                       float rfactor_h, float rfactor_v, float rfactor_d,
                                       float gain_limit)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= half_w || gy >= half_h)
        return;

    const float oh = fadm_read_band_at(ref_band, 1, gy, gx, buf_stride, half_h);
    const float ov = fadm_read_band_at(ref_band, 2, gy, gx, buf_stride, half_h);
    const float od = fadm_read_band_at(ref_band, 3, gy, gx, buf_stride, half_h);
    const float th = fadm_read_band_at(dis_band, 1, gy, gx, buf_stride, half_h);
    const float tv = fadm_read_band_at(dis_band, 2, gy, gx, buf_stride, half_h);
    const float td = fadm_read_band_at(dis_band, 3, gy, gx, buf_stride, half_h);

    /* Angle flag: parenthesisation preserves non-FMA semantics. */
    const float ot_dp = (oh * th) + (ov * tv);
    const float o_mag = (oh * oh) + (ov * ov);
    const float t_mag = (th * th) + (tv * tv);
    const float lhs = ot_dp * ot_dp;
    const float rhs = FADM_COS_1DEG_SQ * (o_mag * t_mag);
    const bool angle_flag = (ot_dp >= 0.0f) && (lhs >= rhs);

    float oarr[3] = {oh, ov, od};
    float tarr[3] = {th, tv, td};
    float rfac[3] = {rfactor_h, rfactor_v, rfactor_d};

#pragma unroll
    for (int b = 0; b < FADM_NUM_BANDS; b++) {
        float k = tarr[b] / (oarr[b] + FADM_EPS);
        k = fmaxf(0.0f, fminf(k, 1.0f));
        float rst = k * oarr[b];
        if (angle_flag && rst > 0.0f)
            rst = fminf(rst * gain_limit, tarr[b]);
        else if (angle_flag && rst < 0.0f)
            rst = fmaxf(rst * gain_limit, tarr[b]);
        const float a_val = tarr[b] - rst;
        const float csf_a_val = rfac[b] * a_val;
        fadm_write_csf(csf_a, b, gy, gx, buf_stride, half_h, csf_a_val);
        fadm_write_csf(csf_f, b, gy, gx, buf_stride, half_h, FADM_ONE_BY_30 * fabsf(csf_a_val));
    }
}

/* ------------------------------------------------------------------
 * Stage 3 — CSF denominator + CM fused.
 *
 * Workgroup grid: (3 * num_active_rows, 1, 1).
 *   band_idx = wg / num_active_rows
 *   row_idx  = wg % num_active_rows
 *
 * Output: 6 floats per WG: [csf_h][csf_v][csf_d][cm_h][cm_v][cm_d].
 *
 * Warp reduction uses FADM_WARP_SIZE=64 (GCN/RDNA) — the shared-mem
 * warp-partial array has FADM_WARPS_PER_BLOCK=4 slots (vs. 8 on CUDA).
 * ------------------------------------------------------------------ */
__device__ static __forceinline__ float fadm_read_csf_f_at(const float *csf_f_buf, int band, int y,
                                                           int x, int half_w, int half_h,
                                                           int buf_stride)
{
    if (x < 0)
        x = -x;
    if (x >= half_w)
        x = 2 * half_w - x - 2;
    if (y < 0)
        y = -y;
    if (y >= half_h)
        y = 2 * half_h - y - 2;
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= half_w)
        x = half_w - 1;
    if (y >= half_h)
        y = half_h - 1;
    const int slice = buf_stride * half_h;
    return csf_f_buf[band * slice + y * buf_stride + x];
}

__device__ static __forceinline__ float fadm_read_csf_a_at(const float *csf_a_buf, int band, int y,
                                                           int x, int half_w, int half_h,
                                                           int buf_stride)
{
    if (x < 0)
        x = 0;
    if (x >= half_w)
        x = half_w - 1;
    if (y < 0)
        y = 0;
    if (y >= half_h)
        y = half_h - 1;
    const int slice = buf_stride * half_h;
    return csf_a_buf[band * slice + y * buf_stride + x];
}

__global__ void float_adm_csf_cm(const float *ref_band, const float *dis_band, const float *csf_a,
                                 const float *csf_f, float *accum_out, int half_w, int half_h,
                                 int buf_stride, int active_left, int active_top, int active_right,
                                 int active_bottom, float rfactor_h, float rfactor_v,
                                 float rfactor_d, float gain_limit)
{
    const int active_h = active_bottom - active_top;
    const int active_w = active_right - active_left;
    if (active_h <= 0 || active_w <= 0)
        return;

    const unsigned wg_id = blockIdx.x;
    const unsigned num_rows = (unsigned)active_h;
    const unsigned band_idx = wg_id / num_rows;
    const unsigned row_idx = wg_id - band_idx * num_rows;
    const int row = active_top + (int)row_idx;

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned lid = ty * FADM_BX + tx;

    const float rfactor_band = (band_idx == 0u) ? rfactor_h :
                               (band_idx == 1u) ? rfactor_v :
                                                  rfactor_d;
    const unsigned WG_SIZE = FADM_BX * FADM_BY;

    float local_csf_sum = 0.0f;
    float local_cm_sum = 0.0f;

    for (int col = active_left + (int)lid; col < active_right; col += (int)WG_SIZE) {
        const float src_ref =
            fadm_read_band_at(ref_band, (int)band_idx + 1, row, col, buf_stride, half_h);
        const float csf_o = fabsf(rfactor_band * src_ref);
        local_csf_sum += csf_o * csf_o * csf_o;

        const float oh = fadm_read_band_at(ref_band, 1, row, col, buf_stride, half_h);
        const float ov = fadm_read_band_at(ref_band, 2, row, col, buf_stride, half_h);
        const float od = fadm_read_band_at(ref_band, 3, row, col, buf_stride, half_h);
        const float th = fadm_read_band_at(dis_band, 1, row, col, buf_stride, half_h);
        const float tv = fadm_read_band_at(dis_band, 2, row, col, buf_stride, half_h);
        const float td = fadm_read_band_at(dis_band, 3, row, col, buf_stride, half_h);
        (void)od;
        (void)td;

        const float ot_dp = (oh * th) + (ov * tv);
        const float o_mag = (oh * oh) + (ov * ov);
        const float t_mag = (th * th) + (tv * tv);
        const float lhs = ot_dp * ot_dp;
        const float rhs = FADM_COS_1DEG_SQ * (o_mag * t_mag);
        const bool angle_flag = (ot_dp >= 0.0f) && (lhs >= rhs);

        float oarr[3] = {oh, ov, od};
        float tarr[3] = {th, tv, td};
        float k = tarr[band_idx] / (oarr[band_idx] + FADM_EPS);
        k = fmaxf(0.0f, fminf(k, 1.0f));
        float r_val = k * oarr[band_idx];
        if (angle_flag && r_val > 0.0f)
            r_val = fminf(r_val * gain_limit, tarr[band_idx]);
        else if (angle_flag && r_val < 0.0f)
            r_val = fmaxf(r_val * gain_limit, tarr[band_idx]);

        /* CM threshold sums csf_f over all 3 bands' 8-neighbours +
         * (1/15)·|csf_a centre| — matches CPU ADM_CM_THRESH_S_I_J. */
        float thr = 0.0f;
#pragma unroll
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
#pragma unroll
            for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0)
                        continue;
                    thr += fadm_read_csf_f_at(csf_f, b, row + dy, col + dx, half_w, half_h,
                                              buf_stride);
                }
            }
        }
        const float own_h = fadm_read_csf_a_at(csf_a, 0, row, col, half_w, half_h, buf_stride);
        const float own_v = fadm_read_csf_a_at(csf_a, 1, row, col, half_w, half_h, buf_stride);
        const float own_d = fadm_read_csf_a_at(csf_a, 2, row, col, half_w, half_h, buf_stride);
        thr += FADM_ONE_BY_15 * fabsf(own_h);
        thr += FADM_ONE_BY_15 * fabsf(own_v);
        thr += FADM_ONE_BY_15 * fabsf(own_d);

        const float x_val = rfactor_band * r_val;
        float xa = fabsf(x_val) - thr;
        if (xa < 0.0f)
            xa = 0.0f;
        local_cm_sum += xa * xa * xa;
    }

    /* Warp + cross-warp reduction. FADM_WARPS_PER_BLOCK = 4 (64-lane warps). */
    __shared__ float s_csf[FADM_WARPS_PER_BLOCK];
    __shared__ float s_cm[FADM_WARPS_PER_BLOCK];
    const float wn_csf = fadm_warp_reduce(local_csf_sum);
    const float wn_cm = fadm_warp_reduce(local_cm_sum);
    const unsigned lane = lid % FADM_WARP_SIZE;
    const unsigned warp_id = lid / FADM_WARP_SIZE;
    if (lane == 0u) {
        s_csf[warp_id] = wn_csf;
        s_cm[warp_id] = wn_cm;
    }
    __syncthreads();
    if (lid == 0u) {
        float total_csf = 0.0f;
        float total_cm = 0.0f;
#pragma unroll
        for (unsigned i = 0u; i < FADM_WARPS_PER_BLOCK; i++) {
            total_csf += s_csf[i];
            total_cm += s_cm[i];
        }
        const unsigned slot_base = wg_id * FADM_ACCUM_SLOTS;
        accum_out[slot_base + band_idx] = total_csf;
        accum_out[slot_base + 3u + band_idx] = total_cm;
    }
}

} /* extern "C" */
