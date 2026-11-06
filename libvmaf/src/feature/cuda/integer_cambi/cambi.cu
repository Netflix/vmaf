/**
 *
 *  Copyright 2026 luoye-cpu
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
 */

/*
 * CAMBI device code.
 *
 * IMPORTANT: this translation unit is *only* compiled to a fatbin by nvcc and
 * then embedded as a byte array by bin2c (see src/meson.build, cuda_cu_sources
 * / ptx_bin2c_*). Host code placed here is never compiled or linked. All host
 * side orchestration lives in feature/cuda/integer_cambi_cuda.c, which loads
 * the `cambi_ptx` module through cuModuleLoadData().
 *
 * Every kernel below is a direct port of the corresponding routine in
 * feature/cambi.c (the CPU reference), including its edge-clamping and
 * integer/float rounding behaviour. Any semantic change here breaks
 * CPU/CUDA numerical parity.
 */

#include "cuda_helper.cuh"

/*
 * Kernels are looked up by name via cuModuleGetFunction(), so they must have
 * unmangled (C) linkage -- same convention as integer_motion and integer_adm.
 */
extern "C" {

/* ------------------------------------------------------------------------- */
/* Preprocessing                                                             */
/* ------------------------------------------------------------------------- */

/*
 * Convert the source picture to 10-bit at the exact encoding resolution.
 * Mirrors decimate_generic_*_and_convert_to_10b() for the in_w == out_w case.
 *
 * mode 0: 8-bit input,  out = v << shift
 * mode 1: 16-bit input, out = v << shift          (bpc 9 -> shift 1, bpc 10 -> 0)
 * mode 2: 16-bit input, out = (v + rounding) >> shift
 *
 * src_stride is in elements of the source type (bytes for mode 0).
 */
__global__ void cambi_convert_10b_kernel(const void *src, int src_stride,
                                         uint16_t *dst, int w, int h,
                                         int mode, int shift, int rounding)
{
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= h || c >= w) return;

    uint32_t v;
    if (mode == 0) {
        v = ((const uint8_t *)src)[(size_t)r * src_stride + c];
        v <<= shift;
    } else if (mode == 1) {
        v = ((const uint16_t *)src)[(size_t)r * src_stride + c];
        v <<= shift;
    } else {
        v = ((const uint16_t *)src)[(size_t)r * src_stride + c];
        v = (v + (uint32_t)rounding) >> shift;
    }
    dst[(size_t)r * w + c] = (uint16_t)v;
}

/*
 * Convert + downscale to the encoding resolution (in_w != out_w case).
 * One thread per output row, replicating the CPU's incremental float
 * accumulation of x/y exactly (a multiply would not be bit-identical).
 */
__global__ void cambi_convert_10b_scaled_kernel(const void *src, int src_stride,
                                                uint16_t *dst, int out_w, int out_h,
                                                int in_w, int in_h,
                                                int mode, int shift, int rounding)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_h) return;

    const float ratio_x = (float)in_w / out_w;
    const float ratio_y = (float)in_h / out_h;
    const float start_x = ratio_x / 2 - 0.5;
    const float start_y = ratio_y / 2 - 0.5;

    float y = start_y;
    for (int k = 0; k < i; k++) y += ratio_y;

    float x = start_x;
    for (int j = 0; j < out_w; j++) {
        const unsigned ori_y = (unsigned)(int)(y + 0.5);
        const unsigned ori_x = (unsigned)(int)(x + 0.5);
        uint32_t v;
        if (mode == 0) {
            v = ((const uint8_t *)src)[(size_t)ori_y * src_stride + ori_x];
            v <<= shift;
        } else if (mode == 1) {
            v = ((const uint16_t *)src)[(size_t)ori_y * src_stride + ori_x];
            v <<= shift;
        } else {
            v = ((const uint16_t *)src)[(size_t)ori_y * src_stride + ori_x];
            v = (v + (uint32_t)rounding) >> shift;
        }
        dst[(size_t)i * out_w + j] = (uint16_t)v;
        x += ratio_x;
    }
}

/*
 * anti_dithering_filter(): out-of-place 2x2 box filter with the CPU's edge
 * handling. In the CPU the filter runs in raster order in place, but every tap
 * it reads is still unmodified at that point, so an out-of-place kernel is
 * exactly equivalent. Running it in place on the GPU would be a data race.
 */
__global__ void cambi_anti_dithering_kernel(const uint16_t *src, uint16_t *dst,
                                            int w, int h, int stride)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h || j >= w) return;

    const size_t p = (size_t)i * stride + j;
    uint16_t val;
    if (i < h - 1 && j < w - 1) {
        val = (src[p] + src[p + 1] +
               src[p + stride] + src[p + stride + 1]) >> 2;
    } else if (i < h - 1) {
        val = (src[p] + src[p + stride]) >> 1;
    } else if (j < w - 1) {
        val = (src[p] + src[p + 1]) >> 1;
    } else {
        val = src[p];
    }
    dst[p] = val;
}

/* ------------------------------------------------------------------------- */
/* Spatial mask (dynamic-programming box filter over the derivative)          */
/* ------------------------------------------------------------------------- */

/*
 * get_derivative_data_for_row() fused with the horizontal prefix sum.
 * Writes, for row r:
 *   Z[r][0] = 0
 *   Z[r][c + 1] = sum_{c' <= c} derivative(r, c')
 * The vertical integration of Z (see cambi_integrate_rows_kernel) turns this
 * into a 2D summed-area table, so the mask becomes a plain box query. This is
 * algebraically identical to the CPU's cyclic DP row buffer.
 */
__global__ void cambi_row_prefix_kernel(const uint16_t *image, int img_pitch,
                                        int w, int h, int z_pitch, int32_t *Z)
{
    __shared__ int32_t ssum[256];
    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    if (r >= h) return;

    const uint16_t *row = image + (size_t)r * img_pitch;
    const uint16_t *nrow = row + ((r + 1 < h) ? img_pitch : 0);
    int32_t *out = Z + (size_t)r * z_pitch;

    const int chunk = (w + blockDim.x - 1) / blockDim.x;
    const int c0 = tid * chunk;
    int c1 = c0 + chunk;
    if (c1 > w) c1 = w;

    int32_t acc = 0;
    for (int c = c0; c < c1; c++) {
        const uint16_t v = row[c];
        const bool h_eq = (c == w - 1) || (v == row[c + 1]);
        const bool v_eq = (r == h - 1) || (v == nrow[c]);
        acc += (h_eq && v_eq) ? 1 : 0;
        out[c + 1] = acc;
    }
    const int32_t local_sum = acc;

    ssum[tid] = local_sum;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int32_t t = (tid >= off) ? ssum[tid - off] : 0;
        __syncthreads();
        ssum[tid] += t;
        __syncthreads();
    }
    const int32_t excl = ssum[tid] - local_sum;
    for (int c = c0; c < c1; c++) out[c + 1] += excl;
    if (tid == 0) out[0] = 0;
}

/* Column-wise inclusive scan over rows: Z[r][m] = sum_{r' <= r} Z[r'][m]. */
__global__ void cambi_integrate_rows_kernel(int32_t *Z, int z_pitch, int h, int w1)
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= w1) return;
    int32_t acc = 0;
    for (int r = 0; r < h; r++) {
        acc += Z[(size_t)r * z_pitch + m];
        Z[(size_t)r * z_pitch + m] = acc;
    }
}

/*
 * compute_mask_row(): mask[i][j] = 1 iff the 7x7 box sum of the derivative
 * around (i, j) exceeds mask_index, with the window truncated at the borders
 * exactly like the CPU's sliding window.
 */
__global__ void cambi_mask_kernel(const int32_t *Z, uint16_t *mask,
                                  int img_pitch, int z_pitch,
                                  int w, int h, int pad, unsigned mask_index)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h || j >= w) return;

    int r1 = i - pad;
    if (r1 < 0) r1 = 0;
    int r2 = i + pad;
    if (r2 > h - 1) r2 = h - 1;

    int c0 = j - pad;
    if (c0 < 0) c0 = 0;
    int c1 = j + pad + 1;
    if (c1 > w) c1 = w;

    const int32_t *zr2 = Z + (size_t)r2 * z_pitch;
    unsigned result;
    if (r1 > 0) {
        const int32_t *zr1 = Z + (size_t)(r1 - 1) * z_pitch;
        result = (unsigned)((zr2[c1] - zr2[c0]) - (zr1[c1] - zr1[c0]));
    } else {
        result = (unsigned)(zr2[c1] - zr2[c0]);
    }
    mask[(size_t)i * img_pitch + j] = (uint16_t)(result > mask_index);
}

/* ------------------------------------------------------------------------- */
/* Banding detection                                                         */
/* ------------------------------------------------------------------------- */

/*
 * decimate(): image[i][j] = image[2i][2j]. Out-of-place; the CPU version is in
 * place but only ever reads pixels that have not been overwritten yet.
 * pitch is the *fixed* working stride, not the scaled width.
 */
__global__ void cambi_decimate_kernel(const uint16_t *src, uint16_t *dst,
                                      int out_w, int out_h, int pitch)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_h || j >= out_w) return;
    dst[(size_t)i * pitch + j] = src[(size_t)(i << 1) * pitch + (j << 1)];
}

__device__ __forceinline__ uint16_t cambi_min3(uint16_t a, uint16_t b, uint16_t c)
{
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}

__device__ __forceinline__ uint16_t cambi_mode3(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return cambi_min3(a, b, c);
}

__device__ __forceinline__ uint16_t cambi_horizontal_mode3(const uint16_t *row, int w, int c)
{
    if (c == 0 || c == w - 1) return row[c];
    return cambi_mode3(row[c - 1], row[c], row[c + 1]);
}

/*
 * filter_mode(): horizontal mode3 of every row, followed by a vertical mode3
 * over three consecutive horizontally filtered rows, written back to row i-1.
 * The CPU does this with a 3-line rolling buffer over the *original* image, so
 * the fused form below produces the same result. Rows 0 and h-1 are never
 * written by the CPU and therefore keep their unfiltered values.
 */
__global__ void cambi_filter_mode_kernel(const uint16_t *image, uint16_t *out,
                                         int w, int h, int img_pitch)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h || j >= w) return;

    const size_t p = (size_t)i * img_pitch + j;
    if (i == 0 || i == h - 1) {
        out[p] = image[p];
        return;
    }

    const uint16_t a = cambi_horizontal_mode3(image + (size_t)(i - 1) * img_pitch, w, j);
    const uint16_t b = cambi_horizontal_mode3(image + (size_t)i * img_pitch, w, j);
    const uint16_t d = cambi_horizontal_mode3(image + (size_t)(i + 1) * img_pitch, w, j);
    out[p] = cambi_mode3(a, b, d);
}

/* Largest |value difference| CAMBI ever looks at, for num_diffs <= 32. */
#define CAMBI_MAX_LOCAL (2 * 32 + 1)

/*
 * calculate_c_values(): for every masked output pixel, build the histogram of
 * the window_size x window_size square around it, restricted to the luma band
 * [v_band_base, v_band_base + v_band_size), then evaluate the suprathreshold
 * contrast response. This reproduces c_value_pixel() and the sliding-window
 * histogram of calculate_c_values() exactly: the histogram is queried only at
 * value offsets v +/- (d + 1), which are the only cells the CPU's incremental
 * update ever populates for a given output pixel.
 *
 * image/mask use the fixed working pitch; c_values is compact (scaled width).
 */
__global__ void cambi_calc_c_values_kernel(const uint16_t *image, const uint16_t *mask,
                                           float *c_values,
                                           int img_pitch, int c_pitch,
                                           int w, int h, int pad,
                                           unsigned num_diffs, unsigned vlt_luma,
                                           const uint16_t *tvi_for_diff,
                                           const int *all_diffs,
                                           const float *recip_lut,
                                           unsigned v_band_base, unsigned v_band_size,
                                           unsigned v_band_offset,
                                           const int *diff_weights)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h || j >= w) return;

    float *const c_out = c_values + (size_t)i * c_pitch + j;

    if (!mask[(size_t)i * img_pitch + j]) {
        *c_out = 0.0f;
        return;
    }

    const uint16_t v = image[(size_t)i * img_pitch + j];
    const unsigned value = (unsigned)v + num_diffs;
    const int compact = (int)value - (int)v_band_offset;
    if ((unsigned)compact >= v_band_size) {
        *c_out = 0.0f;
        return;
    }

    int local[CAMBI_MAX_LOCAL];
    for (unsigned k = 0; k < 2 * num_diffs + 1; k++) local[k] = 0;

    int r0 = i - pad;
    if (r0 < 0) r0 = 0;
    int r1 = i + pad;
    if (r1 > h - 1) r1 = h - 1;
    int c0 = j - pad;
    if (c0 < 0) c0 = 0;
    int c1 = j + pad;
    if (c1 > w - 1) c1 = w - 1;

    for (int r = r0; r <= r1; r++) {
        const uint16_t *irow = image + (size_t)r * img_pitch;
        const uint16_t *mrow = mask + (size_t)r * img_pitch;
        for (int c = c0; c <= c1; c++) {
            if (!mrow[c]) continue;
            const int val = (int)irow[c];
            if ((unsigned)(val - (int)v_band_base) >= v_band_size) continue;
            const int idx = val - (int)v + (int)num_diffs;
            if (idx >= 0 && idx <= (int)(2 * num_diffs)) local[idx]++;
        }
    }

    const unsigned p_0 = (unsigned)local[num_diffs];
    float c_value = 0.0f;
    for (unsigned d = 0; d < num_diffs; d++) {
        if (value <= tvi_for_diff[d] &&
            ((int)value + all_diffs[num_diffs + d + 1]) > (int)vlt_luma) {
            const int idx2 = compact + all_diffs[num_diffs - d - 1];
            const unsigned p_1 = (unsigned)local[num_diffs + d + 1];
            const unsigned p_2 = (idx2 >= 0) ? (unsigned)local[num_diffs - d - 1] : 0u;
            float val;
            if (p_1 > p_2) {
                val = (float)(diff_weights[d] * (int)p_0 * (int)p_1) *
                      recip_lut[p_1 + p_0];
            } else {
                val = (float)(diff_weights[d] * (int)p_0 * (int)p_2) *
                      recip_lut[p_2 + p_0];
            }
            if (val > c_value) c_value = val;
        }
    }
    *c_out = c_value;
}

/* ------------------------------------------------------------------------- */
/* Spatial pooling (top-k average)                                           */
/* ------------------------------------------------------------------------- */

/*
 * c_values are non-negative, so their IEEE-754 bit patterns are monotonic in
 * value and can be used as unsigned integer keys for radix selection.
 */
__global__ void cambi_pool_count_kernel(const float *c_values, int n,
                                        unsigned key_threshold,
                                        unsigned long long *out_count)
{
    __shared__ unsigned long long s[256];
    unsigned long long cnt = 0;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (__float_as_uint(c_values[i]) >= key_threshold) cnt++;
    }
    s[threadIdx.x] = cnt;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out_count, s[0]);
}

/*
 * Counts the elements strictly above the k-th largest key and accumulates
 * their sum. The per-thread accumulation order, the block reduction tree and
 * the final combination of block partials are all fixed, so the result is
 * deterministic (floating-point addition is not associative).
 */
__global__ void cambi_pool_sum_kernel(const float *c_values, int n,
                                      unsigned key_threshold,
                                      unsigned long long *out_count_gt,
                                      double *partials)
{
    __shared__ unsigned long long sc[256];
    __shared__ double ss[256];
    unsigned long long cnt = 0;
    double sum = 0.0;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (__float_as_uint(c_values[i]) > key_threshold) {
            cnt++;
            sum += (double)c_values[i];
        }
    }
    sc[threadIdx.x] = cnt;
    ss[threadIdx.x] = sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sc[threadIdx.x] += sc[threadIdx.x + off];
            ss[threadIdx.x] += ss[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_count_gt, sc[0]);
        partials[blockIdx.x] = ss[0];
    }
}

__global__ void cambi_pool_partials_kernel(const double *partials, int n, double *out_sum)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += partials[i];
    *out_sum = s;
}

} /* extern "C" */
