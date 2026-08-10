/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include "cuda_helper.cuh"
#include "common.h"

/*
 * Float vif_* primitives, as used by filter_and_downscale() in speed.c.
 *
 * These are the FLOAT primitives (vif_filter1d_s / vif_dec16_s). The existing
 * CUDA VIF extractor covers the INTEGER path, so there is no GPU counterpart
 * for these yet; they are also reusable outside SpEED.
 *
 * BIT-EXACTNESS
 * -------------
 * Every output here is a fixed-length accumulation in a fixed order, so the
 * result is reproducible -- but only if the compiler does not contract
 * `accum += f * x` into an FMA, which rounds once instead of twice. This
 * file must be compiled with -fmad=false; the explicit __fmul_rn/__fadd_rn
 * below make that independent of the flag.
 *
 * The CPU reference dispatches to convolution_f32_avx_s when AVX2 is present
 * and fwidth is small, and that vectorised path may accumulate in a different
 * order than the scalar fallback reproduced here. Gate against the scalar
 * path first (libvmaf's --cpumask can disable AVX2) before assuming a
 * mismatch is a bug in the kernel.
 *
 * Strides are in float ELEMENTS. The CPU functions take them in bytes and
 * divide by sizeof(float) internally; the host does that conversion here.
 */

extern "C" {

/*
 * Vertical half of vif_filter1d_s. Mirror boundary without repeating the
 * edge sample: ii < 0 -> -ii, ii >= h -> 2*h - ii - 2.
 *
 * The CPU keeps only one row of intermediate; a full-frame tmp is equivalent
 * because row i of the vertical pass depends only on src, and the horizontal
 * pass for row i reads only row i of the intermediate.
 */
__global__ void speed_filter1d_v_kernel(const float *f, const float *src,
                                        float *tmp, int w, int h,
                                        ptrdiff_t src_stride_px,
                                        ptrdiff_t tmp_stride_px, int fwidth)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= w || i >= h)
        return;

    float accum = 0.0f;
    for (int fi = 0; fi < fwidth; ++fi) {
        int ii = i - fwidth / 2 + fi;
        ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 2 : ii);
        accum = __fadd_rn(accum, __fmul_rn(f[fi], src[(ptrdiff_t) ii * src_stride_px + j]));
    }
    tmp[(ptrdiff_t) i * tmp_stride_px + j] = accum;
}

/* Horizontal half, same mirror rule on the column index. */
__global__ void speed_filter1d_h_kernel(const float *f, const float *tmp,
                                        float *dst, int w, int h,
                                        ptrdiff_t tmp_stride_px,
                                        ptrdiff_t dst_stride_px, int fwidth)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= w || i >= h)
        return;

    const float *trow = tmp + (ptrdiff_t) i * tmp_stride_px;
    float accum = 0.0f;
    for (int fj = 0; fj < fwidth; ++fj) {
        int jj = j - fwidth / 2 + fj;
        jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 2 : jj);
        accum = __fadd_rn(accum, __fmul_rn(f[fj], trow[jj]));
    }
    dst[(ptrdiff_t) i * dst_stride_px + j] = accum;
}

/* vif_dec16_s: take every 16th sample in each direction. Output extent is
 * src_h/16 x src_w/16, matching the CPU's integer division. */
__global__ void speed_dec16_kernel(const float *src, float *dst,
                                   int src_w, int src_h,
                                   ptrdiff_t src_stride_px,
                                   ptrdiff_t dst_stride_px)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= src_w / 16 || i >= src_h / 16)
        return;

    dst[(ptrdiff_t) i * dst_stride_px + j] =
        src[(ptrdiff_t)(i * 16) * src_stride_px + (j * 16)];
}

/* subtract_image: im1 -= im2, in place, both at the same stride. */
__global__ void speed_subtract_kernel(float *im1, const float *im2,
                                      int w, int h, ptrdiff_t stride_px)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= w || i >= h)
        return;

    const ptrdiff_t off = (ptrdiff_t) i * stride_px + j;
    im1[off] = __fsub_rn(im1[off], im2[off]);
}


/*
 * picture_copy(): plane -> float with a constant offset. SpEED passes
 * offset = -128 so the data is centred near zero.
 *
 * The high-bitdepth form divides by a scaler (4.0 for 10-bit, 16.0 for
 * 12-bit, 256.0 for 16-bit) before adding the offset. Those divisors are
 * exact powers of two so the division is exact either way, but it is spelled
 * with __fdiv_rn alongside __fadd_rn to keep the arithmetic pinned for the
 * same reason as the filter kernels.
 *
 * Source strides are in ELEMENTS of the source type; dst_stride is in floats.
 */
__global__ void speed_picture_copy_u8_kernel(const unsigned char *src,
                                             float *dst, int w, int h,
                                             ptrdiff_t src_stride,
                                             ptrdiff_t dst_stride, int offset)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= w || i >= h)
        return;

    const float v = (float) src[(ptrdiff_t) i * src_stride + j];
    dst[(ptrdiff_t) i * dst_stride + j] = __fadd_rn(v, (float) offset);
}

__global__ void speed_picture_copy_u16_kernel(const uint16_t *src,
                                              float *dst, int w, int h,
                                              ptrdiff_t src_stride,
                                              ptrdiff_t dst_stride,
                                              int offset, float scaler)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= w || i >= h)
        return;

    const float v = (float) src[(ptrdiff_t) i * src_stride + j];
    dst[(ptrdiff_t) i * dst_stride + j] =
        __fadd_rn(__fdiv_rn(v, scaler), (float) offset);
}

} // extern "C"
