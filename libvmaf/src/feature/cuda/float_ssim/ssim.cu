/**
 *
 *  Copyright 2026 Bardie Høgh Joensen
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

/*
 * GPU port of the float_ssim feature extractor (feature/ssim.c + feature/iqa).
 *
 * The kernels below replicate the iqa reference operation-for-operation so
 * per-pixel intermediates match the CPU extractor to float precision:
 *   - decimation samples at (x*factor, y*factor) with a box kernel and
 *     symmetric boundary mirroring (iqa/decimate.c, _iqa_filter_pixel)
 *   - the 11-tap separable Gaussian runs in valid mode (output shrinks by
 *     kernel-1) with double accumulation per 1-D pass and a float round
 *     in between, exactly like _iqa_convolve with IQA_CONVOLVE_1D
 *   - per-pixel l/c/s uses the same float/double mixing as _iqa_ssim;
 *     float-typed reference operations use explicit __f*_rn intrinsics so
 *     nvcc cannot contract them into fma with different rounding
 * The frame mean is a per-block tree reduction; the block partials are
 * summed sequentially on the host, so results are deterministic run-to-run.
 */

#include "cuda_helper.cuh"

#include "common.h"

__constant__ float gaussian_k[11] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f, 0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

// Mirrors an idx along its valid [0, sup) range like iqa's KBND_SYMMETRIC
__device__ __forceinline__ int mirror_sym(int idx, const int sup)
{
    if (idx < 0) idx = -1 - idx;
    else if (idx >= sup) idx = (sup - (idx - sup)) - 1;
    return idx;
}

extern "C" {

__global__ void ssim_normalize_8bpc(const VmafPicture pic, VmafCudaBuffer out,
        unsigned w, unsigned h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const uint8_t v = (reinterpret_cast<const uint8_t*>(pic.data[0]) +
            y * pic.stride[0])[x];
    reinterpret_cast<float*>(out.data)[y * w + x] = static_cast<float>(v);
}

__global__ void ssim_normalize_16bpc(const VmafPicture pic, VmafCudaBuffer out,
        unsigned w, unsigned h, float scaler)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const uint16_t v = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const uint8_t*>(pic.data[0]) +
            y * pic.stride[0])[x];
    // picture_copy: (float)data / scaler; scaler is a power of two, exact
    reinterpret_cast<float*>(out.data)[y * w + x] =
        __fdiv_rn(static_cast<float>(v), scaler);
}

// iqa/decimate.c: out[y*sw+x] = box(in, x*factor, y*factor), symmetric
// bounds — fused with the float conversion of picture_copy so the full-res
// float image is never materialized. The conversion is exact per sample
// (power-of-two divide), so the box filter sees bit-identical floats.
__global__ void ssim_decimate_8bpc(const VmafPicture pic, VmafCudaBuffer out,
        int w, int h, int sw, int sh, int factor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sw || y >= sh) return;

    const uint8_t *base = reinterpret_cast<const uint8_t*>(pic.data[0]);
    const int cx = x * factor;
    const int cy = y * factor;
    const int uc = factor / 2;
    const int k_even = (factor & 1) ? 0 : 1;
    const float k_val = 1.0f / (factor * factor);

    double sum = 0.0;
    if (cx >= uc && cy >= uc && cx < w - uc && cy < h - uc) {
        for (int v = -uc; v <= uc - k_even; v++) {
            const uint8_t *row = base + (cy + v) * pic.stride[0] + cx;
            for (int u = -uc; u <= uc - k_even; u++) {
                const float px = static_cast<float>(row[u]);
                sum += px * k_val;
            }
        }
    } else {
        for (int v = -uc; v <= uc - k_even; v++) {
            const int yy = mirror_sym(cy + v, h);
            const uint8_t *row = base + yy * pic.stride[0];
            for (int u = -uc; u <= uc - k_even; u++) {
                const float px = static_cast<float>(row[mirror_sym(cx + u, w)]);
                sum += px * k_val;
            }
        }
    }
    // _iqa_filter_pixel returns (float)(sum * kscale) with kscale == 1.0
    reinterpret_cast<float*>(out.data)[y * sw + x] = static_cast<float>(sum);
}

__global__ void ssim_decimate_16bpc(const VmafPicture pic, VmafCudaBuffer out,
        int w, int h, int sw, int sh, int factor, float scaler)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sw || y >= sh) return;

    const uint8_t *base = reinterpret_cast<const uint8_t*>(pic.data[0]);
    const int cx = x * factor;
    const int cy = y * factor;
    const int uc = factor / 2;
    const int k_even = (factor & 1) ? 0 : 1;
    const float k_val = 1.0f / (factor * factor);

    double sum = 0.0;
    if (cx >= uc && cy >= uc && cx < w - uc && cy < h - uc) {
        for (int v = -uc; v <= uc - k_even; v++) {
            const uint16_t *row = reinterpret_cast<const uint16_t*>(
                    base + (cy + v) * pic.stride[0]) + cx;
            for (int u = -uc; u <= uc - k_even; u++) {
                const float px = __fdiv_rn(static_cast<float>(row[u]), scaler);
                sum += px * k_val;
            }
        }
    } else {
        for (int v = -uc; v <= uc - k_even; v++) {
            const int yy = mirror_sym(cy + v, h);
            const uint16_t *row = reinterpret_cast<const uint16_t*>(
                    base + yy * pic.stride[0]);
            for (int u = -uc; u <= uc - k_even; u++) {
                const float px = __fdiv_rn(static_cast<float>(
                            row[mirror_sym(cx + u, w)]), scaler);
                sum += px * k_val;
            }
        }
    }
    reinterpret_cast<float*>(out.data)[y * sw + x] = static_cast<float>(sum);
}

__global__ void ssim_products(const VmafCudaBuffer ref, const VmafCudaBuffer cmp,
        VmafCudaBuffer ref2, VmafCudaBuffer cmp2, VmafCudaBuffer both, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float r = reinterpret_cast<const float*>(ref.data)[i];
    const float c = reinterpret_cast<const float*>(cmp.data)[i];
    reinterpret_cast<float*>(ref2.data)[i] = __fmul_rn(r, r);
    reinterpret_cast<float*>(cmp2.data)[i] = __fmul_rn(c, c);
    reinterpret_cast<float*>(both.data)[i] = __fmul_rn(r, c);
}

// _iqa_convolve horizontal pass (valid in x): rows span the full height so
// the vertical pass has its apron, exactly like the CPU img_cache
__global__ void ssim_conv_h(const VmafCudaBuffer in, VmafCudaBuffer cache,
        int w, int h, int dst_w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ky = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || ky >= h) return;

    const float *img = reinterpret_cast<const float*>(in.data);
    const int kx = x + 5;
    const int img_offset = ky * w + kx;

    double sum = 0.0;
#pragma unroll
    for (int u = -5; u <= 5; u++)
        sum += img[img_offset + u] * gaussian_k[u + 5];
    reinterpret_cast<float*>(cache.data)[img_offset] = static_cast<float>(sum);
}

__global__ void ssim_conv_v(const VmafCudaBuffer cache, VmafCudaBuffer out,
        int w, int dst_w, int dst_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const float *c = reinterpret_cast<const float*>(cache.data);
    const int img_offset = (y + 5) * w + x + 5;

    double sum = 0.0;
#pragma unroll
    for (int v = -5; v <= 5; v++)
        sum += c[img_offset + v * w] * gaussian_k[v + 5];
    reinterpret_cast<float*>(out.data)[y * dst_w + x] = static_cast<float>(sum);
}

// _iqa_ssim per-pixel l/c/s and frame sums; one partial per block, laid out
// as partials[block][4] = {ssim, l, c, s}
__global__ void ssim_map_reduce(const VmafCudaBuffer mu1_buf,
        const VmafCudaBuffer mu2_buf, const VmafCudaBuffer cref2_buf,
        const VmafCudaBuffer ccmp2_buf, const VmafCudaBuffer cboth_buf,
        VmafCudaBuffer partials, int n, float c1, float c2, float c3)
{
    __shared__ double sh[256][4];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = threadIdx.x;

    double lcs = 0.0, l = 0.0, c = 0.0, s = 0.0;
    if (i < n) {
        const float mu1 = reinterpret_cast<const float*>(mu1_buf.data)[i];
        const float mu2 = reinterpret_cast<const float*>(mu2_buf.data)[i];

        // sigma buffers are float and computed as conv(x^2) - mu^2 in float
        float ref_sigma_sqd = __fsub_rn(
                reinterpret_cast<const float*>(cref2_buf.data)[i],
                __fmul_rn(mu1, mu1));
        float cmp_sigma_sqd = __fsub_rn(
                reinterpret_cast<const float*>(ccmp2_buf.data)[i],
                __fmul_rn(mu2, mu2));
        ref_sigma_sqd = MAX(0.0f, ref_sigma_sqd);
        cmp_sigma_sqd = MAX(0.0f, cmp_sigma_sqd);
        const float sigma_both = __fsub_rn(
                reinterpret_cast<const float*>(cboth_buf.data)[i],
                __fmul_rn(mu1, mu2));

        // float sqrt over a double-promoted float product, as in _iqa_ssim
        const float sigma_ref_sigma_cmp = static_cast<float>(sqrt(
                    static_cast<double>(__fmul_rn(ref_sigma_sqd, cmp_sigma_sqd))));

        // l and c divide a double numerator by a float-summed denominator
        l = (2.0 * mu1 * mu2 + c1) /
            static_cast<double>(__fadd_rn(__fadd_rn(__fmul_rn(mu1, mu1),
                        __fmul_rn(mu2, mu2)), c1));
        c = (2.0 * sigma_ref_sigma_cmp + c2) /
            static_cast<double>(__fadd_rn(__fadd_rn(ref_sigma_sqd,
                        cmp_sigma_sqd), c2));

        const float clamped_sigma_both =
            (sigma_both < 0.0f && sigma_ref_sigma_cmp <= 0.0f) ?
            0.0f : sigma_both;
        // s is a float division in the reference, promoted on assignment
        s = static_cast<double>(__fdiv_rn(__fadd_rn(clamped_sigma_both, c3),
                    __fadd_rn(sigma_ref_sigma_cmp, c3)));

        lcs = l * c * s;
    }

    sh[t][0] = lcs;
    sh[t][1] = l;
    sh[t][2] = c;
    sh[t][3] = s;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
#pragma unroll
            for (int j = 0; j < 4; j++)
                sh[t][j] += sh[t + stride][j];
        }
        __syncthreads();
    }

    if (t == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++)
            reinterpret_cast<double*>(partials.data)[blockIdx.x * 4 + j] =
                sh[0][j];
    }
}

}
