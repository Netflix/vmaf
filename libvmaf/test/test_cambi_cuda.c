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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "test.h"
#include "feature/cuda/cambi_cuda.h"

/* ------------------------------------------------------------------ */
/* CPU reference: a verbatim copy of get_derivative_data_for_row()     */
/* from src/feature/cambi.c. Kept literal on purpose -- if the         */
/* reference changes upstream, this test must be updated deliberately  */
/* rather than silently tracking it.                                   */
/* ------------------------------------------------------------------ */
static void ref_derivative_row(const uint16_t *image_data,
                               uint16_t *derivative_buffer,
                               int width, int height, int row, int stride)
{
    for (int col = 0; col < width; col++) {
        bool horizontal_derivative =
            (col == width - 1 ||
             image_data[row * stride + col] == image_data[row * stride + col + 1]);
        bool vertical_derivative =
            (row == height - 1 ||
             image_data[row * stride + col] == image_data[(row + 1) * stride + col]);
        derivative_buffer[col] = horizontal_derivative && vertical_derivative;
    }
}

/* ------------------------------------------------------------------ */
/* Test image generators. Each targets a different failure mode.       */
/* ------------------------------------------------------------------ */
/* LESSON (from deliberately breaking the derivative kernel's last-column
 * case): structured inputs beat random ones for boundary bugs. Flat and
 * gradient caught every affected pixel; random caught 1 of 2073600, because
 * random data usually differs from the out-of-bounds read anyway and lands
 * on the correct answer by accident. Weight toward flat plateaus and smooth
 * gradients -- which is also what real banding content looks like. */
enum pattern {
    PAT_FLAT,      /* all equal: every derivative should be 1          */
    PAT_GRADIENT,  /* smooth ramp: banding-like, mixed                 */
    PAT_CHECKER,   /* alternating: every derivative should be 0        */
    PAT_RANDOM,    /* full-range noise                                 */
    PAT_COUNT
};

static void fill_pattern(uint16_t *buf, int width, int height,
                         ptrdiff_t stride, enum pattern p, unsigned seed)
{
    unsigned s = seed ? seed : 1;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint16_t v;
            switch (p) {
            case PAT_FLAT:     v = 512; break;
            case PAT_GRADIENT: v = (uint16_t)((j * 1023) / (width > 1 ? width - 1 : 1)); break;
            case PAT_CHECKER:  v = ((i + j) & 1) ? 1023 : 0; break;
            default:
                s = s * 1103515245u + 12345u;
                v = (uint16_t)((s >> 16) & 0x3FF); /* 10-bit, matches preprocessing */
                break;
            }
            buf[i * stride + j] = v;
        }
    }
}

#define CU_CHECK(call)                                                    \
    do {                                                                  \
        CUresult _e = (call);                                             \
        if (_e != CUDA_SUCCESS) {                                         \
            const char *_n = NULL;                                        \
            cuGetErrorName(_e, &_n);                                      \
            fprintf(stderr, "\n  %s failed: %s\n", #call, _n ? _n : "?"); \
            return "cuda driver call failed";                             \
        }                                                                 \
    } while (0)

static char *test_cambi_derivative_cuda(void)
{
    /* Dimensions chosen to exercise the edges that broke the existing
     * CUDA extractors: non-multiple-of-32 widths, odd widths and heights,
     * and the degenerate width==1 / height==1 cases. */
    const int dims[][2] = {
        { 64, 64 }, { 1920, 1080 }, { 1919, 1079 }, { 33, 17 },
        { 1, 16 }, { 16, 1 }, { 1, 1 }, { 3, 3 },
    };
    const int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, cambi_derivative_ptx));
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, "cambi_derivative_kernel"));

    unsigned long total_mismatch = 0;

    for (int d = 0; d < n_dims; d++) {
        const int width = dims[d][0], height = dims[d][1];
        /* Deliberately use a padded stride: a stride == width test would
         * never catch a stride bug. */
        const ptrdiff_t stride = width + 7;

        for (enum pattern p = 0; p < PAT_COUNT; p++) {
            const size_t img_elems = (size_t)stride * height;
            const size_t out_elems = (size_t)width * height;

            uint16_t *h_img = malloc(img_elems * sizeof(uint16_t));
            uint16_t *h_ref = malloc(out_elems * sizeof(uint16_t));
            uint16_t *h_gpu = malloc(out_elems * sizeof(uint16_t));
            if (!h_img || !h_ref || !h_gpu) {
                free(h_img); free(h_ref); free(h_gpu);
                return "allocation failed";
            }
            memset(h_img, 0, img_elems * sizeof(uint16_t));

            fill_pattern(h_img, width, height, stride, p, 1234u + d * 17u + p);

            for (int row = 0; row < height; row++)
                ref_derivative_row(h_img, &h_ref[(size_t)row * width],
                                   width, height, row, (int)stride);

            CUdeviceptr d_img, d_out;
            CU_CHECK(cuMemAlloc(&d_img, img_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemAlloc(&d_out, out_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemcpyHtoD(d_img, h_img, img_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemsetD8(d_out, 0xAB, out_elems * sizeof(uint16_t)));

            int w = width, h = height;
            ptrdiff_t src_stride = stride, dst_stride = width;
            void *args[] = { &d_img, &d_out, &w, &h, &src_stride, &dst_stride };

            const unsigned bx = 32, by = 8;
            CU_CHECK(cuLaunchKernel(kernel,
                                    (width + bx - 1) / bx, (height + by - 1) / by, 1,
                                    bx, by, 1, 0, NULL, args, NULL));
            CU_CHECK(cuCtxSynchronize());
            CU_CHECK(cuMemcpyDtoH(h_gpu, d_out, out_elems * sizeof(uint16_t)));

            unsigned long mismatch = 0;
            size_t first = (size_t)-1;
            for (size_t k = 0; k < out_elems; k++) {
                if (h_gpu[k] != h_ref[k]) {
                    if (!mismatch) first = k;
                    mismatch++;
                }
            }
            if (mismatch) {
                fprintf(stderr,
                        "\n  %dx%d stride=%td pattern=%d: %lu / %zu mismatch, "
                        "first at (row %zu, col %zu): cpu=%u gpu=%u\n",
                        width, height, stride, p, mismatch, out_elems,
                        first / (size_t)width, first % (size_t)width,
                        h_ref[first], h_gpu[first]);
            }
            total_mismatch += mismatch;

            cuMemFree(d_img);
            cuMemFree(d_out);
            free(h_img); free(h_ref); free(h_gpu);
        }
    }

    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    mu_assert("cambi derivative: CUDA diverges from CPU reference",
              total_mismatch == 0);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* CPU reference: decimate() from src/feature/cambi.c, unrolled to a   */
/* separate destination (the CPU version is in-place; see the kernel   */
/* comment for why that is not safe on GPU).                           */
/* ------------------------------------------------------------------ */
static void ref_decimate(const uint16_t *src, uint16_t *dst,
                         int width, int height,
                         ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            dst[i * dst_stride + j] = src[(i << 1) * src_stride + (j << 1)];
}

static char *test_cambi_decimate_cuda(void)
{
    /* SOURCE dimensions. Odd ones matter: the multiscale loop derives the
     * decimated size as (w + 1) >> 1, so a 33-wide source yields 17 columns
     * and output col 16 must read source col 32 -- the last valid one. */
    const int dims[][2] = {
        { 128, 128 }, { 1920, 1080 }, { 1919, 1079 }, { 33, 17 },
        { 2, 32 }, { 32, 2 }, { 1, 1 }, { 7, 5 },
    };
    const int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, cambi_decimate_ptx));
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, "cambi_decimate_kernel"));

    unsigned long total_mismatch = 0;

    for (int d = 0; d < n_dims; d++) {
        const int src_w = dims[d][0], src_h = dims[d][1];
        const int width = (src_w + 1) >> 1;
        const int height = (src_h + 1) >> 1;
        /* Distinct padding on each side: equal strides would hide a bug. */
        const ptrdiff_t src_stride = src_w + 7;
        const ptrdiff_t dst_stride = width + 3;

        for (enum pattern p = 0; p < PAT_COUNT; p++) {
            const size_t src_elems = (size_t)src_stride * src_h;
            const size_t dst_elems = (size_t)dst_stride * height;

            uint16_t *h_src = malloc(src_elems * sizeof(uint16_t));
            uint16_t *h_ref = malloc(dst_elems * sizeof(uint16_t));
            uint16_t *h_gpu = malloc(dst_elems * sizeof(uint16_t));
            if (!h_src || !h_ref || !h_gpu) {
                free(h_src); free(h_ref); free(h_gpu);
                return "allocation failed";
            }
            memset(h_src, 0, src_elems * sizeof(uint16_t));
            memset(h_ref, 0xAB, dst_elems * sizeof(uint16_t));

            fill_pattern(h_src, src_w, src_h, src_stride, p,
                         777u + d * 31u + p);

            ref_decimate(h_src, h_ref, width, height, src_stride, dst_stride);

            CUdeviceptr d_src, d_dst;
            CU_CHECK(cuMemAlloc(&d_src, src_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemAlloc(&d_dst, dst_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemcpyHtoD(d_src, h_src, src_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemsetD8(d_dst, 0xAB, dst_elems * sizeof(uint16_t)));

            int w = width, h = height;
            ptrdiff_t ss = src_stride, ds = dst_stride;
            void *args[] = { &d_src, &d_dst, &w, &h, &ss, &ds };

            const unsigned bx = 32, by = 8;
            CU_CHECK(cuLaunchKernel(kernel,
                                    (width + bx - 1) / bx,
                                    (height + by - 1) / by, 1,
                                    bx, by, 1, 0, NULL, args, NULL));
            CU_CHECK(cuCtxSynchronize());
            CU_CHECK(cuMemcpyDtoH(h_gpu, d_dst, dst_elems * sizeof(uint16_t)));

            unsigned long mismatch = 0;
            size_t first_r = 0, first_c = 0;
            bool have_first = false;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    size_t k = (size_t)i * dst_stride + j;
                    if (h_gpu[k] != h_ref[k]) {
                        if (!have_first) {
                            first_r = i; first_c = j; have_first = true;
                        }
                        mismatch++;
                    }
                }
            }
            if (mismatch) {
                size_t fk = first_r * dst_stride + first_c;
                fprintf(stderr,
                        "\n  src %dx%d -> %dx%d ss=%td ds=%td pattern=%d: "
                        "%lu / %d mismatch, first at (row %zu, col %zu): "
                        "cpu=%u gpu=%u\n",
                        src_w, src_h, width, height, src_stride, dst_stride, p,
                        mismatch, width * height, first_r, first_c,
                        h_ref[fk], h_gpu[fk]);
            }
            total_mismatch += mismatch;

            cuMemFree(d_src);
            cuMemFree(d_dst);
            free(h_src); free(h_ref); free(h_gpu);
        }
    }

    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    mu_assert("cambi decimate: CUDA diverges from CPU reference",
              total_mismatch == 0);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* CPU reference: filter_mode() from src/feature/cambi.c, verbatim     */
/* apart from taking a raw pointer + stride instead of a VmafPicture.  */
/* Deliberately NOT rewritten -- running the real fused ring-buffer    */
/* version is the whole point, since the GPU splits it into two passes */
/* and the equivalence is exactly what is under test.                  */
/* ------------------------------------------------------------------ */
static uint16_t ref_min3(uint16_t a, uint16_t b, uint16_t c) {
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}

static uint16_t ref_mode3(uint16_t a, uint16_t b, uint16_t c) {
    if (a == b || a == c) return a;
    if (b == c) return b;
    return ref_min3(a, b, c);
}

static void ref_filter_mode(uint16_t *data, int width, int height,
                            ptrdiff_t stride, uint16_t *buffer)
{
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        buffer[curr_line * width + 0] = data[i * stride + 0];
        for (int j = 1; j < width - 1; j++) {
            buffer[curr_line * width + j] = ref_mode3(data[i * stride + j - 1],
                                                      data[i * stride + j],
                                                      data[i * stride + j + 1]);
        }
        buffer[curr_line * width + width - 1] = data[i * stride + width - 1];

        if (i > 1) {
            for (int j = 0; j < width; j++) {
                data[(i - 1) * stride + j] = ref_mode3(buffer[0 * width + j],
                                                       buffer[1 * width + j],
                                                       buffer[2 * width + j]);
            }
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

static char *test_cambi_filter_mode_cuda(void)
{
    /* height 1 and 2 matter: the vertical pass never runs below height 3,
     * so the image must come back completely unmodified. width 1 and 2
     * matter for the same reason on the horizontal pass. */
    const int dims[][2] = {
        { 64, 64 }, { 1920, 1080 }, { 1919, 1079 }, { 33, 17 },
        { 1, 16 }, { 16, 1 }, { 16, 2 }, { 2, 16 }, { 3, 3 }, { 1, 1 },
    };
    const int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, cambi_filter_mode_ptx));
    CUfunction k_h, k_v;
    CU_CHECK(cuModuleGetFunction(&k_h, module, "cambi_filter_mode_h_kernel"));
    CU_CHECK(cuModuleGetFunction(&k_v, module, "cambi_filter_mode_v_kernel"));

    unsigned long total_mismatch = 0;

    for (int d = 0; d < n_dims; d++) {
        const int width = dims[d][0], height = dims[d][1];
        const ptrdiff_t stride = width + 7;
        const ptrdiff_t tmp_stride = width + 5;

        for (enum pattern p = 0; p < PAT_COUNT; p++) {
            const size_t elems = (size_t)stride * height;
            const size_t tmp_elems = (size_t)tmp_stride * height;

            uint16_t *h_src = malloc(elems * sizeof(uint16_t));
            uint16_t *h_ref = malloc(elems * sizeof(uint16_t));
            uint16_t *h_gpu = malloc(elems * sizeof(uint16_t));
            uint16_t *scratch = malloc((size_t)3 * width * sizeof(uint16_t));
            if (!h_src || !h_ref || !h_gpu || !scratch) {
                free(h_src); free(h_ref); free(h_gpu); free(scratch);
                return "allocation failed";
            }
            memset(h_src, 0, elems * sizeof(uint16_t));
            memset(scratch, 0, (size_t)3 * width * sizeof(uint16_t));

            fill_pattern(h_src, width, height, stride, p, 4242u + d * 13u + p);

            /* reference runs in place on a copy */
            memcpy(h_ref, h_src, elems * sizeof(uint16_t));
            ref_filter_mode(h_ref, width, height, stride, scratch);

            CUdeviceptr d_src, d_tmp, d_dst;
            CU_CHECK(cuMemAlloc(&d_src, elems * sizeof(uint16_t)));
            CU_CHECK(cuMemAlloc(&d_tmp, tmp_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemAlloc(&d_dst, elems * sizeof(uint16_t)));
            CU_CHECK(cuMemcpyHtoD(d_src, h_src, elems * sizeof(uint16_t)));
            CU_CHECK(cuMemsetD8(d_tmp, 0xAB, tmp_elems * sizeof(uint16_t)));
            CU_CHECK(cuMemsetD8(d_dst, 0xAB, elems * sizeof(uint16_t)));

            int w = width, h = height;
            ptrdiff_t ss = stride, ts = tmp_stride, ds = stride;

            const unsigned bx = 32, by = 8;
            const unsigned gx = (width + bx - 1) / bx;
            const unsigned gy = (height + by - 1) / by;

            void *args_h[] = { &d_src, &d_tmp, &w, &h, &ss, &ts };
            CU_CHECK(cuLaunchKernel(k_h, gx, gy, 1, bx, by, 1,
                                    0, NULL, args_h, NULL));

            void *args_v[] = { &d_src, &d_tmp, &d_dst, &w, &h, &ss, &ts, &ds };
            CU_CHECK(cuLaunchKernel(k_v, gx, gy, 1, bx, by, 1,
                                    0, NULL, args_v, NULL));

            CU_CHECK(cuCtxSynchronize());
            CU_CHECK(cuMemcpyDtoH(h_gpu, d_dst, elems * sizeof(uint16_t)));

            unsigned long mismatch = 0;
            int first_r = 0, first_c = 0;
            bool have_first = false;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    size_t k = (size_t)i * stride + j;
                    if (h_gpu[k] != h_ref[k]) {
                        if (!have_first) {
                            first_r = i; first_c = j; have_first = true;
                        }
                        mismatch++;
                    }
                }
            }
            if (mismatch) {
                size_t fk = (size_t)first_r * stride + first_c;
                fprintf(stderr,
                        "\n  %dx%d stride=%td pattern=%d: %lu / %d mismatch, "
                        "first at (row %d, col %d): cpu=%u gpu=%u\n",
                        width, height, stride, p, mismatch, width * height,
                        first_r, first_c, h_ref[fk], h_gpu[fk]);
            }
            total_mismatch += mismatch;

            cuMemFree(d_src);
            cuMemFree(d_tmp);
            cuMemFree(d_dst);
            free(h_src); free(h_ref); free(h_gpu); free(scratch);
        }
    }

    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    mu_assert("cambi filter_mode: CUDA diverges from CPU reference",
              total_mismatch == 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_cambi_derivative_cuda);
    mu_run_test(test_cambi_decimate_cuda);
    mu_run_test(test_cambi_filter_mode_cuda);
    return NULL;
}
