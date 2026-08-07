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

char *run_tests(void)
{
    mu_run_test(test_cambi_derivative_cuda);
    return NULL;
}
