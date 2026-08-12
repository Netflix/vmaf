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

#include <ffnvcodec/dynlink_cuda.h>
#include <ffnvcodec/dynlink_loader.h>

#include "test.h"
#include "cuda/speed_cuda.h"

/* libvmaf reaches the driver API through the ffnvcodec dynlink loader rather
 * than linking the CUDA toolkit, so a build machine needs only
 * nv-codec-headers. CU_CHECK takes a bare driver call and dispatches it
 * through the loaded table, mirroring CHECK_CUDA in cuda/cuda_helper.cuh. */
static CudaFunctions *g_cu_f = NULL;

static int speed_test_cuda_load(void)
{
    if (g_cu_f) return 0;
    return cuda_load_functions(&g_cu_f, NULL);
}

#define CU_CHECK(call)                                                    \
    do {                                                                  \
        CUresult _e = g_cu_f->call;                                       \
        if (_e != CUDA_SUCCESS) {                                         \
            const char *_n = NULL;                                        \
            g_cu_f->cuGetErrorName(_e, &_n);                              \
            fprintf(stderr, "\n  %s failed: %s\n", #call, _n ? _n : "?"); \
            return "cuda driver call failed";                             \
        }                                                                 \
    } while (0)

/* ------------------------------------------------------------------ */
/* CPU references, copied from the scalar paths in vif_tools.c and     */
/* speed.c. vif_filter1d_s dispatches to convolution_f32_avx_s on x86  */
/* with AVX2; these kernels target the SCALAR reference, so that is    */
/* what we compare against. Whether the AVX2 path agrees with scalar   */
/* bit-for-bit is a separate question -- see test_avx2_vs_scalar.      */
/* ------------------------------------------------------------------ */
static void ref_filter1d(const float *f, const float *src, float *dst,
                         int w, int h, ptrdiff_t src_px, ptrdiff_t dst_px,
                         int fwidth)
{
    float *tmp = malloc((size_t) w * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float accum = 0;
            for (int fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 2 : ii);
                accum += f[fi] * src[ii * src_px + j];
            }
            tmp[j] = accum;
        }
        for (int j = 0; j < w; ++j) {
            float accum = 0;
            for (int fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 2 : jj);
                accum += f[fj] * tmp[jj];
            }
            dst[i * dst_px + j] = accum;
        }
    }
    free(tmp);
}

static void ref_dec16(const float *src, float *dst, int src_w, int src_h,
                      ptrdiff_t src_px, ptrdiff_t dst_px)
{
    for (int i = 0; i < src_h / 16; ++i)
        for (int j = 0; j < src_w / 16; ++j)
            dst[i * dst_px + j] = src[(i * 16) * src_px + (j * 16)];
}

static void ref_subtract(float *im1, const float *im2, int w, int h,
                         ptrdiff_t px)
{
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            im1[i * px + j] -= im2[i * px + j];
}

/* ------------------------------------------------------------------ */
/* Test data. Filter taps are drawn to sum near 1 like a real Gaussian */
/* kernel; image values span a wide dynamic range so cancellation in   */
/* the accumulation is exercised rather than avoided.                  */
/* ------------------------------------------------------------------ */
static void fill_image(float *buf, int w, int h, ptrdiff_t px, unsigned seed)
{
    unsigned s = seed ? seed : 1;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            s = s * 1103515245u + 12345u;
            /* mix of a smooth ramp and noise, centred near zero like the
             * -128 offset picture_copy applies */
            float ramp = (float)((i * 7 + j * 3) % 256) - 128.0f;
            float noise = (float)((int)((s >> 16) & 0xFF) - 128) * 0.25f;
            buf[i * px + j] = ramp + noise;
        }
    }
}

static void fill_filter(float *f, int fwidth, unsigned seed)
{
    unsigned s = seed ? seed : 1;
    float sum = 0;
    for (int k = 0; k < fwidth; k++) {
        s = s * 1103515245u + 12345u;
        f[k] = (float)((s >> 20) & 0x3FF) + 1.0f;
        sum += f[k];
    }
    for (int k = 0; k < fwidth; k++)
        f[k] /= sum;
}

static unsigned long diff_count(const float *a, const float *b,
                                int w, int h, ptrdiff_t px, size_t *first)
{
    unsigned long n = 0;
    *first = (size_t) -1;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            size_t k = (size_t) i * px + j;
            if (memcmp(&a[k], &b[k], sizeof(float)) != 0) {
                if (*first == (size_t) -1) *first = k;
                n++;
            }
        }
    }
    return n;
}

/* ------------------------------------------------------------------ */
static char *test_speed_filter1d_cuda(void)
{
    const struct { int w, h, fwidth; } cases[] = {
        { 256, 128,  5 }, { 256, 128, 17 }, { 129,  65,  9 },
        {  64,  64, 33 }, { 480, 270,  7 }, {  17,  17, 15 },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    if (speed_test_cuda_load() < 0)
        return "could not load the CUDA driver API (is a driver installed?)";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxPushCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, speed_filter_ptx));
    CUfunction k_v, k_h;
    CU_CHECK(cuModuleGetFunction(&k_v, module, "speed_filter1d_v_kernel"));
    CU_CHECK(cuModuleGetFunction(&k_h, module, "speed_filter1d_h_kernel"));

    unsigned long total = 0;

    for (int c = 0; c < n_cases; c++) {
        const int w = cases[c].w, h = cases[c].h, fwidth = cases[c].fwidth;
        /* padded stride: equal stride and width would hide a stride bug */
        const ptrdiff_t px = w + 5;
        const size_t elems = (size_t) px * h;

        float *h_src = malloc(elems * sizeof(float));
        float *h_ref = malloc(elems * sizeof(float));
        float *h_gpu = malloc(elems * sizeof(float));
        float *filt = malloc((size_t) fwidth * sizeof(float));
        if (!h_src || !h_ref || !h_gpu || !filt) {
            free(h_src); free(h_ref); free(h_gpu); free(filt);
            return "allocation failed";
        }
        memset(h_src, 0, elems * sizeof(float));
        memset(h_ref, 0, elems * sizeof(float));
        fill_image(h_src, w, h, px, 4242u + c);
        fill_filter(filt, fwidth, 77u + c);

        ref_filter1d(filt, h_src, h_ref, w, h, px, px, fwidth);

        CUdeviceptr d_src, d_tmp, d_dst, d_f;
        CU_CHECK(cuMemAlloc(&d_src, elems * sizeof(float)));
        CU_CHECK(cuMemAlloc(&d_tmp, elems * sizeof(float)));
        CU_CHECK(cuMemAlloc(&d_dst, elems * sizeof(float)));
        CU_CHECK(cuMemAlloc(&d_f, (size_t) fwidth * sizeof(float)));
        CU_CHECK(cuMemcpyHtoDAsync(d_src, h_src, elems * sizeof(float), 0));
        CU_CHECK(cuMemcpyHtoDAsync(d_f, filt, (size_t) fwidth * sizeof(float), 0));
        CU_CHECK(cuMemsetD8Async(d_tmp, 0xAB, elems * sizeof(float), 0));
        CU_CHECK(cuMemsetD8Async(d_dst, 0xAB, elems * sizeof(float), 0));

        int wi = w, hi = h, fw = fwidth;
        ptrdiff_t sp = px, tp = px, dp = px;
        void *av[] = { &d_f, &d_src, &d_tmp, &wi, &hi, &sp, &tp, &fw };
        void *ah[] = { &d_f, &d_tmp, &d_dst, &wi, &hi, &tp, &dp, &fw };

        const unsigned bx = 32, by = 8;
        const unsigned gx = (w + bx - 1) / bx, gy = (h + by - 1) / by;
        CU_CHECK(cuLaunchKernel(k_v, gx, gy, 1, bx, by, 1, 0, 0, av, NULL));
        CU_CHECK(cuLaunchKernel(k_h, gx, gy, 1, bx, by, 1, 0, 0, ah, NULL));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuMemcpyDtoHAsync(h_gpu, d_dst, elems * sizeof(float), 0));
        CU_CHECK(cuCtxSynchronize());

        size_t first;
        unsigned long n = diff_count(h_ref, h_gpu, w, h, px, &first);
        if (n) {
            fprintf(stderr,
                    "\n  %dx%d fwidth=%d: %lu / %d differ, first at "
                    "(row %zu, col %zu): cpu=%.9g gpu=%.9g\n",
                    w, h, fwidth, n, w * h,
                    first / (size_t) px, first % (size_t) px,
                    h_ref[first], h_gpu[first]);
        }
        total += n;

        g_cu_f->cuMemFree(d_src); g_cu_f->cuMemFree(d_tmp);
        g_cu_f->cuMemFree(d_dst); g_cu_f->cuMemFree(d_f);
        free(h_src); free(h_ref); free(h_gpu); free(filt);
    }

    g_cu_f->cuCtxPopCurrent(NULL);
    g_cu_f->cuDevicePrimaryCtxRelease(dev);

    mu_assert("speed filter1d: CUDA diverges from the scalar CPU reference",
              total == 0);
    return NULL;
}

/* ------------------------------------------------------------------ */
static char *test_speed_dec16_cuda(void)
{
    /* dimensions that are and are not multiples of 16, so the src_h/16
     * truncation is exercised */
    const struct { int w, h; } cases[] = {
        { 256, 256 }, { 1920, 1080 }, { 255, 129 }, { 16, 16 }, { 31, 47 },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    if (speed_test_cuda_load() < 0)
        return "could not load the CUDA driver API";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxPushCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, speed_filter_ptx));
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, "speed_dec16_kernel"));

    unsigned long total = 0;

    for (int c = 0; c < n_cases; c++) {
        const int sw = cases[c].w, sh = cases[c].h;
        const int dw = sw / 16, dh = sh / 16;
        if (dw <= 0 || dh <= 0) continue;
        const ptrdiff_t spx = sw + 7, dpx = dw + 3;
        const size_t se = (size_t) spx * sh, de = (size_t) dpx * (dh ? dh : 1);

        float *h_src = malloc(se * sizeof(float));
        float *h_ref = malloc(de * sizeof(float));
        float *h_gpu = malloc(de * sizeof(float));
        if (!h_src || !h_ref || !h_gpu) {
            free(h_src); free(h_ref); free(h_gpu);
            return "allocation failed";
        }
        memset(h_src, 0, se * sizeof(float));
        memset(h_ref, 0, de * sizeof(float));
        fill_image(h_src, sw, sh, spx, 909u + c);

        ref_dec16(h_src, h_ref, sw, sh, spx, dpx);

        CUdeviceptr d_src, d_dst;
        CU_CHECK(cuMemAlloc(&d_src, se * sizeof(float)));
        CU_CHECK(cuMemAlloc(&d_dst, de * sizeof(float)));
        CU_CHECK(cuMemcpyHtoDAsync(d_src, h_src, se * sizeof(float), 0));
        CU_CHECK(cuMemsetD8Async(d_dst, 0xAB, de * sizeof(float), 0));

        int swi = sw, shi = sh;
        ptrdiff_t sp = spx, dp = dpx;
        void *args[] = { &d_src, &d_dst, &swi, &shi, &sp, &dp };
        const unsigned bx = 32, by = 8;
        CU_CHECK(cuLaunchKernel(kernel, (dw + bx - 1) / bx, (dh + by - 1) / by,
                                1, bx, by, 1, 0, 0, args, NULL));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuMemcpyDtoHAsync(h_gpu, d_dst, de * sizeof(float), 0));
        CU_CHECK(cuCtxSynchronize());

        size_t first;
        unsigned long n = diff_count(h_ref, h_gpu, dw, dh, dpx, &first);
        if (n)
            fprintf(stderr, "\n  src %dx%d -> %dx%d: %lu differ\n",
                    sw, sh, dw, dh, n);
        total += n;

        g_cu_f->cuMemFree(d_src); g_cu_f->cuMemFree(d_dst);
        free(h_src); free(h_ref); free(h_gpu);
    }

    g_cu_f->cuCtxPopCurrent(NULL);
    g_cu_f->cuDevicePrimaryCtxRelease(dev);

    mu_assert("speed dec16: CUDA diverges from CPU", total == 0);
    return NULL;
}

/* ------------------------------------------------------------------ */
static char *test_speed_subtract_cuda(void)
{
    const struct { int w, h; } cases[] = {
        { 256, 128 }, { 1920, 1080 }, { 33, 17 }, { 1, 1 },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    if (speed_test_cuda_load() < 0)
        return "could not load the CUDA driver API";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxPushCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, speed_filter_ptx));
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, "speed_subtract_kernel"));

    unsigned long total = 0;

    for (int c = 0; c < n_cases; c++) {
        const int w = cases[c].w, h = cases[c].h;
        const ptrdiff_t px = w + 5;
        const size_t elems = (size_t) px * h;

        float *a = malloc(elems * sizeof(float));
        float *b = malloc(elems * sizeof(float));
        float *ref = malloc(elems * sizeof(float));
        float *gpu = malloc(elems * sizeof(float));
        if (!a || !b || !ref || !gpu) {
            free(a); free(b); free(ref); free(gpu);
            return "allocation failed";
        }
        memset(a, 0, elems * sizeof(float));
        memset(b, 0, elems * sizeof(float));
        fill_image(a, w, h, px, 31u + c);
        fill_image(b, w, h, px, 991u + c);
        memcpy(ref, a, elems * sizeof(float));
        ref_subtract(ref, b, w, h, px);

        CUdeviceptr d_a, d_b;
        CU_CHECK(cuMemAlloc(&d_a, elems * sizeof(float)));
        CU_CHECK(cuMemAlloc(&d_b, elems * sizeof(float)));
        CU_CHECK(cuMemcpyHtoDAsync(d_a, a, elems * sizeof(float), 0));
        CU_CHECK(cuMemcpyHtoDAsync(d_b, b, elems * sizeof(float), 0));

        int wi = w, hi = h;
        ptrdiff_t sp = px;
        void *args[] = { &d_a, &d_b, &wi, &hi, &sp };
        const unsigned bx = 32, by = 8;
        CU_CHECK(cuLaunchKernel(kernel, (w + bx - 1) / bx, (h + by - 1) / by,
                                1, bx, by, 1, 0, 0, args, NULL));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuMemcpyDtoHAsync(gpu, d_a, elems * sizeof(float), 0));
        CU_CHECK(cuCtxSynchronize());

        size_t first;
        unsigned long n = diff_count(ref, gpu, w, h, px, &first);
        if (n)
            fprintf(stderr, "\n  %dx%d: %lu differ\n", w, h, n);
        total += n;

        g_cu_f->cuMemFree(d_a); g_cu_f->cuMemFree(d_b);
        free(a); free(b); free(ref); free(gpu);
    }

    g_cu_f->cuCtxPopCurrent(NULL);
    g_cu_f->cuDevicePrimaryCtxRelease(dev);

    mu_assert("speed subtract_image: CUDA diverges from CPU", total == 0);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* picture_copy: plane -> float with a constant offset. SpEED uses -128. */
static char *test_speed_picture_copy_cuda(void)
{
    const struct { int w, h, bpc; } cases[] = {
        { 256, 128,  8 }, { 960, 540,  8 }, { 129, 65,  8 },
        { 256, 128, 10 }, { 960, 540, 12 }, { 129, 65, 16 },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    if (speed_test_cuda_load() < 0)
        return "could not load the CUDA driver API";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxPushCurrent(ctx));

    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, speed_filter_ptx));
    CUfunction k8, k16;
    CU_CHECK(cuModuleGetFunction(&k8, module, "speed_picture_copy_u8_kernel"));
    CU_CHECK(cuModuleGetFunction(&k16, module, "speed_picture_copy_u16_kernel"));

    unsigned long total = 0;
    const int offset = -128;

    for (int c = 0; c < n_cases; c++) {
        const int w = cases[c].w, h = cases[c].h, bpc = cases[c].bpc;
        const float scaler = (bpc == 10) ? 4.0f : (bpc == 12) ? 16.0f : 256.0f;
        const ptrdiff_t spx = w + 9, dpx = w + 5;
        const size_t se = (size_t) spx * h, de = (size_t) dpx * h;

        unsigned char *s8 = NULL;
        uint16_t *s16 = NULL;
        float *ref = malloc(de * sizeof(float));
        float *gpu = malloc(de * sizeof(float));
        if (!ref || !gpu) { free(ref); free(gpu); return "allocation failed"; }
        memset(ref, 0, de * sizeof(float));

        unsigned st = 1234u + c;
        if (bpc == 8) {
            s8 = malloc(se);
            if (!s8) { free(ref); free(gpu); return "allocation failed"; }
            memset(s8, 0, se);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++) {
                    st = st * 1103515245u + 12345u;
                    s8[i * spx + j] = (unsigned char)((st >> 16) & 0xFF);
                }
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    ref[i * dpx + j] = (float) s8[i * spx + j] + offset;
        } else {
            const unsigned mask = (1u << bpc) - 1u;
            s16 = malloc(se * sizeof(uint16_t));
            if (!s16) { free(ref); free(gpu); return "allocation failed"; }
            memset(s16, 0, se * sizeof(uint16_t));
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++) {
                    st = st * 1103515245u + 12345u;
                    s16[i * spx + j] = (uint16_t)((st >> 8) & mask);
                }
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    ref[i * dpx + j] =
                        (float) s16[i * spx + j] / scaler + offset;
        }

        CUdeviceptr d_src, d_dst;
        const size_t sbytes = (bpc == 8) ? se : se * sizeof(uint16_t);
        CU_CHECK(cuMemAlloc(&d_src, sbytes));
        CU_CHECK(cuMemAlloc(&d_dst, de * sizeof(float)));
        CU_CHECK(cuMemcpyHtoDAsync(d_src, (bpc == 8) ? (void *) s8 : (void *) s16,
                                   sbytes, 0));
        CU_CHECK(cuMemsetD8Async(d_dst, 0xAB, de * sizeof(float), 0));

        int wi = w, hi = h, off = offset;
        ptrdiff_t sp = spx, dp = dpx;
        float sc = scaler;
        const unsigned bx = 32, by = 8;
        if (bpc == 8) {
            void *a[] = { &d_src, &d_dst, &wi, &hi, &sp, &dp, &off };
            CU_CHECK(cuLaunchKernel(k8, (w + bx - 1) / bx, (h + by - 1) / by,
                                    1, bx, by, 1, 0, 0, a, NULL));
        } else {
            void *a[] = { &d_src, &d_dst, &wi, &hi, &sp, &dp, &off, &sc };
            CU_CHECK(cuLaunchKernel(k16, (w + bx - 1) / bx, (h + by - 1) / by,
                                    1, bx, by, 1, 0, 0, a, NULL));
        }
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuMemcpyDtoHAsync(gpu, d_dst, de * sizeof(float), 0));
        CU_CHECK(cuCtxSynchronize());

        size_t first;
        unsigned long n = diff_count(ref, gpu, w, h, dpx, &first);
        if (n)
            fprintf(stderr, "\n  %dx%d bpc=%d: %lu differ, first cpu=%.9g gpu=%.9g\n",
                    w, h, bpc, n, ref[first], gpu[first]);
        total += n;

        g_cu_f->cuMemFree(d_src); g_cu_f->cuMemFree(d_dst);
        free(s8); free(s16); free(ref); free(gpu);
    }

    g_cu_f->cuCtxPopCurrent(NULL);
    g_cu_f->cuDevicePrimaryCtxRelease(dev);

    mu_assert("speed picture_copy: CUDA diverges from CPU", total == 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_speed_filter1d_cuda);
    mu_run_test(test_speed_dec16_cuda);
    mu_run_test(test_speed_subtract_cuda);
    mu_run_test(test_speed_picture_copy_cuda);
    return NULL;
}
