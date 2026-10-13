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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "cuda/common.h"
#include "cuda/picture_cuda.h"
#include "ref.h"

#define FRAMES 33
#define KEYS 4
static const char *keys[KEYS] = {
    "psnr_y", "psnr_cb", "psnr_cr", "float_ssim",
};

static void fill(VmafPicture *ref, VmafPicture *dist, unsigned index)
{
    const unsigned scale = 1U << (ref->bpc - 8);
    for (unsigned p = 0; p < 3; p++)
        for (unsigned y = 0; y < ref->h[p]; y++) {
            uint8_t *r = (uint8_t *)ref->data[p] + y * ref->stride[p];
            uint8_t *d = (uint8_t *)dist->data[p] + y * dist->stride[p];
            for (unsigned x = 0; x < ref->w[p]; x++) {
                const unsigned a = (p ? 48 + (x * 17 + y * 23 + index * 7 + p * 31) % 128
                                      : 48 + index) * scale;
                /* Identical luma makes missing chroma copies conspicuous. */
                const unsigned b = a + (p ? (1 + index % 5) * scale : 0);
                if (ref->bpc == 8) { r[x] = a; d[x] = b; }
                else { ((uint16_t *)r)[x] = a; ((uint16_t *)d)[x] = b; }
            }
        }
}

static int copy_picture(CudaFunctions *f, VmafPicture *src, VmafPicture *dst,
                        int device_source)
{
    for (unsigned p = 0; p < 3; p++) {
        CUDA_MEMCPY2D m = {
            .srcMemoryType = device_source ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST,
            .srcHost = device_source ? NULL : src->data[p],
            .srcDevice = device_source ? (CUdeviceptr)src->data[p] : 0,
            .srcPitch = src->stride[p], .dstMemoryType = CU_MEMORYTYPE_DEVICE,
            .dstDevice = (CUdeviceptr)dst->data[p], .dstPitch = dst->stride[p],
            .WidthInBytes = src->w[p] * (src->bpc > 8 ? 2 : 1),
            .Height = src->h[p],
        };
        if (f->cuMemcpy2D(&m)) return -1;
    }
    return 0;
}

static void CUDAAPI delay_download(void *unused)
{
    (void)unused;
    /* Force a queued device-to-host copy to remain pending while the API
     * dispatches CPU work. This does not synchronize the extraction pipeline. */
    const struct timespec delay = { .tv_nsec = 2000000 };
    nanosleep(&delay, NULL);
}

/* mode: 0=CPU host, 1=CUDA host, 2=CUDA device, 3=mixed device. */
static char *run(unsigned mode, unsigned threads, unsigned bpc,
                 unsigned subsample, double scores[FRAMES][KEYS])
{
    VmafContext *vmaf;
    int err = vmaf_init(&vmaf, (VmafConfiguration){
        .log_level = VMAF_LOG_LEVEL_ERROR, .n_threads = threads,
        .n_subsample = subsample,
    });
    mu_assert("context init failed", !err);
    VmafCudaState *state = NULL;
    VmafPicture source[2] = { 0 };
    VmafCudaCookie cookie = { 0 };
    if (mode) {
        err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){ 0 });
        if (err) { vmaf_close(vmaf); exit(77); }
        err = vmaf_cuda_import_state(vmaf, state);
        mu_assert("CUDA import failed", !err);
        if (mode >= 2) {
            VmafCudaPictureConfiguration cfg = {
                .pic_params = { .w = 64, .h = 48, .bpc = bpc,
                                .pix_fmt = VMAF_PIX_FMT_YUV420P },
                .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
            };
            err = vmaf_cuda_preallocate_pictures(vmaf, cfg);
            mu_assert("device pool initialization failed", !err);
            cookie = (VmafCudaCookie){ .w = 64, .h = 48, .bpc = bpc,
                .pix_fmt = VMAF_PIX_FMT_YUV420P, .state = state };
            for (unsigned i = 0; i < 2; i++) {
                err = vmaf_cuda_picture_alloc(&source[i], &cookie);
                mu_assert("producer allocation failed", !err);
            }
        }
    }
    /* VIF exercises GPU completion; CPU PSNR/SSIM check downloaded pictures. */
    if (mode)
        err = vmaf_use_feature(vmaf, "vif_cuda", NULL);
    if (mode == 0 || mode == 3) {
        err |= vmaf_use_feature(vmaf, "psnr", NULL);
        err |= vmaf_use_feature(vmaf, "float_ssim", NULL);
    }
    mu_assert("feature registration failed", !err);
    for (unsigned i = 0; i < FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        mu_assert("host allocation failed", !err);
        fill(&ref, &dist, i);
        if (mode >= 2) {
            CudaFunctions *f = state->f;
            err = f->cuCtxPushCurrent(state->ctx);
            err |= copy_picture(f, &ref, &source[0], 0);
            err |= copy_picture(f, &dist, &source[1], 0);
            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dist);
            err |= vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
            err |= vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
            /* Like FFmpeg, the producer uses legacy-stream D2D copies.
             * These calls need not wait for the device copies to finish. */
            err |= copy_picture(f, &source[0], &ref, 1);
            err |= copy_picture(f, &source[1], &dist, 1);
            if (mode == 3) {
                err |= f->cuLaunchHostFunc(vmaf_cuda_picture_get_stream(&ref), delay_download, NULL);
                err |= f->cuLaunchHostFunc(vmaf_cuda_picture_get_stream(&dist), delay_download, NULL);
            }
            err |= f->cuCtxPopCurrent(NULL);
            mu_assert("device producer failed", !err);
        }
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("device-input extraction failed", !err);
    }
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("device-input flush failed", !err);
    for (unsigned i = 0; i < FRAMES; i++) {
        if (mode && (subsample <= 1 || !(i % subsample))) {
            double score;
            err = vmaf_feature_score_at_index(vmaf,
                    "VMAF_integer_feature_vif_scale0_score", &score, i);
            mu_assert("missing GPU completion score", !err);
        }
        if (mode == 1 || mode == 2) continue;
        for (unsigned k = 0; k < KEYS; k++) {
            if (k == 3 && subsample > 1 && i % subsample) continue;
            err = vmaf_feature_score_at_index(vmaf, keys[k], &scores[i][k], i);
            mu_assert("missing device-input score", !err);
        }
    }
    if (mode >= 2) {
        for (unsigned i = 0; i < 2; i++)
            vmaf_cuda_picture_free(&source[i], &cookie);
        VmafPicture pic;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &pic);
        mu_assert("could not inspect reused picture", !err);
        const int refs = vmaf_ref_load(pic.ref);
        vmaf_picture_unref(&pic);
        if (refs != 2) {
            fprintf(stderr, "mode=%u threads=%u picture references=%d (expected 2)\n",
                    mode, threads, refs);
            vmaf_close(vmaf);
            free(state);
            return "device input reference leaked during threaded extraction";
        }
    }
    err = vmaf_close(vmaf);
    free(state);
    mu_assert("device-input close failed", !err);
    return NULL;
}

static char *test_device_input(void)
{
    const unsigned modes[] = { 3, 2, 1 };
    for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
        for (unsigned sub = 1; sub <= 3; sub++) {
            double cpu[FRAMES][KEYS], gpu[FRAMES][KEYS];
            char *fail = run(0, 0, bpc, sub, cpu);
            if (fail) return fail;
            for (unsigned t = 0; t <= 2; t += 2)
                for (unsigned m = 0; m < 3; m++) {
                    fail = run(modes[m], t, bpc, sub, gpu);
                    if (fail) return fail;
                    if (modes[m] != 3) continue;
                    for (unsigned i = 0; i < FRAMES; i++)
                        for (unsigned k = 0; k < KEYS; k++) {
                            if (k == 3 && sub > 1 && i % sub) continue;
                            const double eps = 1e-12;
                            if (!isfinite(cpu[i][k]) || !isfinite(gpu[i][k]) ||
                                fabs(cpu[i][k] - gpu[i][k]) > eps) {
                                fprintf(stderr, "mode=%u threads=%u bpc=%u sub=%u frame=%u %s: CPU=%.12f CUDA=%.12f\n",
                                        modes[m], t, bpc, sub, i, keys[k], cpu[i][k], gpu[i][k]);
                                return "device-input score mismatch";
                            }
                        }
                }
        }
    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_device_input);
    return NULL;
}
