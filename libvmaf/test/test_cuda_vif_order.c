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
#include <time.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "cuda/common.h"

static CudaFunctions driver;
static unsigned delayed_resets;

static void CUDAAPI delay(void *unused)
{
    (void)unused;
    const struct timespec interval = { .tv_nsec = 50000000 };
    nanosleep(&interval, NULL);
}

static CUresult CUDAAPI delayed_memset(CUdeviceptr dst, unsigned char value,
                                      size_t size, CUstream stream)
{
    /* Delay the reset on its own stream so unordered kernels can run first. */
    CUresult err = driver.cuLaunchHostFunc(stream, delay, NULL);
    if (err) return err;
    delayed_resets++;
    return driver.cuMemsetD8Async(dst, value, size, stream);
}

static char *run_vif(unsigned bpc)
{
    VmafContext *ctx;
    int err = vmaf_init(&ctx, (VmafConfiguration){
        .log_level = VMAF_LOG_LEVEL_ERROR,
    });
    mu_assert("context init failed", !err);
    VmafCudaState *state = NULL;
    err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){0});
    if (err) {
        vmaf_close(ctx);
        if (state) cuda_free_functions(&state->f);
        free(state);
        exit(77);
    }
    driver = *state->f;
    state->f->cuMemsetD8Async = delayed_memset;
    delayed_resets = 0;
    err = vmaf_cuda_import_state(ctx, state);
    mu_assert("CUDA import failed", !err);
    err = vmaf_use_feature(ctx, "vif_cuda", NULL);
    mu_assert("VIF registration failed", !err);

    const unsigned frames = 4;
    for (unsigned i = 0; i < frames; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        mu_assert("picture allocation failed", !err);
        for (unsigned p = 0; p < 3; p++) {
            for (unsigned y = 0; y < ref.h[p]; y++) {
                uint8_t *r = (uint8_t *)ref.data[p] + y * ref.stride[p];
                uint8_t *d = (uint8_t *)dist.data[p] + y * dist.stride[p];
                const unsigned value = (100 + i) << (bpc - 8);
                for (unsigned x = 0; x < ref.w[p]; x++) {
                    if (bpc == 8) r[x] = d[x] = value;
                    else ((uint16_t *)r)[x] = ((uint16_t *)d)[x] = value;
                }
            }
        }
        err = vmaf_read_pictures(ctx, &ref, &dist, i);
        mu_assert("VIF extraction failed", !err);
    }
    err = vmaf_read_pictures(ctx, NULL, NULL, 0);
    mu_assert("VIF flush failed", !err);
    int mismatch = 0;
    for (unsigned i = 0; i < frames; i++) {
        for (unsigned scale = 0; scale < 4; scale++) {
            char name[64];
            snprintf(name, sizeof(name), "VMAF_integer_feature_vif_scale%u_score",
                     scale);
            double score;
            err = vmaf_feature_score_at_index(ctx, name, &score, i);
            mu_assert("missing VIF score", !err);
            if (!isfinite(score) || fabs(score - 1.0) > 1e-6) {
                fprintf(stderr, "bpc=%u frame=%u scale=%u VIF=%g expected=1\n",
                        bpc, i, scale, score);
                mismatch = 1;
            }
        }
    }
    err = vmaf_close(ctx);
    cuda_free_functions(&state->f);
    free(state);
    mu_assert("VIF close failed", !err);
    mu_assert("reset delay was not exercised", delayed_resets == frames);
    mu_assert("VIF kernels did not wait for their accumulator reset", !mismatch);
    return NULL;
}

static char *test_vif_order(void)
{
    char *fail = run_vif(8);
    return fail ? fail : run_vif(10);
}

char *run_tests(void)
{
    mu_run_test(test_vif_order);
    return NULL;
}
