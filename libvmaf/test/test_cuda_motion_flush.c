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

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"

static char *run_motion(unsigned threads, unsigned bpc, unsigned frames,
                        int cuda, int mixed)
{
    VmafContext *vmaf;
    const VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR, .n_threads = threads,
    };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);
    VmafCudaState *state = NULL;
    if (cuda) {
        err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){ 0 });
        if (err) {
            vmaf_close(vmaf);
            fprintf(stderr, "CUDA unavailable, skipping motion flush test\n");
            exit(77);
        }
        err = vmaf_cuda_import_state(vmaf, state);
        mu_assert("vmaf_cuda_import_state failed", !err);
    }
    err = vmaf_use_feature(vmaf, cuda ? "motion_cuda" : "motion", NULL);
    mu_assert("could not register motion", !err);
    if (mixed) {
        err = vmaf_use_feature(vmaf, "psnr", NULL);
        mu_assert("could not register CPU PSNR", !err);
    }
    for (unsigned i = 0; i < frames; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        mu_assert("reference allocation failed", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        mu_assert("distorted allocation failed", !err);
        for (unsigned p = 0; p < 3; p++) {
            for (unsigned y = 0; y < ref.h[p]; y++) {
                uint8_t *r = (uint8_t *)ref.data[p] + y * ref.stride[p];
                uint8_t *d = (uint8_t *)dist.data[p] + y * dist.stride[p];
                for (unsigned x = 0; x < ref.w[p]; x++) {
                    /* Keep this flush test independent of motion ordering. */
                    const unsigned value = 32U << (bpc - 8);
                    if (bpc == 8) { r[x] = value; d[x] = value + 1; }
                    else {
                        ((uint16_t *)r)[x] = value;
                        ((uint16_t *)d)[x] = value + 1;
                    }
                }
            }
        }
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("feature extraction failed", !err);
    }
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) {
        fprintf(stderr, "threads=%u bpc=%u frames=%u cuda=%d mixed=%d flush=%d\n",
                threads, bpc, frames, cuda, mixed, err);
        vmaf_close(vmaf);
        free(state);
        return "motion flush failed (possible duplicate final score)";
    }
    for (unsigned i = 0; i < frames; i++) {
        double score;
        err = vmaf_feature_score_at_index(vmaf,
                "VMAF_integer_feature_motion2_score", &score, i);
        mu_assert("missing motion2 score, including the final frame", !err);
        mu_assert("stationary pictures must have zero motion2", score == 0.0);
        if (mixed) {
            err = vmaf_feature_score_at_index(vmaf, "psnr_y", &score, i);
            mu_assert("missing mixed CPU PSNR score", !err);
            const double peak = (1U << bpc) - 1;
            mu_assert("incorrect mixed CPU PSNR score",
                      isfinite(score) && fabs(score - 10 * log10(peak * peak)) < 1e-12);
        }
    }
    err = vmaf_close(vmaf);
    free(state);
    mu_assert("vmaf_close failed", !err);
    return NULL;
}

static char *test_motion_flush()
{
    const unsigned threads[] = { 0, 1, 4 };
    const unsigned frames[] = { 1, 2, 12 };
    for (unsigned t = 0; t < 3; t++)
        for (unsigned n = 0; n < 3; n++)
            for (unsigned bpc = 8; bpc <= 10; bpc += 2)
                for (int cuda = 0; cuda <= 1; cuda++)
                    for (int mixed = 0; mixed <= 1; mixed++) {
                        char *fail = run_motion(threads[t], bpc, frames[n], cuda, mixed);
                        if (fail) return fail;
                    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_motion_flush);
    return NULL;
}
