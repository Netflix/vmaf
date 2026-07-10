/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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
 * CPU vs CUDA parity test for the psnr_cuda and ssim_cuda feature
 * extractors. Runs the same deterministic synthetic frames through the CPU
 * extractors (psnr, float_ssim) and the CUDA extractors (psnr_cuda,
 * ssim_cuda) and asserts per-frame score equality within a small epsilon.
 *
 * Skips (passes with a notice) when no CUDA device is available so CI
 * without a GPU stays green.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"

#define N_FRAMES 5
// 768x432 makes float_ssim pick decimation scale 2, so the full pipeline
// (decimate + convolve) is exercised, not just the scale=1 path
#define TEST_W 768
#define TEST_H 432

#define PSNR_EPS 1e-9
#define SSIM_EPS 1e-7

static const char *score_keys[] = { "psnr_y", "psnr_cb", "psnr_cr",
                                    "float_ssim" };
#define N_KEYS 4

static uint32_t lcg_state;

static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 16;
}

static void fill_pictures(VmafPicture *ref, VmafPicture *dist, unsigned bpc,
                          unsigned index)
{
    const unsigned peak = (1 << bpc) - 1;
    lcg_state = 12345u + index * 7919u;

    for (unsigned p = 0; p < 3; p++) {
        if (bpc == 8) {
            uint8_t *r = ref->data[p];
            uint8_t *d = dist->data[p];
            for (unsigned i = 0; i < ref->h[p]; i++) {
                for (unsigned j = 0; j < ref->w[p]; j++) {
                    const int v = (i + j + lcg_next()) % (peak + 1);
                    const int noise = (int)(lcg_next() % 15) - 7;
                    int vd = v + noise;
                    if (vd < 0) vd = 0;
                    if (vd > (int)peak) vd = peak;
                    r[j] = v;
                    d[j] = vd;
                }
                r += ref->stride[p];
                d += dist->stride[p];
            }
        } else {
            uint16_t *r = ref->data[p];
            uint16_t *d = dist->data[p];
            for (unsigned i = 0; i < ref->h[p]; i++) {
                for (unsigned j = 0; j < ref->w[p]; j++) {
                    const int v = (i + j + lcg_next()) % (peak + 1);
                    const int noise = (int)(lcg_next() % 61) - 30;
                    int vd = v + noise;
                    if (vd < 0) vd = 0;
                    if (vd > (int)peak) vd = peak;
                    r[j] = v;
                    d[j] = vd;
                }
                r += ref->stride[p] / 2;
                d += dist->stride[p] / 2;
            }
        }
    }
}

// returns 0 on success, 1 when CUDA is unavailable (caller should skip)
static int run_pass(int use_cuda, unsigned bpc,
                    double scores[N_FRAMES][N_KEYS], char **fail)
{
    int err = 0;
    *fail = NULL;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR,
        .n_threads = use_cuda ? 2 : 0, // threads + CUDA: the double-flush path
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) { *fail = "problem during vmaf_init"; return 0; }

    if (use_cuda) {
        VmafCudaState *cu_state;
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            vmaf_close(vmaf);
            return 1; // no CUDA device, skip
        }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { *fail = "problem during vmaf_cuda_import_state"; return 0; }
    }

    // enable_apsnr makes the temporal flush emit aggregates; combined with
    // n_threads it regression-tests flush idempotency under the double-flush
    VmafFeatureDictionary *dict = NULL;
    err = vmaf_feature_dictionary_set(&dict, "enable_apsnr", "true");
    if (err) { *fail = "problem during vmaf_feature_dictionary_set"; return 0; }

    err = vmaf_use_feature(vmaf, use_cuda ? "psnr_cuda" : "psnr", dict);
    if (err) { *fail = "problem during vmaf_use_feature psnr"; return 0; }
    err = vmaf_use_feature(vmaf, use_cuda ? "ssim_cuda" : "float_ssim", NULL);
    if (err) { *fail = "problem during vmaf_use_feature ssim"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc,
                                 TEST_W, TEST_H);
        err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc,
                                  TEST_W, TEST_H);
        if (err) { *fail = "problem during vmaf_picture_alloc"; return 0; }
        fill_pictures(&ref, &dist, bpc, i);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) { *fail = "problem during vmaf_read_pictures"; return 0; }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) { *fail = "problem during vmaf_read_pictures flush"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            err = vmaf_feature_score_at_index(vmaf, score_keys[k],
                                              &scores[i][k], i);
            if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
        }
    }

    err = vmaf_close(vmaf);
    if (err) { *fail = "problem during vmaf_close"; return 0; }

    return 0;
}

static char *parity(unsigned bpc)
{
    double cpu[N_FRAMES][N_KEYS], gpu[N_FRAMES][N_KEYS];
    char *fail = NULL;

    run_pass(0, bpc, cpu, &fail);
    if (fail) return fail;

    if (run_pass(1, bpc, gpu, &fail)) {
        fprintf(stderr, "no CUDA device available, skipping\n");
        return NULL;
    }
    if (fail) return fail;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            const double eps = k < 3 ? PSNR_EPS : SSIM_EPS;
            if (fabs(cpu[i][k] - gpu[i][k]) > eps) {
                fprintf(stderr, "mismatch %u bpc, frame %u, %s: "
                        "cpu=%.9f gpu=%.9f\n", bpc, i, score_keys[k],
                        cpu[i][k], gpu[i][k]);
                return "cpu/cuda score mismatch";
            }
        }
    }

    return NULL;
}

static char *test_psnr_ssim_cuda_parity_8bpc(void)
{
    return parity(8);
}

static char *test_psnr_ssim_cuda_parity_10bpc(void)
{
    return parity(10);
}

char *run_tests()
{
    mu_run_test(test_psnr_ssim_cuda_parity_8bpc);
    mu_run_test(test_psnr_ssim_cuda_parity_10bpc);
    return NULL;
}
