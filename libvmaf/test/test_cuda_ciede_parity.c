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
 * CPU vs CUDA parity test for the ciede_cuda feature extractor. Unlike the
 * psnr_cuda/ssim_cuda parity test this asserts a small tolerance rather than
 * bit-exactness: ciede is dominated by libm transcendentals, which differ
 * between glibc and CUDA in the low bits, and the device math runs in
 * float32 (the CPU reference truncates every intermediate to float anyway).
 *
 * Uses YUV420P input so the fused nearest-neighbor chroma upsampling in the
 * kernel is exercised against the CPU's scale_chroma_planes. Also checks the
 * identical-frame case, where both implementations must return +inf.
 *
 * Exits with meson's SKIP code (77) when no CUDA device is available so CI
 * without a GPU reports the test as skipped.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"

#define N_FRAMES 4
#define TEST_W 768
#define TEST_H 432

#define CIEDE_EPS 1e-3

static uint32_t lcg_state;

static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 16;
}

// frame N_FRAMES-1 is generated identical (ref == dist) to exercise the
// de00_sum == 0 -> +inf path
static void fill_pictures(VmafPicture *ref, VmafPicture *dist, unsigned bpc,
                          unsigned index)
{
    const unsigned peak = (1 << bpc) - 1;
    const int identical = index == N_FRAMES - 1;
    lcg_state = 54321u + index * 7919u;

    for (unsigned p = 0; p < 3; p++) {
        for (unsigned i = 0; i < ref->h[p]; i++) {
            for (unsigned j = 0; j < ref->w[p]; j++) {
                const int v = (i + j + lcg_next()) % (peak + 1);
                const int noise = identical ? 0 : (int)(lcg_next() % 31) - 15;
                int vd = v + noise;
                if (vd < 0) vd = 0;
                if (vd > (int)peak) vd = peak;
                if (bpc == 8) {
                    ((uint8_t*)ref->data[p])[i * ref->stride[p] + j] = v;
                    ((uint8_t*)dist->data[p])[i * dist->stride[p] + j] = vd;
                } else {
                    ((uint16_t*)ref->data[p])[i * (ref->stride[p] / 2) + j] = v;
                    ((uint16_t*)dist->data[p])[i * (dist->stride[p] / 2) + j] = vd;
                }
            }
        }
    }
}

// returns 0 on success, 1 when CUDA is unavailable (caller should skip)
static int run_pass(int use_cuda, unsigned bpc, double scores[N_FRAMES],
                    char **fail)
{
    int err = 0;
    *fail = NULL;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR,
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

    err = vmaf_use_feature(vmaf, use_cuda ? "ciede_cuda" : "ciede", NULL);
    if (err) { *fail = "problem during vmaf_use_feature"; return 0; }

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
        err = vmaf_feature_score_at_index(vmaf, "ciede2000", &scores[i], i);
        if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
    }

    err = vmaf_close(vmaf);
    if (err) { *fail = "problem during vmaf_close"; return 0; }

    return 0;
}

static char *parity(unsigned bpc)
{
    double cpu[N_FRAMES], gpu[N_FRAMES];
    char *fail = NULL;

    run_pass(0, bpc, cpu, &fail);
    if (fail) return fail;

    if (run_pass(1, bpc, gpu, &fail)) {
        fprintf(stderr, "no CUDA device available, skipping\n");
        exit(77);
    }
    if (fail) return fail;

    // the identical last frame must be +inf on both sides
    mu_assert("cpu identical-frame score must be +inf",
              isinf(cpu[N_FRAMES - 1]) && cpu[N_FRAMES - 1] > 0);
    mu_assert("cuda identical-frame score must be +inf",
              isinf(gpu[N_FRAMES - 1]) && gpu[N_FRAMES - 1] > 0);

    for (unsigned i = 0; i < N_FRAMES - 1; i++) {
        if (fabs(cpu[i] - gpu[i]) > CIEDE_EPS) {
            fprintf(stderr, "mismatch %u bpc, frame %u: cpu=%.9f gpu=%.9f\n",
                    bpc, i, cpu[i], gpu[i]);
            return "cpu/cuda ciede2000 score mismatch";
        }
    }

    return NULL;
}

static char *test_ciede_cuda_parity_8bpc(void)
{
    return parity(8);
}

static char *test_ciede_cuda_parity_10bpc(void)
{
    return parity(10);
}

char *run_tests()
{
    mu_run_test(test_ciede_cuda_parity_8bpc);
    mu_run_test(test_ciede_cuda_parity_10bpc);
    return NULL;
}
