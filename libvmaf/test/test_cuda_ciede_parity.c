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

/* Compare CPU/CUDA CIEDE on noisy, inverted and identical frames.
 * Float color conversion requires a fixture-specific tolerance.
 * A missing CUDA device is reported as a Meson SKIP (77). */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"
#include "cuda/common.h"

#define N_FRAMES 5
#define CIEDE_EPS 1e-3

static uint32_t lcg_state;

static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 16;
}

/* Noisy, chroma-only and inverted pairs, then identical texture and black. */
static void fill_pictures(VmafPicture *ref, VmafPicture *dist, unsigned bpc,
                          unsigned index)
{
    const unsigned peak = (1 << bpc) - 1;
    lcg_state = 54321u + index * 7919u;

    for (unsigned p = 0; p < 3; p++) {
        for (unsigned i = 0; i < ref->h[p]; i++) {
            for (unsigned j = 0; j < ref->w[p]; j++) {
                const int v = index == N_FRAMES - 1 ? 0 :
                              (i + j + lcg_next()) % (peak + 1);
                const int noise = (int)(lcg_next() % 31) - 15;
                int vd = v + noise;
                if (vd < 0) vd = 0;
                if (vd > (int)peak) vd = peak;
                if (index == 1) vd = p ? (int)peak - v : v;
                if (index == 2) vd = peak - v;
                if (index >= N_FRAMES - 2) vd = v;
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

/* Return 1 when CUDA is unavailable; report test failures through fail. */
static int run_pass(int use_cuda, enum VmafPixelFormat pix_fmt, unsigned bpc,
                    unsigned w, unsigned h, unsigned index_step,
                    double scores[N_FRAMES], char **fail)
{
    int err = 0;
    *fail = NULL;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR,
        .n_threads = use_cuda ? 2 : 0,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) { *fail = "problem during vmaf_init"; return 0; }

    VmafCudaState *cu_state = NULL;
    if (use_cuda) {
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            vmaf_close(vmaf);
            if (cu_state) cuda_free_functions(&cu_state->f);
            free(cu_state);
            return 1; // no CUDA device, skip
        }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { *fail = "problem during vmaf_cuda_import_state"; return 0; }
    }

    err = vmaf_use_feature(vmaf, use_cuda ? "ciede_cuda" : "ciede", NULL);
    if (err) { *fail = "problem during vmaf_use_feature"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, pix_fmt, bpc, w, h);
        err |= vmaf_picture_alloc(&dist, pix_fmt, bpc, w, h);
        if (err) { *fail = "problem during vmaf_picture_alloc"; return 0; }
        fill_pictures(&ref, &dist, bpc, i);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i * index_step);
        if (err) { *fail = "problem during vmaf_read_pictures"; return 0; }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) { *fail = "problem during vmaf_read_pictures flush"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        err = vmaf_feature_score_at_index(vmaf, "ciede2000", &scores[i],
                                          i * index_step);
        if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
    }

    err = vmaf_close(vmaf);
    if (cu_state) cuda_free_functions(&cu_state->f);
    free(cu_state);
    if (err) { *fail = "problem during vmaf_close"; return 0; }

    return 0;
}

static char *parity(enum VmafPixelFormat pix_fmt, unsigned bpc,
                    unsigned w, unsigned h, unsigned index_step)
{
    double cpu[N_FRAMES], gpu[N_FRAMES];
    char *fail = NULL;

    run_pass(0, pix_fmt, bpc, w, h, index_step, cpu, &fail);
    if (fail) return fail;

    if (run_pass(1, pix_fmt, bpc, w, h, index_step, gpu, &fail)) {
        fprintf(stderr, "no CUDA device available, skipping\n");
        exit(77);
    }
    if (fail) return fail;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        if (i >= N_FRAMES - 2) {
            mu_assert("cpu identical-frame score must be +inf",
                      isinf(cpu[i]) && cpu[i] > 0);
            mu_assert("cuda identical-frame score must be +inf",
                      isinf(gpu[i]) && gpu[i] > 0);
            continue;
        }
        if (!isfinite(cpu[i]) || !isfinite(gpu[i]) ||
            fabs(cpu[i] - gpu[i]) > CIEDE_EPS) {
            fprintf(stderr, "mismatch format %d, %u bpc, %ux%u, frame %u: "
                    "cpu=%.9f gpu=%.9f\n", pix_fmt, bpc, w, h, i,
                    cpu[i], gpu[i]);
            return "cpu/cuda ciede2000 score mismatch";
        }
    }

    return NULL;
}

static char *test_ciede_cuda_parity_8bpc(void)
{
    return parity(VMAF_PIX_FMT_YUV420P, 8, 768, 432, 1);
}

static char *test_ciede_cuda_parity_10bpc(void)
{
    return parity(VMAF_PIX_FMT_YUV420P, 10, 768, 432, 1);
}

static char *test_ciede_cuda_formats(void)
{
    const enum VmafPixelFormat formats[] = {
        VMAF_PIX_FMT_YUV420P, VMAF_PIX_FMT_YUV422P, VMAF_PIX_FMT_YUV444P,
    };
    const unsigned depths[] = { 8, 10, 12, 16 };
    for (unsigned f = 0; f < 3; f++)
        for (unsigned d = 0; d < 4; d++) {
            char *fail = parity(formats[f], depths[d], f == 2 ? 65 : 66,
                                f == 0 ? 50 : 49, 1);
            if (fail) return fail;
        }
    return NULL;
}

static char *test_ciede_cuda_skipped_indices(void)
{
    return parity(VMAF_PIX_FMT_YUV444P, 16, 65, 49, 2);
}

#include "test_cuda_score_error.h"

static char *test_ciede_cuda_score_error(void)
{
    return test_cuda_score_error("ciede_cuda", "ciede2000");
}

char *run_tests()
{
    mu_run_test(test_ciede_cuda_parity_8bpc);
    mu_run_test(test_ciede_cuda_parity_10bpc);
    mu_run_test(test_ciede_cuda_formats);
    mu_run_test(test_ciede_cuda_skipped_indices);
    mu_run_test(test_ciede_cuda_score_error);
    return NULL;
}
