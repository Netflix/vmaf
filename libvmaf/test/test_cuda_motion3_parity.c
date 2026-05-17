/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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
 * CHUG audit gap-fill — motion3 CPU vs. CUDA parity test.
 *
 * The motion3 post-process is a host-side moving-average derived from
 * motion2, reproduced independently in integer_motion.c (CPU path) and
 * integer_motion_cuda.c (CUDA path). No cross-backend assertion existed
 * before this test; boundary-condition drift in the moving-average formula
 * would silently pollute the CHUG-extracted motion3_mean/std columns.
 *
 * This test allocates a 256x144 YUV420P 8-bpc synthetic fixture,
 * feeds two frames through both extractors, and asserts that
 * VMAF_integer_feature_motion3_score at frame index 1 matches to within
 * 1e-4 (places=4, per ADR-0214 cross-backend gate).
 *
 * Skip behaviour: if vmaf_cuda_state_init() fails (no CUDA driver or no
 * device visible) the test emits "[skip: no CUDA device]" and passes.
 * This mirrors the pattern used in test_cuda_buffer_alloc_oom.c.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"

/* Test fixture geometry — large enough for the 5-tap Gaussian, small enough
 * for a fast CI run. Motion3 requires ≥ 2 frames (index 0 and index 1). */
#define FIXTURE_W 256u
#define FIXTURE_H 144u
#define FIXTURE_BPC 8u
#define NUM_FRAMES 2u

/* Tolerance matching ADR-0214 cross-backend gate (places=4 → 1e-4). */
#define PARITY_TOL 1e-4

/* Fill a YUV420P 8-bpc picture with a deterministic pattern so that
 * frame 0 and frame 1 differ, causing a non-zero motion score. */
static int fill_fixture(VmafPicture *pic, unsigned frame_idx)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, FIXTURE_BPC, FIXTURE_W, FIXTURE_H);
    if (err)
        return err;

    /* Luma plane: simple ramp with frame-dependent offset so successive
     * frames differ. Chroma planes are uniform (motion is luma-only). */
    uint8_t *y = (uint8_t *)pic->data[0];
    for (unsigned row = 0; row < pic->h[0]; row++) {
        for (unsigned col = 0; col < pic->w[0]; col++) {
            y[row * pic->stride[0] + col] = (uint8_t)((row + col + frame_idx * 13u) & 0xFFu);
        }
    }

    for (unsigned p = 1; p < 3; p++) {
        uint8_t *plane = (uint8_t *)pic->data[p];
        for (unsigned row = 0; row < pic->h[p]; row++) {
            memset(plane + row * pic->stride[p], 128, pic->w[p]);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* CPU path — run the "motion" extractor for NUM_FRAMES frames.        */
/* Returns the motion3_score at frame index 1 via *out_score.         */
/* ------------------------------------------------------------------ */
static char *run_cpu_motion3(double *out_score)
{
    int err = 0;

    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE};
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("CPU: vmaf_init failed", !err);

    err = vmaf_use_feature(vmaf, "motion", NULL);
    mu_assert("CPU: vmaf_use_feature(motion) failed", !err);

    for (unsigned i = 0; i < NUM_FRAMES; i++) {
        VmafPicture ref, dist;
        err = fill_fixture(&ref, i);
        mu_assert("CPU: fill_fixture(ref) failed", !err);
        err = fill_fixture(&dist, i);
        mu_assert("CPU: fill_fixture(dist) failed", !err);

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("CPU: vmaf_read_pictures failed", !err);
    }

    /* Signal end-of-stream so flush() runs and emits motion3 at index 1. */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("CPU: vmaf_read_pictures(EOS) failed", !err);

    err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", out_score, 1u);
    mu_assert("CPU: vmaf_feature_score_at_index(motion3, idx=1) failed", !err);

    err = vmaf_close(vmaf);
    mu_assert("CPU: vmaf_close failed", !err);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* CUDA path — run the "motion_cuda" extractor for NUM_FRAMES frames. */
/* Returns the motion3_score at frame index 1 via *out_score.         */
/* Returns a skip sentinel (out_score = NaN) if no CUDA device.      */
/* ------------------------------------------------------------------ */
static char *run_cuda_motion3(double *out_score)
{
    *out_score = NAN;
    int err = 0;

    VmafCudaState *cu_state = NULL;
    VmafCudaConfiguration cuda_cfg = {0};
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    if (err != 0 || cu_state == NULL) {
        /* No CUDA runtime / no device — caller treats NaN as skip. */
        (void)fprintf(stderr, "[skip: no CUDA device] ");
        return NULL;
    }

    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE};
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("CUDA: vmaf_init failed", !err);

    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("CUDA: vmaf_cuda_import_state failed", !err);

    err = vmaf_use_feature(vmaf, "motion_cuda", NULL);
    mu_assert("CUDA: vmaf_use_feature(motion_cuda) failed", !err);

    for (unsigned i = 0; i < NUM_FRAMES; i++) {
        VmafPicture ref, dist;
        err = fill_fixture(&ref, i);
        mu_assert("CUDA: fill_fixture(ref) failed", !err);
        err = fill_fixture(&dist, i);
        mu_assert("CUDA: fill_fixture(dist) failed", !err);

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("CUDA: vmaf_read_pictures failed", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("CUDA: vmaf_read_pictures(EOS) failed", !err);

    err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", out_score, 1u);
    mu_assert("CUDA: vmaf_feature_score_at_index(motion3, idx=1) failed", !err);

    err = vmaf_close(vmaf);
    mu_assert("CUDA: vmaf_close failed", !err);

    err = vmaf_cuda_state_free(cu_state);
    mu_assert("CUDA: vmaf_cuda_state_free failed", !err);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Top-level parity assertion.                                         */
/* ------------------------------------------------------------------ */
static char *test_motion3_cpu_cuda_parity(void)
{
    double cpu_score = 0.0;
    double cuda_score = NAN;

    char *msg = run_cpu_motion3(&cpu_score);
    if (msg)
        return msg;

    msg = run_cuda_motion3(&cuda_score);
    if (msg)
        return msg;

    /* If no CUDA device was found, cuda_score is NaN — skip the assertion. */
    if (isnan(cuda_score))
        return NULL;

    double delta = fabs(cpu_score - cuda_score);
    if (delta > PARITY_TOL) {
        (void)fprintf(stderr, "\nmotion3 parity FAIL: cpu=%.8f cuda=%.8f delta=%.2e tol=%.2e\n",
                      cpu_score, cuda_score, delta, PARITY_TOL);
    }
    mu_assert("motion3 CPU vs. CUDA delta exceeds places=4 tolerance (1e-4)", delta <= PARITY_TOL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_motion3_cpu_cuda_parity);
    return NULL;
}
