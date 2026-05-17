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
 * T3-15(c) gap-fill — motion3 CPU vs. HIP parity test (ADR-0219).
 *
 * The motion3 post-process is a host-side moving-average derived from
 * motion2, reproduced independently in integer_motion.c (CPU path) and
 * integer_motion_hip.c (HIP path).  No cross-backend assertion existed for
 * the HIP variant before this test; boundary-condition drift in the
 * moving-average formula would silently pollute per-frame motion3 scores.
 *
 * This test allocates a 256x144 YUV420P 8-bpc synthetic fixture,
 * feeds two frames through both extractors, and asserts that
 * VMAF_integer_feature_motion3_score at frame index 1 matches to within
 * 1e-4 (places=4, per ADR-0214 cross-backend gate).
 *
 * Skip behaviour: if vmaf_hip_state_init() fails (no HIP/ROCm driver or
 * no device visible) the test emits "[skip: no HIP device]" and passes
 * cleanly.  This mirrors the pattern used in test_hip_smoke.c,
 * test_cuda_motion3_parity.c, and test_sycl_motion3_parity.c.
 *
 * Reproducer (manual):
 *   tools/vmaf --reference testdata/yuv/ref_256x144_2f.yuv \
 *              --distorted testdata/yuv/ref_256x144_2f.yuv \
 *              --width 256 --height 144 --pixel_format 420 --bitdepth 8 \
 *              --feature motion \
 *              --feature motion_hip
 *   # diff VMAF_integer_feature_motion3_score columns; expected delta < 1e-4.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_hip.h"
#include "libvmaf/picture.h"

/* Test fixture geometry — large enough for the 5-tap Gaussian, small enough
 * for a fast CI run. Motion3 requires >= 2 frames (index 0 and index 1). */
#define FIXTURE_W 256u
#define FIXTURE_H 144u
#define FIXTURE_BPC 8u
#define NUM_FRAMES 2u

/* Tolerance matching ADR-0214 cross-backend gate (places=4 -> 1e-4). */
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

/* ---------------------------------------------------------------------- */
/* CPU path — run the "motion" extractor for NUM_FRAMES frames.           */
/* Returns the motion3_score at frame index 1 via *out_score.            */
/* ---------------------------------------------------------------------- */
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

/* ---------------------------------------------------------------------- */
/* HIP path — run the "motion_hip" extractor for NUM_FRAMES frames.       */
/* Returns the motion3_score at frame index 1 via *out_score.            */
/* Returns a skip sentinel (out_score = NaN) if no HIP device.           */
/* ---------------------------------------------------------------------- */
static char *run_hip_motion3(double *out_score)
{
    *out_score = NAN;
    int err = 0;

    VmafHipState *hip_state = NULL;
    VmafHipConfiguration hip_cfg = {.device_index = -1};
    err = vmaf_hip_state_init(&hip_state, hip_cfg);
    if (err != 0 || hip_state == NULL) {
        /* No HIP/ROCm runtime or no device — caller treats NaN as skip. */
        (void)fprintf(stderr, "[skip: no HIP device] ");
        return NULL;
    }

    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE};
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("HIP: vmaf_init failed", !err);

    err = vmaf_hip_import_state(vmaf, hip_state);
    mu_assert("HIP: vmaf_hip_import_state failed", !err);

    err = vmaf_use_feature(vmaf, "motion_hip", NULL);
    mu_assert("HIP: vmaf_use_feature(motion_hip) failed", !err);

    for (unsigned i = 0; i < NUM_FRAMES; i++) {
        VmafPicture ref, dist;
        err = fill_fixture(&ref, i);
        mu_assert("HIP: fill_fixture(ref) failed", !err);
        err = fill_fixture(&dist, i);
        mu_assert("HIP: fill_fixture(dist) failed", !err);

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("HIP: vmaf_read_pictures failed", !err);
    }

    /* Signal end-of-stream so flush() runs and emits motion3 at index 1. */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("HIP: vmaf_read_pictures(EOS) failed", !err);

    err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", out_score, 1u);
    mu_assert("HIP: vmaf_feature_score_at_index(motion3, idx=1) failed", !err);

    err = vmaf_close(vmaf);
    mu_assert("HIP: vmaf_close failed", !err);

    vmaf_hip_state_free(&hip_state);
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Top-level parity assertion.                                             */
/* ---------------------------------------------------------------------- */
static char *test_motion3_cpu_hip_parity(void)
{
    double cpu_score = 0.0;
    double hip_score = NAN;

    char *msg = run_cpu_motion3(&cpu_score);
    if (msg)
        return msg;

    msg = run_hip_motion3(&hip_score);
    if (msg)
        return msg;

    /* If no HIP device was found, hip_score is NaN — skip the assertion. */
    if (isnan(hip_score))
        return NULL;

    double delta = fabs(cpu_score - hip_score);
    if (delta > PARITY_TOL) {
        (void)fprintf(stderr, "\nmotion3 parity FAIL: cpu=%.8f hip=%.8f delta=%.2e tol=%.2e\n",
                      cpu_score, hip_score, delta, PARITY_TOL);
    }
    mu_assert("motion3 CPU vs. HIP delta exceeds places=4 tolerance (1e-4)", delta <= PARITY_TOL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_motion3_cpu_hip_parity);
    return NULL;
}
