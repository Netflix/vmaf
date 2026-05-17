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
 * T3-15(c) gap-fill — motion3 CPU vs. Vulkan parity test (ADR-0219).
 *
 * The motion3 post-process is a host-side moving-average derived from
 * motion2, reproduced independently in integer_motion.c (CPU path) and
 * motion_vulkan.c (Vulkan path).  No cross-backend assertion existed for
 * the Vulkan variant before this test; boundary-condition drift in the
 * moving-average formula would silently pollute per-frame motion3 scores.
 *
 * This test allocates a 256x144 YUV420P 8-bpc synthetic fixture,
 * feeds two frames through both extractors, and asserts that
 * VMAF_integer_feature_motion3_score at frame index 1 matches to within
 * 1e-4 (places=4, per ADR-0214 cross-backend gate).
 *
 * Skip behaviour: if vmaf_vulkan_state_init() fails (no Vulkan compute
 * device visible) the test emits "[skip: no Vulkan device]" and passes
 * cleanly.  This mirrors the pattern used in test_vulkan_smoke.c.
 *
 * Companion tests: PR #922 (CUDA), PR #927 (SYCL).
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_vulkan.h"
#include "libvmaf/picture.h"

/* Test fixture geometry — large enough for the 5-tap Gaussian blur
 * (requires at least 3 px in each dimension), small enough for a fast
 * CI run.  motion3 requires at least 2 frames (index 0 and index 1). */
#define FIXTURE_W 256u
#define FIXTURE_H 144u
#define FIXTURE_BPC 8u
#define NUM_FRAMES 2u

/* Tolerance matching ADR-0214 cross-backend gate (places=4 → 1e-4). */
#define PARITY_TOL 1e-4

/* Fill a YUV420P 8-bpc picture with a deterministic pattern so that
 * frame 0 and frame 1 differ, causing a non-zero motion score.  Chroma
 * planes are set to mid-grey (128) because motion is luma-only. */
static int fill_fixture(VmafPicture *pic, unsigned frame_idx)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, FIXTURE_BPC, FIXTURE_W, FIXTURE_H);
    if (err)
        return err;

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
    int err;

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
/* Vulkan path — run the "motion_vulkan" extractor for NUM_FRAMES.    */
/* Returns motion3_score at frame index 1 via *out_score.             */
/* Sets *out_score to NaN and returns NULL (success) if no Vulkan      */
/* compute device is present.                                          */
/* ------------------------------------------------------------------ */
static char *run_vulkan_motion3(double *out_score)
{
    *out_score = NAN;
    int err;

    if (!vmaf_vulkan_available())
        return NULL;

    VmafVulkanState *vk_state = NULL;
    VmafVulkanConfiguration vk_cfg = {.device_index = -1, .enable_validation = 0};
    err = vmaf_vulkan_state_init(&vk_state, vk_cfg);
    if (err != 0 || vk_state == NULL) {
        /* No Vulkan GPU visible — caller treats NaN as skip. */
        (void)fprintf(stderr, "[skip: no Vulkan device] ");
        return NULL;
    }

    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE};
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("Vulkan: vmaf_init failed", !err);

    err = vmaf_vulkan_import_state(vmaf, vk_state);
    mu_assert("Vulkan: vmaf_vulkan_import_state failed", !err);

    err = vmaf_use_feature(vmaf, "motion_vulkan", NULL);
    mu_assert("Vulkan: vmaf_use_feature(motion_vulkan) failed", !err);

    for (unsigned i = 0; i < NUM_FRAMES; i++) {
        VmafPicture ref, dist;
        err = fill_fixture(&ref, i);
        mu_assert("Vulkan: fill_fixture(ref) failed", !err);
        err = fill_fixture(&dist, i);
        mu_assert("Vulkan: fill_fixture(dist) failed", !err);

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("Vulkan: vmaf_read_pictures failed", !err);
    }

    /* Signal end-of-stream so flush() runs and emits motion3 at index 1. */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("Vulkan: vmaf_read_pictures(EOS) failed", !err);

    err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", out_score, 1u);
    mu_assert("Vulkan: vmaf_feature_score_at_index(motion3, idx=1) failed", !err);

    err = vmaf_close(vmaf);
    mu_assert("Vulkan: vmaf_close failed", !err);

    vmaf_vulkan_state_free(&vk_state);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Top-level parity assertion.                                         */
/* ------------------------------------------------------------------ */
static char *test_motion3_cpu_vulkan_parity(void)
{
    double cpu_score = 0.0;
    double vk_score = NAN;

    char *msg = run_cpu_motion3(&cpu_score);
    if (msg)
        return msg;

    msg = run_vulkan_motion3(&vk_score);
    if (msg)
        return msg;

    /* If no Vulkan device was found, vk_score is NaN — skip the assertion. */
    if (isnan(vk_score))
        return NULL;

    double delta = fabs(cpu_score - vk_score);
    if (delta > PARITY_TOL) {
        (void)fprintf(stderr,
                      "\nmotion3 parity FAIL: cpu=%.8f vulkan=%.8f "
                      "delta=%.2e tol=%.2e\n",
                      cpu_score, vk_score, delta, PARITY_TOL);
    }
    mu_assert("motion3 CPU vs. Vulkan delta exceeds places=4 tolerance (1e-4)",
              delta <= PARITY_TOL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_motion3_cpu_vulkan_parity);
    return NULL;
}
