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
 * Unit test: integer_vif vif_skip_scale0 option (CPU path).
 *
 * Gap closed: the audit at .workingdir/audit-test-coverage-2026-05-16.md §1
 * noted that GPU backends (CUDA, SYCL) do not expose vif_skip_scale0,
 * meaning their output for the scale0 score diverges from the CPU path when
 * a model requests this option.  The Python golden tests
 * (feature_extractor_test.py) cover the CPU path via bindings but there is
 * no C-level assertion that the option is plumbed correctly.
 *
 * This test closes the C-unit-test gap by verifying via the public API:
 *   1. vif_skip_scale0=true causes the scale0 score to be exactly 0.0
 *      (CPU reference behaviour, integer_vif.c:714).
 *   2. Without the option the scale0 score is a finite positive value,
 *      confirming the default=false path is exercised too.
 *
 * Feature-naming note: when vif_skip_scale0 is set the option system
 * appends the option alias "_ssclz" to each provided feature name.
 * The alias of "VMAF_integer_feature_vif_scale0_score" is
 * "integer_vif_scale0" (alias.c:109), so the score is stored under
 * "integer_vif_scale0_ssclz" when the option is active.
 *
 * The assertions establish the CPU ground truth that GPU backends must
 * match once vif_skip_scale0 is ported to CUDA/SYCL/Vulkan.
 *
 * No GPU hardware required; runs in the fast suite.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "libvmaf/feature.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

#include "test.h"

/* Minimum frame size that satisfies VIF's 4-level pyramid (2^3 = 8 pixel
 * minimum along each axis after three halvings).  64x64 divides cleanly
 * and is small enough to keep the test fast. */
#define TEST_W 64u
#define TEST_H 64u

/* Fill a YUV420P 8-bpc picture with a flat luma value and neutral chroma. */
static int alloc_flat_pic(VmafPicture *pic, uint8_t luma_val)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, 8, TEST_W, TEST_H);
    if (err)
        return err;
    for (unsigned p = 0; p < 3; p++) {
        unsigned pw = (p == 0) ? TEST_W : TEST_W / 2;
        unsigned ph = (p == 0) ? TEST_H : TEST_H / 2;
        uint8_t val = (p == 0) ? luma_val : 128u;
        uint8_t *plane = (uint8_t *)pic->data[p];
        for (unsigned r = 0; r < ph; r++) {
            memset(plane + r * (size_t)pic->stride[p], val, pw);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 1: vif_skip_scale0=true -> scale0 score == 0.0                */
/* ------------------------------------------------------------------ */
static char *test_vif_skip_scale0_true(void)
{
    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE, .n_threads = 1};
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init should succeed", err == 0);
    if (err)
        return NULL;

    VmafFeatureDictionary *opts = NULL;
    err = vmaf_feature_dictionary_set(&opts, "vif_skip_scale0", "true");
    mu_assert("dictionary_set vif_skip_scale0 should succeed", err == 0);

    err = vmaf_use_feature(vmaf, "vif", opts);
    mu_assert("vmaf_use_feature(vif) with vif_skip_scale0 should succeed", err == 0);
    if (err) {
        (void)vmaf_feature_dictionary_free(&opts);
        (void)vmaf_close(vmaf);
        return NULL;
    }

    VmafPicture ref_pic, dis_pic;
    err = alloc_flat_pic(&ref_pic, 100u);
    mu_assert("ref picture alloc", err == 0);
    err = alloc_flat_pic(&dis_pic, 120u);
    mu_assert("dis picture alloc", err == 0);

    err = vmaf_read_pictures(vmaf, &ref_pic, &dis_pic, 0);
    mu_assert("vmaf_read_pictures should succeed", err == 0);

    /* Flush: pass NULL, NULL to trigger the EOS path in vmaf_read_pictures. */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("flush should succeed", err == 0);

    /* When vif_skip_scale0=true the FEATURE_PARAM flag causes the option
     * system to remap feature names: the alias "ssclz" is appended.
     * The alias of "VMAF_integer_feature_vif_scale0_score" is
     * "integer_vif_scale0", so the stored key is "integer_vif_scale0_ssclz". */
    double scale0 = -1.0;
    err = vmaf_feature_score_at_index(vmaf, "integer_vif_scale0_ssclz", &scale0, 0);
    mu_assert("scale0 score retrieval should succeed", err == 0);
    mu_assert("vif_skip_scale0=true: scale0_score must be exactly 0.0", scale0 == 0.0);

    /* Scales 1-3 must still be finite and non-negative. */
    double scale1 = -1.0;
    err = vmaf_feature_score_at_index(vmaf, "integer_vif_scale1_ssclz", &scale1, 0);
    mu_assert("scale1 score retrieval should succeed", err == 0);
    mu_assert("scale1_score should be finite and non-negative", isfinite(scale1) && scale1 >= 0.0);

    (void)vmaf_close(vmaf);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Test 2: default (vif_skip_scale0=false) -> scale0 score > 0        */
/* ------------------------------------------------------------------ */
static char *test_vif_skip_scale0_false(void)
{
    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE, .n_threads = 1};
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init should succeed", err == 0);
    if (err)
        return NULL;

    /* No opts_dict -> vif_skip_scale0 defaults to false; standard feature names. */
    err = vmaf_use_feature(vmaf, "vif", NULL);
    mu_assert("vmaf_use_feature(vif) default should succeed", err == 0);
    if (err) {
        (void)vmaf_close(vmaf);
        return NULL;
    }

    VmafPicture ref_pic, dis_pic;
    err = alloc_flat_pic(&ref_pic, 100u);
    mu_assert("ref picture alloc", err == 0);
    err = alloc_flat_pic(&dis_pic, 120u);
    mu_assert("dis picture alloc", err == 0);

    err = vmaf_read_pictures(vmaf, &ref_pic, &dis_pic, 0);
    mu_assert("vmaf_read_pictures should succeed", err == 0);

    /* Flush: pass NULL, NULL to trigger the EOS path in vmaf_read_pictures. */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("flush should succeed", err == 0);

    double scale0 = -1.0;
    err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_vif_scale0_score", &scale0, 0);
    mu_assert("scale0 score retrieval should succeed", err == 0);
    /* Without skip, scale0_score is a finite positive ratio. */
    mu_assert("vif_skip_scale0=false: scale0_score must be finite and > 0",
              isfinite(scale0) && scale0 > 0.0);

    (void)vmaf_close(vmaf);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_vif_skip_scale0_true);
    mu_run_test(test_vif_skip_scale0_false);
    return NULL;
}
