/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Smoke test for the integer_motion_vulkan feature extractor.
 *
 *  Validates the registration and init() contract:
 *    - "integer_motion_vulkan" is resolvable by name.
 *    - init() rejects frames smaller than 3x3 with -EINVAL.
 *    - init() rejects motion_five_frame_window=true with -ENOTSUP.
 *    - init() accepts a normal 576x324 frame when a Vulkan device is present
 *      (skipped when no compute-capable Vulkan device is available).
 *
 *  GPU end-to-end correctness (score parity vs CPU) is validated by the
 *  cross-backend gate (ADR-0214) and the validate-scores skill.
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "libvmaf/libvmaf_vulkan.h"

/* Helper: allocate priv, set the named option if provided, call init(),
 * call close(), free priv.  Returns init()'s return code. */
static int invoke_init_opt(VmafFeatureExtractor *fex, unsigned w, unsigned h, const char *opt_name,
                           bool opt_val)
{
    void *priv = calloc(1, fex->priv_size);
    if (!priv)
        return -1;
    fex->priv = priv;

    /* Apply the boolean option by scanning the options table directly.
     * This mirrors the approach used in test_motion_min_dim.c — we do not
     * go through the full VmafFeatureExtractorContext (which requires a live
     * VmafContext) so we write the field by offset. */
    if (opt_name != NULL) {
        for (const VmafOption *o = fex->options; o && o->name; o++) {
            if (strcmp(o->name, opt_name) == 0 && o->type == VMAF_OPT_TYPE_BOOL) {
                *(bool *)((char *)priv + o->offset) = opt_val;
                break;
            }
        }
    }

    int rc = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8u, w, h);
    if (fex->close)
        (void)fex->close(fex);
    free(priv);
    fex->priv = NULL;
    return rc;
}

static int invoke_init(VmafFeatureExtractor *fex, unsigned w, unsigned h)
{
    return invoke_init_opt(fex, w, h, NULL, false);
}

/* ------------------------------------------------------------------ */
/* Registration.                                                        */
/* ------------------------------------------------------------------ */

static char *test_extractor_is_registered(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("integer_motion_vulkan extractor is registered", fex != NULL);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Minimum-dimension contract.                                          */
/* ------------------------------------------------------------------ */

static char *test_rejects_1x1(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init(fex, 1u, 1u);
    mu_assert("init(1x1) must return -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_rejects_2x2(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init(fex, 2u, 2u);
    mu_assert("init(2x2) must return -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_rejects_1xN(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init(fex, 64u, 1u);
    mu_assert("init(64x1) must return -EINVAL", rc == -EINVAL);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* five-frame-window rejection.                                         */
/* ------------------------------------------------------------------ */

static char *test_rejects_five_frame_window(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init_opt(fex, 64u, 64u, "motion_five_frame_window", true);
    mu_assert("motion_five_frame_window=true must return -ENOTSUP", rc == -ENOTSUP);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Happy-path: accepts 3x3 minimum and 576x324 Netflix golden size.   */
/* Skipped when no Vulkan compute device is available.                 */
/* ------------------------------------------------------------------ */

static char *test_accepts_3x3(void)
{
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init(fex, 3u, 3u);
    mu_assert("init(3x3) must succeed (exact minimum)", rc == 0);
    return NULL;
}

static char *test_accepts_576x324(void)
{
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("integer_motion_vulkan");
    mu_assert("extractor missing", fex != NULL);
    int rc = invoke_init(fex, 576u, 324u);
    mu_assert("init(576x324) must succeed (Netflix golden resolution)", rc == 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_extractor_is_registered);
    mu_run_test(test_rejects_1x1);
    mu_run_test(test_rejects_2x2);
    mu_run_test(test_rejects_1xN);
    mu_run_test(test_rejects_five_frame_window);
    mu_run_test(test_accepts_3x3);
    mu_run_test(test_accepts_576x324);
    return NULL;
}
