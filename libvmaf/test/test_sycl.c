/**
 *
 *  Copyright 2024-2026 Lusoris and Claude (Anthropic)
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

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "config.h"
#include "test.h"

#if HAVE_SYCL

#include "libvmaf/libvmaf_sycl.h"
#include "feature/feature_extractor.h"

static VmafSyclState *sycl = NULL;
static int sycl_init_failed = 0;

static char *test_sycl_state_init(void)
{
    VmafSyclConfiguration cfg = { .device_index = -1 };
    int err = vmaf_sycl_state_init(&sycl, cfg);
    if (err) {
        /* No SYCL GPU available — skip device-dependent tests */
        fprintf(stderr, "  [SKIP] SYCL state init failed (err=%d), "
                "no GPU available — skipping device tests\n", err);
        sycl_init_failed = 1;
        sycl = NULL;
        return NULL;
    }
    mu_assert("sycl_state should be non-NULL", sycl != NULL);
    return NULL;
}

static char *test_sycl_state_init_invalid(void)
{
    /* NULL pointer should be rejected */
    VmafSyclConfiguration cfg = { .device_index = -1 };
    int err = vmaf_sycl_state_init(NULL, cfg);
    mu_assert("NULL pointer should return EINVAL", err < 0);

    /* Out-of-range device index */
    VmafSyclState *tmp = NULL;
    VmafSyclConfiguration bad_cfg = { .device_index = 9999 };
    err = vmaf_sycl_state_init(&tmp, bad_cfg);
    mu_assert("invalid device_index should fail", err < 0);
    mu_assert("state should be NULL on failure", tmp == NULL);

    return NULL;
}

static char *test_sycl_import_state(void)
{
    if (sycl_init_failed) {
        fprintf(stderr, "  [SKIP] test_sycl_import_state (no GPU)\n");
        return NULL;
    }

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init should succeed", err == 0);

    err = vmaf_sycl_import_state(vmaf, sycl);
    mu_assert("vmaf_sycl_import_state should succeed", err == 0);

    vmaf_close(vmaf);
    return NULL;
}

static char *test_sycl_feature_extractor_lookup(void)
{
    VmafFeatureExtractor *fex;

    /* Lookup by extractor name */
    fex = vmaf_get_feature_extractor_by_name("adm_sycl");
    mu_assert("adm_sycl should be registered", fex != NULL);
    mu_assert("adm_sycl name should match",
              !strcmp(fex->name, "adm_sycl"));

    fex = vmaf_get_feature_extractor_by_name("vif_sycl");
    mu_assert("vif_sycl should be registered", fex != NULL);
    mu_assert("vif_sycl name should match",
              !strcmp(fex->name, "vif_sycl"));

    fex = vmaf_get_feature_extractor_by_name("motion_sycl");
    mu_assert("motion_sycl should be registered", fex != NULL);
    mu_assert("motion_sycl name should match",
              !strcmp(fex->name, "motion_sycl"));

    /* Lookup by feature name with SYCL flag */
    unsigned flags = VMAF_FEATURE_EXTRACTOR_SYCL;
    fex = vmaf_get_feature_extractor_by_feature_name(
            "VMAF_integer_feature_adm2_score", flags);
    mu_assert("SYCL ADM should be found by feature name", fex != NULL);
    mu_assert("should be adm_sycl",
              !strcmp(fex->name, "adm_sycl"));

    return NULL;
}

static char *test_sycl_state_release(void)
{
    if (sycl_init_failed || sycl == NULL) {
        fprintf(stderr, "  [SKIP] test_sycl_state_release (no GPU)\n");
        return NULL;
    }

    vmaf_sycl_state_free(&sycl);
    mu_assert("state should be NULL after free", sycl == NULL);

    return NULL;
}

char *run_tests(void)
{
    /* Invalid-argument tests work without a GPU */
    mu_run_test(test_sycl_state_init_invalid);

    /* Feature extractor registration is compile-time */
    mu_run_test(test_sycl_feature_extractor_lookup);

    /* Device-dependent tests (gracefully skipped if no GPU) */
    mu_run_test(test_sycl_state_init);
    mu_run_test(test_sycl_import_state);

    /* Cleanup (always last) */
    mu_run_test(test_sycl_state_release);

    return NULL;
}

#else /* !HAVE_SYCL */

char *run_tests(void)
{
    fprintf(stderr, "SYCL not enabled, skipping tests\n");
    return NULL;
}

#endif /* HAVE_SYCL */
