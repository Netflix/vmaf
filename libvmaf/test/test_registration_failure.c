/* Copyright 2026 Lusoris
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 */

#include <errno.h>
#include <stdlib.h>

#include "config.h"
#include "dict.h"
#include "feature/feature_extractor.h"
#include "libvmaf/libvmaf.h"
#include "test.h"

static int fail_copy, fail_create, track_copy;
static unsigned fail_malloc, input_frees, copy_frees;
static VmafDictionary *tracked_input, *tracked_copy;

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
int __real_vmaf_dictionary_copy(VmafDictionary **src, VmafDictionary **dst);
int __real_vmaf_dictionary_free(VmafDictionary **dict);
int __real_vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **ctx,
    VmafFeatureExtractor *fex, VmafDictionary *options);

void *__wrap_malloc(size_t size)
{
    if (fail_malloc && !--fail_malloc) return NULL;
    return __real_malloc(size);
}

/* Optimized malloc + memset pairs may become calloc calls. */
void *__wrap_calloc(size_t count, size_t size)
{
    if (fail_malloc && !--fail_malloc) return NULL;
    return __real_calloc(count, size);
}

int __wrap_vmaf_dictionary_copy(VmafDictionary **src, VmafDictionary **dst)
{
    if (!src || !*src ||
        !vmaf_dictionary_get(src, "motion_force_zero", 0))
        return __real_vmaf_dictionary_copy(src, dst);
    int err;
    if (fail_copy) {
        fail_copy = 0;
        /* Leave a real partial destination, as a later insertion failure can. */
        if (!src || !*src || (*src)->cnt < 2) return -EINVAL;
        err = vmaf_dictionary_set(dst, (*src)->entry[0].key,
                                  (*src)->entry[0].val, 0);
        if (!err) err = -ENOMEM;
    } else {
        err = __real_vmaf_dictionary_copy(src, dst);
    }
    if (track_copy) {
        tracked_copy = *dst;
        track_copy = 0;
    }
    return err;
}

int __wrap_vmaf_dictionary_free(VmafDictionary **dict)
{
    if (dict && *dict) {
        if (*dict == tracked_input) {
            input_frees++;
            tracked_input = NULL;
        }
        if (*dict == tracked_copy) {
            copy_frees++;
            tracked_copy = NULL;
        }
    }
    return __real_vmaf_dictionary_free(dict);
}

int __wrap_vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **ctx,
    VmafFeatureExtractor *fex, VmafDictionary *options)
{
    if (fail_create && options &&
        vmaf_dictionary_get(&options, "motion_force_zero", 0)) {
        fail_create = 0;
        *ctx = NULL;
        return -ENOMEM;
    }
    return __real_vmaf_feature_extractor_context_create(ctx, fex, options);
}

static int motion_options(VmafFeatureDictionary **options)
{
    int err = vmaf_feature_dictionary_set(options, "motion_force_zero", "true");
    if (!err)
        err = vmaf_feature_dictionary_set(options, "motion_blend_offset", "10");
    return err;
}

static char *check_explicit_failure(int copy_failure)
{
    VmafContext *ctx;
    VmafConfiguration cfg = { 0 };
    mu_assert("context initialization failed", !vmaf_init(&ctx, cfg));
    VmafFeatureDictionary *options = NULL;
    mu_assert("options creation failed", !motion_options(&options));
    tracked_input = (VmafDictionary *) options;
    input_frees = copy_frees = 0;
    fail_copy = copy_failure;
    fail_create = !copy_failure;
    track_copy = 1;
    int err = vmaf_use_feature(ctx, "motion", options);
    mu_assert("failure was not propagated", err == -ENOMEM);
    mu_assert("private dictionary was not freed", copy_frees == 1);
    mu_assert("supplied dictionary ownership changed",
              input_frees == (copy_failure ? 0 : 1));
    if (!copy_failure) {
        options = NULL;
        mu_assert("retry options creation failed", !motion_options(&options));
    }
    err = vmaf_use_feature(ctx, "motion", options);
    mu_assert("registration retry failed", !err);
    vmaf_close(ctx);
    return NULL;
}

static char *test_explicit_copy_failure() { return check_explicit_failure(1); }
static char *test_explicit_create_failure() { return check_explicit_failure(0); }

#if VMAF_BUILT_IN_MODELS
static char *check_model_failure(int copy_failure)
{
    VmafContext *ctx;
    VmafConfiguration cfg = { 0 };
    mu_assert("context initialization failed", !vmaf_init(&ctx, cfg));
    VmafModel *model;
    VmafModelConfig model_cfg = { 0 };
    mu_assert("model load failed", !vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1"));
    VmafFeatureDictionary *options = NULL;
    mu_assert("options creation failed", !motion_options(&options));
    mu_assert("model override failed", !vmaf_model_feature_overload(model, "motion", options));
    copy_frees = 0;
    fail_copy = copy_failure;
    fail_create = !copy_failure;
    track_copy = 1;
    int err = vmaf_use_features_from_model(ctx, model);
    mu_assert("failure was not propagated", err == -ENOMEM);
    mu_assert("private dictionary was not freed", copy_frees == 1);
    err = vmaf_use_features_from_model(ctx, model);
    mu_assert("model options did not survive retry", !err);
    vmaf_close(ctx);
    vmaf_model_destroy(model);
    return NULL;
}

static char *test_model_copy_failure() { return check_model_failure(1); }
static char *test_model_create_failure() { return check_model_failure(0); }
#endif

static char *test_constructor_allocation_failure()
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion");
    mu_assert("motion extractor not found", fex);
    VmafDictionary *options = NULL;
    mu_assert("option creation failed",
              !vmaf_dictionary_set(&options, "motion_force_zero", "true", 0));
    for (unsigned allocation = 1; allocation <= 3; allocation++) {
        VmafFeatureExtractorContext *ctx = NULL;
        fail_malloc = allocation;
        int err = vmaf_feature_extractor_context_create(&ctx, fex, options);
        mu_assert("allocation failure was not propagated", err == -ENOMEM);
        mu_assert("failed context must not be published", !ctx);
        mu_assert("caller options were consumed",
                  vmaf_dictionary_get(&options, "motion_force_zero", 0));
    }
    vmaf_dictionary_free(&options);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_explicit_copy_failure);
    mu_run_test(test_explicit_create_failure);
#if VMAF_BUILT_IN_MODELS
    mu_run_test(test_model_copy_failure);
    mu_run_test(test_model_create_failure);
#endif
    mu_run_test(test_constructor_allocation_failure);
    return NULL;
}
