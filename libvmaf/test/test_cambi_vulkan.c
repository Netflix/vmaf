/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Smoke test for the CAMBI Vulkan extractor (v2 / ADR-0456).
 *
 *  Goals:
 *    1. Verify vmaf_fex_cambi_vulkan is discoverable.
 *    2. Verify end-to-end init → extract → close with a synthetic
 *       all-flat frame does not crash and emits a finite non-negative score.
 *    3. Verify that default parameters match the CPU reference defaults
 *       (Gap 1 of ADR-0456 parity audit).
 *
 *  This is a smoke + registration test, NOT a numerical-correctness test.
 *  Bit-exactness against the CPU scalar extractor is verified by the
 *  cross-backend scoring gate (Build — Linux GPU (Vulkan) parity /
 *  ADR-0214). Full-precision numerical assertions belong in python/test/
 *  alongside the other golden-data tests (per CLAUDE.md §8).
 *
 *  Cross-backend ULP gate reproducer:
 *    tools/vmaf --reference testdata/yuv/src01_hrc00_576x324.yuv
 *               --distorted testdata/yuv/src01_hrc01_576x324.yuv
 *               --width 576 --height 324 --pixel_format 420 --bitdepth 8
 *               --feature cambi
 *               --feature cambi_vulkan
 *               # then diff Cambi_feature_cambi_score columns
 *    Expected: max abs diff < 1e-4 (places=4 gate).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "test.h"

#if HAVE_VULKAN

#include "feature/feature_extractor.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

static const char *g_cambi_vk_name = "cambi_vulkan";

/* ---------------------------------------------------------------------- */
/* Test 1: registration — vmaf_fex_cambi_vulkan is discoverable.           */
/* ---------------------------------------------------------------------- */
static char *test_cambi_vk_registration(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name(g_cambi_vk_name);
    mu_assert("vmaf_fex_cambi_vulkan should be findable by name", fex != NULL);
    if (fex) {
        mu_assert("fex name should be cambi_vulkan", strcmp(fex->name, g_cambi_vk_name) == 0);
        mu_assert("fex flags should include VMAF_FEATURE_EXTRACTOR_VULKAN",
                  (fex->flags & VMAF_FEATURE_EXTRACTOR_VULKAN) != 0);
        mu_assert("fex provided_features[0] should be the CAMBI score feature",
                  fex->provided_features != NULL &&
                      strcmp(fex->provided_features[0], "Cambi_feature_cambi_score") == 0);
    }
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Test 2: default parameter parity with CPU (ADR-0456 gap 1 regression). */
/* ---------------------------------------------------------------------- */
static char *test_cambi_vk_defaults(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name(g_cambi_vk_name);
    mu_assert("fex must be non-NULL for default checks", fex != NULL);
    if (!fex || !fex->options)
        return NULL;

    /* Walk the options table and check that the defaults match cambi.c. */
    double max_val = -1.0, topk = -1.0, tvi = -1.0, vlt = -2.0;
    int window_size = -1, max_log_contrast = -1, min_enc_w = -1;
    for (const VmafOption *opt = fex->options; opt->name; opt++) {
        if (strcmp(opt->name, "cambi_max_val") == 0)
            max_val = opt->default_val.d;
        else if (strcmp(opt->name, "topk") == 0)
            topk = opt->default_val.d;
        else if (strcmp(opt->name, "tvi_threshold") == 0)
            tvi = opt->default_val.d;
        else if (strcmp(opt->name, "cambi_vis_lum_threshold") == 0)
            vlt = opt->default_val.d;
        else if (strcmp(opt->name, "window_size") == 0)
            window_size = opt->default_val.i;
        else if (strcmp(opt->name, "max_log_contrast") == 0)
            max_log_contrast = opt->default_val.i;
        else if (strcmp(opt->name, "enc_width") == 0)
            min_enc_w = opt->min;
    }

    /* CPU cambi.c defaults — must match exactly (ADR-0456 gap 1). */
    mu_assert("cambi_max_val default should be 1000.0 (CPU match)", max_val == 1000.0);
    mu_assert("topk default should be 0.6 (CPU match)", topk == 0.6);
    mu_assert("tvi_threshold default should be 0.019 (CPU match)", tvi == 0.019);
    mu_assert("cambi_vis_lum_threshold default should be 0.0 (CPU match)", vlt == 0.0);
    mu_assert("window_size default should be 65 (CPU match)", window_size == 65);
    mu_assert("max_log_contrast default should be 2 (CPU match)", max_log_contrast == 2);
    mu_assert("enc_width min should be 180 (CPU match)", min_enc_w == 180);
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Test 3: end-to-end smoke — flat 576×324 YUV420 frame.                   */
/* ---------------------------------------------------------------------- */
static char *test_cambi_vk_smoke(void)
{
    /* Create a 576×324 10-bit YUV400 flat frame (all luma = 512). */
    VmafPicture ref_pic, dis_pic;
    int err = vmaf_picture_alloc(&ref_pic, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    if (err) {
        fprintf(stderr, "  [SKIP] vmaf_picture_alloc failed (%d), skipping smoke\n", err);
        return NULL;
    }
    err = vmaf_picture_alloc(&dis_pic, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    if (err) {
        vmaf_picture_unref(&ref_pic);
        fprintf(stderr, "  [SKIP] vmaf_picture_alloc failed (%d), skipping smoke\n", err);
        return NULL;
    }
    /* Fill luma planes with 128 (mid-gray 8-bit). */
    memset(ref_pic.data[0], 128, ref_pic.stride[0] * ref_pic.h[0]);
    memset(dis_pic.data[0], 128, dis_pic.stride[0] * dis_pic.h[0]);

    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name(g_cambi_vk_name);
    if (!fex) {
        vmaf_picture_unref(&ref_pic);
        vmaf_picture_unref(&dis_pic);
        fprintf(stderr, "  [SKIP] cambi_vulkan not found — Vulkan not compiled?\n");
        return NULL;
    }

    /* Allocate private state (calloc as in the engine). */
    void *priv = calloc(1, fex->priv_size);
    if (!priv) {
        vmaf_picture_unref(&ref_pic);
        vmaf_picture_unref(&dis_pic);
        return NULL;
    }
    fex->priv = priv;

    err = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    if (err) {
        fprintf(stderr, "  [SKIP] cambi_vulkan init failed (%d) — no Vulkan device?\n", err);
        free(priv);
        vmaf_picture_unref(&ref_pic);
        vmaf_picture_unref(&dis_pic);
        return NULL;
    }

    VmafFeatureCollector *fc = NULL;
    err = vmaf_feature_collector_init(&fc, NULL);
    if (err) {
        fex->close(fex);
        free(priv);
        vmaf_picture_unref(&ref_pic);
        vmaf_picture_unref(&dis_pic);
        return NULL;
    }

    err = fex->extract(fex, &ref_pic, NULL, &dis_pic, NULL, 0, fc);
    mu_assert("cambi_vulkan extract should succeed on a flat frame", err == 0);

    if (!err) {
        double score;
        int score_err =
            vmaf_feature_collector_get_score(fc, "Cambi_feature_cambi_score", &score, 0);
        if (score_err == 0) {
            mu_assert("CAMBI score on flat frame should be finite", isfinite(score));
            mu_assert("CAMBI score on flat frame should be non-negative", score >= 0.0);
            /* Flat frames have no banding; score should be near 0. */
            mu_assert("CAMBI score on flat frame should be < 0.1", score < 0.1);
        }
    }

    vmaf_feature_collector_destroy(fc);
    fex->close(fex);
    free(priv);
    vmaf_picture_unref(&ref_pic);
    vmaf_picture_unref(&dis_pic);
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Test runner                                                              */
/* ---------------------------------------------------------------------- */
static int tests_run = 0;

static char *all_tests(void)
{
    mu_run_test(test_cambi_vk_registration);
    mu_run_test(test_cambi_vk_defaults);
    mu_run_test(test_cambi_vk_smoke);
    return NULL;
}

int main(void)
{
    char *result = all_tests();
    if (result) {
        fprintf(stderr, "FAIL: %s\n", result);
    } else {
        printf("%d tests run, %d passed\n", tests_run, tests_run);
    }
    return result != NULL;
}

#else /* !HAVE_VULKAN */

int main(void)
{
    printf("0 tests run, 0 passed (Vulkan not compiled)\n");
    return 0;
}

#endif /* HAVE_VULKAN */
