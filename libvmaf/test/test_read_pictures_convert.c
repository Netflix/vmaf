/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "libvmaf/picture.h"

#define W 256
#define H 144
#define BPC 10
#define N_FRAMES 3

static const VmafColor pq_bt2020nc_color = {
    .range = VMAF_COLOR_RANGE_LIMITED,
    .primaries = VMAF_COLOR_PRIMARIES_BT2020,
    .trc = VMAF_COLOR_TRC_SMPTE2084,
    .matrix = VMAF_COLOR_MATRIX_BT2020_NCL,
};

static const VmafColor pq_ictcp_color = {
    .range = VMAF_COLOR_RANGE_LIMITED,
    .primaries = VMAF_COLOR_PRIMARIES_BT2020,
    .trc = VMAF_COLOR_TRC_SMPTE2084,
    .matrix = VMAF_COLOR_MATRIX_ICTCP,
};

#define COLORSPACE_ICTCP                                                    \
    "\"colorspace\": {\"range\": \"limited\", \"primaries\": \"bt2020\", "     \
    "\"trc\": \"smpte2084\", \"matrix\": \"ictcp\"}"

/* the target's colorspace only: format and depth follow the source */
static const char *target_color_only =
    "\"conversion_target\": {" COLORSPACE_ICTCP "}, ";

/* the target also pins 4:2:0 at the source's own 10 bits */
static const char *target_420_10bit =
    "\"conversion_target\": {" COLORSPACE_ICTCP ", \"pixel_format\": \"420\", "
    "\"bit_depth\": 10}, ";

/* the target also pins 4:2:0 at 16 bits */
static const char *target_420_16bit =
    "\"conversion_target\": {" COLORSPACE_ICTCP ", \"pixel_format\": \"420\", "
    "\"bit_depth\": 16}, ";

/*
 * vmaf_v0.6.1, optionally with a conversion_target spliced into its
 * model_dict. This keeps the test on integer features so it does not depend
 * on float feature extractors.
 */
static int load_model(VmafModel **model, const char *target_block)
{
    FILE *f = fopen(JSON_MODEL_PATH "vmaf_v0.6.1.json", "rb");
    if (!f) return -ENOENT;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = calloc(n + 1, 1);
    if (!buf || fread(buf, 1, n, f) != (size_t) n) {
        fclose(f);
        free(buf);
        return -EIO;
    }
    fclose(f);

    const char *key = "\"model_dict\": {";
    char *at = strstr(buf, key);
    if (!at) {
        free(buf);
        return -EINVAL;
    }
    at += strlen(key);

    /* plain stdio keeps this portable (no mkstemp/unistd.h on MSVC) */
    const char *path = "vmaf_convert_model.json";
    FILE *out = fopen(path, "wb");
    if (!out) {
        free(buf);
        return -EIO;
    }
    fwrite(buf, 1, at - buf, out);
    if (target_block)
        fputs(target_block, out);
    fputs(at, out);
    fclose(out);
    free(buf);

    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(model, &cfg, path);
    remove(path);
    return err;
}

static void fill_picture(VmafPicture *pic, unsigned frame, unsigned noise)
{
    uint32_t seed = 12345u + frame;
    for (unsigned p = 0; p < 3; p++) {
        for (unsigned y = 0; y < pic->h[p]; y++) {
            uint16_t *row =
                (uint16_t *) ((uint8_t *) pic->data[p] + y * pic->stride[p]);
            for (unsigned x = 0; x < pic->w[p]; x++) {
                int v = 200 + (int) ((x * 3 + y * 2 + frame * 5) % 600);
                if (noise) {
                    seed = seed * 1664525u + 1013904223u;
                    v += (int) ((seed >> 16) % (2 * noise + 1)) - (int) noise;
                }
                row[x] = (uint16_t) (v < 0 ? 0 : v > 1023 ? 1023 : v);
            }
        }
    }
}

/*
 * Score N_FRAMES synthetic frames with the given colors set on ref and dist.
 * Returns the first vmaf_read_pictures error, or 0 and fills `score`.
 */
static int run_fmt(const char *target_block, enum VmafPixelFormat pix_fmt,
                   const VmafColor *ref_color, const VmafColor *dist_color,
                   double *score, double *psnr_cb)
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    int err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    VmafModel *model;
    err = load_model(&model, target_block);
    if (err) return err;
    err = vmaf_use_features_from_model(vmaf, model);
    if (err) return err;
    /* the model is luma-only; chroma PSNR is what shows a chroma conversion */
    if (psnr_cb) {
        err = vmaf_use_feature(vmaf, "psnr", NULL);
        if (err) return err;
    }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, pix_fmt, BPC, W, H);
        if (err) return err;
        err = vmaf_picture_alloc(&dist, pix_fmt, BPC, W, H);
        if (err) return err;
        fill_picture(&ref, i, 0);
        fill_picture(&dist, i, 20);
        if (ref_color) ref.color = *ref_color;
        if (dist_color) dist.color = *dist_color;

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) {
            /* ownership stays with the caller on failure */
            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dist);
            vmaf_model_destroy(model);
            vmaf_close(vmaf);
            return err;
        }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (!err)
        err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN, score, 0,
                                N_FRAMES - 1);
    if (!err && psnr_cb)
        err = vmaf_feature_score_at_index(vmaf, "psnr_cb", psnr_cb, 0);
    vmaf_model_destroy(model);
    vmaf_close(vmaf);
    return err;
}

static int run(const char *target_block, const VmafColor *ref_color,
               const VmafColor *dist_color, double *score)
{
    return run_fmt(target_block, VMAF_PIX_FMT_YUV420P, ref_color, dist_color,
                   score, NULL);
}

static char *test_no_model_target_never_converts()
{
    double untagged, tagged;
    mu_assert("untagged run failed", !run(NULL, NULL, NULL, &untagged));
    mu_assert("tagged run failed",
              !run(NULL, &pq_bt2020nc_color, &pq_bt2020nc_color, &tagged));
    mu_assert("a model without a conversion target should ignore picture "
              "colors", untagged == tagged);

    return NULL;
}

#ifdef HAVE_ZIMG
static char *test_model_target_converts_source()
{
    double unconverted, converted;
    mu_assert("baseline run failed",
              !run(NULL, NULL, NULL, &unconverted));
    mu_assert("conversion run failed",
              !run(target_color_only, &pq_bt2020nc_color, &pq_bt2020nc_color, &converted));
    mu_assert("source should be converted to the model's target, but the "
              "score matches the untouched pictures",
              unconverted != converted);

    return NULL;
}

static char *test_ref_and_dist_are_converted_independently()
{
    double both_converted, both_matching, ref_only, dist_only;
    mu_assert("both-converted run failed",
              !run(target_color_only, &pq_bt2020nc_color, &pq_bt2020nc_color,
                   &both_converted));
    mu_assert("both-matching run failed",
              !run(target_color_only, &pq_ictcp_color, &pq_ictcp_color, &both_matching));
    mu_assert("reference-only conversion run failed",
              !run(target_color_only, &pq_bt2020nc_color, &pq_ictcp_color, &ref_only));
    mu_assert("distorted-only conversion run failed",
              !run(target_color_only, &pq_ictcp_color, &pq_bt2020nc_color, &dist_only));

    /*
     * Converting just one picture must differ from converting both or
     * neither, and which side is converted must matter.
     */
    mu_assert("reference-only conversion should differ from converting both",
              ref_only != both_converted);
    mu_assert("reference-only conversion should differ from converting "
              "neither", ref_only != both_matching);
    mu_assert("converting the reference or the distorted input alone should "
              "give different scores", ref_only != dist_only);

    return NULL;
}
#endif

static char *test_source_matching_target_is_not_converted()
{
    double unconverted, matching;
    mu_assert("baseline run failed",
              !run(NULL, NULL, NULL, &unconverted));
    mu_assert("matching run failed",
              !run(target_color_only, &pq_ictcp_color, &pq_ictcp_color, &matching));
    mu_assert("a source already in the target colorspace should be left "
              "untouched", unconverted == matching);

    return NULL;
}

#ifndef HAVE_ZIMG
static char *test_conversion_without_zimg_is_an_error()
{
    double score;
    int err = run(target_color_only, &pq_bt2020nc_color, &pq_bt2020nc_color, &score);
    mu_assert("a conversion that is needed should fail without zimg",
              err == -ENOTSUP);

    return NULL;
}
#endif

static char *test_source_matching_pinned_format_is_not_converted()
{
    /* the synthetic frames are 4:2:0 at 10 bits, which this target pins */
    double unconverted, pinned;
    mu_assert("baseline run failed",
              !run(NULL, NULL, NULL, &unconverted));
    mu_assert("pinned-format run failed",
              !run(target_420_10bit, &pq_ictcp_color, &pq_ictcp_color,
                   &pinned));
    mu_assert("a source already in the pinned format and colorspace should "
              "be left untouched", unconverted == pinned);

    return NULL;
}

#ifdef HAVE_ZIMG
static char *test_target_pixel_format_is_applied()
{
    /* same colorspace as the target, so only the chroma subsampling differs */
    double score, unconverted, subsampled;
    mu_assert("4:4:4 baseline run failed",
              !run_fmt(NULL, VMAF_PIX_FMT_YUV444P, NULL, NULL, &score,
                       &unconverted));
    mu_assert("4:2:0 target run failed",
              !run_fmt(target_420_10bit, VMAF_PIX_FMT_YUV444P, &pq_ictcp_color,
                       &pq_ictcp_color, &score, &subsampled));
    mu_assert("a 4:4:4 source should be converted to a pinned 4:2:0 target, "
              "but its chroma matches the untouched pictures",
              unconverted != subsampled);

    return NULL;
}

static char *test_target_bit_depth_converts_without_error()
{
    /*
     * Widening 10 to 16 bits is an exact scale that the integer features
     * normalize away, so the score is not a signal here; this only checks
     * that the conversion to a deeper target runs.
     */
    double score;
    mu_assert("16-bit target run failed",
              !run(target_420_16bit, &pq_ictcp_color, &pq_ictcp_color, &score));

    return NULL;
}

static char *test_models_with_different_formats_cannot_share_a_run()
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafModel *ten_bit, *sixteen_bit;
    mu_assert("load 10-bit model", !load_model(&ten_bit, target_420_10bit));
    mu_assert("load 16-bit model", !load_model(&sixteen_bit, target_420_16bit));

    mu_assert("first model should register",
              !vmaf_use_features_from_model(vmaf, ten_bit));
    mu_assert("a model pinning a different bit depth cannot share a run",
              vmaf_use_features_from_model(vmaf, sixteen_bit) == -EINVAL);

    vmaf_model_destroy(ten_bit);
    vmaf_model_destroy(sixteen_bit);
    vmaf_close(vmaf);
    return NULL;
}
#endif

static char *test_untagged_source_with_target_is_rejected()
{
    double score;
    mu_assert("untagged source should be rejected when the model has a "
              "target", run(target_color_only, NULL, NULL, &score) == -EINVAL);

    VmafColor partial = pq_bt2020nc_color;
    partial.matrix = VMAF_COLOR_MATRIX_UNKNOWN;
    mu_assert("partly tagged source should be rejected when the model has "
              "a target",
              run(target_color_only, &partial, &partial, &score) == -EINVAL);
    mu_assert("only one picture tagged should be rejected when the model has "
              "a target",
              run(target_color_only, &pq_bt2020nc_color, NULL, &score) == -EINVAL);

    return NULL;
}

static char *test_models_must_share_a_conversion_target()
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafModel *with_target, *without_target;
    mu_assert("load model with target", !load_model(&with_target, target_color_only));
    mu_assert("load model without target", !load_model(&without_target, NULL));

    mu_assert("first model should register",
              !vmaf_use_features_from_model(vmaf, with_target));
    mu_assert("a model with no target cannot share a run with one that has "
              "a target",
              vmaf_use_features_from_model(vmaf, without_target) == -EINVAL);

    vmaf_model_destroy(with_target);
    vmaf_model_destroy(without_target);
    vmaf_close(vmaf);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_no_model_target_never_converts);
#ifdef HAVE_ZIMG
    mu_run_test(test_model_target_converts_source);
    mu_run_test(test_ref_and_dist_are_converted_independently);
#else
    mu_run_test(test_conversion_without_zimg_is_an_error);
#endif
    mu_run_test(test_source_matching_target_is_not_converted);
    mu_run_test(test_source_matching_pinned_format_is_not_converted);
#ifdef HAVE_ZIMG
    mu_run_test(test_target_pixel_format_is_applied);
    mu_run_test(test_target_bit_depth_converts_without_error);
    mu_run_test(test_models_with_different_formats_cannot_share_a_run);
#endif
    mu_run_test(test_untagged_source_with_target_is_rejected);
    mu_run_test(test_models_must_share_a_conversion_target);
    return NULL;
}
