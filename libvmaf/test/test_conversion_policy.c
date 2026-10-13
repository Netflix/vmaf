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
#include <string.h>

#include "test.h"
#include "libvmaf/picture.h"
#include "conversion_policy.h"

static const VmafColor bt709_color = {
    .range = VMAF_COLOR_RANGE_LIMITED,
    .primaries = VMAF_COLOR_PRIMARIES_BT709,
    .trc = VMAF_COLOR_TRC_BT709,
    .matrix = VMAF_COLOR_MATRIX_BT709,
};

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

static VmafPicture make_pic(const VmafColor *color,
                            enum VmafPixelFormat pix_fmt, unsigned bpc)
{
    VmafPicture pic;
    memset(&pic, 0, sizeof(pic));
    pic.color = *color;
    pic.pix_fmt = pix_fmt;
    pic.bpc = bpc;
    return pic;
}

/* a model target that pins only the colorspace */
static VmafPictureConvertTarget color_target(const VmafColor *color)
{
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    target.color = *color;
    return target;
}

static char *test_no_model_target_is_pass_through()
{
    /* a model without a target never converts, however pictures are tagged */
    const VmafColor unset = { 0 };
    const VmafColor *colors[] = { &unset, &bt709_color, &pq_bt2020nc_color };
    for (unsigned i = 0; i < sizeof(colors) / sizeof(*colors); i++) {
        VmafPicture pic = make_pic(colors[i], VMAF_PIX_FMT_YUV420P, 10);
        bool needs_conversion = true;
        VmafPictureConvertTarget target;
        memset(&target, 0, sizeof(target));
        int err = vmaf_conversion_policy_target(&pic, &pic, NULL,
                                                &needs_conversion, &target);
        mu_assert("policy should accept any color without a model target",
                  !err);
        mu_assert("no model target should not need conversion",
                  !needs_conversion);
        mu_assert("target should be left untouched",
                  target.color.trc == VMAF_COLOR_TRC_UNKNOWN);
    }

    return NULL;
}

static char *test_target_picks_model_colorspace()
{
    VmafPicture pic = make_pic(&pq_bt2020nc_color, VMAF_PIX_FMT_YUV420P, 10);
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    bool needs_conversion = false;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    int err = vmaf_conversion_policy_target(&pic, &pic, &model,
                                            &needs_conversion, &target);
    mu_assert("policy should accept fully specified colors", !err);
    mu_assert("differing source and target should need conversion",
              needs_conversion);
    mu_assert("target should be the model's colorspace",
              vmaf_conversion_policy_color_equal(&target.color,
                                                 &pq_ictcp_color));
    mu_assert("an unpinned pixel format should stay unknown",
              target.pix_fmt == VMAF_PIX_FMT_UNKNOWN);
    mu_assert("an unpinned bit depth should stay 0", target.bpc == 0);

    return NULL;
}

static char *test_source_already_matching_target_skips_conversion()
{
    VmafPicture pic = make_pic(&pq_ictcp_color, VMAF_PIX_FMT_YUV420P, 10);
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    bool needs_conversion = true;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    int err = vmaf_conversion_policy_target(&pic, &pic, &model,
                                            &needs_conversion, &target);
    mu_assert("policy should accept matching colors", !err);
    mu_assert("source already in the target should not need conversion",
              !needs_conversion);

    return NULL;
}

static char *test_pinned_format_and_depth_are_part_of_the_target()
{
    VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    model.pix_fmt = VMAF_PIX_FMT_YUV420P;
    model.bpc = 16;

    bool needs_conversion = false;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));

    /* right colorspace, wrong bit depth */
    VmafPicture ten_bit = make_pic(&pq_ictcp_color, VMAF_PIX_FMT_YUV420P, 10);
    int err = vmaf_conversion_policy_target(&ten_bit, &ten_bit, &model,
                                            &needs_conversion, &target);
    mu_assert("policy should accept the pictures", !err);
    mu_assert("a different bit depth should need conversion",
              needs_conversion);
    mu_assert("the pinned pixel format should be in the target",
              target.pix_fmt == VMAF_PIX_FMT_YUV420P);
    mu_assert("the pinned bit depth should be in the target",
              target.bpc == 16);

    /* right colorspace, wrong chroma subsampling */
    VmafPicture yuv444 = make_pic(&pq_ictcp_color, VMAF_PIX_FMT_YUV444P, 16);
    err = vmaf_conversion_policy_target(&yuv444, &yuv444, &model,
                                        &needs_conversion, &target);
    mu_assert("policy should accept the pictures", !err);
    mu_assert("a different pixel format should need conversion",
              needs_conversion);

    /* everything matches */
    VmafPicture exact = make_pic(&pq_ictcp_color, VMAF_PIX_FMT_YUV420P, 16);
    err = vmaf_conversion_policy_target(&exact, &exact, &model,
                                        &needs_conversion, &target);
    mu_assert("policy should accept the pictures", !err);
    mu_assert("pictures matching colorspace, format and depth should not "
              "need conversion", !needs_conversion);

    return NULL;
}

static char *test_unpinned_format_and_depth_follow_the_source()
{
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    bool needs_conversion = false;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    const struct { enum VmafPixelFormat fmt; unsigned bpc; } formats[] = {
        { VMAF_PIX_FMT_YUV420P, 8 }, { VMAF_PIX_FMT_YUV444P, 12 },
        { VMAF_PIX_FMT_YUV422P, 16 },
    };
    for (unsigned i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
        VmafPicture pic = make_pic(&pq_ictcp_color, formats[i].fmt,
                                   formats[i].bpc);
        int err = vmaf_conversion_policy_target(&pic, &pic, &model,
                                                &needs_conversion, &target);
        mu_assert("policy should accept the pictures", !err);
        mu_assert("a target that pins neither format nor depth should accept "
                  "any", !needs_conversion);
    }

    return NULL;
}

static char *test_one_side_differing_still_needs_conversion()
{
    VmafPicture matching = make_pic(&pq_ictcp_color, VMAF_PIX_FMT_YUV420P, 10);
    VmafPicture other = make_pic(&pq_bt2020nc_color, VMAF_PIX_FMT_YUV420P, 10);
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    bool needs_conversion = false;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    int err = vmaf_conversion_policy_target(&matching, &other, &model,
                                            &needs_conversion, &target);
    mu_assert("policy should accept fully specified colors", !err);
    mu_assert("one differing picture should need conversion",
              needs_conversion);

    return NULL;
}

static char *test_ref_and_dist_may_differ_from_each_other()
{
    VmafPicture sdr = make_pic(&bt709_color, VMAF_PIX_FMT_YUV420P, 10);
    VmafPicture hdr = make_pic(&pq_bt2020nc_color, VMAF_PIX_FMT_YUV420P, 10);
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);
    bool needs_conversion = false;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));
    int err = vmaf_conversion_policy_target(&sdr, &hdr, &model,
                                            &needs_conversion, &target);
    mu_assert("each picture converts to the shared target, so differing "
              "sources are not an error", !err);
    mu_assert("differing sources should need conversion", needs_conversion);

    return NULL;
}

static char *test_unspecified_source_with_target_is_an_error()
{
    const VmafColor unset = { 0 };
    VmafColor partial_color = pq_bt2020nc_color;
    partial_color.range = VMAF_COLOR_RANGE_UNKNOWN;

    VmafPicture none = make_pic(&unset, VMAF_PIX_FMT_YUV420P, 10);
    VmafPicture full = make_pic(&pq_bt2020nc_color, VMAF_PIX_FMT_YUV420P, 10);
    VmafPicture partial = make_pic(&partial_color, VMAF_PIX_FMT_YUV420P, 10);
    const VmafPictureConvertTarget model = color_target(&pq_ictcp_color);

    const struct { const VmafPicture *ref, *dist; } cases[] = {
        { &none, &none }, { &none, &full }, { &full, &none },
        { &partial, &full }, { &full, &partial },
    };

    for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        bool needs_conversion = true;
        VmafPictureConvertTarget target;
        memset(&target, 0, sizeof(target));
        int err = vmaf_conversion_policy_target(cases[i].ref, cases[i].dist,
                                                &model, &needs_conversion,
                                                &target);
        mu_assert("an unspecified or partly specified source should be "
                  "rejected when the model has a target", err == -EINVAL);
    }

    return NULL;
}

static char *test_target_equal_compares_format_and_depth()
{
    VmafPictureConvertTarget a = color_target(&pq_ictcp_color);
    VmafPictureConvertTarget b = color_target(&pq_ictcp_color);
    mu_assert("identical targets should be equal",
              vmaf_conversion_policy_target_equal(&a, &b));
    b.bpc = 16;
    mu_assert("targets with different bit depth should differ",
              !vmaf_conversion_policy_target_equal(&a, &b));
    b.bpc = 0;
    b.pix_fmt = VMAF_PIX_FMT_YUV420P;
    mu_assert("targets with different pixel format should differ",
              !vmaf_conversion_policy_target_equal(&a, &b));
    b.pix_fmt = VMAF_PIX_FMT_UNKNOWN;
    b.color = bt709_color;
    mu_assert("targets with different color should differ",
              !vmaf_conversion_policy_target_equal(&a, &b));

    return NULL;
}

static char *test_null_arguments_are_rejected()
{
    VmafPicture pic = make_pic(&bt709_color, VMAF_PIX_FMT_YUV420P, 10);
    bool needs_conversion;
    VmafPictureConvertTarget target;
    memset(&target, 0, sizeof(target));

    mu_assert("null ref should be rejected",
              vmaf_conversion_policy_target(NULL, &pic, NULL,
                                            &needs_conversion, &target) ==
                  -EINVAL);
    mu_assert("null dist should be rejected",
              vmaf_conversion_policy_target(&pic, NULL, NULL,
                                            &needs_conversion, &target) ==
                  -EINVAL);
    mu_assert("null needs_conversion should be rejected",
              vmaf_conversion_policy_target(&pic, &pic, NULL, NULL, &target) ==
                  -EINVAL);
    mu_assert("null target should be rejected",
              vmaf_conversion_policy_target(&pic, &pic, NULL,
                                            &needs_conversion, NULL) ==
                  -EINVAL);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_no_model_target_is_pass_through);
    mu_run_test(test_target_picks_model_colorspace);
    mu_run_test(test_source_already_matching_target_skips_conversion);
    mu_run_test(test_pinned_format_and_depth_are_part_of_the_target);
    mu_run_test(test_unpinned_format_and_depth_follow_the_source);
    mu_run_test(test_one_side_differing_still_needs_conversion);
    mu_run_test(test_ref_and_dist_may_differ_from_each_other);
    mu_run_test(test_unspecified_source_with_target_is_an_error);
    mu_run_test(test_target_equal_compares_format_and_depth);
    mu_run_test(test_null_arguments_are_rejected);
    return NULL;
}
