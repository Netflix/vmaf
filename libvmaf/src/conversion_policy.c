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

#include "conversion_policy.h"
#include "log.h"

static bool color_is_fully_specified(const VmafColor *color)
{
    return color->range != VMAF_COLOR_RANGE_UNKNOWN &&
           color->primaries != VMAF_COLOR_PRIMARIES_UNKNOWN &&
           color->trc != VMAF_COLOR_TRC_UNKNOWN &&
           color->matrix != VMAF_COLOR_MATRIX_UNKNOWN;
}

static bool color_is_untagged(const VmafColor *color)
{
    return color->range == VMAF_COLOR_RANGE_UNKNOWN &&
           color->primaries == VMAF_COLOR_PRIMARIES_UNKNOWN &&
           color->trc == VMAF_COLOR_TRC_UNKNOWN &&
           color->matrix == VMAF_COLOR_MATRIX_UNKNOWN;
}

int vmaf_conversion_policy_assume_color(VmafPicture *ref, VmafPicture *dist,
                                        const VmafColor *assumed,
                                        bool *assumed_applied)
{
    if (!ref || !dist || !assumed || !assumed_applied) return -EINVAL;
    *assumed_applied = false;

    const bool ref_untagged = color_is_untagged(&ref->color);
    const bool dist_untagged = color_is_untagged(&dist->color);
    if (ref_untagged != dist_untagged) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "the %s picture has no colorimetry set but the %s picture "
                 "has; the model assumes a colorimetry only when neither has "
                 "one, so set both or neither\n",
                 ref_untagged ? "reference" : "distorted",
                 ref_untagged ? "distorted" : "reference");
        return -EINVAL;
    }
    if (!ref_untagged) return 0;

    ref->color = *assumed;
    dist->color = *assumed;
    *assumed_applied = true;
    return 0;
}

bool vmaf_conversion_policy_color_equal(const VmafColor *a, const VmafColor *b)
{
    return a->range == b->range && a->primaries == b->primaries &&
           a->trc == b->trc && a->matrix == b->matrix;
}

static void log_missing_color(const char *which, const VmafColor *color)
{
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "the model requires source colorimetry, but the %s picture has "
             "unspecified:%s%s%s%s. Set --color_range_ref, "
             "--color_primaries_ref, --color_trc_ref and --color_matrix_ref "
             "(--color_*_dist for the distorted input) on the CLI, or "
             "VmafPicture.color in the library.\n", which,
             color->range == VMAF_COLOR_RANGE_UNKNOWN ? " range" : "",
             color->primaries == VMAF_COLOR_PRIMARIES_UNKNOWN ?
                 " primaries" : "",
             color->trc == VMAF_COLOR_TRC_UNKNOWN ? " trc" : "",
             color->matrix == VMAF_COLOR_MATRIX_UNKNOWN ? " matrix" : "");
}

bool vmaf_conversion_policy_picture_matches(const VmafPicture *pic,
                                            const VmafPictureConvertTarget *target)
{
    return vmaf_conversion_policy_color_equal(&pic->color, &target->color) &&
           (!target->pix_fmt || pic->pix_fmt == target->pix_fmt) &&
           (!target->bpc || pic->bpc == target->bpc) &&
           (!target->w || pic->w[0] == target->w) &&
           (!target->h || pic->h[0] == target->h);
}

bool vmaf_conversion_policy_target_equal(const VmafPictureConvertTarget *a,
                                         const VmafPictureConvertTarget *b)
{
    return vmaf_conversion_policy_color_equal(&a->color, &b->color) &&
           a->pix_fmt == b->pix_fmt && a->bpc == b->bpc;
}

int vmaf_conversion_policy_target(const VmafPicture *ref,
                                  const VmafPicture *dist,
                                  const VmafPictureConvertTarget *model_target,
                                  bool *needs_conversion,
                                  VmafPictureConvertTarget *target)
{
    if (!ref || !dist || !needs_conversion || !target) return -EINVAL;

    if (!model_target) {
        *needs_conversion = false;
        return 0;
    }

    if (!color_is_fully_specified(&ref->color)) {
        log_missing_color("reference", &ref->color);
        return -EINVAL;
    }
    if (!color_is_fully_specified(&dist->color)) {
        log_missing_color("distorted", &dist->color);
        return -EINVAL;
    }

    target->color = model_target->color;
    target->pix_fmt = model_target->pix_fmt;
    target->bpc = model_target->bpc;
    *needs_conversion =
        !vmaf_conversion_policy_picture_matches(ref, target) ||
        !vmaf_conversion_policy_picture_matches(dist, target);
    return 0;
}

/* the long sides of two pictures scaled to the same short side may differ
 * by this much from rounding to even (854x480 vs 1280x720 gives 1922 vs 1920) */
#define SCALE_ASPECT_TOLERANCE 2

/**
 * Size `w`x`h` scales to for `display_resolution`.
 */
static int scaled_size(unsigned w, unsigned h, unsigned display_resolution,
                       unsigned *scaled_w, unsigned *scaled_h)
{
    if (!w || !h) return -EINVAL;

    const bool portrait = h > w;
    const unsigned short_side = portrait ? w : h;
    const unsigned long_side = portrait ? h : w;

    /* already at the display size: keep the source's dimensions exactly */
    if (short_side == display_resolution) {
        *scaled_w = w;
        *scaled_h = h;
        return 0;
    }

    /* long side at the source's aspect ratio, rounded to the nearest even */
    const unsigned long long scaled =
        ((unsigned long long) long_side * display_resolution + short_side) /
        (2ull * short_side) * 2;
    /* wider than 4:1 (32:9 is ~3.6:1) is almost certainly a bad input, and
     * the scaled picture would be huge */
    if (scaled > 4ull * display_resolution) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "scaling %ux%u to a display resolution of %u would give a "
                 "long side of %llu, over the limit of %u\n",
                 w, h, display_resolution, scaled, 4 * display_resolution);
        return -EINVAL;
    }

    *scaled_w = portrait ? display_resolution : (unsigned) scaled;
    *scaled_h = portrait ? (unsigned) scaled : display_resolution;
    return 0;
}

/**
 * Filter to resize `w`x`h` to `final_w`x`final_h`: none to pick when the size
 * is kept, bicubic to upscale and lanczos to downscale. This looks at the
 * final size, which can differ by a few pixels from `scaled_size()` for a
 * picture that is already at the display size.
 */
static enum VmafResampleFilter resample_filter(unsigned w, unsigned h,
                                               unsigned final_w,
                                               unsigned final_h)
{
    if (w == final_w && h == final_h) return VMAF_RESAMPLE_DEFAULT;
    return (unsigned long long) w * h < (unsigned long long) final_w * final_h ?
        VMAF_RESAMPLE_BICUBIC : VMAF_RESAMPLE_LANCZOS;
}

static bool within_tolerance(unsigned a, unsigned b)
{
    return (a > b ? a - b : b - a) <= SCALE_ASPECT_TOLERANCE;
}

int vmaf_conversion_policy_scale(const VmafPicture *ref,
                                 const VmafPicture *dist,
                                 unsigned display_resolution,
                                 bool *needs_conversion,
                                 VmafPictureConvertTarget *ref_target,
                                 VmafPictureConvertTarget *dist_target)
{
    if (!ref || !dist || !needs_conversion || !ref_target || !dist_target ||
        !display_resolution)
    {
        return -EINVAL;
    }

    unsigned ref_w, ref_h, dist_w, dist_h;
    int err = scaled_size(ref->w[0], ref->h[0], display_resolution,
                          &ref_w, &ref_h);
    if (err) return err;
    err = scaled_size(dist->w[0], dist->h[0], display_resolution,
                      &dist_w, &dist_h);
    if (err) return err;

    /* a landscape and a portrait picture never match, however close their
     * scaled sizes are (1081x1080 against 1080x1081); a square matches both */
    const bool mixed_orientation =
        (ref->w[0] > ref->h[0] && dist->h[0] > dist->w[0]) ||
        (ref->h[0] > ref->w[0] && dist->w[0] > dist->h[0]);
    if (mixed_orientation ||
        !within_tolerance(ref_w, dist_w) || !within_tolerance(ref_h, dist_h))
    {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "the reference (%ux%u) and distorted (%ux%u) pictures do not "
                 "have the same orientation and aspect ratio, so they cannot "
                 "be scaled to a common size\n",
                 ref->w[0], ref->h[0], dist->w[0], dist->h[0]);
        return -EINVAL;
    }

    /* both pictures are scaled to the size the reference scales to */
    *dist_target = *ref_target;
    ref_target->w = dist_target->w = ref_w;
    ref_target->h = dist_target->h = ref_h;
    ref_target->resample_filter =
        resample_filter(ref->w[0], ref->h[0], ref_w, ref_h);
    dist_target->resample_filter =
        resample_filter(dist->w[0], dist->h[0], ref_w, ref_h);

    if (ref->w[0] != ref_w || ref->h[0] != ref_h ||
        dist->w[0] != ref_w || dist->h[0] != ref_h)
    {
        *needs_conversion = true;
    }
    return 0;
}
