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

#ifndef __VMAF_SRC_CONVERSION_POLICY_H__
#define __VMAF_SRC_CONVERSION_POLICY_H__

#include <stdbool.h>

#include "libvmaf/picture.h"

/**
 * Decide whether `ref` and `dist` need conversion before feature
 * extraction, given the target the model(s) in use declare.
 *
 * `model_target` is the model's declared conversion target, or NULL when
 * the model declares none. A model without a target is not
 * colorspace-aware, so this is a pass-through: `*needs_conversion` is set
 * to false and `target` is left untouched, regardless of how `ref` and
 * `dist` are tagged.
 *
 * When the model declares a target, `ref` and `dist` must each have all of
 * `range`, `primaries`, `trc` and `matrix` specified. Guessing the source
 * colorimetry would yield a plausible but wrong score, so anything less
 * (including a caller that never set `VmafPicture.color`, which is
 * indistinguishable from `*_UNKNOWN`) returns -EINVAL and logs which
 * attributes are missing.
 *
 * Otherwise `target` is filled from `*model_target`: `color` always, and
 * `pix_fmt` / `bpc` when the model pins them (UNKNOWN / 0 otherwise, meaning
 * "keep each picture's own"). `*needs_conversion` is true unless both `ref`
 * and `dist` already match the target. `ref` and `dist` are allowed to
 * differ from each other: each is converted to the same target. The caller
 * is responsible for `target->w`/`h`/`resample_filter`, which
 * vmaf_conversion_policy_scale() fills in when the model declares a
 * display resolution.
 */
int vmaf_conversion_policy_target(const VmafPicture *ref,
                                  const VmafPicture *dist,
                                  const VmafPictureConvertTarget *model_target,
                                  bool *needs_conversion,
                                  VmafPictureConvertTarget *target);

/**
 * Fill in `ref_target->w`/`h`/`resample_filter` and `dist_target` for a model
 * that declares a display resolution, after vmaf_conversion_policy_target().
 * `ref_target` is the target that call returned.
 *
 * `display_resolution` is the short side both pictures are scaled to (1080 or
 * 2160). The orientation follows the source, so portrait video scores at e.g.
 * 1080x1920, and the long side keeps the source's aspect ratio, rounded to
 * an even number. Upscaling uses bicubic, downscaling lanczos.
 *
 * `ref` and `dist` may differ in size, and are both scaled to the size the
 * reference scales to. Their orientations and aspect ratios must match,
 * within the rounding of the long side to even (a landscape and a portrait
 * picture never match; a square matches either); otherwise -EINVAL. An aspect
 * ratio that would scale the long side beyond 4x `display_resolution` is also
 * -EINVAL.
 *
 * The filter depends on each picture's own scaling direction, so `ref_target`
 * holds the reference's and `dist_target` the distorted picture's. `dist_target`
 * is set from `ref_target`, so the two are otherwise identical. `*needs_conversion` is set to true when either picture
 * is not at the target size, and left as is otherwise.
 */
int vmaf_conversion_policy_scale(const VmafPicture *ref,
                                 const VmafPicture *dist,
                                 unsigned display_resolution,
                                 bool *needs_conversion,
                                 VmafPictureConvertTarget *ref_target,
                                 VmafPictureConvertTarget *dist_target);

/** True when all of `a`'s color attributes equal `b`'s. */
bool vmaf_conversion_policy_color_equal(const VmafColor *a, const VmafColor *b);

/**
 * True when `pic` already matches `target`: same color, and the same pixel
 * format, bit depth and size wherever `target` pins them.
 */
bool vmaf_conversion_policy_picture_matches(const VmafPicture *pic,
                                            const VmafPictureConvertTarget *target);

/** True when two targets are the same in color, pixel format and bit depth. */
bool vmaf_conversion_policy_target_equal(const VmafPictureConvertTarget *a,
                                         const VmafPictureConvertTarget *b);

#endif /* __VMAF_SRC_CONVERSION_POLICY_H__ */
