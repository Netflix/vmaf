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
 * is responsible for `target->w`/`h`/`resample_filter`.
 *
 * NOTE: a resolution-mismatch policy (deciding `target->w`/`h` when
 * reference and distorted differ) is expected to land here too once
 * that's in scope, rather than in a separate module - hence
 * `conversion_policy` rather than `colorspace_policy`.
 */
int vmaf_conversion_policy_target(const VmafPicture *ref,
                                  const VmafPicture *dist,
                                  const VmafPictureConvertTarget *model_target,
                                  bool *needs_conversion,
                                  VmafPictureConvertTarget *target);

/** True when all of `a`'s color attributes equal `b`'s. */
bool vmaf_conversion_policy_color_equal(const VmafColor *a, const VmafColor *b);

/**
 * True when `pic` already matches `target`: same color, and the same pixel
 * format and bit depth wherever `target` pins them.
 */
bool vmaf_conversion_policy_picture_matches(const VmafPicture *pic,
                                            const VmafPictureConvertTarget *target);

/** True when two targets are the same in color, pixel format and bit depth. */
bool vmaf_conversion_policy_target_equal(const VmafPictureConvertTarget *a,
                                         const VmafPictureConvertTarget *b);

#endif /* __VMAF_SRC_CONVERSION_POLICY_H__ */
