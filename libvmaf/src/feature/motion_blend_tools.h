/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include "common/macros.h"

#pragma once

#ifndef MOTION_BLEND_TOOLS_H_
#define MOTION_BLEND_TOOLS_H_

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static FORCE_INLINE inline double motion_blend(double motion_score, double blend_factor, double blend_offset) {
    /* return a blended motion score */
    return motion_score * blend_factor + (1 - blend_factor) * MIN(blend_offset, motion_score);
}

#endif /* MOTION_BLEND_TOOLS_H_ */
