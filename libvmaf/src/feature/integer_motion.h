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

#ifndef FEATURE_MOTION_H_
#define FEATURE_MOTION_H_

#include <stdbool.h>
#include <stdint.h>

static const uint16_t filter[5] = { 3571, 16004, 26386, 16004, 3571 };
static const int filter_width = sizeof(filter) / sizeof(filter[0]);

/* Whole-sample symmetric ("reflect-101") boundary index.
 * Reflects repeatedly so that any size >= 1 is safe; for size >= 3 and
 * |overshoot| <= 2 (the 5-tap/radius-2 call contract) at most one
 * reflection is taken, so behavior is bit-identical to the historical
 * single-bounce version. size <= 1 must be special-cased: the reflection
 * period 2*(size-1) is 0 (or negative) and the loop would not terminate.
 * size is always >= 1 in the real call graph, so covering size <= 0 as well
 * only makes the function total; it changes no reachable result. */
static inline int motion_mirror(int idx, int size)
{
    if (size <= 1) return 0;
    while (idx < 0 || idx >= size) {
        if (idx < 0) idx = -idx;
        else idx = 2 * size - idx - 2;
    }
    return idx;
}

#endif /* _FEATURE_MOTION_H_ */
