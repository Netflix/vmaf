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

static inline uint32_t
edge_16(bool horizontal, const uint16_t *src, int width,
        int height, int stride, int i, int j)
{
    const int radius = filter_width / 2;
    uint32_t accum = 0;

    // MIRROR | ЯOЯЯIM
    for (int k = 0; k < filter_width; ++k) {
        int i_tap = horizontal ? i : i - radius + k;
        int j_tap = horizontal ? j - radius + k : j;

        if (horizontal) {
            if (j_tap < 0)
                j_tap = -j_tap;
            else if (j_tap >= width)
                j_tap = width - (j_tap - width + 1);
        } else {
            if (i_tap < 0)
                i_tap = -i_tap;
            else if (i_tap >= height)
                i_tap = height - (i_tap - height + 1);
        }
        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

#endif /* _FEATURE_MOTION_H_ */
