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

#ifndef LUMINANCE_TOOLS_H_
#define LUMINANCE_TOOLS_H_

typedef double (*EOTF)(double V);

/*
 * Limited pixel range means that only values between 16 and 235 will be used in 8 bits
 * (rescale the bounds appropriately for other bitdepths).
 * Full pixel range means that values from 0 to 2^bitdepth - 1 will be used.
 */
typedef enum  {
    VMAF_PIXEL_RANGE_LIMITED,
    VMAF_PIXEL_RANGE_FULL,
} PixelRange;

/*
 * Contains the necessary information to normalize a luma value down to [0, 1].
 */
typedef struct LumaRange {
    int foot;
    int head;
} LumaRange;

/*
 * Constructor for the LumaRange struct.
 */
LumaRange LumaRange_init(int bitdepth, PixelRange pix_range);

/*
 * Determines the lowest and highest value possible for a given bitdepth and PixelRange.
 * Is used in the constructor for LumaRange, not to be used directly.
 */
void range_foot_head(int bitdepth, PixelRange pix_range, int *foot, int *head);

/*
 * Takes a luma value and a LumaRange struct and returns a normalized value in the [0, 1] range.
 */
double normalize_range(int sample, LumaRange range);

/*
 * Takes a normalized luma value in the [0, 1] range and returns a luminance value.
 */
double bt1886_eotf(double V);

/*
 * Takes a luma value, normalizes it and applies the given EOTF to return a luminance value.
 */
double get_luminance(int sample, LumaRange luma_range, EOTF eotf);

#endif