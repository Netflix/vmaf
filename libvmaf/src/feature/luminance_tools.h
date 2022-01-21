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

#ifndef VMAF_LUMINANCE_TOOLS_H_
#define VMAF_LUMINANCE_TOOLS_H_

typedef double (*VmafEOTF)(double V);

/*
 * Limited pixel range means that only values between 16 and 235 will be used in 8 bits
 * (rescale the bounds appropriately for other bitdepths).
 * Full pixel range means that values from 0 to 2^bitdepth - 1 will be used.
 */
enum VmafPixelRange {
    VMAF_PIXEL_RANGE_UNKNOWN,
    VMAF_PIXEL_RANGE_LIMITED,
    VMAF_PIXEL_RANGE_FULL,
};

/*
 * Contains the necessary information to normalize a luma value down to [0, 1].
 */
typedef struct VmafLumaRange {
    int foot;
    int head;
} VmafLumaRange;

/*
 * Constructor for the LumaRange struct.
 */
int vmaf_luminance_init_luma_range(VmafLumaRange *luma_range, int bitdepth, enum VmafPixelRange pix_range);

/*
 * Returns the EOTF corresponding to the string given.
 * eotf_str must be one of ['bt1886', 'pq']
 */
int vmaf_luminance_init_eotf(VmafEOTF *eotf, const char *eotf_str);


/*
 * Takes a normalized luma value in the [0, 1] range and returns a luminance value.
 */
double vmaf_luminance_bt1886_eotf(double V);

/*
 * Takes a normalized luma value in the [0, 1] range and returns a luminance value.
 */
double vmaf_luminance_pq_eotf(double V);

/*
 * Takes a luma value, normalizes it and applies the given VmafEOTF
 * to return a luminance value.
 */
double vmaf_luminance_get_luminance(int sample, VmafLumaRange luma_range, VmafEOTF eotf);

#endif