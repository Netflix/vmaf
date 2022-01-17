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

#include <errno.h>
#include <math.h>

#include "log.h"
#include "luminance_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define BT1886_GAMMA (2.4)
#define BT1886_LW (300.0)
#define BT1886_LB (0.01)

static inline int clip(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

/*
 * Standard range for 8 bit: [16, 235]
 * Standard range for 10 bit: [64, 940]
 * Full range for 8 bit: [0, 255]
 * Full range for 10 bit: [0, 1023]
 */
static inline int range_foot_head(int bitdepth, enum VmafPixelRange pix_range, int *foot, int *head) {
    switch (pix_range) {
    case VMAF_PIXEL_RANGE_LIMITED:
        *foot = 16 * (1 << (bitdepth - 8));
        *head = 235 * (1 << (bitdepth - 8));
        break;
    case VMAF_PIXEL_RANGE_FULL:
        *foot = 0;
        *head = (1 << bitdepth) - 1;
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "unknown pixel range received");
        return -EINVAL;
    }
    return 0;
}

static inline double normalize_range(int sample, VmafLumaRange range) {
    int clipped_sample = clip(sample, range.foot, range.head);
    return (double)(clipped_sample - range.foot) / (range.head - range.foot);
}

int vmaf_luminance_init_luma_range(VmafLumaRange *luma_range, int bitdepth, enum VmafPixelRange pix_range) {
    int err = range_foot_head(bitdepth, pix_range, &(luma_range->foot), &(luma_range->head));
    return err;
}

double vmaf_luminance_bt1886_eotf(double V) {
    double a = pow(pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA), BT1886_GAMMA);
    double b = pow(BT1886_LB, 1.0 / BT1886_GAMMA) / (pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA));
    double L = a * pow(MAX(V + b, 0), BT1886_GAMMA);
    return L;
}

double vmaf_luminance_get_luminance(int sample, VmafLumaRange luma_range, VmafEOTF eotf) {
    double normalized = normalize_range(sample, luma_range);
    return eotf(normalized);
}