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
#include <string.h>

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

int vmaf_luminance_init_eotf(VmafEOTF *eotf, const char *eotf_str) {
    if (strcmp(eotf_str, "bt1886") == 0) {
        *eotf = vmaf_luminance_bt1886_eotf;
    }
    else if (strcmp(eotf_str, "pq") == 0) {
        *eotf = vmaf_luminance_pq_eotf;
    }
    else {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "unknown EOTF received");
        return -EINVAL;
    }
    return 0;
}

double vmaf_luminance_bt1886_eotf(double V) {
    double a = pow(pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA), BT1886_GAMMA);
    double b = pow(BT1886_LB, 1.0 / BT1886_GAMMA) / (pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA));
    return a * pow(MAX(V + b, 0), BT1886_GAMMA);
}

double vmaf_luminance_pq_eotf(double V) {
    const double m_1 = 0.1593017578125;
    const double m_2 = 78.84375;
    const double c_1 = 0.8359375;
    const double c_2 = 18.8515625;
    const double c_3 = 18.6875;  // c_3 = c_1 + c_2 - 1
    double num = pow(V, 1.0 / m_2) - c_1;
    double num_clipped = MAX(num, 0);
    double den = c_2 - c_3 * pow(V, 1.0 / m_2);
    return 10000 * pow(num_clipped / den, 1.0 / m_1);
}

double vmaf_luminance_get_luminance(int sample, VmafLumaRange luma_range, VmafEOTF eotf) {
    double normalized = normalize_range(sample, luma_range);
    return eotf(normalized);
}