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

#include <math.h>

#include "luminance_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define BT1886_GAMMA (2.4)
#define BT1886_LW (300.0)
#define BT1886_LB (0.01)

inline double bt1886_eotf(double V) {
    double a = pow(pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA), BT1886_GAMMA);
    double b = pow(BT1886_LB, 1.0 / BT1886_GAMMA) / (pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA));
    double L = a * pow(MAX(V + b, 0), BT1886_GAMMA);
    return L;
}