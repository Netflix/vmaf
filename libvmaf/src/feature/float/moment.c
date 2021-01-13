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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mem.h"
#include "moment_options.h"

int compute_1st_moment(const float *pic, int w, int h, int stride, double *score)
{
    double cum = 0;
    float pic_;

    int stride_ = stride / sizeof(float);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            pic_ = pic[i * stride_ + j];
            cum += pic_;
        }
    }
    cum /= (w * h);

    *score = cum;

    return 0;
}

int compute_2nd_moment(const float *pic, int w, int h, int stride, double *score)
{
    double cum = 0;
    float pic_;

    int stride_ = stride / sizeof(float);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            pic_ = pic[i * stride_ + j];
            cum += pic_ * pic_;
        }
    }
    cum /= (w * h);

    *score = cum;

    return 0;
}
