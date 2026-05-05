/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include "test.h"
#include "ref.h"
#include "feature/motion_blend_tools.h"

#define EPS 0.00001

/* Test support function */
int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static char *test_motion_blend()
{
    mu_assert("motion blend", almost_equal(motion_blend(50.0, 0.5, 40.0), 45.0));
    mu_assert("motion blend offset higher", almost_equal(motion_blend(40.0, 0.5, 50.0), 40.0));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_motion_blend);
    return NULL;
}
