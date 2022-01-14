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

#include "test.h"
#include "ref.h"
#include "feature/luminance_tools.h"

#define EPS 0.00001

/* Test support function */
int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static char *test_bt1886_eotf()
{
    double L = bt1886_eotf(0.5);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 58.716634));
    L = bt1886_eotf(0.1);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 1.576653));
    L = bt1886_eotf(0.9);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 233.819503));

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_bt1886_eotf);

    return NULL;
}