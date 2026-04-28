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
#include "feature/adm_csf_tools.h"

#define EPS 0.00001

/* Test support function */
int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static char *test_adm_csf()
{
    mu_assert("adm csf mismatch", almost_equal(adm_native_csf(3, 3.0, 1080, 0), 0.986264592442799));
    mu_assert("adm csf mismatch", almost_equal(adm_native_csf(3, 3.0, 1080, 45), 0.8773599546532113));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_adm_csf);
    return NULL;
}
