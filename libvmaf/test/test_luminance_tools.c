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

static char *test_range_foot_head()
{
    int foot, head;

    range_foot_head(8, "standard", &foot, &head);
    mu_assert("wrong 'standard' 8b range computation", (foot==16 && head==235));
    range_foot_head(8, "full", &foot, &head);
    mu_assert("wrong 'full' 8b range computation", (foot==0 && head==255));
    range_foot_head(10, "standard", &foot, &head);
    mu_assert("wrong 'standard' 10b range computation", (foot==64 && head==940));

    return NULL;
}

static char *test_get_luminance()
{
    LumaRange range_8b_standard = LumaRange_init(8, "standard");
    LumaRange range_10b_standard = LumaRange_init(10, "standard");
    LumaRange range_10b_full = LumaRange_init(10, "full");

    double L;
    L = get_luminance(100, range_8b_standard, bt1886_eotf);
    mu_assert("wrong 'standard' 8b luminance bt1886", almost_equal(L, 31.68933962217197));
    L = get_luminance(400, range_10b_standard, bt1886_eotf);
    mu_assert("wrong 'standard' 10b luminance bt1886", almost_equal(L, 31.68933962217197));
    L = get_luminance(400, range_10b_full, bt1886_eotf);
    printf("%.15lf\n", L);
    mu_assert("wrong 'full' 10b luminance bt1886", almost_equal(L, 33.133003757557773));

    return NULL;
}

static char *test_normalize_range()
{
    LumaRange range_8b_standard = LumaRange_init(8, "standard");
    LumaRange range_8b_full = LumaRange_init(8, "full");
    LumaRange range_10b_standard = LumaRange_init(10, "standard");

    double n = normalize_range(0, range_8b_full);
    mu_assert("wrong 'full' 8b normalize range", almost_equal(n, 0.0));
    n = normalize_range(128, range_8b_standard);
    mu_assert("wrong 'standard' 8b normalize range", almost_equal(n, 0.5114155251141552));
    n = normalize_range(255, range_8b_standard);
    mu_assert("wrong 'standard' 8b normalize range", almost_equal(n, 1.0));

    n = normalize_range(65, range_10b_standard);
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.001141552511415525));
    n = normalize_range(512, range_10b_standard);
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.5114155251141552));
    n = normalize_range(939, range_10b_standard);
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.9988584474885844));

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_bt1886_eotf);
    mu_run_test(test_range_foot_head);
    mu_run_test(test_get_luminance);
    mu_run_test(test_normalize_range);

    return NULL;
}