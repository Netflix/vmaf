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
#include "feature/ciede.c"

static int close_enough(float a, float b)
{
    const float epsilon = 1e-9f;
    return fabs(a - b) < epsilon;
}

static const KSubArgs default_ksub = { .l = 0.65, .c = 1.0, .h = 4.0 };

static char *test_ciede()
{
    const LABColor color_1 = { .l = 0.052488625, .a = -0.587470829, .b = -8.98771572 };
    const LABColor color_2 = { .l = 0.465437293, .a = 0.386364758, .b = -12.7648535 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 2.54780269",
              close_enough(de00, 2.54780269));

    return NULL;
}

static char *test_ciede2()
{
    const LABColor color_1 = { .l = 87.156334, .a = -12.049645, .b = -1.205325 };
    const LABColor color_2 = { .l = 83.455727, .a = -9.040445, .b = -8.894289 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.22714281",
              close_enough(de00, 4.22714281));

    return NULL;
}

static char *test_ciede3()
{
    const LABColor color_1 = { .l = 79.718491, .a = 9.109915, .b = 13.727915 };
    const LABColor color_2 = { .l = 78.717224, .a = 7.526546, .b = 5.597448 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.26012468",
              close_enough(de00, 4.26012468));

    return NULL;
}

static char *test_ciede4()
{
    const LABColor color_1 = { .l = 99.205299, .a = -3.339410, .b = 1.205873 };
    const LABColor color_2 = { .l = 97.991730, .a = -2.497345, .b = 2.473533 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 1.26915979",
              close_enough(de00, 1.26915979));

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_ciede);
    mu_run_test(test_ciede2);
    mu_run_test(test_ciede3);
    mu_run_test(test_ciede4);
    return NULL;
}
