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
#include "feature/barten_csf_tools.h"

#define EPS 0.00001

/* Test support function */
int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static char *test_barten_csf()
{
    mu_assert("barten rod/cone sensitivity mismatch", almost_equal(barten_rod_cone_sens(150.0), 30.141537));
    mu_assert("barten rod/cone sensitivity mismatch", almost_equal(barten_rod_cone_sens(100.0), 30.121951521378232));
    mu_assert("barten mtf mismatch", almost_equal(barten_mtf(3.0), 0.5792610023712589));
    mu_assert("barten csf mismatch lum 100", almost_equal(barten_csf(3, 3.0, 1080, 100.0, 1.0), 26.97588185355269));
    mu_assert("barten csf mismatch lum 100", almost_equal(barten_csf(3, 3.0, 1080, 100.0, 0.5), 13.4879409268));
    mu_assert("barten csf mismatch lum 80", almost_equal(barten_csf(3, 3.0, 1080, 80.0, 1.0), 26.949246017567383));
    mu_assert("barten csf mismatch lum 50", almost_equal(barten_csf(3, 3.0, 1080, 50.0, 1.0), 26.9958));
    // although we clamp below 0.002 for the interpolation, we still use the un-clamped in the rest of the CSF
    mu_assert("barten csf mismatch lum 0.001", almost_equal(barten_csf(3, 3.0, 1080, 0.001, 1.0), 0.112742));
    mu_assert("barten csf mismatch lum 0.002", almost_equal(barten_csf(3, 3.0, 1080, 0.002, 1.0), 0.154019));
    mu_assert("barten csf mismatch lum 0.01", almost_equal(barten_csf(3, 3.0, 1080, 0.01, 1.0), 0.439781));
    mu_assert("barten csf mismatch lum 0.05", almost_equal(barten_csf(3, 3.0, 1080, 0.05, 1.0), 1.240470));
    mu_assert("barten csf mismatch lum 1.0", almost_equal(barten_csf(3, 3.0, 1080, 1.0, 1.0), 9.063454));
    mu_assert("barten csf mismatch lum 10.0", almost_equal(barten_csf(3, 3.0, 1080, 10.0, 1.0), 23.722300));
    mu_assert("barten csf mismatch lum 150.0", almost_equal(barten_csf(3, 3.0, 1080, 150.0, 1.0), 27.113543));
    mu_assert("barten csf mismatch lum 155.0", almost_equal(barten_csf(3, 3.0, 1080, 155.0, 1.0), 27.114515));
    mu_assert("linear interpolate", almost_equal(linear_interpolate(log10(20), barten_csf_params[4][1], log10(150.0),
                                                                    barten_csf_params[5][1], log10(50.0)), 0.361679));
    mu_assert("csf blend scale 0 H/V 1080p 3H", almost_equal(barten_watson_blend_csf(0, 0, 3.0, 1080), 0.01183));
    mu_assert("csf blend scale 0 D 1080p 3H", almost_equal(barten_watson_blend_csf(0, 1, 3.0, 1080), 0.004302));
    mu_assert("csf blend scale 1 H/V 1080p 3H", almost_equal(barten_watson_blend_csf(1, 0, 3.0, 1080), 0.025026));
    mu_assert("csf blend scale 1 D 1080p 3H", almost_equal(barten_watson_blend_csf(1, 1, 3.0, 1080), 0.011778));
    mu_assert("csf blend scale 2 H/V 1080p 3H", almost_equal(barten_watson_blend_csf(2, 0, 3.0, 1080), 0.04295));
    mu_assert("csf blend scale 2 D 1080p 3H", almost_equal(barten_watson_blend_csf(2, 1, 3.0, 1080), 0.023918));
    mu_assert("csf blend scale 3 H/V 1080p 3H", almost_equal(barten_watson_blend_csf(3, 0, 3.0, 1080), 0.058621));
    mu_assert("csf blend scale 3 D 1080p 3H", almost_equal(barten_watson_blend_csf(3, 1, 3.0, 1080), 0.035901));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_barten_csf);
    return NULL;
}
