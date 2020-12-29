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

#include <stddef.h>
#include <string.h>

#include "test.h"

#include "feature/feature_name.h"

static char *test_feature_name()
{
    char buf[VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE];
    char *feature_name;

    char *name = "VMAF_integer_feature_vif_scale0_score";
    char *key = "vif_enhn_gain_limit";
    const double val = 1.0;

    feature_name = vmaf_feature_name(name, NULL, val, &buf[0],
                                     VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE);
    mu_assert("name should not be modified", !strcmp(feature_name, name));

    feature_name = vmaf_feature_name(name, key, val, &buf[0],
                                     VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE);
    mu_assert("name should have been formatted according to key/val",
        !strcmp(feature_name, "integer_vif_scale0_egl_1"));

    feature_name = vmaf_feature_name(name, key, 1.23, &buf[0],
                                     VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE);
    mu_assert("name should have been formatted according to key/val",
        !strcmp(feature_name, "integer_vif_scale0_egl_1.23"));

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_feature_name);
    return NULL;
}
