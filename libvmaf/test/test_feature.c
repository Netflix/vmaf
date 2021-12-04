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
#include <stdlib.h>
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

/*
static char *test_feature_name_from_options()
{
    typedef struct TestState {
        bool opt_bool;
        double opt_double;
        int opt_int;
    } TestState;

#define opt_bool_default false
#define opt_double_default 3.14
#define opt_int_default 200

    static VmafOption options[] = {
        {
            .name = "opt_bool",
            .offset = offsetof(TestState, opt_bool),
            .type = VMAF_OPT_TYPE_BOOL,
            .default_val.b = opt_bool_default,
        },
        {
            .name = "opt_double",
            .alias = "opt_double_alias",
            .offset = offsetof(TestState, opt_double),
            .type = VMAF_OPT_TYPE_DOUBLE,
            .default_val.d = opt_double_default,
        },
        {
            .name = "opt_int",
            .alias = "opt_int_alias",
            .offset = offsetof(TestState, opt_int),
            .type = VMAF_OPT_TYPE_INT,
            .default_val.i = opt_int_default,
        },
        { 0 }
    };

    TestState s1 = {
        .opt_bool = opt_bool_default,
        .opt_double = opt_double_default,
        .opt_int = opt_int_default,
    };

    char *feature_name1 =
        vmaf_feature_name_from_options("feature_name", options, &s1, 3,
                                       &s1.opt_bool, &s1.opt_double, &s1.opt_int);

    mu_assert("when all options are default, feature_name should not change",
              !strcmp(feature_name1, "feature_name"));

    free(feature_name1);

    TestState s2 = {
        .opt_bool = !opt_bool_default,
        .opt_double = opt_double_default,
        .opt_int = opt_int_default,
    };

    char *feature_name2 =
        vmaf_feature_name_from_options("feature_name", options, &s2, 3,
                                       &s2.opt_bool, &s2.opt_double, &s2.opt_int);

    mu_assert("when opt_bool has a non-default value, "
              "feature_name should have a non-aliased opt_bool suffix",
              !strcmp(feature_name2, "feature_name_opt_bool_1"));

    free(feature_name2);

    TestState s3 = {
        .opt_bool = opt_bool_default,
        .opt_double = opt_double_default + 1,
        .opt_int = opt_int_default,
    };

    char *feature_name3 =
        vmaf_feature_name_from_options("feature_name", options, &s3, 3,
                                       &s3.opt_bool, &s3.opt_double, &s3.opt_int);

    mu_assert("when opt_double has a non-default value, "
              "feature_name should have a aliased opt_double_alias suffix",
              !strcmp(feature_name3, "feature_name_opt_double_alias_4.14"));

    free(feature_name3);

    TestState s4 = {
        .opt_bool = !opt_bool_default,
        .opt_double = opt_double_default + 1,
        .opt_int = opt_int_default + 1,
    };

    char *feature_name4 =
        vmaf_feature_name_from_options("feature_name", options, &s4, 3,
                                       &s4.opt_bool, &s4.opt_double, &s4.opt_int);


    mu_assert("when all opts have a non-default value, "
              "feature_name should have a suffix with aliases and values",
              !strcmp(feature_name4, "feature_name_opt_bool_1_opt_double_alias_4.14_opt_int_alias_201"));

    free(feature_name4);

    TestState s5 = s4;

    char *feature_name5 =
        vmaf_feature_name_from_options("feature_name", options, &s5, 3,
                                       &s5.opt_double, &s5.opt_bool, &s5.opt_int);

    mu_assert("feature_name should have a suffix with aliases and values, "
              "ordering should not follow the ordering of variadac params,"
              "rather it should follow the order of options",
              !strcmp(feature_name5, "feature_name_opt_bool_1_opt_double_alias_4.14_opt_int_alias_201"));

    free(feature_name5);

    TestState s6 = s4;

    char *feature_name6 =
        vmaf_feature_name_from_options("feature_name", options, &s6, 1,
                                       &s6.opt_double);

    mu_assert("feature_name should have a single opt_double_alias suffix, "
              "although all members of TestState have non-default values, "
              "since just one variadic param is passed",
              !strcmp(feature_name6, "feature_name_opt_double_alias_4.14"));

    free(feature_name6);

    return NULL;
}
*/

char *run_tests()
{
    mu_run_test(test_feature_name);
    //mu_run_test(test_feature_name_from_options);
    return NULL;
}
