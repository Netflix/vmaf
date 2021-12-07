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

#include "feature/feature_name.c"

static char *test_feature_name_from_options()
{
    typedef struct TestState {
        bool opt_bool;
        double opt_double;
        int opt_int;
        bool opt_bool2;
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
            .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        },
        {
            .name = "opt_double",
            .alias = "opt_double_alias",
            .offset = offsetof(TestState, opt_double),
            .type = VMAF_OPT_TYPE_DOUBLE,
            .default_val.d = opt_double_default,
            .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        },
        {
            .name = "opt_int",
            .alias = "opt_int_alias",
            .offset = offsetof(TestState, opt_int),
            .type = VMAF_OPT_TYPE_INT,
            .default_val.i = opt_int_default,
            .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        },
        {
            .name = "opt_bool2",
            .offset = offsetof(TestState, opt_bool),
            .type = VMAF_OPT_TYPE_BOOL,
            .default_val.b = opt_bool_default,
            .flags = 0,
        },
        { 0 }
    };

    TestState s1 = {
        .opt_bool = opt_bool_default,
        .opt_double = opt_double_default,
        .opt_int = opt_int_default,
        .opt_bool2 = opt_bool_default,
    };

    char *feature_name1 =
        vmaf_feature_name_from_options("feature_name", options, &s1);

    mu_assert("when all options are default, feature_name should not change",
              !strcmp(feature_name1, "feature_name"));

    free(feature_name1);

    TestState s2 = {
        .opt_bool = !opt_bool_default,
        .opt_double = opt_double_default,
        .opt_int = opt_int_default,
        .opt_bool2 = opt_bool_default,
    };

    char *feature_name2 =
        vmaf_feature_name_from_options("feature_name", options, &s2);

    mu_assert("when opt_bool has a non-default value, "
              "feature_name should have a non-aliased opt_bool suffix",
              !strcmp(feature_name2, "feature_name_opt_bool"));

    free(feature_name2);

    TestState s3 = {
        .opt_bool = opt_bool_default,
        .opt_double = opt_double_default + 1,
        .opt_int = opt_int_default,
        .opt_bool2 = opt_bool_default,
    };

    char *feature_name3 =
        vmaf_feature_name_from_options("feature_name", options, &s3);

    mu_assert("when opt_double has a non-default value, "
              "feature_name should have a aliased opt_double_alias suffix",
              !strcmp(feature_name3, "feature_name_opt_double_alias_4.14"));

    free(feature_name3);

    TestState s4 = {
        .opt_bool = !opt_bool_default,
        .opt_double = opt_double_default + 1,
        .opt_int = opt_int_default + 1,
        .opt_bool2 = !opt_bool_default,
    };

    char *feature_name4 =
        vmaf_feature_name_from_options("feature_name", options, &s4);


    mu_assert("when all opts have a non-default value, "
              "feature_name should have a suffix with aliases and values. "
              "opt_bool2 should not parameterize since its flags are unset.",
              !strcmp(feature_name4, "feature_name_opt_bool_opt_double_alias_4.14_opt_int_alias_201"));

    free(feature_name4);

    TestState s5 = s4;

    char *feature_name5 =
        vmaf_feature_name_from_options("feature_name", options, &s5);

    mu_assert("feature_name should have a suffix with aliases and values, "
              "ordering should not follow the ordering of variadac params,"
              "rather it should follow the order of options",
              !strcmp(feature_name5, "feature_name_opt_bool_opt_double_alias_4.14_opt_int_alias_201"));

    free(feature_name5);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_feature_name_from_options);
    return NULL;
}
