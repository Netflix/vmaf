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
#include <stdio.h>
#include <string.h>

#include "alias.h"

typedef struct {
    const char *name;
    struct { const char *name, *alias; } key;
} Template;

static const Template template_list[] = {
    {
        .name = "VMAF_integer_feature_motion2_score",
        .key = { .name = "motion_force_zero", .alias = "force" },
    },
    {
        .name = "VMAF_integer_feature_motion_score",
        .key = { .name = "motion_force_zero", .alias = "force" },
    },
    {
        .name = "VMAF_integer_feature_adm2_score",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_scale0",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_scale1",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_scale2",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_scale3",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_num",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_den",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_num_scale0",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_den_scale0",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_num_scale1",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_den_scale1",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_num_scale2",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_den_scale2",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_num_scale3",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_adm_den_scale3",
        .key = { .name = "adm_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "VMAF_integer_feature_vif_scale0_score",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "VMAF_integer_feature_vif_scale1_score",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "VMAF_integer_feature_vif_scale2_score",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "VMAF_integer_feature_vif_scale3_score",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_num",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_den",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_num_scale0",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_den_scale0",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_num_scale1",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_den_scale1",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_num_scale2",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_den_scale2",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_num_scale3",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
    {
        .name = "integer_vif_den_scale3",
        .key = { .name = "vif_enhn_gain_limit", .alias = "egl" },
    },
};

static int in_template_list(char *name)
{
    const unsigned n = sizeof(template_list) / sizeof(template_list[0]);
    for(unsigned i = 0; i < n; i++) {
        if (!strcmp(name, template_list[i].name))
            return 1;
    }
    return 0;
}

static const char *key_alias(char *key)
{
    const unsigned n = sizeof(template_list) / sizeof(template_list[0]);
    for(unsigned i = 0; i < n; i++) {
        if (!strcmp(key, template_list[i].key.name))
            return template_list[i].key.alias;
    }
    return key;
}

char *vmaf_feature_name(char *name, char *key, double val,
                        char *buf, size_t buf_sz)
{
    if (!key) return name;
    if (!in_template_list(name)) return name;

    memset(buf, 0, buf_sz);
    snprintf(buf, buf_sz - 1, "%s_%s_%.2f",
             vmaf_feature_name_alias(name), key_alias(key), val);
    return buf;
}
