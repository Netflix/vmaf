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
    const char *name, *alias;
} Alias;

static const Alias alias_list[] = {
    {
        .name = "motion_force_zero",
        .alias = "force",
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdh",
    },
};

static const char *key_alias(char *key)
{
    const unsigned n = sizeof(alias_list) / sizeof(alias_list[0]);
    for(unsigned i = 0; i < n; i++) {
        if (!strcmp(key, alias_list[i].name))
            return alias_list[i].alias;
    }
    return NULL;
}

char *vmaf_feature_name(char *name, char *key, double val,
                        char *buf, size_t buf_sz)
{
    if (!key) return name;
    if (!key_alias(key)) return name;

    memset(buf, 0, buf_sz);
    snprintf(buf, buf_sz - 1, "%s_%s_%g",
             vmaf_feature_name_alias(name), key_alias(key), val);
    return buf;
}
