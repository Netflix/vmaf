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

#include <stdbool.h>
#include <string.h>

typedef struct {
    const char *name, *alias;
} Alias;

static Alias alias_map[] = {
    {
        .name = "VMAF_feature_adm2_score",
        .alias = "adm2",
    },
    {
        .name = "VMAF_feature_adm_scale0_score",
        .alias = "adm_scale0",
    },
    {
        .name = "VMAF_feature_adm_scale1_score",
        .alias = "adm_scale1",
    },
    {
        .name = "VMAF_feature_adm_scale2_score",
        .alias = "adm_scale2",
    },
    {
        .name = "VMAF_feature_adm_scale3_score",
        .alias = "adm_scale3",
    },
    {
        .name = "VMAF_feature_motion_score",
        .alias = "motion",
    },
    {
        .name = "VMAF_feature_motion2_score",
        .alias = "motion2",
    },
    {
        .name = "VMAF_feature_vif_scale0_score",
        .alias = "vif_scale0",
    },
    {
        .name = "VMAF_feature_vif_scale1_score",
        .alias = "vif_scale1",
    },
    {
        .name = "VMAF_feature_vif_scale2_score",
        .alias = "vif_scale2",
    },
    {
        .name = "VMAF_feature_vif_scale3_score",
        .alias = "vif_scale3",
    },
    {
        .name = "VMAF_integer_feature_adm2_score",
        .alias = "integer_adm2",
    },
    {
        .name = "VMAF_integer_feature_motion_score",
        .alias = "integer_motion",
    },
    {
        .name = "VMAF_integer_feature_motion2_score",
        .alias = "integer_motion2",
    },
    {
        .name = "VMAF_integer_feature_vif_scale0_score",
        .alias = "integer_vif_scale0",
    },
    {
        .name = "VMAF_integer_feature_vif_scale1_score",
        .alias = "integer_vif_scale1",
    },
    {
        .name = "VMAF_integer_feature_vif_scale2_score",
        .alias = "integer_vif_scale2",
    },
    {
        .name = "VMAF_integer_feature_vif_scale3_score",
        .alias = "integer_vif_scale3",
    },
};

const char *vmaf_feature_name_alias(const char *feature_name)
{
    unsigned alias_cnt = sizeof(alias_map) / sizeof(alias_map[0]);

    for (unsigned i = 0; i < alias_cnt; i++) {
       if (!strcmp(feature_name, alias_map[i].name))
           return alias_map[i].alias;
    }

    return feature_name;
}
