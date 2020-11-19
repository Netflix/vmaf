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
        .name = "'VMAF_feature_adm2_score'",
        .alias = "adm2",
    },
    {
        .name = "'VMAF_feature_adm2_integer_score'",
        .alias = "integer_adm2",
    },
    {
        .name = "'VMAF_feature_adm_scale0_score'",
        .alias = "adm_scale0",
    },
    {
        .name = "'VMAF_feature_adm_scale1_score'",
        .alias = "adm_scale1",
    },
    {
        .name = "'VMAF_feature_adm_scale2_score'",
        .alias = "adm_scale2",
    },
    {
        .name = "'VMAF_feature_adm_scale3_score'",
        .alias = "adm_scale3",
    },
    {
        .name = "'VMAF_feature_motion_score'",
        .alias = "motion",
    },
    {
        .name = "'VMAF_feature_motion2_score'",
        .alias = "motion2",
    },
    {
        .name = "'VMAF_feature_motion2_integer_score'",
        .alias = "integer_motion2",
    },
    {
        .name = "'VMAF_feature_vif_scale0_score'",
        .alias = "vif_scale0",
    },
    {
        .name = "'VMAF_feature_vif_scale1_score'",
        .alias = "vif_scale1",
    },
    {
        .name = "'VMAF_feature_vif_scale2_score'",
        .alias = "vif_scale2",
    },
    {
        .name = "'VMAF_feature_vif_scale3_score'",
        .alias = "vif_scale3",
    },
    {
        .name = "'VMAF_feature_vif_scale0_integer_score'",
        .alias = "integer_vif_scale0",
    },
    {
        .name = "'VMAF_feature_vif_scale1_integer_score'",
        .alias = "integer_vif_scale1",
    },
    {
        .name = "'VMAF_feature_vif_scale2_integer_score'",
        .alias = "integer_vif_scale2",
    },
    {
        .name = "'VMAF_feature_vif_scale3_integer_score'",
        .alias = "integer_vif_scale3",
    },
};

static Alias internal_alias_map[] = {
    {
        .name = "VMAF_feature_adm2_score",
        .alias = "'VMAF_feature_adm2_score'",
    },
    {
        .name = "VMAF_feature_adm2_integer_score",
        .alias = "'VMAF_feature_adm2_integer_score'",
    },
    {
        .name = "VMAF_feature_motion2_score",
        .alias = "'VMAF_feature_motion2_score'",
    },
    {
        .name = "VMAF_feature_motion2_integer_score",
        .alias = "'VMAF_feature_motion2_integer_score'",
    },
    {
        .name = "VMAF_feature_vif_scale0_score",
        .alias = "'VMAF_feature_vif_scale0_score'",
    },
    {
        .name = "VMAF_feature_vif_scale1_score",
        .alias = "'VMAF_feature_vif_scale1_score'",
    },
    {
        .name = "VMAF_feature_vif_scale2_score",
        .alias = "'VMAF_feature_vif_scale2_score'",
    },
    {
        .name = "VMAF_feature_vif_scale3_score",
        .alias = "'VMAF_feature_vif_scale3_score'",
    },
    {
        .name = "VMAF_feature_vif_scale0_integer_score",
        .alias = "'VMAF_feature_vif_scale0_integer_score'",
    },
    {
        .name = "VMAF_feature_vif_scale1_integer_score",
        .alias = "'VMAF_feature_vif_scale1_integer_score'",
    },
    {
        .name = "VMAF_feature_vif_scale2_integer_score",
        .alias = "'VMAF_feature_vif_scale2_integer_score'",
    },
    {
        .name = "VMAF_feature_vif_scale3_integer_score",
        .alias = "'VMAF_feature_vif_scale3_integer_score'",
    },
};

static Alias vmafossexec_alias_map[] = {
    {
        .name = "float_psnr",
        .alias = "psnr",
    },
    {
        .name = "float_ssim",
        .alias = "ssim",
    },
    {
        .name = "float_ms_ssim",
        .alias = "ms_ssim",
    },
};

static bool vmafossexec_aliases = false;
void vmaf_use_vmafossexec_aliases(void)
{
    vmafossexec_aliases = true;
}

static const char *vmafossexec_feature_name_alias(const char *feature_name)
{
    if (!vmafossexec_aliases) return feature_name;

    unsigned alias_cnt =
        sizeof(vmafossexec_alias_map) / sizeof(vmafossexec_alias_map[0]);

    for (unsigned i = 0; i < alias_cnt; i++) {
       if (!strcmp(feature_name, vmafossexec_alias_map[i].name))
           return vmafossexec_alias_map[i].alias;
    }

    return feature_name;
}

const char *vmaf_feature_name_alias(const char *feature_name)
{
    unsigned alias_cnt = sizeof(alias_map) / sizeof(alias_map[0]);

    for (unsigned i = 0; i < alias_cnt; i++) {
       if (!strcmp(feature_name, alias_map[i].name))
           return vmafossexec_feature_name_alias(alias_map[i].alias);
    }

    return vmafossexec_feature_name_alias(feature_name);
}

const char *vmaf_internal_feature_name_alias(const char *feature_name)
{
    unsigned alias_cnt =
        sizeof(internal_alias_map) / sizeof(internal_alias_map[0]);

    for (unsigned i = 0; i < alias_cnt; i++) {
       if (!strcmp(feature_name, internal_alias_map[i].name))
           return internal_alias_map[i].alias;
    }

    return feature_name;
}
