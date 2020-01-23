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
        .name = "'VMAF_feature_motion2_score'",
        .alias = "motion2",
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
