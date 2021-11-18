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

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alias.h"
#include "feature_name.h"
#include "opt.h"

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

static int option_is_default(const VmafOption *opt, const void *data)
{
    if (!opt) return -EINVAL;
    if (!data) return -EINVAL;

    switch (opt->type) {
    case VMAF_OPT_TYPE_BOOL:
        return opt->default_val.b == *((bool*)data);
    case VMAF_OPT_TYPE_INT:
        return opt->default_val.i == *((int*)data);
    case VMAF_OPT_TYPE_DOUBLE:
        return opt->default_val.d == *((double*)data);
    default:
        return -EINVAL;
    }
}

static size_t snprintfcat(char* buf, size_t buf_sz, char const* fmt, ...)
{
    va_list args;
    const size_t len = strnlen(buf, buf_sz);
    va_start(args, fmt);
    const size_t result = vsnprintf(buf + len, buf_sz - len, fmt, args);
    va_end(args);

    return result + len;
}

char *vmaf_feature_name_from_options(char *name, VmafOption *opts, void *obj,
                                     unsigned param_cnt, ...)
{
    if (!name) return NULL;
    if (!opts) return NULL;
    if (!obj) return NULL;
    if (!param_cnt) return NULL;

    const size_t buf_sz = VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE;
    char buf[VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE + 1] = { 0 };

    snprintfcat(buf, buf_sz, "%s", name);

    va_list(args);
    va_start(args, param_cnt);

    for (unsigned i = 0; i < param_cnt; i++) {
        VmafOption *opt = NULL;
        const void *param = va_arg(args, void*);
        for (unsigned j = 0; (opt = &opts[j]); j++) {
            if (!opt->name) break;
            const void *data = (uint8_t*)obj + opt->offset;
            if (data != param) continue;
            if (option_is_default(opt, data)) continue;

            const char *key = opt->alias ? opt->alias : opt->name;
            const void *val = data;

            switch (opt->type) {
            case VMAF_OPT_TYPE_BOOL:
            case VMAF_OPT_TYPE_INT:
                snprintfcat(buf, buf_sz, "_%s_%d", key, *((int*)val));
                break;
            case VMAF_OPT_TYPE_DOUBLE:
                snprintfcat(buf, buf_sz, "_%s_%g", key, *((double*)val));
                break;
            default:
                break;
            }
        }
    }

    va_end(args);

    const size_t dst_sz = strnlen(buf, buf_sz) + 1;
    char *dst = malloc(dst_sz);
    if (!dst) return NULL;
    strncpy(dst, buf, dst_sz);
    return dst;
}
