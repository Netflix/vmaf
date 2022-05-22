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
#include <string.h>
#include <stdlib.h>

#include "opt.h"

static int set_option_bool(bool *dst, bool default_val, const char *val)
{
    *dst = default_val;
    if (!val) return 0;

    if (!strcmp(val, "true")) *dst = true;
    else if (!strcmp(val, "false")) *dst = false;
    else return -EINVAL;

    return 0;
}

static int set_option_int(int *dst, int default_val, const char *val,
                          int min, int max)
{
    *dst = default_val;
    if (!val) return 0;

    const int n = atoi(val);
    if (n == 0 && val[0] != '0') return -EINVAL;
    if (n < min) return -EINVAL;
    if (n > max) return -EINVAL;
    *dst = n;
    return 0;
}

static int set_option_double(double *dst, double default_val, const char *val,
                             double min, double max)
{
    *dst = default_val;
    if (!val) return 0;

    const double n = atof(val);
    if (n == 0 && val[0] != '0') return -EINVAL;
    if (n < min) return -EINVAL;
    if (n > max) return -EINVAL;
    *dst = n;
    return 0;
}

static int set_option_string(char **dst, char *default_val, const char *val) {
    *dst = default_val;
    if (!val) return 0;
    *dst = (char*)val;
    return 0;
}

int vmaf_option_set(const VmafOption *opt, void *obj, const char *val)
{
    if (!obj) return -EINVAL;
    if (!opt) return -EINVAL;

    void *dst = (uint8_t*)obj + opt->offset;

    switch (opt->type) {
    case VMAF_OPT_TYPE_BOOL:
        return set_option_bool(dst, opt->default_val.b, val);
    case VMAF_OPT_TYPE_INT:
        return set_option_int(dst, opt->default_val.i, val, opt->min, opt->max);
    case VMAF_OPT_TYPE_DOUBLE:
        return set_option_double(dst, opt->default_val.d, val, opt->min,
                                 opt->max);
    case VMAF_OPT_TYPE_STRING:
        return set_option_string(dst, opt->default_val.s, val);
    default:
        return -EINVAL;
    }
}
