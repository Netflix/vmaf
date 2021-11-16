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

#ifndef __VMAF_SRC_OPT_H__
#define __VMAF_SRC_OPT_H__

#include <stdint.h>
#include <stdbool.h>

enum VmafOptionType {
    VMAF_OPT_TYPE_BOOL,
    VMAF_OPT_TYPE_INT,
    VMAF_OPT_TYPE_DOUBLE,
    VMAF_OPT_TYPE_STRING,
};

enum VmafOptionFlag {
    VMAF_OPT_FLAG_FEATURE_PARAM = 1 << 0,
};

typedef struct VmafOption {
    const char *name;
    const char *help;
    const char *alias;
    int offset;
    enum VmafOptionType type;
    union {
        bool b;
        int i;
        double d;
        char *s;
    } default_val;
    double min, max;
    uint64_t flags;
} VmafOption;

int vmaf_option_set(const VmafOption *opt, void *obj, const char *val);

#endif /* __VMAF_SRC_OPT_H__ */
