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

char *vmaf_feature_name(char *name, char *key, double val,
                        char *buf, size_t buf_sz)
{
    if (!key) return name; 

    memset(buf, 0, buf_sz);
    snprintf(buf, buf_sz - 1, "%s_%s_%.2f", name, key, val);
    return buf;
}
