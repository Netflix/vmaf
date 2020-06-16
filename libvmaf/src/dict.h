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

#ifndef __VMAF_SRC_DICT_H__
#define __VMAF_SRC_DICT_H__

#include <stdint.h>

typedef struct VmafDictionary VmafDictionary;

typedef struct VmafDictionaryEntry {
    const char *key, *val;
} VmafDictionaryEntry;

enum VmafDictionaryFlags {
    VMAF_DICT_DO_NOT_OVERWRITE = 1 << 0,
};

int vmaf_dictionary_set(VmafDictionary **dict, const char *key, const char *val,
                        uint64_t flags);

const VmafDictionaryEntry *vmaf_dictionary_get(VmafDictionary **dict,
                                               const char *key, uint64_t flags);

int vmaf_dictionary_free(VmafDictionary **dict);

#endif /* __VMAF_SRC_DICT_H__ */
