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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafDictionaryEntry {
    const char *key, *val;
} VmafDictionaryEntry;

typedef struct VmafDictionary {
    VmafDictionaryEntry *entry;
    unsigned size, cnt;
} VmafDictionary;

enum VmafDictionaryFlags {
    VMAF_DICT_DO_NOT_OVERWRITE = 1 << 0,
    VMAF_DICT_NORMALIZE_NUMERICAL_VALUES = 1 << 1,
};

int vmaf_dictionary_set(VmafDictionary **dict, const char *key, const char *val,
                        uint64_t flags);

VmafDictionaryEntry *vmaf_dictionary_get(VmafDictionary **dict,
                                         const char *key, uint64_t flags);

int vmaf_dictionary_copy(VmafDictionary **src, VmafDictionary **dst);

VmafDictionary *vmaf_dictionary_merge(VmafDictionary **dict_a,
                                      VmafDictionary **dict_b,
                                      uint64_t flags);

int vmaf_dictionary_compare(VmafDictionary *dict_a, VmafDictionary *dict_b);

void vmaf_dictionary_alphabetical_sort(VmafDictionary *dict);

int vmaf_dictionary_free(VmafDictionary **dict);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_SRC_DICT_H__ */
