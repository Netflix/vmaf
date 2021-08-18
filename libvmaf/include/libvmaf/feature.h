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

#ifndef __VMAF_FEATURE_H__
#define __VMAF_FEATURE_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafFeatureDictionary VmafFeatureDictionary;

int vmaf_feature_dictionary_set(VmafFeatureDictionary **dict, const char *key,
                                const char *val);

int vmaf_feature_dictionary_free(VmafFeatureDictionary **dict);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_FEATURE_H__ */
