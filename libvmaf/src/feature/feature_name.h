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

#ifndef __VMAF_FEATURE_NAME_H__
#define __VMAF_FEATURE_NAME_H__

#include "dict.h"
#include "opt.h"

char *vmaf_feature_name_from_options(const char *name, const VmafOption *opts,
                                     void *obj);

VmafDictionary *
vmaf_feature_name_dict_from_provided_features(const char **provided_features,
                                              const VmafOption *opts, void *obj);

#endif /* __VMAF_FEATURE_NAME_H__ */
