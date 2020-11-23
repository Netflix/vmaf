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

#ifndef __VMAF_JSON_MODEL_H__
#define __VMAF_JSON_MODEL_H__

#include "model.h"

int vmaf_read_json_model_from_buffer(VmafModel **model, VmafModelConfig *cfg,
                                     const char *data, const int data_len);

int vmaf_read_json_model_collection_from_buffer(VmafModel **model,
                                         VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *data, const int data_len);

int vmaf_read_json_model_from_path(VmafModel **model, VmafModelConfig *cfg,
                                   const char *path);

int vmaf_read_json_model_collection_from_path(VmafModel **model,
                                              VmafModelCollection **model_collection,
                                              VmafModelConfig *cfg,
                                              const char *path);

#endif /* __VMAF_JSON_MODEL_H__ */
