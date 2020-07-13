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

#ifndef __VMAF_MODEL_H__
#define __VMAF_MODEL_H__

#include <stdint.h>

typedef struct VmafModel VmafModel;

enum VmafModelFlags {
    VMAF_MODEL_FLAGS_DEFAULT = 0,
    VMAF_MODEL_FLAG_DISABLE_CLIP = (1 << 0),
    VMAF_MODEL_FLAG_ENABLE_TRANSFORM = (1 << 1),
    VMAF_MODEL_FLAG_ENABLE_CONFIDENCE_INTERVAL = (1 << 2),
};

typedef struct VmafModelConfig {
    char *name;
    char *path;
    uint64_t flags;
} VmafModelConfig;

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg);

void vmaf_model_destroy(VmafModel *model);

#endif /* __VMAF_MODEL_H__ */
