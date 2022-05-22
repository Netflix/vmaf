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

#include "feature.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafModel VmafModel;

enum VmafModelFlags {
    VMAF_MODEL_FLAGS_DEFAULT = 0,
    VMAF_MODEL_FLAG_DISABLE_CLIP = (1 << 0),
    VMAF_MODEL_FLAG_ENABLE_TRANSFORM = (1 << 1),
    VMAF_MODEL_FLAG_DISABLE_TRANSFORM = (1 << 2),
};

typedef struct VmafModelConfig {
    const char *name;
    uint64_t flags;
} VmafModelConfig;

int vmaf_model_load(VmafModel **model, VmafModelConfig *cfg,
                    const char *version);

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg,
                              const char *path);

int vmaf_model_feature_overload(VmafModel *model, const char *feature_name,
                                VmafFeatureDictionary *opts_dict);

void vmaf_model_destroy(VmafModel *model);

typedef struct VmafModelCollection VmafModelCollection;

enum VmafModelCollectionScoreType {
    VMAF_MODEL_COLLECTION_SCORE_UNKNOWN = 0,
    VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP,
};

typedef struct VmafModelCollectionScore {
    enum VmafModelCollectionScoreType type;
    struct {
        double bagging_score;
        double stddev;
        struct {
            struct { double lo, hi; } p95;
        } ci;
    } bootstrap;
} VmafModelCollectionScore;

int vmaf_model_collection_load(VmafModel **model,
                               VmafModelCollection **model_collection,
                               VmafModelConfig *cfg,
                               const char *version);

int vmaf_model_collection_load_from_path(VmafModel **model,
                                         VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *path);

int vmaf_model_collection_feature_overload(VmafModel *model,
                                           VmafModelCollection **model_collection,
                                           const char *feature_name,
                                           VmafFeatureDictionary *opts_dict);

void vmaf_model_collection_destroy(VmafModelCollection *model_collection);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_MODEL_H__ */
