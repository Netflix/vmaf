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

#ifndef __VMAF_SRC_FEX_CTX_VECTOR_H__
#define __VMAF_SRC_FEX_CTX_VECTOR_H__

#include "feature/feature_extractor.h"

typedef struct {
    VmafFeatureExtractorContext **fex_ctx;
    unsigned cnt, capacity;
} RegisteredFeatureExtractors;

int feature_extractor_vector_init(RegisteredFeatureExtractors *rfe);

int feature_extractor_vector_append(RegisteredFeatureExtractors *rfe,
                                    VmafFeatureExtractorContext *fex_ctx,
                                    uint64_t flags);

void feature_extractor_vector_destroy(RegisteredFeatureExtractors *rfe);

#endif /* __VMAF_SRC_FEX_CTX_VECTOR_H__ */
