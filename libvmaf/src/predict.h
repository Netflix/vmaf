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

#ifndef __VMAF_PREDICT_H__
#define __VMAF_PREDICT_H__

#include "feature/feature_collector.h"
#include "model.h"

int vmaf_predict_score_at_index(VmafModel *model,
                                VmafFeatureCollector *feature_collector,
                                unsigned index, double *vmaf_score,
                                bool write_prediction,
                                enum VmafModelFlags flags);

int vmaf_predict_score_at_index_model_collection(
                                VmafModelCollection *model_collection,
                                VmafFeatureCollector *feature_collector,
                                unsigned index,
                                VmafModelCollectionScore *score);

#endif /* __VMAF_PREDICT_H__ */
