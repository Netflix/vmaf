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

#include <stdint.h>

#include "test.h"
#include "predict.h"

#include <libvmaf/model.h>

static char *test_predict_score_at_index()
{
    int err;

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);

    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(feature_collector,
                                            model->feature[i].name, 60., 0);
        mu_assert("problem during vmaf_feature_collector_append", !err);
    }

    double vmaf_score = 0.;
    err = vmaf_predict_score_at_index(model, feature_collector, 0, &vmaf_score);
    mu_assert("problem during vmaf_predict_score_at_index", !err);

    vmaf_model_destroy(model);
    vmaf_feature_collector_destroy(feature_collector);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_predict_score_at_index);
    return NULL;
}
