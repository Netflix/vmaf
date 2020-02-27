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

#include <errno.h>
#include <stdlib.h>

#include "feature/feature_collector.h"
#include "model.h"
#include "svm.h"

static int normalize(VmafModel *model, double slope, double intercept,
                     double *feature_score)
{
    switch (model->norm_type) {
    case(VMAF_MODEL_NORMALIZATION_TYPE_NONE):
        break;
    case(VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE):
        *feature_score = slope * (*feature_score) + intercept;
        break;
    default:
        return -EINVAL;
    }

    return 0;
}

static int denormalize(VmafModel *model, double *prediction)
{
    switch (model->norm_type) {
    case(VMAF_MODEL_NORMALIZATION_TYPE_NONE):
        break;
    case(VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE):
        *prediction = (*prediction - model->intercept) / model->slope;
        break;
    default:
        return -EINVAL;
    }

    return 0;
}


static int transform(VmafModel *model, double *prediction)
{
    if (!model->score_transform.enabled)
        return 0;

    double value = 0.;
    double p = *prediction;

    if (model->score_transform.p0.enabled)
        value += model->score_transform.p0.value;
    if (model->score_transform.p1.enabled)
        value += model->score_transform.p1.value * p;
    if (model->score_transform.p2.enabled)
        value += model->score_transform.p2.value * p * p;
    if (model->score_transform.out_lte_in)
        value = (value > p) ? p : value;
    if (model->score_transform.out_gte_in)
        value = (value < p) ? p : value;

    *prediction = value;
    return 0;
}

static int clip(VmafModel *model, double *prediction)
{
    if (!model->score_clip.enabled)
        return 0;

    *prediction = (*prediction < model->score_clip.min) ?
        model->score_clip.min : *prediction;
    *prediction = (*prediction > model->score_clip.max) ?
        model->score_clip.max : *prediction;

    return 0;
}

int vmaf_predict_score_at_index(VmafModel *model,
                                VmafFeatureCollector *feature_collector,
                                unsigned index, double *vmaf_score)
{
    if (!model) return -EINVAL;
    if (!feature_collector) return -EINVAL;
    if (!vmaf_score) return -EINVAL;

    int err = 0;

    struct svm_node *node = malloc(sizeof(*node) * (model->n_features + 1));
    if (!node) return -ENOMEM;

    for (unsigned i = 0; i < model->n_features; i++) {
        double feature_score;

        err = vmaf_feature_collector_get_score(feature_collector,
                                               model->feature[i].name,
                                               &feature_score, index);
        if (err) goto free_node;
        err = normalize(model, model->feature[i].slope,
                        model->feature[i].intercept, &feature_score);
        if (err) goto free_node;

        node[i].index = i + 1;
        node[i].value = feature_score;
    }
    node[model->n_features].index = -1;

    double prediction = svm_predict(model->svm, node);

    err = denormalize(model, &prediction);
    if (err) goto free_node;
    err = transform(model, &prediction);
    if (err) goto free_node;
    err = clip(model, &prediction);
    if (err) goto free_node;

    err = vmaf_feature_collector_append(feature_collector, model->name,
                                        prediction, index);
    if (err) goto free_node;

    *vmaf_score = prediction;

free_node:
    free(node);
    return err;
}
