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

#include "feature/alias.h"
#include "feature/feature_collector.h"
#include "model.h"
#include "predict.h"
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


static int transform(VmafModel *model, double *prediction,
                     enum VmafModelFlags flags)
{
    if (!model->score_transform.enabled)
        return 0;
    if (flags & VMAF_MODEL_FLAG_DISABLE_TRANSFORM)
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

static int clip(VmafModel *model, double *prediction,
                enum VmafModelFlags flags)
{
    if (!model->score_clip.enabled)
        return 0;
    if (flags & VMAF_MODEL_FLAG_DISABLE_CLIP)
        return 0;

    *prediction = (*prediction < model->score_clip.min) ?
        model->score_clip.min : *prediction;
    *prediction = (*prediction > model->score_clip.max) ?
        model->score_clip.max : *prediction;

    return 0;
}

int vmaf_predict_score_at_index(VmafModel *model,
                                VmafFeatureCollector *feature_collector,
                                unsigned index, double *vmaf_score,
                                bool write_prediction,
                                enum VmafModelFlags flags)
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
                       vmaf_internal_feature_name_alias(model->feature[i].name),
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

    err = transform(model, &prediction, flags);
    if (err) goto free_node;

    err = clip(model, &prediction, flags);
    if (err) goto free_node;

    if (write_prediction) {
        err = vmaf_feature_collector_append(feature_collector, model->name,
                                            prediction, index);
        if (err) goto free_node;
    }

    *vmaf_score = prediction;

free_node:
    free(node);
    return err;
}


static int score_compare(const void *a, const void *b)
{
    const double *x = a;
    const double *y = b;
    if (*x > *y) return 1;
    else if (*x < *y) return -1;
    else return 0;
}

static double percentile(double *scores, unsigned n_scores, double perc)
{
    const double p = perc * (n_scores - 1) / 100.;
    const int idx_l = floor(p);
    const int idx_r = ceil(p);

    return (idx_l == idx_r) ? scores[idx_l] :
        scores[idx_l] * (idx_r - p) + scores[idx_r] * (p - idx_l);
}

static int vmaf_bootstrap_predict_score_at_index(
                                        VmafModelCollection *model_collection,
                                        VmafFeatureCollector *feature_collector,
                                        unsigned index,
                                        VmafModelCollectionScore *score)
{
    int err = 0;
    double scores[model_collection->cnt];

    for (unsigned i = 0; i < model_collection->cnt; i++) {
        // mean, stddev, etc. are calculated on untransformed/unclipped scores
        // gather the unclipped scores, for the purposes of these calculations
        // but do not write them to the feature collector
        const unsigned flags =
            VMAF_MODEL_FLAG_DISABLE_CLIP | VMAF_MODEL_FLAG_DISABLE_TRANSFORM;
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index,
                                          &scores[i], false,
                                          flags);
        if (err) return err;

        // do not override the model's transform/clip behavior
        // write the scores to the feature collector
        double score;
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index,
                                          &score, true, 0);
        if (err) return err;
    }

    score->type = VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP;

    double sum = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        sum += scores[i];
    const double mean = sum / model_collection->cnt;
    score->bootstrap.bagging_score = mean;

    const double delta = 0.01;
    double score_plus_delta = mean + delta;
    double score_minus_delta = mean - delta;

    double ssd = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        ssd += pow(scores[i] - mean, 2);
    score->bootstrap.stddev = sqrt(ssd / model_collection->cnt);

    qsort(scores, model_collection->cnt, sizeof(double), score_compare);
    score->bootstrap.ci.p95.lo = percentile(scores, model_collection->cnt, 2.5);
    score->bootstrap.ci.p95.hi = percentile(scores, model_collection->cnt, 97.5);

    const VmafModel *model = model_collection->model[0];
    transform(model, &score->bootstrap.bagging_score, 0);
    clip(model, &score->bootstrap.bagging_score, 0);
    transform(model, &score->bootstrap.ci.p95.lo, 0);
    clip(model, &score->bootstrap.ci.p95.lo, 0);
    transform(model, &score->bootstrap.ci.p95.hi, 0);
    clip(model, &score->bootstrap.ci.p95.hi, 0);
    transform(model, &score_plus_delta, 0);
    clip(model, &score_plus_delta, 0);
    transform(model, &score_minus_delta, 0);
    clip(model, &score_minus_delta, 0);

    const double slope = (score_plus_delta - score_minus_delta) / (2.0 * delta);
    score->bootstrap.stddev *= slope;

    //TODO: dedupe, vmaf_score_pooled_model_collection()
    const char *suffix_lo = "_ci_p95_lo";
    const char *suffix_hi = "_ci_p95_hi";
    const char *suffix_bagging = "_bagging";
    const char *suffix_stddev = "_stddev";
    const size_t name_sz =
        strlen(model_collection->name) + strlen(suffix_lo) + 1;
    const char name[name_sz];
    memset(name, 0, name_sz);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_bagging);
    err = vmaf_feature_collector_append(feature_collector, name,
                                        score->bootstrap.bagging_score, index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_stddev);
    err = vmaf_feature_collector_append(feature_collector, name,
                                        score->bootstrap.stddev, index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->bootstrap.ci.p95.lo, index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->bootstrap.ci.p95.hi,
                                         index);
    return err;
}

int vmaf_predict_score_at_index_model_collection(
                                VmafModelCollection *model_collection,
                                VmafFeatureCollector *feature_collector,
                                unsigned index,
                                VmafModelCollectionScore *score)
{
    switch (model_collection->type) {
    case VMAF_MODEL_BOOTSTRAP_SVM_NUSVR:
    case VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR:
        return vmaf_bootstrap_predict_score_at_index(model_collection,
                                                     feature_collector,
                                                     index, score);
    default:
        return -EINVAL;
    }
}
