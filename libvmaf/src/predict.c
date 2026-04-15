/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "feature/alias.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "feature/feature_name.h"
#include "log.h"
#include "model.h"
#include "predict.h"
#include "svm.h"

static int normalize(const VmafModel *model, double slope, double intercept,
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

static int denormalize(const VmafModel *model, double *prediction)
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

static int find_linear_function_parameters(VmafPoint p1, VmafPoint p2, double *alpha, double *beta) {

    if (!(p1.x <= p2.x && p1.y <= p2.y))
        return -EINVAL;  // first_point coordinates need to be smaller or equal to second_point coordinates

    if (p2.x - p1.x == 0 || p2.y - p1.y == 0) {
        if (!(p1.x == p2.x && p1.y == p2.y))
            return -EINVAL;  // first_point and second_point cannot lie on a horizontal or vertical line
        *alpha = 1.0;  // both points are the same
        *beta = 0.0;
    }
    else if (p1.x == 0) {
        *beta = p1.y;
        *alpha = (p2.y - *beta) / p2.x;
    }
    else {
        *alpha = (p2.y - p1.y) / (p2.x - p1.x);
        *beta = p1.y - (p1.x * (*alpha));
    }

    return 0;
}

/* NOLINTNEXTLINE(readability-function-size) */
static int piecewise_linear_mapping(double x, VmafPoint *knots, unsigned n_knots, double *y) {

    // initialize them to prevent uninitialized variable warnings
    double slope = 0.0;
    double offset = 0.0;

    if (n_knots <= 1) {
        return EINVAL;
    }
    unsigned n_seg = n_knots - 1;

    *y = 0.0;

    // construct the function
    for (unsigned idx = 0; idx < n_seg; idx++) {
        if (!(knots[idx].x <  knots[idx + 1].x &&
              knots[idx].y <= knots[idx + 1].y))
            return EINVAL;

        bool cond0 = knots[idx].x <= x;
        bool cond1 = x <= knots[idx + 1].x;

        if (knots[idx].y == knots[idx + 1].y) {  // the segment is horizontal
            if (cond0 && cond1) {
                *y = knots[idx].y;
            }

            if (idx == 0) {
                // for points below the defined range
                if (x < knots[idx].x)
                    *y = knots[idx].y;
            }

            if (idx == n_seg - 1) {
                // for points above the defined range
                if (x > knots[idx + 1].x)
                    *y = knots[idx].y;
            }
        } else {
            find_linear_function_parameters(knots[idx], knots[idx + 1],
                                            &slope, &offset);

            if (cond0 && cond1){
                *y = slope * x + offset;
            }

            if (idx == 0) {
                // for points below the defined range
                if (x < knots[idx].x)
                    *y = slope * x + offset;
            }

            if (idx == n_seg - 1) {
                // for points above the defined range
                if (x > knots[idx + 1].x)
                    *y = slope * x + offset;
            }
        }
    }

    return 0;
}

/*  Reproducing the logic in quality_runner.VmafQualityRunner.transform_score().
    Transform final quality score in the following optional steps (in this
    order):
    1) polynomial mapping. e.g. {'p0': 1, 'p1': 1, 'p2': 0.5} means
    transform through 1 + x + 0.5 * x^2. For now, only support polynomail
    up to 2nd-order.
    2) piecewise-linear mapping, where the change points are defined in
    'knots', in the form of [[x0, y0], [x1, y1], ...].
    3) rectification, supporting 'out_lte_in' (output is less than or equal
    to input) and 'out_gte_in' (output is greater than or equal to input).
 */
static int transform(const VmafModel *model, double *y_in,
                     enum VmafModelFlags flags)
{
    if (!model->score_transform.enabled)
        return 0;
    if (flags & VMAF_MODEL_FLAG_DISABLE_TRANSFORM)
        return 0;

    double y_stage;
    double y_out;

    // polynomial mapping
    y_stage = *y_in;
    if (model->score_transform.p0.enabled ||
        model->score_transform.p1.enabled ||
        model->score_transform.p2.enabled)
    {
        y_out = 0.;
        if (model->score_transform.p0.enabled)
            y_out += model->score_transform.p0.value;
        if (model->score_transform.p1.enabled)
            y_out += model->score_transform.p1.value * y_stage;
        if (model->score_transform.p2.enabled)
            y_out += model->score_transform.p2.value * y_stage * y_stage;
    }
    else {
        y_out = y_stage;
    }

    // piecewise-linear mapping
    y_stage = y_out;
    if (model->score_transform.knots.enabled) {
        piecewise_linear_mapping(y_stage,
                                 model->score_transform.knots.list,
                                 model->score_transform.knots.n_knots,
                                 &y_out);
    }
    else {
        y_out = y_stage;
    }

    // rectification
    if (model->score_transform.out_lte_in)
        y_out = (y_out > *y_in) ? *y_in : y_out;
    if (model->score_transform.out_gte_in)
        y_out = (y_out < *y_in) ? *y_in : y_out;

    *y_in = y_out;
    return 0;
}

static int clip(const VmafModel *model, double *prediction,
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

/* NOLINTNEXTLINE(readability-function-size) */
int vmaf_predict_score_at_index(VmafModel *model,
                                VmafFeatureCollector *feature_collector,
                                unsigned index, double *vmaf_score,
                                bool write_prediction,
                                bool propagate_metadata,
                                enum VmafModelFlags flags)
{
    if (!model) return -EINVAL;
    if (!feature_collector) return -EINVAL;
    if (!vmaf_score) return -EINVAL;

    int err = 0;

    // Lazily build the cached feature name lookup table
    if (!model->predict_feature_names) {
        model->predict_feature_names =
            (char **) calloc(model->n_features, sizeof(char *));
        if (!model->predict_feature_names) return -ENOMEM;

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(
                    model->feature[i].name, 0);
            if (!fex) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR,
                         "vmaf_predict_score_at_index(): no feature extractor "
                         "providing feature '%s'\n", model->feature[i].name);
                return -EINVAL;
            }

            VmafDictionary *opts_dict = NULL;
            if (model->feature[i].opts_dict) {
                err = vmaf_dictionary_copy(&model->feature[i].opts_dict,
                                           &opts_dict);
                if (err) return err;
            }

            VmafFeatureExtractorContext *fex_ctx;
            err = vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                        opts_dict);
            if (err) {
                vmaf_dictionary_free(&opts_dict);
                return err;
            }

            model->predict_feature_names[i] =
                vmaf_feature_name_from_options(model->feature[i].name,
                        fex_ctx->fex->options, fex_ctx->fex->priv);
            vmaf_feature_extractor_context_destroy(fex_ctx);

            if (!model->predict_feature_names[i]) return -ENOMEM;
        }
    }

    // Use pre-allocated node array, or allocate on first call
    if (!model->predict_nodes) {
        model->predict_nodes =
            malloc(sizeof(struct svm_node) * (model->n_features + 1));
        if (!model->predict_nodes) return -ENOMEM;
    }

    // Lazily cache FeatureVector pointers for direct score access
    if (!model->predict_feature_vectors) {
        model->predict_feature_vectors =
            (void **) calloc(model->n_features, sizeof(void *));
        if (!model->predict_feature_vectors) return -ENOMEM;
        for (unsigned i = 0; i < model->n_features; i++) {
            model->predict_feature_vectors[i] =
                vmaf_feature_collector_find(feature_collector,
                                            model->predict_feature_names[i]);
        }
    }

    struct svm_node *node = model->predict_nodes;

    for (unsigned i = 0; i < model->n_features; i++) {
        double feature_score;
        FeatureVector *fv = model->predict_feature_vectors[i];

        if (fv) {
            err = vmaf_feature_vector_get_score(fv, &feature_score, index);
        } else {
            // Fallback: feature not yet registered, try normal lookup
            err = vmaf_feature_collector_get_score(feature_collector,
                                                   model->predict_feature_names[i],
                                                   &feature_score, index);
            if (!err) {
                // Cache for future calls
                model->predict_feature_vectors[i] =
                    vmaf_feature_collector_find(feature_collector,
                                                model->predict_feature_names[i]);
            }
        }

        if (err) {
            if (!propagate_metadata) {
              vmaf_log(VMAF_LOG_LEVEL_ERROR,
                       "vmaf_predict_score_at_index(): no feature '%s' "
                       "at index %d\n", model->predict_feature_names[i],
                       index);
            }
            return err;
        }

        err = normalize(model, model->feature[i].slope,
                        model->feature[i].intercept, &feature_score);
        if (err) return err;

        node[i].index = i + 1;
        node[i].value = feature_score;
    }
    node[model->n_features].index = -1;

    double prediction = svm_predict(model->svm, node);

    err = denormalize(model, &prediction);
    if (err) return err;

    err = transform(model, &prediction, flags);
    if (err) return err;

    err = clip(model, &prediction, flags);
    if (err) return err;

    if (write_prediction) {
        err = vmaf_feature_collector_append(feature_collector, model->name,
                                            prediction, index);
        if (err) return err;
    }

    *vmaf_score = prediction;

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

/* NOLINTNEXTLINE(readability-function-size) */
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
        /* Bitmask of VmafModelFlags values is not itself a named enumerator. */
        // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index,
                                          &scores[i], false,
                                          false, flags);
        // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
        if (err) return err;

        // do not override the model's transform/clip behavior
        // write the scores to the feature collector
        double score;
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index,
                                          &score, true, false, 0);
        if (err) return err;
    }

    score->type = VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP;

    double sum = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++) {
        sum += scores[i];
    }
    const double mean = sum / model_collection->cnt;
    score->bootstrap.bagging_score = mean;

    const double delta = 0.01;
    double score_plus_delta = mean + delta;
    double score_minus_delta = mean - delta;

    double ssd = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++) {
        ssd += pow(scores[i] - mean, 2);
    }
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
    char name[name_sz];
    memset(name, 0, name_sz);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_bagging);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->bootstrap.bagging_score, index);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_stddev);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->bootstrap.stddev, index);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->bootstrap.ci.p95.lo, index);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
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
