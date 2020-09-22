#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feature/feature_collector.h"
#include "model.h"
#include "predict.h"

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

static double clip_score(double score, double clip_min, double clip_max)
{
    score = score < clip_min ? clip_min : score;
    score = score > clip_max ? clip_max : score;
    return score;
}

typedef struct BootstrapClipParams {
    bool enabled;
    double min, max;
} BootstrapClipParams;

static double bootstrap_clip(double score, BootstrapClipParams clip)
{
    if (clip.enabled) {
        score = score < clip.min ? clip.min : score;
        score = score > clip.max ? clip.max : score;
    }

    return score;
}

int vmaf_bootstrap_predict_score_at_index(VmafModelCollection *model_collection,
                                          VmafFeatureCollector *feature_collector,
                                          unsigned index,
                                          VmafModelCollectionScore *score)
{
    if (model_collection->cnt == 0) return -EINVAL;
    BootstrapClipParams clip = {
        .enabled = model_collection->model[0]->score_clip.enabled,
        .min = model_collection->model[0]->score_clip.min,
        .max = model_collection->model[0]->score_clip.max,
    };

    int err = 0;
    double scores[model_collection->cnt];

    for (unsigned i = 0; i < model_collection->cnt; i++) {
        // mean, stddev, etc. are calculated on the array of unclipped scores
        // gather the unclipped scores, for the purposes of these calculations
        // but do no write them to the feature collector
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index,
                                          &scores[i], false,
                                          VMAF_MODEL_FLAG_DISABLE_CLIP);
        if (err) return err;

        // do not override clip behavior
        // write the clipped scores to the feature collector
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

    double ssd = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        ssd += pow(scores[i] - mean, 2);
    score->bootstrap.stddev = sqrt(ssd / model_collection->cnt);

    qsort(scores, model_collection->cnt, sizeof(double), score_compare);
    score->bootstrap.ci.p95.lo = percentile(scores, model_collection->cnt, 2.5);
    score->bootstrap.ci.p95.hi = percentile(scores, model_collection->cnt, 97.5);

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
                           bootstrap_clip(score->bootstrap.bagging_score, clip),
                           index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_stddev);
    err = vmaf_feature_collector_append(feature_collector, name,
                                  bootstrap_clip(score->bootstrap.stddev, clip),
                                  index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_collector_append(feature_collector, name,
                               bootstrap_clip(score->bootstrap.ci.p95.lo, clip),
                               index);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
    err |= vmaf_feature_collector_append(feature_collector, name,
                               bootstrap_clip(score->bootstrap.ci.p95.hi, clip),
                               index);
    return err;
}
