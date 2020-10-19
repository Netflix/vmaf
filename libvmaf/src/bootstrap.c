#include <math.h>
#include <stdlib.h>

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

int vmaf_bootstrap_predict_score_at_index(VmafModelCollection *model_collection,
                                          VmafFeatureCollector *feature_collector,
                                          unsigned index,
                                          VmafModelCollectionScore *score)
{
    int err = 0;
    double scores[model_collection->cnt];

    for (unsigned i = 0; i < model_collection->cnt; i++) {
        err = vmaf_predict_score_at_index(model_collection->model[i],
                                          feature_collector, index, &scores[i]);
        if (err) return err;
    }

    double sum = 0.;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        sum += scores[i]; 
    const double mean = sum / model_collection->cnt;
    score->score = mean;

    //double ssd = 0.;
    //for (unsigned i = 0; i < model_collection->cnt; i++)
    //    ssd += pow(scores[i] - mean, 2);
    //const double std = sqrt(ssd / model_collection->cnt);

    qsort(scores, model_collection->cnt, sizeof(double), score_compare);
    score->ci.p95.lo = percentile(scores, model_collection->cnt, 2.5);
    score->ci.p95.hi = percentile(scores, model_collection->cnt, 97.5);

    err = vmaf_feature_collector_append(feature_collector,
                                        model_collection->name,
                                        score->score, index);
    const char *suffix_lo = "_ci_p95_lo";
    const char *suffix_hi = "_ci_p95_hi";
    const size_t name_sz =
        strlen(model_collection->name) + strlen(suffix_lo) + 1;
    const char name[name_sz];
    memset(name, 0, name_sz);
    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->ci.p95.lo, index);
    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
    err |= vmaf_feature_collector_append(feature_collector, name,
                                         score->ci.p95.hi, index);
    return err;
}
