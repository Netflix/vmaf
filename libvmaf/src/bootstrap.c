#include <math.h>
#include <stdlib.h>

#include "feature_collector.h"
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

int vmaf_bootstrap_predict_score_at_index(VmafModel *model, unsigned model_cnt,
                                          VmafFeatureCollector *feature_collector,
                                          unsigned index, double *vmaf_score,
                                          double *ci_lo, double *ci_hi)
{
    int err = 0;
    double scores[model_cnt]; //FIXME: without VLA

    for (unsigned i = 0; i < model_cnt; i++) {
        err = vmaf_predict_score_at_index(&model[i], feature_collector, index,
                                          &scores[i]);
        if (err) return err;
    }

    double sum = 0.;
    for (unsigned i = 0; i < model_cnt; i++)
        sum += scores[i]; 
    const double mean = sum / model_cnt;
    *vmaf_score = mean;

    //double ssd = 0.;
    //for (unsigned i = 0; i < model_cnt; i++)
    //    ssd += pow(scores[i] - mean, 2);
    //const double std = sqrt(ssd / model_cnt);

    qsort(scores, model_cnt, sizeof(double), score_compare);
    const double c95_lo = percentile(scores, model_cnt, 2.5);
    const double c95_hi = percentile(scores, model_cnt, 97.5);
    *ci_lo = c95_lo;
    *ci_hi = c95_hi;

    return 0;
}
