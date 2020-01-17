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
