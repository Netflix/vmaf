#include <stdint.h>
#include "test.h"
#include "model.c"

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .name = "some_vmaf",
        .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Score transform must be enabled.\n", model->score_transform.enabled);
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n", !strcmp(model->feature[0].name, "'VMAF_feature_adm2_score'"));

    /*
    for (unsigned i = 0; i < model->n_features; i++)
        fprintf(stderr, "feature name: %s slope: %f intercept: %f\n",
                model->feature[i].name,
                model->feature[i].slope,
                model->feature[i].intercept
        );
    */

    vmaf_model_destroy(model);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_model_load_and_destroy);
    return NULL;
}
