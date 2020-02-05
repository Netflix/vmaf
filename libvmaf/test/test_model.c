#include <stdint.h>
#include "test.h"
#include "model.c"

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);

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

static char *test_model_check_default_behavior()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .name = "some_vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Disable clip must be true by default.\n", !model->score_clip.enabled);
    mu_assert("Enable score transform must be false by default.\n", !model->score_transform.enabled);
    /* TODO: add check for confidence interval */
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n", !strcmp(model->feature[0].name, "'VMAF_feature_adm2_score'"));

    vmaf_model_destroy(model);

    return NULL;
}

static char *test_model_set_flags()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Enable score transform must be true.\n",
              model->score_transform.enabled);
    mu_assert("Disable clip must be false.\n",
              model->score_clip.enabled);
    vmaf_model_destroy(model);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_model_load_and_destroy);
    mu_run_test(test_model_check_default_behavior);
    mu_run_test(test_model_set_flags);
    return NULL;
}
