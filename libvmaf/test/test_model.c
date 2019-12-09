#include <stdint.h>

#include "test.h"
#include "model.c"

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    err = vmaf_model_load_from_path(&model, "../../model/vmaf_v0.6.1.pkl");
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

char *run_tests()
{
    mu_run_test(test_model_load_and_destroy);
    return NULL;
}
