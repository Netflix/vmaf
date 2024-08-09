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

#include <stdint.h>

#include "feature/feature_collector.h"
#include "metadata_handler.h"
#include "test.h"
#include "predict.h"
#include "predict.c"

#include <libvmaf/model.h>
#include <math.h>

typedef struct {
    VmafDictionary **metadata;
    int flags;
} MetaStruct;


static char *test_predict_score_at_index()
{
    int err;

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(feature_collector,
                                            model->feature[i].name, 60., 0);
        mu_assert("problem during vmaf_feature_collector_append", !err);
    }

    double vmaf_score = 0.;
    err = vmaf_predict_score_at_index(model, feature_collector, 0, &vmaf_score,
                                      true, false, 0);
    mu_assert("problem during vmaf_predict_score_at_index", !err);

    vmaf_model_destroy(model);
    vmaf_feature_collector_destroy(feature_collector);
    return NULL;
}


void set_meta(void *data, VmafMetadata *metadata)
{
    if (!data) return;
    MetaStruct *meta = data;
    char key[128], value[128];
    snprintf(key, sizeof(value), "%s_%d", metadata->feature_name,
             metadata->picture_index);
    snprintf(value, sizeof(value), "%f", metadata->score);
    vmaf_dictionary_set(meta->metadata, key, value, meta->flags);
}

static char* test_propagate_metadata()
{
    int err;

    VmafDictionary *dict = NULL;
    MetaStruct meta_data = {
        .metadata = &dict,
        .flags    = 0,
    };

    VmafMetadataConfiguration m = {
        .feature_name = "vmaf",
        .callback = set_meta,
        .data     = &meta_data,
    };

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m);
    mu_assert("problem during vmaf_feature_collector_register_metadata_0", !err);

    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);
    err = vmaf_feature_collector_mount_model(feature_collector, model);
    mu_assert("problem during vmaf_mount_model", !err);

    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(feature_collector,
                                            model->feature[i].name, 60., 0);
        mu_assert("problem during vmaf_feature_collector_append", !err);
    }

    VmafDictionaryEntry *e = vmaf_dictionary_get(&dict, "vmaf_0", 0);
    mu_assert("error on propagaton metadata: propagated key not found!",
              e);
    mu_assert("error on propagaton metadata: propagated key wrong!",
              !strcmp(e->key, "vmaf_0"));
    mu_assert("error on propagaton metadata: propagated data wrong!",
              !strcmp(e->val, "100.000000"));

    vmaf_feature_collector_destroy(feature_collector);

    m.data = NULL;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m);
    mu_assert("problem during vmaf_feature_collector_register_metadata_1", !err);

    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(feature_collector,
                                            model->feature[i].name, 60., 0);
        mu_assert("problem during vmaf_feature_collector_append", !err);
    }

    vmaf_feature_collector_destroy(feature_collector);

    m.callback = NULL;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m);
    mu_assert("problem during vmaf_feature_collector_register_metadata_2", err);

    vmaf_feature_collector_destroy(feature_collector);

    vmaf_model_destroy(model);
    return NULL;

}

static char *test_find_linear_function_parameters()
{
    int err;

    double a, b;

    VmafPoint p1 = { .x = 1, .y = 1 }, p2 = { .x = 0, .y = 0 };
    err = find_linear_function_parameters(p1, p2, &a, &b);
    mu_assert("first_point coordinates need to be smaller or equal to second_point coordinates", err);

    VmafPoint p3 = { .x = 0, .y = 1 }, p4 = { .x = 0, .y = 0 };
    err = find_linear_function_parameters(p3, p4, &a, &b);
    mu_assert("first_point coordinates need to be smaller or equal to second_point coordinates", err);

    VmafPoint p5 = { .x = 1, .y = 0 }, p6 = { .x = 0, .y = 0 };
    err = find_linear_function_parameters(p5, p6, &a, &b);
    mu_assert("first_point coordinates need to be smaller or equal to second_point coordinates", err);

    VmafPoint p7 = { .x = 50, .y = 30 }, p8 = { .x = 50, .y = 100 };
    err = find_linear_function_parameters(p7, p8, &a, &b);
    mu_assert("first_point and second_point cannot lie on a horizontal or vertical line", err);

    VmafPoint p9 = { .x = 50, .y = 30 }, p10 = { .x = 100, .y = 30 };
    err = find_linear_function_parameters(p9, p10, &a, &b);
    mu_assert("first_point and second_point cannot lie on a horizontal or vertical line", err);

    VmafPoint p11 = { .x = 50, .y = 20 }, p12 = { .x = 110, .y = 110 };
    err = find_linear_function_parameters(p11, p12, &a, &b);
    mu_assert("error code should be 0", !err);
    mu_assert("returned a does not match", a == 1.5);
    mu_assert("returned b does not match", b == -55.0);

    VmafPoint p13 = { .x = 50, .y = 30 }, p14 = { .x = 110, .y = 110 };
    err = find_linear_function_parameters(p13, p14, &a, &b);
    mu_assert("error code should be 0", !err);
    mu_assert("returned a does not match", fabs(a - 1.333333333333333) < 1e-8);
    mu_assert("returned b does not match", fabs(b - (-36.666666666666664)) < 1e-8);

    VmafPoint p15 = { .x = 50, .y = 30 }, p16 = { .x = 50, .y = 30 };
    err = find_linear_function_parameters(p15, p16, &a, &b);
    mu_assert("error code should be 0", !err);
    mu_assert("returned a does not match", a == 1.0);
    mu_assert("returned b does not match", b == 0.0);

    VmafPoint p17 = { .x = 10, .y = 10 }, p18 = { .x = 50, .y = 110 };
    err = find_linear_function_parameters(p17, p18, &a, &b);
    mu_assert("error code should be 0", !err);
    mu_assert("returned a does not match", a == 2.5);
    mu_assert("returned b does not match", b == -15.0);

    return NULL;
}

static char *test_piecewise_linear_mapping()
{
    int err;

    double y, y0, y1, y0_true, y1_true;

    VmafPoint knots1[] = {{ .x = 0, .y = 1 }, { .x = 1, .y = 2 }, { .x = 1, .y = 3 }};
    err = piecewise_linear_mapping(0, knots1, 3, &y);
    mu_assert("The x-coordinate of each point need to be greater that the x-coordinate of the previous point, the y-coordinate needs to be greater or equal", err);

    VmafPoint knots2[] = {{ .x = 0, .y = 2 }, { .x = 1, .y = 1 }};
    err = piecewise_linear_mapping(0, knots2, 2, &y);
    mu_assert("The x-coordinate of each point need to be greater that the x-coordinate of the previous point, the y-coordinate needs to be greater or equal", err);

    VmafPoint knots2160p[] = {{ .x = 0.0, .y = -55.0  }, { .x = 95.0, .y = 87.5  }, { .x = 105.0, .y = 105.0 }, { .x = 110.0, .y = 110.0 }};
    VmafPoint knots1080p[] = {{ .x = 0.0, .y = -36.66 }, { .x = 90.0, .y = 83.04 }, { .x = 95.0,  .y = 95.0  }, { .x = 100.0, .y = 100.0 }};

    for (double x0 = 0.0; x0 < 95.0; x0 += 0.1) {
        y0_true = 1.5 * x0 - 55.0;
        piecewise_linear_mapping(x0, knots2160p, 4, &y0);
        mu_assert("returned y0 does not match y0_true", fabs(y0 - y0_true) < 1e-8);
    }
    for (double x1 = 0.0; x1 < 90.0; x1 += 0.1) {
        y1_true = 1.33 * x1 - 36.66;
        piecewise_linear_mapping(x1, knots1080p, 4, &y1);
        mu_assert("returned y1 does not match y1_true", fabs(y1 - y1_true) < 1e-8);
    }

    for (double x0 = 95.0; x0 < 105.0; x0 += 0.1) {
        y0_true = 1.75 * x0 - 78.75;
        piecewise_linear_mapping(x0, knots2160p, 4, &y0);
        mu_assert("returned y0 does not match y0_true", fabs(y0 - y0_true) < 1e-8);
    }
    for (double x1 = 90.0; x1 < 95.0; x1 += 0.1) {
        y1_true = 2.392 * x1 - 132.24;
        piecewise_linear_mapping(x1, knots1080p, 4, &y1);
        mu_assert("returned y1 does not match y1_true", fabs(y1 - y1_true) < 1e-8);
    }

    for (double x0 = 105.0; x0 < 110.0; x0 += 0.1) {
        piecewise_linear_mapping(x0, knots2160p, 4, &y0);
        mu_assert("returned y0 does not match y0_true", fabs(y0 - x0) < 1e-8);
    }
    for (double x1 = 95.0; x1 < 100.0; x1 += 0.1) {
        piecewise_linear_mapping(x1, knots1080p, 4, &y1);
        mu_assert("returned y1 does not match x1", fabs(y1 - x1) < 1e-8);
    }

    VmafPoint knots_single[] = {{ .x = 10.0, .y = 10.0  }, { .x = 50.0, .y = 60.0  }};
    for (double x0 = 0.0; x0 < 110.0; x0 += 0.1) {
        piecewise_linear_mapping(x0, knots_single, 2, &y0);
        y0_true = 1.25 * x0 - 2.5;
        mu_assert("returned y0 does not match y0_true", fabs(y0 - y0_true) < 1e-8);
    }

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_predict_score_at_index);
    mu_run_test(test_find_linear_function_parameters);
    mu_run_test(test_piecewise_linear_mapping);
    mu_run_test(test_propagate_metadata);
    return NULL;
}
