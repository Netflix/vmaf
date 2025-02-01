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

#include "test.h"
#include "metadata_handler.h"
#include "feature/feature_collector.h"
#include "predict.h"
#include "predict.c"

#include <libvmaf/model.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    VmafDictionary **metadata;
    int flags;
} MetaStruct;

static void set_meta(void *data, VmafMetadata *metadata)
{
    if (!data) return;
    MetaStruct *meta = data;
    char key[128], value[128];
    snprintf(key, sizeof(value), "%s_%d", metadata->feature_name,
             metadata->picture_index);
    snprintf(value, sizeof(value), "%f", metadata->score);
    vmaf_dictionary_set(meta->metadata, key, value, meta->flags);
}

static int g_callback_order[4] = {-1, -1, -1, -1};
static int g_callback_count = 0;

static void test_non_monotonic_callback(void *data, VmafMetadata *m)
{
    if (!data) return;
    MetaStruct *meta = data;

    if (!strcmp("vmaf", m->feature_name))
        // Track callback order
        g_callback_order[m->picture_index] = g_callback_count++;

    // Store in dictionary for verification
    char key[32], value[32];
    snprintf(key, sizeof(key), "vmaf_%d", m->picture_index);
    snprintf(value, sizeof(value), "%f", m->score);
    vmaf_dictionary_set(meta->metadata, key, value, meta->flags);
}

static char *test_propagate_metadata_init()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);
    mu_assert("problem during vmaf_propagate_metadata_init, metadata is NULL",
              propagate_metadata);

    vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

    return NULL;
}

static char *test_propagate_metadata_destroy()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);
    mu_assert("problem during vmaf_propagate_metadata_init, metadata is NULL",
              propagate_metadata);

    err = vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

  return NULL;
}

static char *test_propagate_metadata_append()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);

    VmafMetadataConfiguration metadata_config = {0};
    metadata_config.callback = set_meta;

    err = vmaf_metadata_append(propagate_metadata, metadata_config);
    mu_assert("problem during vmaf_propagate_metadata_append", !err);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head is NULL",
              propagate_metadata->head);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head->next is not NULL",
              !propagate_metadata->head->next);

    err = vmaf_metadata_append(propagate_metadata, metadata_config);
    mu_assert("problem during vmaf_propagate_metadata_append", !err);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head->next is NULL",
              propagate_metadata->head->next);

    err = vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

    return NULL;
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
    m.feature_name = "vmaf";
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
    m.feature_name = "vmaf";
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m);
    mu_assert("problem during vmaf_feature_collector_register_metadata_2", err);

    vmaf_feature_collector_destroy(feature_collector);

    vmaf_model_destroy(model);
    return NULL;

}

static char *test_propagate_metadata_non_monotonic()
{
    int err;

    // Reset global counters
    g_callback_count = 0;
    for (int i = 0; i < 4; i++) {
        g_callback_order[i] = -1;
    }

    // Setup dictionary to store callback results
    VmafDictionary *dict = NULL;
    MetaStruct meta_data = {
        .metadata = &dict,
        .flags    = 0,
    };

    VmafMetadataConfiguration m = {
        .feature_name = strdup("vmaf"),
        .callback = test_non_monotonic_callback,
        .data = &meta_data,
    };

    // Initialize feature collector
    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m);
    mu_assert("problem during vmaf_feature_collector_register_metadata", !err);

    // Load VMAF model
    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);
    err = vmaf_feature_collector_mount_model(feature_collector, model);
    mu_assert("problem during vmaf_mount_model", !err);

    // Simulate non-monotonic VMAF score computations
    // Frame order: 3, 0, 2, 1
    for (unsigned i = 0; i < model->n_features; i++) {
        // Frame 3
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 3);
        mu_assert("problem appending frame 3", !err);

        // Frame 0
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 70., 0);
        mu_assert("problem appending frame 0", !err);

        // Frame 2
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 80., 2);
        mu_assert("problem appending frame 2", !err);

        // Frame 1
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 90., 1);
        mu_assert("problem appending frame 1", !err);
    }

    // Verify callback order is monotonic regardless of computation order
    mu_assert("Frame 0 callback not first", g_callback_order[0] == 0);
    mu_assert("Frame 1 callback not second", g_callback_order[1] == 1);
    mu_assert("Frame 2 callback not third", g_callback_order[2] == 2);
    mu_assert("Frame 3 callback not fourth", g_callback_order[3] == 3);

    // Verify all frame scores were propagated
    for (int i = 0; i < 4; i++) {
        char key[32];
        snprintf(key, sizeof(key), "vmaf_%d", i);
        VmafDictionaryEntry *e = vmaf_dictionary_get(&dict, key, 0);
        mu_assert("Missing frame score in metadata", e != NULL);
    }

    vmaf_feature_collector_destroy(feature_collector);
    vmaf_model_destroy(model);
    return NULL;
}

// Structure to track callback invocations for multiple callbacks
typedef struct {
    int callback_id;
    int call_count;
    VmafDictionary **metadata;
} MultiCallbackData;

static void multi_callback(void *data, VmafMetadata *metadata)
{
    if (!data) return;
    MultiCallbackData *cb_data = data;

    char key[128], value[128];
    snprintf(key, sizeof(key), "cb%d_%s_%d",
             cb_data->callback_id,
             metadata->feature_name,
             metadata->picture_index);
    snprintf(value, sizeof(value), "%f", metadata->score);
    cb_data->call_count++;
    vmaf_dictionary_set(cb_data->metadata, key, value, 0);
}

static char *test_multiple_callbacks()
{
    int err;
    VmafDictionary *dict1 = NULL;
    VmafDictionary *dict2 = NULL;

    // Setup two different callback data structures
    MultiCallbackData cb_data1 = {
        .callback_id = 1,
        .call_count = 0,
        .metadata = &dict1
    };

    MultiCallbackData cb_data2 = {
        .callback_id = 2,
        .call_count = 0,
        .metadata = &dict2
    };

    // Register two different callbacks
    VmafMetadataConfiguration m1 = {
        .feature_name = "vmaf",
        .callback = multi_callback,
        .data = &cb_data1,
    };

    VmafMetadataConfiguration m2 = {
        .feature_name = "vmaf",
        .callback = multi_callback,
        .data = &cb_data2,
    };

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m1);
    mu_assert("problem registering first callback", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m2);
    mu_assert("problem registering second callback", !err);

    // Load and mount VMAF model
    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    err = vmaf_feature_collector_mount_model(feature_collector, model);
    mu_assert("problem mounting model", !err);

    // Add some feature data
    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 0);
        mu_assert("problem appending features", !err);
    }

    // Verify both callbacks were called
    mu_assert("first callback not called", cb_data1.call_count > 0);
    mu_assert("second callback not called", cb_data2.call_count > 0);
    mu_assert("callbacks called different number of times",
              cb_data1.call_count == cb_data2.call_count);

    // Verify data in both dictionaries
    VmafDictionaryEntry *e1 = vmaf_dictionary_get(&dict1, "cb1_vmaf_0", 0);
    VmafDictionaryEntry *e2 = vmaf_dictionary_get(&dict2, "cb2_vmaf_0", 0);

    mu_assert("first callback data missing", e1 != NULL);
    mu_assert("second callback data missing", e2 != NULL);
    mu_assert("callback data mismatch", strcmp(e1->val, e2->val) == 0);

    vmaf_feature_collector_destroy(feature_collector);
    vmaf_model_destroy(model);
    return NULL;
}

static char *test_multiple_callbacks_non_monotonic()
{
    int err;
    VmafDictionary *dict1 = NULL;
    VmafDictionary *dict2 = NULL;

    // Setup callback data
    MultiCallbackData cb_data1 = {
        .callback_id = 1,
        .call_count = 0,
        .metadata = &dict1
    };

    MultiCallbackData cb_data2 = {
        .callback_id = 2,
        .call_count = 0,
        .metadata = &dict2
    };

    VmafMetadataConfiguration m1 = {
        .feature_name = "vmaf",
        .callback = multi_callback,
        .data = &cb_data1,
    };

    VmafMetadataConfiguration m2 = {
        .feature_name = "vmaf",
        .callback = multi_callback,
        .data = &cb_data2,
    };

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during init", !err);

    err = vmaf_feature_collector_register_metadata(feature_collector, m1);
    err |= vmaf_feature_collector_register_metadata(feature_collector, m2);
    mu_assert("problem registering callbacks", !err);

    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem loading model", !err);

    err = vmaf_feature_collector_mount_model(feature_collector, model);
    mu_assert("problem mounting model", !err);


    for (unsigned i = 0; i < model->n_features; i++) {
        // Frame 2
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 2);
        mu_assert("problem appending frame 2", !err);

        // Frame 0
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 0);
        mu_assert("problem appending frame 0", !err);

        // Frame 3
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 3);
        mu_assert("problem appending frame 3", !err);

        // Frame 1
        err = vmaf_feature_collector_append(feature_collector,
                                          model->feature[i].name, 60., 1);
        mu_assert("problem appending frame 1", !err);
    }

    for (int i = 0; i < 4; i++) {
        char key1[32], key2[32];
        snprintf(key1, sizeof(key1), "cb1_vmaf_%d", i);
        snprintf(key2, sizeof(key2), "cb2_vmaf_%d", i);

        VmafDictionaryEntry *e1 = vmaf_dictionary_get(&dict1, key1, 0);
        VmafDictionaryEntry *e2 = vmaf_dictionary_get(&dict2, key2, 0);

        mu_assert("missing callback 1 data", e1 != NULL);
        mu_assert("missing callback 2 data", e2 != NULL);

        mu_assert("callback data mismatch", strcmp(e1->val, e2->val) == 0);
    }
    vmaf_feature_collector_destroy(feature_collector);
    vmaf_model_destroy(model);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_propagate_metadata_init);
    mu_run_test(test_propagate_metadata_destroy);
    mu_run_test(test_propagate_metadata_append);
    mu_run_test(test_propagate_metadata);
    mu_run_test(test_propagate_metadata_non_monotonic);
    mu_run_test(test_multiple_callbacks);
    mu_run_test(test_multiple_callbacks_non_monotonic);
    return NULL;
}