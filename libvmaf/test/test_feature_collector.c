#include "test.h"
#include "feature_collector.c"

static char *test_feature_vector_init_append_and_destroy()
{
    int err;

    FeatureVector *feature_vector;
    err = feature_vector_init(&feature_vector, "psnr_y");
    mu_assert("problem during feature_vector_init", !err);

    unsigned initial_capacity = feature_vector->capacity;
    for (int j = initial_capacity - 1; j >= 0; j--) {
        err = feature_vector_append(feature_vector, j, 60.);
        mu_assert("problem during feature_vector_append", !err);
    }
    mu_assert("feature_vector->capacity should not have changed",
              feature_vector->capacity == initial_capacity);
    err = feature_vector_append(feature_vector, initial_capacity, 60.);
    mu_assert("problem during feature_vector_append", !err);
    mu_assert("feature_vector->capacity did not double its allocation",
              feature_vector->capacity == initial_capacity * 2);
    err = feature_vector_append(feature_vector, initial_capacity, 60.);
    mu_assert("feature_vector_append should not overwrite", err);

    feature_vector_destroy(feature_vector);
    return NULL;
}

static char *test_feature_collector_init_append_get_and_destroy()
{
    int err;

    VmafFeatureCollector *feature_collector;
    err = vmaf_feature_collector_init(&feature_collector);
    mu_assert("problem during vmaf_feature_collector_init", !err);
    unsigned initial_capacity = feature_collector->capacity;
    mu_assert("this test assumes an initial capacity of 8",
              initial_capacity == 8);
    err  = vmaf_feature_collector_append(feature_collector, "feature0", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature1", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature2", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature3", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature4", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature5", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature6", 60., 1);
    err |= vmaf_feature_collector_append(feature_collector, "feature7", 60., 1);
    mu_assert("problem during vmaf_feature_collector_append", !err);
    mu_assert("feature_collector->capacity should not have changed",
              feature_collector->capacity == initial_capacity);
    err = vmaf_feature_collector_append(feature_collector, "feature8", 60., 1);
    mu_assert("problem during vmaf_feature_collector_append", !err);
    mu_assert("feature_collector->capacity did not double its allocation",
              feature_collector->capacity == initial_capacity * 2);

    double score;
    err = vmaf_feature_collector_get_score(feature_collector, "feature5",
                                           &score, 1);
    mu_assert("problem during vmaf_feature_collector_get_score", !err);
    mu_assert("vmaf_feature_collector_get_score did not get the expected score",
              score == 60.);
    err = vmaf_feature_collector_get_score(feature_collector, "feature5",
                                           &score, 2);
    mu_assert("vmaf_feature_collector_get_score did not fail with bad index",
              err);

    vmaf_feature_collector_destroy(feature_collector);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_feature_vector_init_append_and_destroy);
    mu_run_test(test_feature_collector_init_append_get_and_destroy);
    return NULL;
}
