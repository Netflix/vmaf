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
#include "feature_collector.c"

static char *test_aggregate_vector_init_append_and_destroy()
{
    int err = 0;

    AggregateVector aggregate_vector;
    err = aggregate_vector_init(&aggregate_vector);
    mu_assert("problem during aggregate_vector_init", !err);
    mu_assert("aggregate_vector is not initialized properly",
             (aggregate_vector.cnt == 0) && (aggregate_vector.capacity == 8));

    err = aggregate_vector_append(&aggregate_vector, "A", 1);
    mu_assert("problem during aggregate_vector_append", !err);
    mu_assert("name and value were incorrectly set",
              (!strcmp("A", aggregate_vector.metric[0].name) &&
               aggregate_vector.metric[0].value == 1));

    err |= aggregate_vector_append(&aggregate_vector, "B", 2);
    err |= aggregate_vector_append(&aggregate_vector, "C", 3);
    err |= aggregate_vector_append(&aggregate_vector, "D", 4);
    err |= aggregate_vector_append(&aggregate_vector, "E", 5);
    err |= aggregate_vector_append(&aggregate_vector, "F", 6);
    err |= aggregate_vector_append(&aggregate_vector, "G", 7);
    err |= aggregate_vector_append(&aggregate_vector, "H", 8);
    mu_assert("problem during aggregate_vector_append", !err);
    mu_assert("aggregate_vector is not sized properly",
             (aggregate_vector.cnt == 8) && (aggregate_vector.capacity == 8));

    err = aggregate_vector_append(&aggregate_vector, "I", 9);
    mu_assert("problem during aggregate_vector_append", !err);
    mu_assert("aggregate_vector has not realloc'd properly",
             (aggregate_vector.cnt == 9) && (aggregate_vector.capacity == 16));
    mu_assert("name and value were incorrectly set",
              (!strcmp("I", aggregate_vector.metric[8].name) &&
               aggregate_vector.metric[8].value == 9));

    aggregate_vector_destroy(&aggregate_vector);
    return NULL;
}

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

    err = vmaf_feature_collector_set_aggregate(feature_collector,
                                               "aggregate0", 100.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate1", 101.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate2", 102.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate3", 103.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate4", 104.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate5", 105.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate6", 106.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate7", 107.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate8", 108.);
    err |= vmaf_feature_collector_set_aggregate(feature_collector,
                                                "aggregate9", 109.);
    mu_assert("problem during vmaf_feature_collector_set_aggregate", !err);

    err = vmaf_feature_collector_get_aggregate(feature_collector,
                                               "aggregate5", &score);
    mu_assert("problem during vmaf_feature_collector_get_aggregate", !err);
    mu_assert("unexpected aggreggate_score", score = 105.);
    err = vmaf_feature_collector_get_aggregate(feature_collector,
                                               "aggregate9", &score);
    mu_assert("problem during vmaf_feature_collector_get_aggregate", !err);
    mu_assert("unexpected aggreggate_score", score = 109.);

    vmaf_feature_collector_destroy(feature_collector);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_feature_vector_init_append_and_destroy);
    mu_run_test(test_feature_collector_init_append_get_and_destroy);
    mu_run_test(test_aggregate_vector_init_append_and_destroy);
    return NULL;
}
