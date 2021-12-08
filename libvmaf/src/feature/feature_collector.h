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

#ifndef __VMAF_FEATURE_COLLECTOR_H__
#define __VMAF_FEATURE_COLLECTOR_H__

#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include "dict.h"

typedef struct {
    char *name;
    struct {
        bool written;
        double value;
    } *score;
    unsigned capacity;
} FeatureVector;

typedef struct {
    struct {
        char *name;
        double value;
    } *metric;
    unsigned cnt, capacity;
} AggregateVector;

typedef struct VmafFeatureCollector {
    FeatureVector **feature_vector;
    AggregateVector aggregate_vector;
    unsigned cnt, capacity;
    struct { clock_t begin, end; } timer;
    pthread_mutex_t lock;
} VmafFeatureCollector;

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector);

int vmaf_feature_collector_append(VmafFeatureCollector *feature_collector,
                                  const char *feature_name, double score,
                                  unsigned index);

int vmaf_feature_collector_append_with_dict(VmafFeatureCollector *fc,
        VmafDictionary *dict, const char *feature_name, double score,
        unsigned index);

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     const char *feature_name, double *score,
                                     unsigned index);

int vmaf_feature_collector_set_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name,
                                         double score);

int vmaf_feature_collector_get_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name,
                                         double *score);

void vmaf_feature_collector_destroy(VmafFeatureCollector *feature_collector);

#endif /* __VMAF_FEATURE_COLLECTOR_H__ */
