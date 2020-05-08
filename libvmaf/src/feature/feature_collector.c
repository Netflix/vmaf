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

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "feature_collector.h"

static int feature_vector_init(FeatureVector **const feature_vector,
                               const char *name)
{
    if (!feature_vector) return -EINVAL;
    if (!name) return -EINVAL;

    FeatureVector *const fv = *feature_vector = malloc(sizeof(*fv));
    if (!fv) goto fail;
    memset(fv, 0, sizeof(*fv));
    fv->name = malloc(strlen(name) + 1);
    if (!fv->name) goto free_fv;
    strcpy(fv->name, name);
    fv->capacity = 8;
    fv->score = malloc(sizeof(fv->score[0]) * fv->capacity);
    if (!fv->score) goto free_name;
    memset(fv->score, 0, sizeof(fv->score[0]) * fv->capacity);
    return 0;

free_name:
    free(fv->name);
free_fv:
    free(fv);
fail:
    return -ENOMEM;
}

static void feature_vector_destroy(FeatureVector *feature_vector)
{
    if (!feature_vector) return;
    free(feature_vector->name);
    free(feature_vector->score);
    free(feature_vector);
}

static int feature_vector_append(FeatureVector *feature_vector,
                                 unsigned index, double score)
{
    if (!feature_vector) return -EINVAL;

    while (index >= feature_vector->capacity) {
        size_t initial_size =
            sizeof(feature_vector->score[0]) * feature_vector->capacity;
        void *score = realloc(feature_vector->score, initial_size * 2);
        if (!score) return -ENOMEM;
        memset(score + initial_size, 0, initial_size);
        feature_vector->score = score;
        feature_vector->capacity *= 2;
    }

    if (feature_vector->score[index].written)
        return -EINVAL;

    feature_vector->score[index].written = true;
    feature_vector->score[index].value = score;

    return 0;
}

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector)
{
    if (!feature_collector) return -EINVAL;

    VmafFeatureCollector *const fc = *feature_collector = malloc(sizeof(*fc));
    if (!fc) goto fail;
    memset(fc, 0, sizeof(*fc));
    fc->capacity = 8;
    fc->feature_vector = malloc(sizeof(*(fc->feature_vector)) * fc->capacity);
    if (!fc->feature_vector) goto free_fc;
    memset(fc->feature_vector, 0, sizeof(*(fc->feature_vector)) * fc->capacity);
    int err = pthread_mutex_init(&(fc->lock), NULL);
    if (err) goto free_feature_vector;
    return 0;

free_feature_vector:
    free(fc->feature_vector);
free_fc:
    free(fc);
fail:
    return -ENOMEM;
}

static FeatureVector *find_feature_vector(VmafFeatureCollector *fc,
                                          char *feature_name)
{
    FeatureVector *feature_vector = NULL;
    for (unsigned i = 0; i < fc->cnt; i++) {
        FeatureVector *fv = fc->feature_vector[i];
        if (!strcmp(fv->name, feature_name)) {
            feature_vector = fv;
            break;
        }
    }
    return feature_vector;
}

int vmaf_feature_collector_append(VmafFeatureCollector *feature_collector,
                                  char *feature_name, double score,
                                  unsigned picture_index)
{
    if (!feature_collector) return -EINVAL;
    if (!feature_name) return -EINVAL;

    pthread_mutex_lock(&(feature_collector->lock));
    int err = 0;

    if (!feature_collector->timer.begin)
        feature_collector->timer.begin = clock();

    FeatureVector *feature_vector =
        find_feature_vector(feature_collector, feature_name);

    if (!feature_vector) {
        err = feature_vector_init(&feature_vector, feature_name);
        if (err) goto unlock;
        if (feature_collector->cnt + 1 > feature_collector->capacity) {
            size_t initial_size = sizeof(feature_collector->feature_vector[0]) *
                feature_vector->capacity;
            FeatureVector **fv =
                realloc(feature_collector->feature_vector,
                sizeof(*(feature_collector->feature_vector)) *
                initial_size * 2);
            if (!fv) {
                err = -ENOMEM;
                goto unlock;
            }
            memset(fv + feature_collector->capacity, 0, initial_size);
            feature_collector->feature_vector = fv;
            feature_collector->capacity *= 2;
        }
        feature_collector->feature_vector[feature_collector->cnt++]
            = feature_vector;
    }

    err = feature_vector_append(feature_vector, picture_index, score);

unlock:
    feature_collector->timer.end = clock();
    pthread_mutex_unlock(&(feature_collector->lock));
    return err;
}

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     char *feature_name, double *score,
                                     unsigned index)
{
    if (!feature_collector) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (!score) return -EINVAL;

    pthread_mutex_lock(&(feature_collector->lock));
    int err = 0;

    FeatureVector *feature_vector =
        find_feature_vector(feature_collector, feature_name);

    if (!feature_vector || index >= feature_vector->capacity) {
        err = -EINVAL;
        goto unlock;
    }

    if (!feature_vector->score[index].written) {
        err = -EINVAL;
        goto unlock;
    }

    *score = feature_vector->score[index].value;

unlock:
    pthread_mutex_unlock(&(feature_collector->lock));
    return err;
}

void vmaf_feature_collector_destroy(VmafFeatureCollector *feature_collector)
{
    if (!feature_collector) return;

    pthread_mutex_lock(&(feature_collector->lock));
    for (unsigned i = 0; i < feature_collector->cnt; i++)
        feature_vector_destroy(feature_collector->feature_vector[i]);
    free(feature_collector->feature_vector);
    pthread_mutex_unlock(&(feature_collector->lock));
    pthread_mutex_destroy(&(feature_collector->lock));
    free(feature_collector);
}
