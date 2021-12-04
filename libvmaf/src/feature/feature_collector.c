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
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_name.h"
#include "log.h"
#include "opt.h"

static int aggregate_vector_init(AggregateVector *aggregate_vector)
{
    if (!aggregate_vector) return -EINVAL;
    memset(aggregate_vector, 0, sizeof(*aggregate_vector));
    const unsigned initial_capacity = 8;
    const size_t metric_vector_sz =
        sizeof(aggregate_vector->metric[0]) * initial_capacity;
    aggregate_vector->metric = malloc(metric_vector_sz);
    if (!aggregate_vector->metric) return -ENOMEM;
    memset(aggregate_vector->metric, 0, metric_vector_sz);
    aggregate_vector->capacity = initial_capacity;

    return 0;
}

static int aggregate_vector_append(AggregateVector *aggregate_vector,
                                   const char *feature_name, double score)
{
    if (!aggregate_vector) return -EINVAL;

    for (unsigned i = 0; i < aggregate_vector->cnt; i++) {
        if (!strcmp(feature_name, aggregate_vector->metric[i].name)) {
            if (aggregate_vector->metric[i].value == score) {
                return 0;
            } else {
                return -EINVAL;
            }
        }
    }

    const unsigned cnt = aggregate_vector->cnt;
    if (cnt >= aggregate_vector->capacity) {
        size_t initial_size =
            sizeof(aggregate_vector->metric[0]) * aggregate_vector->capacity;
        void *metric = realloc(aggregate_vector->metric, initial_size * 2);
        if (!metric) return -ENOMEM;
        memset(metric + initial_size, 0, initial_size);
        aggregate_vector->metric = metric;
        aggregate_vector->capacity *= 2;
    }

    const size_t feature_name_sz = strnlen(feature_name, 2048);
    char *f = malloc(feature_name_sz + 1);
    if (!f) return -EINVAL;
    memset(f, 0, feature_name_sz + 1);
    strncpy(f, feature_name, feature_name_sz);

    aggregate_vector->metric[cnt].name = f;
    aggregate_vector->metric[cnt].value = score;
    aggregate_vector->cnt++;

    return 0;
}

static void aggregate_vector_destroy(AggregateVector *aggregate_vector)
{
    if (!aggregate_vector) return;
    for (unsigned i = 0; i < aggregate_vector->cnt; i++) {
        if (aggregate_vector->metric[i].name)
            free(aggregate_vector->metric[i].name);
    }
    free(aggregate_vector->metric);
}

int vmaf_feature_collector_set_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name, double score)
{
    if (!feature_collector) return -EINVAL;
    if (!feature_name) return -EINVAL;

    pthread_mutex_lock(&(feature_collector->lock));
    int err = aggregate_vector_append(&feature_collector->aggregate_vector,
                                      feature_name, score);
    pthread_mutex_unlock(&(feature_collector->lock));
    return err;
}

int vmaf_feature_collector_get_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name,
                                         double *score)
{
    if (!feature_collector) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (!score) return -EINVAL;

    pthread_mutex_lock(&(feature_collector->lock));
    int err = 0;

    double *s = NULL;
    for (unsigned i = 0; i < feature_collector->aggregate_vector.cnt; i++) {
        const char *f = feature_collector->aggregate_vector.metric[i].name;
        if (!strcmp(f, feature_name)) {
            s = &(feature_collector->aggregate_vector.metric[i].value);
            break;
        }
    }

    if (!s) {
        err = -EINVAL;
        goto unlock;
    };

    *score = *s;

unlock:
    pthread_mutex_unlock(&(feature_collector->lock));
    return err;
}

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

    if (feature_vector->score[index].written) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "feature \"%s\" cannot be overwritten at index %d\n",
                 feature_vector->name, index);
        return -EINVAL;
    }

    feature_vector->score[index].written = true;
    feature_vector->score[index].value = score;

    return 0;
}

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector)
{
    if (!feature_collector) return -EINVAL;
    int err = 0;

    VmafFeatureCollector *const fc = *feature_collector = malloc(sizeof(*fc));
    if (!fc) goto fail;
    memset(fc, 0, sizeof(*fc));
    fc->capacity = 8;
    fc->feature_vector = malloc(sizeof(*(fc->feature_vector)) * fc->capacity);
    if (!fc->feature_vector) goto free_fc;
    memset(fc->feature_vector, 0, sizeof(*(fc->feature_vector)) * fc->capacity);
    err = aggregate_vector_init(&fc->aggregate_vector);
    if (err) goto free_feature_vector;
    err = pthread_mutex_init(&(fc->lock), NULL);
    if (err) goto free_aggregate_vector;
    return 0;

free_aggregate_vector:
    aggregate_vector_destroy(&(fc->aggregate_vector));
free_feature_vector:
    free(fc->feature_vector);
free_fc:
    free(fc);
fail:
    return -ENOMEM;
}

static FeatureVector *find_feature_vector(VmafFeatureCollector *fc,
                                          const char *feature_name)
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
                                  const char *feature_name, double score,
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

int vmaf_feature_collector_append_formatted(VmafFeatureCollector *feature_collector,
                                            double score, unsigned index,
                                            const char *fmt, ...)
{
    if (!feature_collector) return -EINVAL;
    if (!fmt) return -EINVAL;

    int err = 0;

    va_list(args);
    va_start(args, fmt);
    char *feature_name = NULL;
    vasprintf(&feature_name, fmt, args);
    if (err || !feature_name) return -ENOMEM;
    err = vmaf_feature_collector_append(feature_collector, feature_name,
                                        score, index);
    free(feature_name);
    return err;
}

int vmaf_feature_collector_append_templated(VmafFeatureCollector *feature_collector,
                                            const char *feature_name,
                                            const char *key, double val,
                                            double score,
                                            unsigned picture_index)
{
    if (!feature_collector) return -EINVAL;
    if (!feature_name) return -EINVAL;

    char buf[VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE];
    feature_name = vmaf_feature_name(feature_name, key, val, &buf[0],
                                     VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE);

    return vmaf_feature_collector_append(feature_collector, feature_name,
                                         score, picture_index);
}

static int option_is_default(const VmafOption *opt, const void *data)
{
    if (!opt) return -EINVAL;
    if (!data) return -EINVAL;

    switch (opt->type) {
    case VMAF_OPT_TYPE_BOOL:
        return opt->default_val.b == *((bool*)data);
    case VMAF_OPT_TYPE_INT:
        return opt->default_val.i == *((int*)data);
    case VMAF_OPT_TYPE_DOUBLE:
        return opt->default_val.d == *((double*)data);
    default:
        return -EINVAL;
    }
}

int vmaf_feature_collector_append_with_options(VmafFeatureCollector *fc,
        double score, unsigned index, const char *feature_name,
        const VmafOption *opts, const void *obj, unsigned param_cnt, ...)
{
    if (!feature_name) return NULL;
    if (!opts) return NULL;
    if (!obj) return NULL;

    VmafDictionary *opts_dict = NULL;
    char *fn = NULL;

    VmafOption *opt = NULL;
    for (unsigned i = 0; (opt = &opts[i]); i++) {
        if (!opt->name) break;
        va_list(args);
        va_start(args, param_cnt);
        for (unsigned j = 0; j < param_cnt; j++) {
            const void *param = va_arg(args, void*);
            const void *data = (uint8_t*)obj + opt->offset;
            if (data != param) continue;
            if (option_is_default(opt, data)) continue;

            const size_t buf_sz = VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE;
            const char buf[VMAF_FEATURE_NAME_DEFAULT_BUFFER_SIZE + 1] = { 0 };

            switch (opt->type) {
            case VMAF_OPT_TYPE_BOOL:
                snprintf(buf, buf_sz, "%d", *(bool*)data);
                break;
            case VMAF_OPT_TYPE_INT:
                snprintf(buf, buf_sz, "%d", *((int*)data));
                break;
            case VMAF_OPT_TYPE_DOUBLE:
                snprintf(buf, buf_sz, "%g", *((double*)data));
                break;
            default:
                break;
            }

            vmaf_dictionary_set(&opts_dict, opt->name, buf, 0);
        }
        va_end(args);
    }

    fn = vmaf_feature_name_from_opts_dict(feature_name, opts, opts_dict);
    vmaf_dictionary_free(&opts_dict);
    int err = vmaf_feature_collector_append(fc, fn, score, index);
    free(fn);
    return err;
}

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     const char *feature_name, double *score,
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
    aggregate_vector_destroy(&(feature_collector->aggregate_vector));
    for (unsigned i = 0; i < feature_collector->cnt; i++)
        feature_vector_destroy(feature_collector->feature_vector[i]);
    free(feature_collector->feature_vector);
    pthread_mutex_unlock(&(feature_collector->lock));
    pthread_mutex_destroy(&(feature_collector->lock));
    free(feature_collector);
}
