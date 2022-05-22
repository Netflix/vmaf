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

#include "libvmaf/model.h"
#include "model.h"
#include "pdjson.h"
#include "svm.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FEATURE_COUNT 32 //FIXME
#define MAX_KNOT_COUNT 10 //FIXME

static int parse_feature_opts_dicts(json_stream *s, VmafModel *model)
{
    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_OBJECT)
            return -EINVAL;
        if (i >= MAX_FEATURE_COUNT)
            return -EINVAL;

        while (json_peek(s) != JSON_OBJECT_END && !json_get_error(s)) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            char *key = strdup(json_get_string(s, NULL));
            if (!key) return -ENOMEM;
            if (json_peek(s) == JSON_NUMBER) {
                const char *val = json_get_string(s, NULL);
                const uint64_t flags = VMAF_DICT_DO_NOT_OVERWRITE |
                                       VMAF_DICT_NORMALIZE_NUMERICAL_VALUES;
                int err = vmaf_dictionary_set(&(model->feature[i].opts_dict),
                                              key, val, flags);
                free(key);
                if (err) return err;
            } else if (json_peek(s) == JSON_TRUE || json_peek(s) == JSON_FALSE) {
                const uint64_t flags = VMAF_DICT_DO_NOT_OVERWRITE;
                int err;
                if (json_peek(s) == JSON_TRUE) {
                    err = vmaf_dictionary_set(&(model->feature[i].opts_dict),
                                              key, "true", flags);
                } else {
                    err = vmaf_dictionary_set(&(model->feature[i].opts_dict),
                                              key, "false", flags);
                }
                free(key);
                if (err) return err;
            } else {
                return -EINVAL; //TODO
            }
            json_skip(s);
        }
        i++;
        json_skip_until(s, JSON_OBJECT_END);
    }
    json_skip_until(s, JSON_ARRAY_END);

    return 0;
}

static int parse_intercepts(json_stream *s, VmafModel *model)
{
    if (json_next(s) != JSON_NUMBER)
        return -EINVAL;
    model->intercept = json_get_number(s);

    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_NUMBER)
            return -EINVAL;
        if (i >= MAX_FEATURE_COUNT)
            return -EINVAL;
        model->feature[i++].intercept = json_get_number(s);
    }

    return 0;
}

static int parse_slopes(json_stream *s, VmafModel *model)
{
    if (json_next(s) != JSON_NUMBER)
        return -EINVAL;
    model->slope = json_get_number(s);

    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_NUMBER)
            return -EINVAL;
        if (i >= MAX_FEATURE_COUNT)
            return -EINVAL;
        model->feature[i++].slope = json_get_number(s);
    }

    return 0;
}

static int parse_knots_list(struct json_stream *s, struct VmafModel *model, unsigned idx)
{
    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_NUMBER)
            return -EINVAL;
        if (i >= 2)
            return -EINVAL;
        if (i == 0)
            model->score_transform.knots.list[idx].x = json_get_number(s);
        else
            model->score_transform.knots.list[idx].y = json_get_number(s);
        i++;
    }

    return 0;
}

static int parse_knots(json_stream *s, struct VmafModel *model)
{
    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_ARRAY)
            return -EINVAL;
        if (i >= MAX_KNOT_COUNT)
            return -EINVAL;
        int err = parse_knots_list(s, model, i);
        if (err) return err;
        json_skip_until(s, JSON_ARRAY_END);
        i++;
    }
    model->score_transform.knots.n_knots = i;
    model->score_transform.knots.enabled = true;
    return 0;
}

static int append_feature_name(VmafModel *model, const char *name,
                               unsigned index)
{
    if (index >= MAX_FEATURE_COUNT)
        return -EINVAL;
    model->feature[index].name = strdup(name);
    if (!model->feature[index].name)
        return -ENOMEM;
    return 0;
}

static int parse_feature_names(json_stream *s, VmafModel *model)
{
    int err = 0;

    unsigned i = 0;
    while (json_peek(s) != JSON_ARRAY_END && !json_get_error(s)) {
        if (json_next(s) != JSON_STRING)
            return -EINVAL;
        const char *name = json_get_string(s, NULL);
        err = append_feature_name(model, name, i++);
        if (err) return err;
        model->n_features++;
    }

    json_skip_until(s, JSON_ARRAY_END);
    return 0;
}

static int parse_score_transform(json_stream *s, VmafModel *model)
{
    model->score_transform.enabled = false;
    while (json_peek(s) != JSON_OBJECT_END && !json_get_error(s)) {
        if (json_next(s) != JSON_STRING)
            return -EINVAL;

        const char *key = json_get_string(s, NULL);

        if (!strcmp(key, "enabled")) {
            if (json_peek(s) != JSON_TRUE && json_peek(s) != JSON_FALSE)
                return -EINVAL;
            model->score_transform.enabled = (json_next(s) == JSON_TRUE);
            continue;
        }

        if (!strcmp(key, "p0")) {
            if (json_peek(s) == JSON_NULL) {
                model->score_transform.p0.enabled = false;
            } else if (json_next(s) == JSON_NUMBER) {
                model->score_transform.p0.enabled = true;
                model->score_transform.p0.value = json_get_number(s);
            } else {
                return -EINVAL;
            }
            continue;
        }

        if (!strcmp(key, "p1")) {
            if (json_peek(s) == JSON_NULL) {
                model->score_transform.p1.enabled = false;
            } else if (json_next(s) == JSON_NUMBER) {
                model->score_transform.p1.enabled = true;
                model->score_transform.p1.value = json_get_number(s);
            } else {
                return -EINVAL;
            }
            continue;
        }

        if (!strcmp(key, "p2")) {
            if (json_peek(s) == JSON_NULL) {
                model->score_transform.p2.enabled = false;
            } else if (json_next(s) == JSON_NUMBER) {
                model->score_transform.p2.enabled = true;
                model->score_transform.p2.value = json_get_number(s);
            } else {
                return -EINVAL;
            }
            continue;
        }

        if (!strcmp(key, "knots")) {
            if (json_peek(s) == JSON_NULL) {
                model->score_transform.knots.enabled = false;
                model->score_transform.knots.n_knots = 0;
            } else if (json_next(s) == JSON_ARRAY) {
                int err = parse_knots(s, model);
                if (err) return err;
                json_skip_until(s, JSON_ARRAY_END);
            } else {
                return -EINVAL;
            }
            continue;
        }

        if (!strcmp(key, "out_lte_in")) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            const char *out_lte_in = json_get_string(s, NULL);
            if (!strcmp(out_lte_in, "true"))
                model->score_transform.out_lte_in = true;
            continue;
        }

        if (!strcmp(key, "out_gte_in")) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            const char *out_gte_in = json_get_string(s, NULL);
            if (!strcmp(out_gte_in, "true"))
                model->score_transform.out_gte_in = true;
            continue;
        }

        json_skip(s);
    }

    return 0;
}

static int parse_libsvm_model(json_stream *s, VmafModel *model)
{
    size_t sz;
    const char *libsvm_model = json_get_string(s, &sz);
    model->svm = svm_parse_model_from_buffer(libsvm_model, sz);
    if (!model->svm) return -ENOMEM;

    return 0;
}

static int parse_model_dict(json_stream *s, VmafModel *model,
                            enum VmafModelFlags flags)
{
    if (json_next(s) != JSON_OBJECT)
        return -EINVAL;

    while (json_peek(s) != JSON_OBJECT_END && !json_get_error(s)) {
        if (json_next(s) != JSON_STRING)
            return -EINVAL;
        const char *key = json_get_string(s, NULL);

        if (!strcmp(key, "score_transform")) {
            if (json_next(s) != JSON_OBJECT)
                return -EINVAL;

            int err = parse_score_transform(s, model);
            if (err) return err;

            if (!model->score_transform.enabled &&
            (flags & VMAF_MODEL_FLAG_ENABLE_TRANSFORM)) {
                model->score_transform.enabled = true;
            }

            json_skip_until(s, JSON_OBJECT_END);
            continue;
        }

        if (!strcmp(key, "model_type")) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            const char *model_type = json_get_string(s, NULL);
            if (!strcmp(model_type, "RESIDUEBOOTSTRAP_LIBSVMNUSVR"))
                model->type = VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR;
            else if (!strcmp(model_type, "BOOTSTRAP_LIBSVMNUSVR"))
                model->type = VMAF_MODEL_BOOTSTRAP_SVM_NUSVR;
            else if (!strcmp(model_type, "LIBSVMNUSVR"))
                model->type = VMAF_MODEL_TYPE_SVM_NUSVR;
            else
                return -EINVAL;
            continue;
        }

        if (!strcmp(key, "norm_type")) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            const char *norm_type = json_get_string(s, NULL);
            if (!strcmp(norm_type, "linear_rescale"))
                model->norm_type = VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE;
            else if (!strcmp(norm_type, "none"))
                model->norm_type = VMAF_MODEL_NORMALIZATION_TYPE_NONE;
            else
                return -EINVAL;
            continue;
        }

        if (!strcmp(key, "score_clip")) {
            if (json_next(s) != JSON_ARRAY)
                return -EINVAL;
            if (!(flags & VMAF_MODEL_FLAG_DISABLE_CLIP)) {
                model->score_clip.enabled = true;
                if (json_next(s) != JSON_NUMBER)
                    return -EINVAL;
                model->score_clip.min = json_get_number(s);
                if (json_next(s) != JSON_NUMBER)
                    return -EINVAL;
                model->score_clip.max = json_get_number(s);
            }
            json_skip_until(s, JSON_ARRAY_END);
            continue;
        }

        if (!strcmp(key, "slopes")) {
            if (json_next(s) != JSON_ARRAY)
                return -EINVAL;
            int err = parse_slopes(s, model);
            if (err) return err;
            json_skip_until(s, JSON_ARRAY_END);
            continue;
        }

        if (!strcmp(key, "intercepts")) {
            if (json_next(s) != JSON_ARRAY)
                return -EINVAL;
            int err = parse_intercepts(s, model);
            if (err) return err;
            json_skip_until(s, JSON_ARRAY_END);
            continue;
        }

        if (!strcmp(key, "feature_names")) {
            if (json_next(s) != JSON_ARRAY)
                return -EINVAL;
            int err = parse_feature_names(s, model);
            if (err) return err;
            continue;
        }

        if (!strcmp(key, "feature_opts_dicts")) {
            if (json_next(s) != JSON_ARRAY)
                return -EINVAL;
            int err = parse_feature_opts_dicts(s, model);
            if (err) return err;
            continue;
        }

        if (!strcmp(key, "model")) {
            if (json_next(s) != JSON_STRING)
                return -EINVAL;
            int err = parse_libsvm_model(s, model);
            if (err) return err;
            continue;
        }

        json_skip(s);
    }

    json_skip_until(s, JSON_OBJECT_END);
    return 0;
}

static int model_parse(json_stream *s, VmafModel *model,
                       enum VmafModelFlags flags)
{
    int err = -EINVAL;

    if (json_next(s) != JSON_OBJECT)
        return -EINVAL;

    while (json_peek(s) != JSON_OBJECT_END && !json_get_error(s)) {
        if (json_next(s) != JSON_STRING)
            return -EINVAL;
        const char *key = json_get_string(s, NULL);

        if (!strcmp(key, "model_dict")) {
            err = parse_model_dict(s, model, flags);
            if (err) return err;
            continue;
        }

        json_skip(s);
    }

    json_skip_until(s, JSON_OBJECT_END);
    return err;
}

static int vmaf_read_json_model(VmafModel **model, VmafModelConfig *cfg,
                                json_stream *s)
{
    VmafModel *const m = *model = malloc(sizeof(*m));
    if (!m) return -ENOMEM;
    memset(m, 0, sizeof(*m));

    const size_t model_sz = sizeof(*m->feature) * MAX_FEATURE_COUNT;
    m->feature = malloc(model_sz);
    if (!m->feature) return -ENOMEM;
    memset(m->feature, 0, model_sz);

    m->name = vmaf_model_generate_name(cfg);
    if (!m->name) return -ENOMEM;

    const size_t knots_sz = sizeof(VmafPoint) * MAX_KNOT_COUNT;
    m->score_transform.knots.list = malloc(knots_sz);
    if (!m->score_transform.knots.list) return -ENOMEM;
    memset(m->score_transform.knots.list, 0, knots_sz);

    return model_parse(s, m, cfg->flags);
}

int vmaf_read_json_model_from_buffer(VmafModel **model, VmafModelConfig *cfg,
                                     const char *data, const int data_len)
{
    int err = 0;
    json_stream s;
    json_open_buffer(&s, data, data_len);
    err = vmaf_read_json_model(model, cfg, &s);
    json_close(&s);
    return err;
}

int vmaf_read_json_model_from_path(VmafModel **model, VmafModelConfig *cfg,
                                   const char *path)
{
    int err = 0;
    FILE *in = fopen(path, "r");
    if (!in) return -EINVAL;
    json_stream s;
    json_open_stream(&s, in);
    err = vmaf_read_json_model(model, cfg, &s);
    json_close(&s);
    fclose(in);
    return err;
}

static int model_collection_parse(json_stream *s, VmafModel **model,
                                  VmafModelCollection **model_collection,
                                  VmafModelConfig *cfg)
{
    int err = -EINVAL;

    *model_collection = NULL;

    if (json_next(s) != JSON_OBJECT)
        return -EINVAL;

    VmafModelConfig c = *cfg;
    const char *name = c.name = vmaf_model_generate_name(cfg);
    if (!c.name) return -ENOMEM;

    const size_t cfg_name_sz = strlen(name) + 5 + 1;
    char cfg_name[cfg_name_sz];

    const size_t generated_key_sz = 4 + 1;
    char generated_key[generated_key_sz];

    unsigned i = 0;
    while (json_peek(s) != JSON_OBJECT_END && !json_get_error(s)) {
        if (json_next(s) != JSON_STRING)
            return -EINVAL;

        const char *key = json_get_string(s, NULL);
        snprintf(generated_key, generated_key_sz, "%d", i);

        if (!strcmp(key, generated_key)) {
            VmafModel *m;
            err = vmaf_read_json_model(&m, &c, s);
            if (err) return err;

            if (i == 0) {
                *model = m;
                c.name = cfg_name;
            } else {
                err = vmaf_model_collection_append(model_collection, m);
                if (err) return err;
            }

            sprintf((char*)c.name, "%s_%04d", name, ++i);
            continue;
        }

        json_skip(s);
    }

    free((char*)name);
    if (!(*model_collection)) return -EINVAL;
    return err;
}

int vmaf_read_json_model_collection_from_path(VmafModel **model,
                                              VmafModelCollection **model_collection,
                                              VmafModelConfig *cfg,
                                              const char *path)
{
    int err = 0;
    FILE *in = fopen(path, "r");
    if (!in) return -EINVAL;
    json_stream s;
    json_open_stream(&s, in);
    err = model_collection_parse(&s, model, model_collection, cfg);
    json_close(&s);
    fclose(in);
    return err;
}

int vmaf_read_json_model_collection_from_buffer(VmafModel **model,
                                         VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *data, const int data_len)
{
    int err = 0;
    json_stream s;
    json_open_buffer(&s, data, data_len);
    err = model_collection_parse(&s, model, model_collection, cfg);
    json_close(&s);
    return err;
}
