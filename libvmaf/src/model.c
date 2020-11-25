#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <libvmaf/model.h>

#include "config.h"
#include "model.h"
#include "read_json_model.h"
#include "svm.h"
#include "unpickle.h"

typedef struct VmafBuiltInModel {
    const char *version;
    const char *data;
    const int *data_len;
} VmafBuiltInModel;

#if VMAF_BUILT_IN_MODELS
extern const char ___src_______model_vmaf_float_v0_6_1neg_json;
extern const int ___src_______model_vmaf_float_v0_6_1neg_json_len;
extern const char ___src_______model_vmaf_v0_6_1_json;
extern const int ___src_______model_vmaf_v0_6_1_json_len;
extern const char ___src_______model_vmaf_float_v0_6_1_json;
extern const int ___src_______model_vmaf_float_v0_6_1_json_len;
extern const char ___src_______model_vmaf_b_v0_6_3_json;
extern const int ___src_______model_vmaf_b_v0_6_3_json_len;
extern const char ___src_______model_vmaf_v0_6_1neg_json;
extern const int ___src_______model_vmaf_v0_6_1neg_json_len;
extern const char ___src_______model_vmaf_float_b_v0_6_3_json;
extern const int ___src_______model_vmaf_float_b_v0_6_3_json_len;
#endif

static const VmafBuiltInModel built_in_models[] = {
#if VMAF_BUILT_IN_MODELS
    {
        .version = "vmaf_float_v0.6.1neg",
        .data = &___src_______model_vmaf_float_v0_6_1neg_json,
        .data_len = &___src_______model_vmaf_float_v0_6_1neg_json_len,
    },
    {
        .version = "vmaf_v0.6.1",
        .data = &___src_______model_vmaf_v0_6_1_json,
        .data_len = &___src_______model_vmaf_v0_6_1_json_len,
    },
    {
        .version = "vmaf_float_v0.6.1",
        .data = &___src_______model_vmaf_float_v0_6_1_json,
        .data_len = &___src_______model_vmaf_float_v0_6_1_json_len,
    },
    {
        .version = "vmaf_b_v0.6.3",
        .data = &___src_______model_vmaf_b_v0_6_3_json,
        .data_len = &___src_______model_vmaf_b_v0_6_3_json_len,
    },
    {
        .version = "vmaf_v0.6.1neg",
        .data = &___src_______model_vmaf_v0_6_1neg_json,
        .data_len = &___src_______model_vmaf_v0_6_1neg_json_len,
    },
    {
        .version = "vmaf_float_b_v0.6.3",
        .data = &___src_______model_vmaf_float_b_v0_6_3_json,
        .data_len = &___src_______model_vmaf_float_b_v0_6_3_json_len,
    },
#endif
    { 0 }
};

#define BUILT_IN_MODEL_CNT \
    ((sizeof(built_in_models)) / (sizeof(built_in_models[0]))) - 1

int vmaf_model_load(VmafModel **model, VmafModelConfig *cfg,
                    const char *version)
{
    const VmafBuiltInModel *built_in_model = NULL;

    for (unsigned i = 0; i < BUILT_IN_MODEL_CNT; i++) {
        if (!strcmp(version, built_in_models[i].version)) {
            built_in_model = &built_in_models[i];
            break;
        }
    }

    if (!built_in_model) return -EINVAL;

    return vmaf_read_json_model_from_buffer(model, cfg, built_in_model->data,
                                            *built_in_model->data_len);
}

char *vmaf_model_generate_name(VmafModelConfig *cfg)
{
    const char *default_name = "vmaf";
    const size_t name_sz =
        cfg->name ? strlen(cfg->name) + 1 : strlen(default_name) + 1;

    char *name = malloc(name_sz);
    if (!name) return NULL;
    memset(name, 0, name_sz);

    if (!cfg->name)
        strncpy(name, default_name, name_sz);
    else
        strncpy(name, cfg->name, name_sz);

    return name;
}

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg,
                              const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!strcmp(ext, ".json"))
        return vmaf_read_json_model_from_path(model, cfg, path);

    const size_t check_path_sz = strlen(path) + 5 + 1;
    char check_path[check_path_sz];
    memset(check_path, 0, check_path_sz);
    sprintf(check_path, "%s.%04d", path, 1);
    struct stat s;

    // bail, this is a model_collection
    if (!stat(check_path, &s)) return -EAGAIN;

    VmafModel *const m = *model = malloc(sizeof(*m));
    if (!m) goto fail;
    memset(m, 0, sizeof(*m));
    m->path = malloc(strlen(path) + 1);
    if (!m->path) goto free_m;
    strcpy(m->path, path);

    m->name = vmaf_model_generate_name(cfg);
    if (!m->name) goto free_path;

    char *svm_path_suffix = ".model";
    size_t svm_path_sz =
        strlen(path) + strlen(svm_path_suffix) + 1 * sizeof(char);
    char *svm_path = malloc(svm_path_sz);
    if (!svm_path) goto free_name;
    memset(svm_path, 0, svm_path_sz);
    strncat(svm_path, m->path, strlen(m->path));
    strncat(svm_path, svm_path_suffix, strlen(svm_path_suffix));

    if (stat(svm_path, &s)) return -EINVAL;
    m->svm = svm_load_model(svm_path);
    free(svm_path);
    if (!m->svm) goto free_name;
    int err = vmaf_unpickle_model(m, m->path, cfg->flags);
    if (err) goto free_svm;

    return 0;

free_svm:
    svm_free_and_destroy_model(&(m->svm));
free_name:
    free(m->name);
free_path:
    free(m->path);
free_m:
    free(m);
fail:
    return -ENOMEM;
}

static int vmaf_model_load_from_path_internal(VmafModel **model,
                                              VmafModelConfig *cfg,
                                              const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!strcmp(ext, ".json"))
        return vmaf_read_json_model_from_path(model, cfg, path);

    VmafModel *const m = *model = malloc(sizeof(*m));
    if (!m) goto fail;
    memset(m, 0, sizeof(*m));
    m->path = malloc(strlen(path) + 1);
    if (!m->path) goto free_m;
    strcpy(m->path, path);

    m->name = vmaf_model_generate_name(cfg);
    if (!m->name) goto free_path;

    char *svm_path_suffix = ".model";
    size_t svm_path_sz =
        strlen(path) + strlen(svm_path_suffix) + 1 * sizeof(char);
    char *svm_path = malloc(svm_path_sz);
    if (!svm_path) goto free_name;
    memset(svm_path, 0, svm_path_sz);
    strncat(svm_path, m->path, strlen(m->path));
    strncat(svm_path, svm_path_suffix, strlen(svm_path_suffix));

    struct stat s;
    if (stat(svm_path, &s)) return -EINVAL;
    m->svm = svm_load_model(svm_path);
    free(svm_path);
    if (!m->svm) goto free_name;
    int err = vmaf_unpickle_model(m, m->path, cfg->flags);
    if (err) goto free_svm;

    return 0;

free_svm:
    svm_free_and_destroy_model(&(m->svm));
free_name:
    free(m->name);
free_path:
    free(m->path);
free_m:
    free(m);
    *model = NULL;
fail:
    return -ENOMEM;
}

void vmaf_model_destroy(VmafModel *model)
{
    if (!model) return;
    free(model->path);
    free(model->name);
    svm_free_and_destroy_model(&(model->svm));
    for (unsigned i = 0; i < model->n_features; i++) {
        free(model->feature[i].name);
        vmaf_dictionary_free(&model->feature[i].opts_dict);
    }
    free(model->feature);
    free(model);
}

int vmaf_model_collection_append(VmafModelCollection **model_collection,
                                 VmafModel *model)
{
    if (!model_collection) return -EINVAL;
    if (!model) return -EINVAL;

    VmafModelCollection *mc = *model_collection;

    if (!mc) {
        mc = *model_collection = malloc(sizeof(*mc));
        if (!mc) goto fail;
        memset(mc, 0, sizeof(*mc));
        const size_t initial_sz = 8 * sizeof(*mc->model);
        mc->model = malloc(initial_sz);
        if (!mc->model) goto fail_mc;
        memset(mc->model, 0, initial_sz);
        mc->size = 8;
        mc->type = model->type;
        const size_t name_sz = strlen(model->name) - 5 + 1;
        mc->name = malloc(name_sz);
        if (!mc->name) goto fail_model;
        memset(mc->name, 0, name_sz);
        strncpy(mc->name, model->name, name_sz - 1);
    }

    if (mc->type != model->type) return -EINVAL;

    if (mc->cnt == mc->size) {
        const size_t sz = mc->size * sizeof(*mc->model) * 2;
        VmafModel **m = realloc(mc->model, sz);
        if (!m) goto fail;
        mc->model = m;
        mc->size *= 2;
    }

    mc->model[mc->cnt++] = model;
    return 0;

fail_model:
    free(mc->model);
fail_mc:
    free(mc);
fail:
    *model_collection = NULL;
    return -ENOMEM;
}

void vmaf_model_collection_destroy(VmafModelCollection *model_collection)
{
    if (!model_collection) return;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        vmaf_model_destroy(model_collection->model[i]);
    free(model_collection->model);
    free(model_collection->name);
    free(model_collection);
}

int vmaf_model_collection_load(VmafModel **model,
                               VmafModelCollection **model_collection,
                               VmafModelConfig *cfg,
                               const char *version)
{
    const VmafBuiltInModel *built_in_model = NULL;

    for (unsigned i = 0; i < BUILT_IN_MODEL_CNT; i++) {
        if (!strcmp(version, built_in_models[i].version)) {
            built_in_model = &built_in_models[i];
            break;
        }
    }

    if (!built_in_model) return -EINVAL;

    return vmaf_read_json_model_collection_from_buffer(model, model_collection,
                                                     cfg, built_in_model->data,
                                                     *built_in_model->data_len);
}

int vmaf_model_collection_load_from_path(VmafModel **model,
                                         VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!strcmp(ext, ".json")) {
        return vmaf_read_json_model_collection_from_path(model,
                                                   model_collection, cfg, path);
    }

    int err = 0;
    *model_collection = NULL;
    VmafModelConfig c = *cfg;

    char *name = c.name = vmaf_model_generate_name(cfg);
    if (!name) return -ENOMEM;

    // FIXME: this whole function is duplicated just to support
    // implicit parsing of a pkl model collection
    // remove when pkl model collections are depricated
    err = vmaf_model_load_from_path_internal(model, cfg, path);
    if (err) return err;

    // vmaf_rb_v0.6.3.pkl -> vmaf_rb_v0.6.3.pkl.0001 ...
    const size_t check_path_sz = strlen(path) + 5 + 1;
    char check_path[check_path_sz];
    memset(check_path, 0, check_path_sz);

    const size_t cfg_name_sz = strlen(name) + 5 + 1;
    char cfg_name[cfg_name_sz];
    c.name = cfg_name;

    for (unsigned i = 1; i <= 9999; i++) {
        sprintf(c.name, "%s_%04d", name, i);
        sprintf(check_path, "%s.%04d", path, i);
        struct stat s;
        if (stat(check_path, &s))
            break;
        const VmafModel *m;
        err = vmaf_model_load_from_path(&m, &c, check_path);
        if (err)
            goto fail;
        err = vmaf_model_collection_append(model_collection, m);
        if (err) {
            vmaf_model_destroy(m);
            goto fail;
        }
    }

    free(name);
    if (!(*model_collection)) return -EINVAL;
    return 0;

fail:
    vmaf_model_collection_destroy(*model_collection);
    *model_collection = NULL;
    free(name);
    return err;
}
