#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <libvmaf/model.h>

#include "model.h"
#include "svm.h"
#include "unpickle.h"

static char *generate_model_name(VmafModelConfig *cfg)
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
    VmafModel *const m = *model = malloc(sizeof(*m));
    if (!m) goto fail;
    memset(m, 0, sizeof(*m));
    m->path = malloc(strlen(path) + 1);
    if (!m->path) goto free_m;
    strcpy(m->path, path);

    m->name = generate_model_name(cfg);
    if (!m->name) goto free_path;

    char *svm_path_suffix = ".model";
    size_t svm_path_sz =
        strlen(path) + strlen(svm_path_suffix) + 1 * sizeof(char);
    char *svm_path = malloc(svm_path_sz);
    if (!svm_path) goto free_name;
    memset(svm_path, 0, svm_path_sz);
    strncat(svm_path, m->path, strlen(m->path));
    strncat(svm_path, svm_path_suffix, strlen(svm_path_suffix));

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

static int vmaf_model_collection_append(VmafModelCollection **model_collection,
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


int vmaf_model_collection_load_from_path(VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *path)
{
    int err = 0;
    *model_collection = NULL;
    VmafModelConfig tmp_cfg = *cfg;

    // vmaf_rb_v0.6.3.pkl -> vmaf_rb_v0.6.3.pkl.0001 ...
    const size_t check_path_sz = strlen(path) + 5 + 1;
    char check_path[check_path_sz];
    memset(check_path, 0, check_path_sz);

    char *name = generate_model_name(cfg);
    if (!name) return -ENOMEM;
    const size_t cfg_name_sz = strlen(name) + 5 + 1;
    char cfg_name[cfg_name_sz];
    tmp_cfg.name = cfg_name;

    for (unsigned i = 1; i <= 9999; i++) {
        sprintf(tmp_cfg.name, "%s_%04d", name, i);
        sprintf(check_path, "%s.%04d", path, i);
        struct stat s;
        if (stat(check_path, &s))
            break;
        const VmafModel *model;
        err = vmaf_model_load_from_path(&model, &tmp_cfg, check_path);
        if (err)
            goto fail;
        err = vmaf_model_collection_append(model_collection, model);
        if (err) {
            vmaf_model_destroy(model);
            goto fail;
        }
    }

    free(name);
    return 0;

fail:
    vmaf_model_collection_destroy(*model_collection);
    *model_collection = NULL;
    free(name);
    return err;
}
