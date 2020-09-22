#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <libvmaf/model.h>

#include "model.h"
#include "svm.h"
#include "unpickle.h"

static char *generate_model_name(VmafModelConfig *cfg) {

    char *default_name = "vmaf";
    size_t name_sz;
    char *name;

    if (!cfg->name) {
        name_sz = strlen(default_name) + 1 * sizeof(char);
    } else {
        name_sz = strlen(cfg->name) + 1 * sizeof(char);
    }

    name = malloc(name_sz);
    if (!name) {
        return NULL;
    }
    memset(name, 0, name_sz);

    /* if there is no name, create a default one */
    if (!cfg->name) {
        strcpy(name, default_name);
    } else {
        strcpy(name, cfg->name);
    }

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

    /* if config does not have a name, create a default one */
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
        if (!mc->model) {
            free(mc);
            *model_collection = NULL;
            goto fail;
        }
        memset(mc->model, 0, initial_sz);
        mc->size = 8;
    }

    if (mc->cnt == mc->size) {
        const size_t sz = mc->size * sizeof(*mc->model) * 2;
        VmafModel *model = realloc(mc->model, sz);
        if (!model) goto fail;
        mc->model = model;
        mc->size *= 2;
    }

    mc->model[mc->cnt++] = model;
    return 0;

fail:
    return -ENOMEM;
}

void vmaf_model_collection_destroy(VmafModelCollection *model_collection)
{
    if (!model_collection) return;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        vmaf_model_destroy(model_collection->model[i]);
    free(model_collection->model);
    free(model_collection);
    memset(model_collection, 0, sizeof(*model_collection));
}
