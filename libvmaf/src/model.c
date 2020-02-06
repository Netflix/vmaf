#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <libvmaf/model.h>

#include "model.h"
#include "svm.h"
#include "unpickle.h"

char *generate_model_name(VmafModelConfig *cfg) {

    char *clip_phrase = "_clip_disabled";
    char *transform_phrase = "_score_transform_enabled";
    size_t name_sz;
    char *name;

    if (!cfg->name) {
        name_sz = strlen(cfg->path) + strlen(clip_phrase) +
            strlen(transform_phrase) + 1 * sizeof(char);
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
        strncat(name, cfg->path, strlen(cfg->path));
        if (cfg->flags && (cfg->flags & VMAF_MODEL_FLAG_DISABLE_CLIP)) {
            strncat(name, clip_phrase, strlen(clip_phrase));
        }
        if (cfg->flags && (cfg->flags & VMAF_MODEL_FLAG_ENABLE_TRANSFORM)) {
            strncat(name, transform_phrase, strlen(transform_phrase));
        }
    } else {
        strcpy(name, cfg->name);
    }

    return name;

}

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg)
{
    VmafModel *const m = *model = malloc(sizeof(*m));
    if (!m) goto fail;
    memset(m, 0, sizeof(*m));
    m->path = malloc(strlen(cfg->path) + 1);
    if (!m->path) goto free_m;
    strcpy(m->path, cfg->path);

    /* if config does not have a name, create a default one */
    m->name = generate_model_name(cfg);
    if (!m->name) goto free_path;

    // ugly, this shouldn't be implict (but it is)
    char *svm_path_suffix = ".model";
    size_t svm_path_sz =
        strlen(m->path) + strlen(svm_path_suffix) + 1 * sizeof(char);
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
    for (unsigned i = 0; i < model->n_features; i++)
        free(model->feature[i].name);
    free(model->feature);
    free(model);
}
