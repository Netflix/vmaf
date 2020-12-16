#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <libvmaf/model.h>

#include "config.h"
#include "log.h"
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
#if VMAF_FLOAT_FEATURES
extern const char ___src_______model_vmaf_float_v0_6_1neg_json;
extern const int ___src_______model_vmaf_float_v0_6_1neg_json_len;
extern const char ___src_______model_vmaf_float_v0_6_1_json;
extern const int ___src_______model_vmaf_float_v0_6_1_json_len;
extern const char ___src_______model_vmaf_float_b_v0_6_3_json;
extern const int ___src_______model_vmaf_float_b_v0_6_3_json_len;
#endif
extern const char ___src_______model_vmaf_v0_6_1_json;
extern const int ___src_______model_vmaf_v0_6_1_json_len;
extern const char ___src_______model_vmaf_b_v0_6_3_json;
extern const int ___src_______model_vmaf_b_v0_6_3_json_len;
extern const char ___src_______model_vmaf_v0_6_1neg_json;
extern const int ___src_______model_vmaf_v0_6_1neg_json_len;
#endif

static const VmafBuiltInModel built_in_models[] = {
#if VMAF_BUILT_IN_MODELS
#if VMAF_FLOAT_FEATURES
    {
        .version = "vmaf_float_v0.6.1neg",
        .data = &___src_______model_vmaf_float_v0_6_1neg_json,
        .data_len = &___src_______model_vmaf_float_v0_6_1neg_json_len,
    },
    {
        .version = "vmaf_float_v0.6.1",
        .data = &___src_______model_vmaf_float_v0_6_1_json,
        .data_len = &___src_______model_vmaf_float_v0_6_1_json_len,
    },
    {
        .version = "vmaf_float_b_v0.6.3",
        .data = &___src_______model_vmaf_float_b_v0_6_3_json,
        .data_len = &___src_______model_vmaf_float_b_v0_6_3_json_len,
    },
#endif
    {
        .version = "vmaf_v0.6.1",
        .data = &___src_______model_vmaf_v0_6_1_json,
        .data_len = &___src_______model_vmaf_v0_6_1_json_len,
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

    if (!built_in_model) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "no such built-in model: \"%s\"\n", version);
        return -EINVAL;
    }

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
    int err = vmaf_read_json_model_from_path(model, cfg, path);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "could not read model from path: \"%s\"\n", path);
        char *ext = strrchr(path, '.');
        if (ext && !strcmp(ext, ".pkl")) {
            vmaf_log(VMAF_LOG_LEVEL_WARNING,
                     "pkl model files have been deprecated, use json\n");
        }
    }
    return err;
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

    if (!built_in_model) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "no such built-in model collection: \"%s\"\n", version);
        return -EINVAL;
    }

    return vmaf_read_json_model_collection_from_buffer(model, model_collection,
                                                     cfg, built_in_model->data,
                                                     *built_in_model->data_len);
}

int vmaf_model_collection_load_from_path(VmafModel **model,
                                         VmafModelCollection **model_collection,
                                         VmafModelConfig *cfg,
                                         const char *path)
{
    int err =
        vmaf_read_json_model_collection_from_path(model, model_collection,
                                                  cfg, path);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "could not read model collection from path: \"%s\"\n", path);
        char *ext = strrchr(path, '.');
        if (ext && !strcmp(ext, ".pkl")) {
            vmaf_log(VMAF_LOG_LEVEL_WARNING,
                     "pkl model files have been deprecated, use json\n");
        }
    }

    return err;
}
