#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <libvmaf/model.h>

#include "config.h"
#include "feature/feature_extractor.h"
#include "log.h"
#include "model.h"
#include "read_json_model.h"
#include "svm.h"

typedef struct VmafBuiltInModel {
    const char *version;
    const char *data;
    const int *data_len;
} VmafBuiltInModel;

#if VMAF_BUILT_IN_MODELS
#if VMAF_FLOAT_FEATURES
extern const char src_vmaf_float_v0_6_1neg_json;
extern const int src_vmaf_float_v0_6_1neg_json_len;
extern const char src_vmaf_float_v0_6_1_json;
extern const int src_vmaf_float_v0_6_1_json_len;
extern const char src_vmaf_float_b_v0_6_3_json;
extern const int src_vmaf_float_b_v0_6_3_json_len;
extern const char src_vmaf_float_4k_v0_6_1_json;
extern const int src_vmaf_float_4k_v0_6_1_json_len;
#endif
extern const char src_vmaf_v0_6_1_json;
extern const int src_vmaf_v0_6_1_json_len;
extern const char src_vmaf_b_v0_6_3_json;
extern const int src_vmaf_b_v0_6_3_json_len;
extern const char src_vmaf_v0_6_1neg_json;
extern const int src_vmaf_v0_6_1neg_json_len;
extern const char src_vmaf_4k_v0_6_1_json;
extern const int src_vmaf_4k_v0_6_1_json_len;
extern const char src_vmaf_4k_v0_6_1neg_json;
extern const int src_vmaf_4k_v0_6_1neg_json_len;
#endif

static const VmafBuiltInModel built_in_models[] = {
#if VMAF_BUILT_IN_MODELS
#if VMAF_FLOAT_FEATURES
    {
        .version = "vmaf_float_v0.6.1",
        .data = &src_vmaf_float_v0_6_1_json,
        .data_len = &src_vmaf_float_v0_6_1_json_len,
    },
    {
        .version = "vmaf_float_b_v0.6.3",
        .data = &src_vmaf_float_b_v0_6_3_json,
        .data_len = &src_vmaf_float_b_v0_6_3_json_len,
    },
    {
        .version = "vmaf_float_v0.6.1neg",
        .data = &src_vmaf_float_v0_6_1neg_json,
        .data_len = &src_vmaf_float_v0_6_1neg_json_len,
    },
    {
        .version = "vmaf_float_4k_v0.6.1",
        .data = &src_vmaf_float_4k_v0_6_1_json,
        .data_len = &src_vmaf_float_4k_v0_6_1_json_len,
    },
#endif
    {
        .version = "vmaf_v0.6.1",
        .data = &src_vmaf_v0_6_1_json,
        .data_len = &src_vmaf_v0_6_1_json_len,
    },
    {
        .version = "vmaf_b_v0.6.3",
        .data = &src_vmaf_b_v0_6_3_json,
        .data_len = &src_vmaf_b_v0_6_3_json_len,
    },
    {
        .version = "vmaf_v0.6.1neg",
        .data = &src_vmaf_v0_6_1neg_json,
        .data_len = &src_vmaf_v0_6_1neg_json_len,
    },
    {
        .version = "vmaf_4k_v0.6.1",
        .data = &src_vmaf_4k_v0_6_1_json,
        .data_len = &src_vmaf_4k_v0_6_1_json_len,
    },
    {
        .version = "vmaf_4k_v0.6.1neg",
        .data = &src_vmaf_4k_v0_6_1neg_json,
        .data_len = &src_vmaf_4k_v0_6_1neg_json_len,
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
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "could not read model from path: \"%s\"\n", path);
        char *ext = strrchr(path, '.');
        if (ext && !strcmp(ext, ".pkl")) {
            vmaf_log(
                VMAF_LOG_LEVEL_ERROR,
                "support for pkl model files has been removed, use json\n");
        }
    }
    return err;
}

int vmaf_model_feature_overload(VmafModel *model, const char *feature_name,
                                VmafFeatureDictionary *opts_dict)
{
    if (!model) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (!opts_dict) return -EINVAL;

    int err = 0;

    for (unsigned i = 0; i < model->n_features; i++) {
        VmafFeatureExtractor *fex =
            vmaf_get_feature_extractor_by_feature_name(model->feature[i].name, 0);
        if (!fex) continue;
        if (strcmp(feature_name, fex->name)) continue;
        VmafDictionary *d =
            vmaf_dictionary_merge((VmafDictionary**)&model->feature[i].opts_dict,
                                  (VmafDictionary**)&opts_dict, 0);
        if (!d) return -ENOMEM;
        err = vmaf_dictionary_free(&model->feature[i].opts_dict);
        if (err) goto exit;
        model->feature[i].opts_dict = d;
    }

exit:
    err |= vmaf_dictionary_free((VmafDictionary**)&opts_dict);
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
    free(model->score_transform.knots.list);
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
        memset((char*)mc->name, 0, name_sz);
        strncpy((char*)mc->name, model->name, name_sz - 1);
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
    free((char*)model_collection->name);
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
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "could not read model collection from path: \"%s\"\n", path);
        char *ext = strrchr(path, '.');
        if (ext && !strcmp(ext, ".pkl")) {
            vmaf_log(
                VMAF_LOG_LEVEL_ERROR,
                "support for pkl model files has been removed, use json\n");
        }
    }

    return err;
}

int vmaf_model_collection_feature_overload(VmafModel *model,
                                           VmafModelCollection **model_collection,
                                           const char *feature_name,
                                           VmafFeatureDictionary *opts_dict)
{
    if (!model_collection) return -EINVAL;
    VmafModelCollection *mc = *model_collection;

    int err = 0;
    for (unsigned i = 0; i < mc->cnt; i++) {
        VmafFeatureDictionary *d = NULL;
        if (vmaf_dictionary_copy((VmafDictionary**)&opts_dict, (VmafDictionary**)&d)) goto exit;
        err |= vmaf_model_feature_overload(mc->model[i], feature_name, d);
    }

exit:
    err |= vmaf_model_feature_overload(model, feature_name, opts_dict);
    return err;
}

