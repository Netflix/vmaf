#include <errno.h>

#include "chooseser.h"
#include "model.h"

#define VAL_EQUAL_STR(V,S) (Stringize((V)).compare((S))==0)
#define VAL_IS_LIST(V) ((V).tag=='n') /* check ocval.cc */
#define VAL_IS_NONE(V) ((V).tag=='Z') /* check ocval.cc */
#define VAL_IS_DICT(V) ((V).tag=='t') /* check ocval.cc */

static int unpickle(VmafModel *model, const char *pickle_path)
{
    Val pickle_model;
    LoadValFromFile(pickle_path, pickle_model, SERIALIZE_P0);
    Val model_type = pickle_model["model_dict"]["model_type"];
    Val feature_names = pickle_model["model_dict"]["feature_names"];
    Val norm_type = pickle_model["model_dict"]["norm_type"];
    Val slopes = pickle_model["model_dict"]["slopes"];
    Val intercepts = pickle_model["model_dict"]["intercepts"];
    Val score_clip = pickle_model["model_dict"]["score_clip"];
    Val score_transform = pickle_model["model_dict"]["score_transform"];

    if (!((VAL_IS_NONE(score_clip)) || VAL_IS_LIST(score_clip)))
        return -EINVAL;
    model->score_clip.enabled = !(VAL_IS_NONE(score_clip));
    if (model->score_clip.enabled) {
        model->score_clip.min = score_clip[0];
        model->score_clip.max = score_clip[1];
    }

    if (VAL_EQUAL_STR(model_type, "'RESIDUEBOOTSTRAP_LIBSVMNUSVR'"))
        model->type = VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR;
    else if (VAL_EQUAL_STR(model_type, "BOOTSTRAP_LIBSVMNUSVR"))
        model->type = VMAF_MODEL_BOOTSTRAP_SVM_NUSVR;
    else if (VAL_EQUAL_STR(model_type, "'LIBSVMNUSVR'"))
        model->type = VMAF_MODEL_TYPE_SVM_NUSVR;
    else
        return -EINVAL;

    if (VAL_EQUAL_STR(norm_type, "'linear_rescale'"))
        model->norm_type = VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE;
    else if (VAL_EQUAL_STR(norm_type, "'none'"))
        model->norm_type = VMAF_MODEL_NORMALIZATION_TYPE_NONE;
    else
        return -EINVAL;

    if (!(VAL_IS_NONE(score_transform) || VAL_IS_DICT(score_transform)))
        return -EINVAL;
    if (VAL_IS_NONE(score_transform)) {
        model->score_transform.enabled = false;
    } else {
        model->score_transform.enabled = true;

        if (VAL_IS_NONE(score_transform["p0"])) {
            model->score_transform.p0.enabled = false;
        } else {
            model->score_transform.p0.enabled = true;
            model->score_transform.p0.value = score_transform["p0"];
        }
        if (VAL_IS_NONE(score_transform["p1"])) {
            model->score_transform.p1.enabled = false;
        } else {
            model->score_transform.p1.enabled = true;
            model->score_transform.p1.value = score_transform["p1"];
        }
        if (VAL_IS_NONE(score_transform["p2"])) {
            model->score_transform.p2.enabled = false;
        } else {
            model->score_transform.p2.enabled = true;
            model->score_transform.p2.value = score_transform["p2"];
        }
        if (!VAL_IS_NONE(score_transform["out_lte_in"]) &&
            VAL_EQUAL_STR(score_transform["out_lte_in"], "'true'"))
        {
            model->score_transform.out_lte_in = true;
        }
        if (!VAL_IS_NONE(score_transform["out_gte_in"]) &&
            VAL_EQUAL_STR(score_transform["out_gte_in"], "'true'"))
        {
            model->score_transform.out_gte_in = true;
        }
    }

    if (!VAL_IS_LIST(feature_names))
        return -EINVAL;
    model->n_features = feature_names.length();
    model->feature = (VmafModelFeature *)
        malloc(sizeof(*(model->feature)) * model->n_features);
    if (!model->feature) goto fail;
    memset(model->feature, 0, sizeof(*(model->feature)) * model->n_features);
    for (unsigned i = 0; i < model->n_features; i++) {
       model->feature[i].name = strdup(Stringize(feature_names[i]).c_str());
       if (!model->feature[i].name) goto free_name;
       model->feature[i].slope = slopes[i];
       model->feature[i].intercept = intercepts[i];
    }

    return 0;

free_name:
    for (unsigned i = 0; i < model->n_features; i++) {
        if (model->feature[i].name)
            free(model->feature[i].name);
    }
free_feature:
    free(model->feature);
fail:
    return -ENOMEM;
}

extern "C" {

int vmaf_unpickle_model(VmafModel *model, const char *pickle_path)
{
    return unpickle(model, pickle_path);
}

}
