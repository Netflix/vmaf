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

#include <libvmaf/model.h>

#include "dict.h"
#include "chooseser.h"
#include "model.h"

#define VAL_EQUAL_STR(V,S) (Stringize((V)).compare((S))==0)
#define VAL_IS_LIST(V) ((V).tag=='n') /* check ocval.cc */
#define VAL_IS_NONE(V) ((V).tag=='Z') /* check ocval.cc */
#define VAL_IS_DICT(V) ((V).tag=='t') /* check ocval.cc */

static int unpickle(VmafModel *model, const char *pickle_path,
                    enum VmafModelFlags flags)
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
    Val feature_opts_dicts = pickle_model["model_dict"]["feature_opts_dicts"];

    if (!((VAL_IS_NONE(score_clip)) || VAL_IS_LIST(score_clip)))
        return -EINVAL;
    model->score_clip.enabled = !(VAL_IS_NONE(score_clip)) &&
                                !(flags & VMAF_MODEL_FLAG_DISABLE_CLIP);
    if (model->score_clip.enabled) {
        model->score_clip.min = score_clip[0];
        model->score_clip.max = score_clip[1];
    }

    if (VAL_EQUAL_STR(model_type, "'RESIDUEBOOTSTRAP_LIBSVMNUSVR'"))
        model->type = VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR;
    else if (VAL_EQUAL_STR(model_type, "'BOOTSTRAP_LIBSVMNUSVR'"))
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
    if (VAL_IS_NONE(score_transform) ||
            !(flags & VMAF_MODEL_FLAG_ENABLE_TRANSFORM)) {
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
    if (!VAL_IS_LIST(slopes) || slopes.length() != model->n_features + 1)
        return -EINVAL;
    if (!VAL_IS_LIST(intercepts) || intercepts.length() != model->n_features + 1)
        return -EINVAL;
    if (!((VAL_IS_NONE(feature_opts_dicts)) || VAL_IS_LIST(feature_opts_dicts)))
        return -EINVAL;
    if (VAL_IS_LIST(feature_opts_dicts) && feature_opts_dicts.length() != model->n_features)
        return -EINVAL;
    if (VAL_IS_LIST(feature_opts_dicts)) {
        for (unsigned i = 0; i < model->n_features; i++) {
            if (!(VAL_IS_NONE(feature_opts_dicts[i]) || VAL_IS_DICT(feature_opts_dicts[i])))
                return -EINVAL;
        }
    }

    int err = 0;
    model->feature = (VmafModelFeature *)
        malloc(sizeof(*(model->feature)) * model->n_features);
    if (!model->feature) {
        err = -ENOMEM;
        goto fail;
    }
    memset(model->feature, 0, sizeof(*(model->feature)) * model->n_features);

    for (unsigned i = 0; i < model->n_features; i++) {
        model->feature[i].name = strdup(Stringize(feature_names[i]).c_str());
        if (!model->feature[i].name) {
            err = -ENOMEM;
            goto free_name;
        }

        model->feature[i].slope = double(slopes[i + 1]);
        model->feature[i].intercept = double(intercepts[i + 1]);

        if (VAL_IS_LIST(feature_opts_dicts)) {
            Val feature_opts_dict = feature_opts_dicts[i];
            if (!VAL_IS_DICT(feature_opts_dict)) {
                err = -EINVAL;
                goto free_name;
            }

            Tab feature_opts_dict_tab = feature_opts_dict;
            Arr keys = feature_opts_dict_tab.keys();
            for (unsigned j = 0; j < keys.length(); j++) {
                char *key = strdup(Stringize(keys[j]).c_str());
                key[strlen(key) - 1] = 0; //FIXME: ptools
                char *val =
                    strdup(Stringize(feature_opts_dict_tab[keys[j]]).c_str());
                err = vmaf_dictionary_set(&(model->feature[i].opts_dict),
                                          key + 1, val,
                                          VMAF_DICT_DO_NOT_OVERWRITE);
                free(key);
                free(val);
                if (err) goto free_name;
            }
        }
    }

    model->slope = double(slopes[0]);
    model->intercept = double(intercepts[0]);
    return 0;

free_name:
    for (unsigned i = 0; i < model->n_features; i++) {
        if (model->feature[i].name)
            free(model->feature[i].name);
    }
free_feature:
    free(model->feature);
fail:
    return err;
}

extern "C" {

int vmaf_unpickle_model(VmafModel *model, const char *pickle_path,
                        enum VmafModelFlags flags)
{
    return unpickle(model, pickle_path, flags);
}

}
