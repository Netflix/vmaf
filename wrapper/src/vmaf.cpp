/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "vmaf.h"
#include "combo.h"
#include "pugixml/pugixml.hpp"
#include "timer.h"
#include "jsonprint.h"
#include "debug.h"

#define VAL_EQUAL_STR(V,S) (Stringize((V)).compare((S))==0)
#define VAL_IS_LIST(V) ((V).tag=='n') /* check ocval.cc */
#define VAL_IS_NONE(V) ((V).tag=='Z') /* check ocval.cc */
#define VAL_IS_DICT(V) ((V).tag=='t') /* check ocval.cc */

//#define MIN(A,B) ((A)<(B)?(A):(B))

#if !defined(min)
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

inline double _round_to_digit(double val, int digit);
string _get_file_name(const std::string& s);

void SvmDelete::operator()(void *svm)
{
    svm_free_and_destroy_model((svm_model **)&svm);
}

void LibsvmNusvrTrainTestModel::_assert_model_type(Val model_type) {
    if (!VAL_EQUAL_STR(model_type, "'LIBSVMNUSVR'")) {
        printf("Expect model type LIBSVMNUSVR, "
                "but got %s\n", Stringize(model_type).c_str());
        throw VmafException("Incompatible model_type");
    }
}

void LibsvmNusvrTrainTestModel::_read_and_assert_model(const char *model_path, Val& feature_names,
     Val& norm_type, Val& slopes, Val& intercepts, Val& score_clip, Val& score_transform)
{
    /*  TrainTestModel._assert_trained() in Python:
     *  assert 'model_type' in self.model_dict # need this to recover class
        assert 'feature_names' in self.model_dict
        assert 'norm_type' in self.model_dict
        assert 'model' in self.model_dict

        norm_type = self.model_dict['norm_type']
        assert norm_type == 'none' or norm_type == 'linear_rescale'

        if norm_type == 'linear_rescale':
            assert 'slopes' in self.model_dict
            assert 'intercepts' in self.model_dict
     */
    Val model, model_type;
    char errmsg[1024];
    try
    {
        LoadValFromFile(model_path, model, SERIALIZE_P0);

        model_type = model["model_dict"]["model_type"];
        feature_names = model["model_dict"]["feature_names"];
        norm_type = model["model_dict"]["norm_type"];

        slopes = model["model_dict"]["slopes"];
        intercepts = model["model_dict"]["intercepts"];
        score_clip = model["model_dict"]["score_clip"];
        score_transform = model["model_dict"]["score_transform"];
    }
    catch (std::runtime_error& e)
    {
        printf("Input model at %s cannot be read successfully.\n", model_path);
        sprintf(errmsg, "Error loading model (.pkl): %s", e.what());
        throw VmafException(errmsg);
    }

    _assert_model_type(model_type);

    if (!VAL_IS_LIST(feature_names))
    {
        printf("feature_names in model must be a list.\n");
        throw VmafException("Incompatible feature_names");
    }

    if (!(VAL_EQUAL_STR(norm_type, "'none'") ||
          VAL_EQUAL_STR(norm_type, "'linear_rescale'")
          )
        )
    {
        printf("norm_type in model must be either 'none' or 'linear_rescale'.\n");
        throw VmafException("Incompatible norm_type");
    }

    if ( VAL_EQUAL_STR(norm_type, "'linear_rescale'") &&
         (!VAL_IS_LIST(slopes) || !VAL_IS_LIST(intercepts))
        )
    {
        printf("if norm_type in model is 'linear_rescale', "
                "both slopes and intercepts must be a list.\n");
        throw VmafException("Incompatible slopes or intercepts");
    }

    if (!(VAL_IS_NONE(score_clip) || VAL_IS_LIST(score_clip)))
    {
        printf("score_clip in model must be either None or list.\n");
        throw VmafException("Incompatible score_clip");
    }

    if (!(VAL_IS_NONE(score_transform) || VAL_IS_DICT(score_transform)))
    {
        printf("score_transform in model must be either None or dictionary (table).\n");
        throw VmafException("Incompatible score_transform");
    }

}

void LibsvmNusvrTrainTestModel::load_model()
{
    dbg_printf("Read input model (pkl) at %s ...\n", model_path);
    _read_and_assert_model(model_path, feature_names, norm_type, slopes, intercepts, score_clip, score_transform);

    /* follow the convention that if model_path is a/b.c, the libsvm_model_path is always a/b.c.model */
    std::string libsvm_model_path_ = std::string(model_path) + std::string(".model");
    const char *libsvm_model_path = libsvm_model_path_.c_str();
    dbg_printf("Read input model (libsvm) at %s ...\n", libsvm_model_path);
    svm_model_ptr = _read_and_assert_svm_model(libsvm_model_path);
}

std::unique_ptr<svm_model, SvmDelete> LibsvmNusvrTrainTestModel::_read_and_assert_svm_model(const char* libsvm_model_path)
{
    std::unique_ptr<svm_model, SvmDelete> svm_model_ptr { svm_load_model(libsvm_model_path) };
    if (!svm_model_ptr) {
        printf("Error loading SVM model.\n");
        throw VmafException("Error loading SVM model");
    }
    return svm_model_ptr;
}

void LibsvmNusvrTrainTestModel::populate_and_normalize_nodes_at_frm(size_t i_frm,
        svm_node*& nodes, StatVector& adm2,
        StatVector& adm_scale0, StatVector& adm_scale1,
        StatVector& adm_scale2, StatVector& adm_scale3, StatVector& motion,
        StatVector& vif_scale0, StatVector& vif_scale1, StatVector& vif_scale2,
        StatVector& vif_scale3, StatVector& vif, StatVector& motion2) {
    if (VAL_EQUAL_STR(norm_type, "'linear_rescale'")) {
        for (size_t j = 0; j < feature_names.length(); j++) {
            nodes[j].index = j + 1;
            if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm2_score'") == 0)
                nodes[j].value = double(slopes[j + 1]) * adm2.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale0_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * adm_scale0.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale1_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * adm_scale1.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale2_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * adm_scale2.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale3_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * adm_scale3.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_motion_score'") == 0)
                nodes[j].value = double(slopes[j + 1]) * motion.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale0_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * vif_scale0.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale1_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * vif_scale1.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale2_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * vif_scale2.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale3_score'") == 0)
                nodes[j].value = double(slopes[j + 1])
                        * vif_scale3.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_score'") == 0)
                nodes[j].value = double(slopes[j + 1]) * vif.at(i_frm)
                        + double(intercepts[j + 1]);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_motion2_score'") == 0)
                nodes[j].value = double(slopes[j + 1]) * motion2.at(i_frm)
                        + double(intercepts[j + 1]);
            else {
                printf("Unknown feature name: %s.\n",
                        Stringize(feature_names[j]).c_str());
                throw VmafException("Unknown feature name");
            }
        }
    } else {
        for (size_t j = 0; j < feature_names.length(); j++) {
            nodes[j].index = j + 1;
            if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm2_score'") == 0)
                nodes[j].value = adm2.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale0_score'") == 0)
                nodes[j].value = adm_scale0.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale1_score'") == 0)
                nodes[j].value = adm_scale1.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale2_score'") == 0)
                nodes[j].value = adm_scale2.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_adm_scale3_score'") == 0)
                nodes[j].value = adm_scale3.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_motion_score'") == 0)
                nodes[j].value = motion.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale0_score'") == 0)
                nodes[j].value = vif_scale0.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale1_score'") == 0)
                nodes[j].value = vif_scale1.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale2_score'") == 0)
                nodes[j].value = vif_scale2.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_scale3_score'") == 0)
                nodes[j].value = vif_scale3.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_vif_score'") == 0)
                nodes[j].value = vif.at(i_frm);
            else if (strcmp(Stringize(feature_names[j]).c_str(),
                    "'VMAF_feature_motion2_score'") == 0)
                nodes[j].value = motion2.at(i_frm);
            else {
                printf("Unknown feature name: %s.\n",
                        Stringize(feature_names[j]).c_str());
                throw VmafException("Unknown feature name");
            }
        }
    }
}

std::map<VmafPredictionReturnType, double> LibsvmNusvrTrainTestModel::predict(svm_node* nodes) {

    double prediction = svm_predict(svm_model_ptr.get(), nodes);

    /* denormalize score */
    _denormalize_prediction(prediction);

    std::map<VmafPredictionReturnType, double> predictionMap;
    predictionMap[VmafPredictionReturnType::SCORE] = prediction;

    return predictionMap;
}

void LibsvmNusvrTrainTestModel::_denormalize_prediction(double& prediction) {
    if (VAL_EQUAL_STR(norm_type, "'linear_rescale'")) {
        /* denormalize */
        prediction = (prediction - double(intercepts[0]))
                / double(slopes[0]);
    } else {
        ;
    }
}

std::string BootstrapLibsvmNusvrTrainTestModel::_get_model_i_filename(const char* model_path, int i_model)
{
    if (i_model == 0) {
        return std::string(model_path);
    }
    else {
        std::stringstream ss;
        ss << '.' << std::setw(4) << std::setfill('0') << i_model;
        std::string s = ss.str();
        std::string model_path_i = std::string(model_path) + s;
        return model_path_i;
    }
}

void BootstrapLibsvmNusvrTrainTestModel::_assert_model_type(Val model_type) {
    if (!VAL_EQUAL_STR(model_type, "'RESIDUEBOOTSTRAP_LIBSVMNUSVR'") &&
            !VAL_EQUAL_STR(model_type, "'BOOTSTRAP_LIBSVMNUSVR'")) {
        printf("Expect model type BOOTSTRAP_LIBSVMNUSVR or "
                "RESIDUEBOOTSTRAP_LIBSVMNUSVR, but got %s\n", Stringize(model_type).c_str());
        throw VmafException("Incompatible model_type");
    }
}

void BootstrapLibsvmNusvrTrainTestModel::_read_and_assert_model(const char *model_path, Val& feature_names, Val& norm_type, Val& slopes,
        Val& intercepts, Val& score_clip, Val& score_transform, int& numModels)
{
    LibsvmNusvrTrainTestModel::_read_and_assert_model(model_path, feature_names, norm_type, slopes, intercepts, score_clip, score_transform);

    Val model, model_type, numModelsVal;
    char errmsg[1024];
    try
    {
        LoadValFromFile(model_path, model, SERIALIZE_P0);
        numModelsVal = model["param_dict"]["num_models"];
    }
    catch (std::runtime_error& e)
    {
        printf("num_models at %s cannot be read successfully.\n", model_path);
        sprintf(errmsg, "Error loading model (.pkl): %s", e.what());
        throw VmafException(errmsg);
    }
    if (VAL_IS_NONE(numModelsVal))
    {
        printf("num_models cannot be none.\n");
        throw VmafException("num_models cannot be none.");
    }
    numModels = numModelsVal;
}

void BootstrapLibsvmNusvrTrainTestModel::load_model()
{

    int numModels;

    std::string model_path_0 = _get_model_i_filename(this->model_path, 0);

    dbg_printf("Read input model (pkl) at %s ...\n", model_path_0.c_str());
    _read_and_assert_model(model_path_0.c_str(), feature_names, norm_type, slopes, intercepts, score_clip, score_transform, numModels);
    dbg_printf("Number of models: %d\n", numModels);

    for (size_t i=0; i<numModels; i++)
    {
        std::string model_path_i = _get_model_i_filename(this->model_path, i);
        /* follow the convention that if model_path is a/b.c, the libsvm_model_path is always a/b.c.model */
        std::string libsvm_model_path_i = model_path_i + std::string(".model");
        dbg_printf("Read input model (libsvm) at %s ...\n", libsvm_model_path_i.c_str());

        if (i == 0)
        {
            svm_model_ptr = _read_and_assert_svm_model(libsvm_model_path_i.c_str());
        }
        else
        {
            bootstrap_svm_model_ptrs.push_back(_read_and_assert_svm_model(libsvm_model_path_i.c_str()));
        }
    }

}

std::map<VmafPredictionReturnType, double> BootstrapLibsvmNusvrTrainTestModel::predict(svm_node* nodes) {

    std::map<VmafPredictionReturnType, double> predictionMap = LibsvmNusvrTrainTestModel::predict(nodes);

    StatVector predictions;
    double prediction;
    for (size_t i=0; i<bootstrap_svm_model_ptrs.size(); i++) {
        prediction = svm_predict(bootstrap_svm_model_ptrs.at(i).get(), nodes);
        _denormalize_prediction(prediction);
        predictions.append(prediction);
    }
    predictionMap[VmafPredictionReturnType::BAGGING_SCORE] = predictions.mean();
    predictionMap[VmafPredictionReturnType::STDDEV] = predictions.std();
    predictionMap[VmafPredictionReturnType::CI95_LOW] = predictions.percentile(2.5);
    predictionMap[VmafPredictionReturnType::CI95_HIGH] = predictions.percentile(97.5);

    return predictionMap;
}

void VmafQualityRunner::_transform_value(LibsvmNusvrTrainTestModel& model,
        double& prediction) {
    if (!VAL_IS_NONE(model.score_transform)) {
        double value = 0.0;

        /* quadratic transform */
        if (!VAL_IS_NONE(model.score_transform["p0"])) {
            value += double(model.score_transform["p0"]);
        }
        if (!VAL_IS_NONE(model.score_transform["p1"])) {
            value += double(model.score_transform["p1"]) * prediction;
        }
        if (!VAL_IS_NONE(model.score_transform["p2"])) {
            value += double(model.score_transform["p2"]) * prediction
                    * prediction;
        }

        /* rectification */
        if (!VAL_IS_NONE(model.score_transform["out_lte_in"])
                && VAL_EQUAL_STR(model.score_transform["out_lte_in"], "'true'")) {
            if (value > prediction) {
                value = prediction;
            }
        }
        if (!VAL_IS_NONE(model.score_transform["out_gte_in"])
                && VAL_EQUAL_STR(model.score_transform["out_gte_in"], "'true'")) {
            if (value < prediction) {
                value = prediction;
            }
        }

        prediction = value;
    }
}

void VmafQualityRunner::_clip_value(LibsvmNusvrTrainTestModel& model,
        double& prediction) {
    if (!VAL_IS_NONE(model.score_clip)) {
        if (prediction < double(model.score_clip[0])) {
            prediction = double(model.score_clip[0]);
        } else if (prediction > double(model.score_clip[1])) {
            prediction = double(model.score_clip[1]);
        }
    }
}

void VmafQualityRunner::_postproc_predict(
        std::map<VmafPredictionReturnType, double>& predictionMap) {
    ;
}

void VmafQualityRunner::_transform_score(LibsvmNusvrTrainTestModel& model,
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double& prediction = predictionMap[VmafPredictionReturnType::SCORE];
    _transform_value(model, prediction);
}

void VmafQualityRunner::_clip_score(LibsvmNusvrTrainTestModel& model,
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double& prediction = predictionMap[VmafPredictionReturnType::SCORE];
    _clip_value(model, prediction);
}

void VmafQualityRunner::_postproc_transform_clip(
        std::map<VmafPredictionReturnType, double>& predictionMap) {
    ;
}

void VmafQualityRunner::_normalize_predict_denormalize_transform_clip(
        LibsvmNusvrTrainTestModel& model, size_t num_frms,
        StatVector& adm2, StatVector& adm_scale0, StatVector& adm_scale1,
        StatVector& adm_scale2, StatVector& adm_scale3, StatVector& motion,
        StatVector& vif_scale0, StatVector& vif_scale1, StatVector& vif_scale2,
        StatVector& vif_scale3, StatVector& vif, StatVector& motion2,
        bool enable_transform, bool disable_clip,
        std::vector<std::map<VmafPredictionReturnType, double>>& predictionMaps) {

    /* IMPORTANT: always allocate one more spot and put a -1 at the last one's
     * index, so that libsvm will stop looping when seeing the -1 !!!
     * see https://github.com/cjlin1/libsvm */
    svm_node* nodes = (svm_node*) alloca(
            sizeof(svm_node) * (model.feature_names.length() + 1));
    nodes[model.feature_names.length()].index = -1;

    size_t i_subsampled;
    for (size_t i_frm=0; i_frm<num_frms; i_frm++) {
        model.populate_and_normalize_nodes_at_frm(i_frm, nodes, adm2,
                adm_scale0, adm_scale1, adm_scale2, adm_scale3, motion,
                vif_scale0, vif_scale1, vif_scale2, vif_scale3, vif, motion2);

        std::map<VmafPredictionReturnType, double> predictionMap = model.predict(nodes);

        _postproc_predict(predictionMap);

        if (enable_transform) {
            _transform_score(model, predictionMap);
        }

        if (!disable_clip) {
            _clip_score(model, predictionMap);
        }

        _postproc_transform_clip(predictionMap);

        dbg_printf("frame: %zu, ", i);
        dbg_printf("adm2: %f, ", adm2.at(i_frm));
        dbg_printf("adm_scale0: %f, ", adm_scale0.at(i_frm));
        dbg_printf("adm_scale1: %f, ", adm_scale1.at(i_frm));
        dbg_printf("adm_scale2: %f, ", adm_scale2.at(i_frm));
        dbg_printf("adm_scale3: %f, ", adm_scale3.at(i_frm));
        dbg_printf("motion: %f, ", motion.at(i_frm));
        dbg_printf("vif_scale0: %f, ", vif_scale0.at(i_frm));
        dbg_printf("vif_scale1: %f, ", vif_scale1.at(i_frm));
        dbg_printf("vif_scale2: %f, ", vif_scale2.at(i_frm));
        dbg_printf("vif_scale3: %f, ", vif_scale3.at(i_frm));
        dbg_printf("vif: %f, ", vif.at(i_frm));
        dbg_printf("motion2: %f, ", motion2.at(i_frm));

        dbg_printf("\n");

        predictionMaps.push_back(predictionMap);

    }
}

std::unique_ptr<LibsvmNusvrTrainTestModel> VmafQualityRunner::_load_model(const char *model_path)
{
    std::unique_ptr<LibsvmNusvrTrainTestModel> model_ptr = std::unique_ptr<LibsvmNusvrTrainTestModel>(new LibsvmNusvrTrainTestModel(model_path));
    model_ptr->load_model();
    return model_ptr;
}

void VmafQualityRunner::_set_prediction_result(
        std::vector<std::map<VmafPredictionReturnType, double> > predictionMaps,
        Result& result) {
    StatVector score;
    for (size_t i = 0; i < predictionMaps.size(); i++) {
        score.append(predictionMaps.at(i)[VmafPredictionReturnType::SCORE]);
    }
    result.set_scores("vmaf", score);
}

Result VmafQualityRunner::run(Asset asset, int (*read_frame)(float *ref_data, float *main_data, float *temp_data,
                       int stride, void *user_data), void *user_data, bool disable_clip, bool enable_transform,
                       bool do_psnr, bool do_ssim, bool do_ms_ssim, int n_thread, int n_subsample)
{

    std::unique_ptr<LibsvmNusvrTrainTestModel> model_ptr = _load_model(model_path);
    LibsvmNusvrTrainTestModel& model = *model_ptr;

    dbg_printf("Initialize storage arrays...\n");
    int w = asset.getWidth();
    int h = asset.getHeight();
    const char* fmt = asset.getFmt();
    char errmsg[1024];
    DArray adm_num_array, adm_den_array, adm_num_scale0_array,
            adm_den_scale0_array, adm_num_scale1_array, adm_den_scale1_array,
            adm_num_scale2_array, adm_den_scale2_array, adm_num_scale3_array,
            adm_den_scale3_array, motion_array, motion2_array,
            vif_num_scale0_array, vif_den_scale0_array, vif_num_scale1_array,
            vif_den_scale1_array, vif_num_scale2_array, vif_den_scale2_array,
            vif_num_scale3_array, vif_den_scale3_array, vif_array, psnr_array,
            ssim_array, ms_ssim_array;
    /* use the following ptrs as flags to turn on/off optional metrics */
    DArray *psnr_array_ptr, *ssim_array_ptr, *ms_ssim_array_ptr;
    init_array(&adm_num_array, INIT_FRAMES);
    init_array(&adm_den_array, INIT_FRAMES);
    init_array(&adm_num_scale0_array, INIT_FRAMES);
    init_array(&adm_den_scale0_array, INIT_FRAMES);
    init_array(&adm_num_scale1_array, INIT_FRAMES);
    init_array(&adm_den_scale1_array, INIT_FRAMES);
    init_array(&adm_num_scale2_array, INIT_FRAMES);
    init_array(&adm_den_scale2_array, INIT_FRAMES);
    init_array(&adm_num_scale3_array, INIT_FRAMES);
    init_array(&adm_den_scale3_array, INIT_FRAMES);
    init_array(&motion_array, INIT_FRAMES);
    init_array(&motion2_array, INIT_FRAMES);
    init_array(&vif_num_scale0_array, INIT_FRAMES);
    init_array(&vif_den_scale0_array, INIT_FRAMES);
    init_array(&vif_num_scale1_array, INIT_FRAMES);
    init_array(&vif_den_scale1_array, INIT_FRAMES);
    init_array(&vif_num_scale2_array, INIT_FRAMES);
    init_array(&vif_den_scale2_array, INIT_FRAMES);
    init_array(&vif_num_scale3_array, INIT_FRAMES);
    init_array(&vif_den_scale3_array, INIT_FRAMES);
    init_array(&vif_array, INIT_FRAMES);
    init_array(&psnr_array, INIT_FRAMES);
    init_array(&ssim_array, INIT_FRAMES);
    init_array(&ms_ssim_array, INIT_FRAMES);
    /* optional output arrays */
    if (do_psnr) {
        psnr_array_ptr = &psnr_array;
    } else {
        psnr_array_ptr = NULL;
    }
    if (do_ssim) {
        ssim_array_ptr = &ssim_array;
    } else {
        ssim_array_ptr = NULL;
    }
    if (do_ms_ssim) {
        ms_ssim_array_ptr = &ms_ssim_array;
    } else {
        ms_ssim_array_ptr = NULL;
    }
    dbg_printf("Extract atom features...\n");
    int ret = combo(read_frame, user_data, w, h, fmt, &adm_num_array,
            &adm_den_array, &adm_num_scale0_array, &adm_den_scale0_array,
            &adm_num_scale1_array, &adm_den_scale1_array, &adm_num_scale2_array,
            &adm_den_scale2_array, &adm_num_scale3_array, &adm_den_scale3_array,
            &motion_array, &motion2_array, &vif_num_scale0_array,
            &vif_den_scale0_array, &vif_num_scale1_array, &vif_den_scale1_array,
            &vif_num_scale2_array, &vif_den_scale2_array, &vif_num_scale3_array,
            &vif_den_scale3_array, &vif_array, psnr_array_ptr, ssim_array_ptr,
            ms_ssim_array_ptr, errmsg, n_thread, n_subsample);
    if (ret) {
        throw VmafException(errmsg);
    }
    size_t num_frms = motion_array.used;
    bool num_frms_is_consistent = (adm_num_array.used == num_frms)
            && (adm_den_array.used == num_frms)
            && (motion2_array.used == num_frms)
            && (adm_num_scale0_array.used == num_frms)
            && (adm_den_scale0_array.used == num_frms)
            && (adm_num_scale1_array.used == num_frms)
            && (adm_den_scale1_array.used == num_frms)
            && (adm_num_scale2_array.used == num_frms)
            && (adm_den_scale2_array.used == num_frms)
            && (adm_num_scale3_array.used == num_frms)
            && (adm_den_scale3_array.used == num_frms)
            && (vif_num_scale0_array.used == num_frms)
            && (vif_den_scale0_array.used == num_frms)
            && (vif_num_scale1_array.used == num_frms)
            && (vif_den_scale1_array.used == num_frms)
            && (vif_num_scale2_array.used == num_frms)
            && (vif_den_scale2_array.used == num_frms)
            && (vif_num_scale3_array.used == num_frms)
            && (vif_den_scale3_array.used == num_frms)
            && (vif_array.used == num_frms);
    if (psnr_array_ptr != NULL) {
        num_frms_is_consistent = num_frms_is_consistent
                && (psnr_array.used == num_frms);
    }
    if (ssim_array_ptr != NULL) {
        num_frms_is_consistent = num_frms_is_consistent
                && (ssim_array.used == num_frms);
    }
    if (ms_ssim_array_ptr != NULL) {
        num_frms_is_consistent = num_frms_is_consistent
                && (ms_ssim_array.used == num_frms);
    }
    if (!num_frms_is_consistent) {
        sprintf(errmsg,
                "Output feature vectors are of inconsistent dimensions: motion (%zu), motion2 (%zu), adm_num (%zu), adm_den (%zu), vif_num_scale0 (%zu), vif_den_scale0 (%zu), vif_num_scale1 (%zu), vif_den_scale1 (%zu), vif_num_scale2 (%zu), vif_den_scale2 (%zu), vif_num_scale3 (%zu), vif_den_scale3 (%zu), vif (%zu), psnr (%zu), ssim (%zu), ms_ssim (%zu), adm_num_scale0 (%zu), adm_den_scale0 (%zu), adm_num_scale1 (%zu), adm_den_scale1 (%zu), adm_num_scale2 (%zu), adm_den_scale2 (%zu), adm_num_scale3 (%zu), adm_den_scale3 (%zu)",
                motion_array.used, motion2_array.used, adm_num_array.used,
                adm_den_array.used, vif_num_scale0_array.used,
                vif_den_scale0_array.used, vif_num_scale1_array.used,
                vif_den_scale1_array.used, vif_num_scale2_array.used,
                vif_den_scale2_array.used, vif_num_scale3_array.used,
                vif_num_scale3_array.used, vif_array.used, psnr_array.used,
                ssim_array.used, ms_ssim_array.used, adm_num_scale0_array.used,
                adm_den_scale0_array.used, adm_num_scale1_array.used,
                adm_den_scale1_array.used, adm_num_scale2_array.used,
                adm_den_scale2_array.used, adm_num_scale3_array.used,
                adm_num_scale3_array.used);
        throw VmafException(errmsg);
    }
    dbg_printf(
            "Generate final features (including derived atom features)...\n");
    double ADM2_CONSTANT = 0.0;
    double ADM_SCALE_CONSTANT = 0.0;
    StatVector adm2, motion, vif_scale0, vif_scale1, vif_scale2, vif_scale3,
            vif, motion2;
    StatVector adm_scale0, adm_scale1, adm_scale2, adm_scale3;
    StatVector psnr, ssim, ms_ssim;
    std::vector < std::map<VmafPredictionReturnType, double> > predictionMaps;
    for (size_t i = 0; i < num_frms; i += n_subsample) {
        adm2.append(
                (get_at(&adm_num_array, i) + ADM2_CONSTANT)
                        / (get_at(&adm_den_array, i) + ADM2_CONSTANT));
        adm_scale0.append(
                (get_at(&adm_num_scale0_array, i) + ADM_SCALE_CONSTANT)
                        / (get_at(&adm_den_scale0_array, i) + ADM_SCALE_CONSTANT));
        adm_scale1.append(
                (get_at(&adm_num_scale1_array, i) + ADM_SCALE_CONSTANT)
                        / (get_at(&adm_den_scale1_array, i) + ADM_SCALE_CONSTANT));
        adm_scale2.append(
                (get_at(&adm_num_scale2_array, i) + ADM_SCALE_CONSTANT)
                        / (get_at(&adm_den_scale2_array, i) + ADM_SCALE_CONSTANT));
        adm_scale3.append(
                (get_at(&adm_num_scale3_array, i) + ADM_SCALE_CONSTANT)
                        / (get_at(&adm_den_scale3_array, i) + ADM_SCALE_CONSTANT));
        motion.append(get_at(&motion_array, i));
        motion2.append(get_at(&motion2_array, i));
        vif_scale0.append(
                get_at(&vif_num_scale0_array, i)
                        / get_at(&vif_den_scale0_array, i));
        vif_scale1.append(
                get_at(&vif_num_scale1_array, i)
                        / get_at(&vif_den_scale1_array, i));
        vif_scale2.append(
                get_at(&vif_num_scale2_array, i)
                        / get_at(&vif_den_scale2_array, i));
        vif_scale3.append(
                get_at(&vif_num_scale3_array, i)
                        / get_at(&vif_den_scale3_array, i));
        vif.append(get_at(&vif_array, i));

        if (psnr_array_ptr != NULL) {
            psnr.append(get_at(&psnr_array, i));
        }
        if (ssim_array_ptr != NULL) {
            ssim.append(get_at(&ssim_array, i));
        }
        if (ms_ssim_array_ptr != NULL) {
            ms_ssim.append(get_at(&ms_ssim_array, i));
        }
    }
    dbg_printf(
            "Normalize features, SVM regression, denormalize score, clip...\n");
    size_t num_frms_subsampled = 0;
    for (size_t i = 0; i < num_frms; i += n_subsample) {
        num_frms_subsampled++;
    }
    _normalize_predict_denormalize_transform_clip(model, num_frms_subsampled,
            adm2, adm_scale0, adm_scale1, adm_scale2, adm_scale3, motion,
            vif_scale0, vif_scale1, vif_scale2, vif_scale3, vif, motion2,
            enable_transform, disable_clip, predictionMaps);
    Result result { };
    for (size_t j = 0; j < model.feature_names.length(); j++) {
        if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_adm2_score'") == 0)
            result.set_scores("adm2", adm2);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_adm_scale0_score'") == 0)
            result.set_scores("adm_scale0", adm_scale0);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_adm_scale1_score'") == 0)
            result.set_scores("adm_scale1", adm_scale1);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_adm_scale2_score'") == 0)
            result.set_scores("adm_scale2", adm_scale2);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_adm_scale3_score'") == 0)
            result.set_scores("adm_scale3", adm_scale3);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_motion_score'") == 0)
            result.set_scores("motion", motion);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_vif_scale0_score'") == 0)
            result.set_scores("vif_scale0", vif_scale0);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_vif_scale1_score'") == 0)
            result.set_scores("vif_scale1", vif_scale1);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_vif_scale2_score'") == 0)
            result.set_scores("vif_scale2", vif_scale2);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_vif_scale3_score'") == 0)
            result.set_scores("vif_scale3", vif_scale3);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_vif_score'") == 0)
            result.set_scores("vif", vif);
        else if (strcmp(Stringize(model.feature_names[j]).c_str(),
                "'VMAF_feature_motion2_score'") == 0)
            result.set_scores("motion2", motion2);
        else {
            printf("Unknown feature name: %s.\n",
                    Stringize(model.feature_names[j]).c_str());
            throw VmafException("Unknown feature name");
        }
    }

    if (psnr_array_ptr != NULL) {
        result.set_scores("psnr", psnr);
    }
    if (ssim_array_ptr != NULL) {
        result.set_scores("ssim", ssim);
    }
    if (ms_ssim_array_ptr != NULL) {
        result.set_scores("ms_ssim", ms_ssim);
    }

    _set_prediction_result(predictionMaps, result);

    free_array(&adm_num_array);
    free_array(&adm_den_array);
    free_array(&adm_num_scale0_array);
    free_array(&adm_den_scale0_array);
    free_array(&adm_num_scale1_array);
    free_array(&adm_den_scale1_array);
    free_array(&adm_num_scale2_array);
    free_array(&adm_den_scale2_array);
    free_array(&adm_num_scale3_array);
    free_array(&adm_den_scale3_array);
    free_array(&motion_array);
    free_array(&motion2_array);
    free_array(&vif_num_scale0_array);
    free_array(&vif_den_scale0_array);
    free_array(&vif_num_scale1_array);
    free_array(&vif_den_scale1_array);
    free_array(&vif_num_scale2_array);
    free_array(&vif_den_scale2_array);
    free_array(&vif_num_scale3_array);
    free_array(&vif_den_scale3_array);
    free_array(&vif_array);
    free_array(&psnr_array);
    free_array(&ssim_array);
    free_array(&ms_ssim_array);

    return result;
}

std::unique_ptr<LibsvmNusvrTrainTestModel> BootstrapVmafQualityRunner::_load_model(const char *model_path)
{
    std::unique_ptr<LibsvmNusvrTrainTestModel> model_ptr = std::unique_ptr<BootstrapLibsvmNusvrTrainTestModel>(new BootstrapLibsvmNusvrTrainTestModel(model_path));
    model_ptr->load_model();
    return model_ptr;
}

void BootstrapVmafQualityRunner::_postproc_predict(
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double baggingScore = predictionMap[VmafPredictionReturnType::BAGGING_SCORE];
    double scorePlusDelta = baggingScore + BootstrapVmafQualityRunner::DELTA;
    double scoreMinusDelta = baggingScore - BootstrapVmafQualityRunner::DELTA;

    predictionMap[VmafPredictionReturnType::PLUS_DELTA] = scorePlusDelta;
    predictionMap[VmafPredictionReturnType::MINUS_DELTA] = scoreMinusDelta;
}

void BootstrapVmafQualityRunner::_transform_score(LibsvmNusvrTrainTestModel& model,
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double& score = predictionMap[VmafPredictionReturnType::SCORE];
    double& baggingScore = predictionMap[VmafPredictionReturnType::BAGGING_SCORE];
    double& ci95LowScore = predictionMap[VmafPredictionReturnType::CI95_LOW];
    double& ci95HighScore = predictionMap[VmafPredictionReturnType::CI95_HIGH];
    double& scorePlusDelta = predictionMap[VmafPredictionReturnType::PLUS_DELTA];
    double& scoreMinusDelta = predictionMap[VmafPredictionReturnType::MINUS_DELTA];

    _transform_value(model, score);
    _transform_value(model, baggingScore);
    _transform_value(model, ci95LowScore);
    _transform_value(model, ci95HighScore);
    _transform_value(model, scorePlusDelta);
    _transform_value(model, scoreMinusDelta);
}

void BootstrapVmafQualityRunner::_clip_score(LibsvmNusvrTrainTestModel& model,
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double& score = predictionMap[VmafPredictionReturnType::SCORE];
    double& baggingScore = predictionMap[VmafPredictionReturnType::BAGGING_SCORE];
    double& ci95LowScore = predictionMap[VmafPredictionReturnType::CI95_LOW];
    double& ci95HighScore = predictionMap[VmafPredictionReturnType::CI95_HIGH];
    double& scorePlusDelta = predictionMap[VmafPredictionReturnType::PLUS_DELTA];
    double& scoreMinusDelta = predictionMap[VmafPredictionReturnType::MINUS_DELTA];

    _clip_value(model, score);
    _clip_value(model, baggingScore);
    _clip_value(model, ci95LowScore);
    _clip_value(model, ci95HighScore);
    _clip_value(model, scorePlusDelta);
    _clip_value(model, scoreMinusDelta);
}

void BootstrapVmafQualityRunner::_postproc_transform_clip(
        std::map<VmafPredictionReturnType, double>& predictionMap) {

    double scorePlusDelta = predictionMap[VmafPredictionReturnType::PLUS_DELTA];
    double scoreMinusDelta = predictionMap[VmafPredictionReturnType::MINUS_DELTA];
    double& scoreStdDev = predictionMap[VmafPredictionReturnType::STDDEV];

    double slope = (scorePlusDelta - scoreMinusDelta) / (2.0 * BootstrapVmafQualityRunner::DELTA);
    scoreStdDev *= slope;
}

void BootstrapVmafQualityRunner::_set_prediction_result(
        std::vector<std::map<VmafPredictionReturnType, double> > predictionMaps,
        Result& result) {

    VmafQualityRunner::_set_prediction_result(predictionMaps, result);

    StatVector baggingScore, stdDev, ci95LowScore, ci95HighScore;
    for (size_t i = 0; i < predictionMaps.size(); i++) {
        baggingScore.append(predictionMaps.at(i)[VmafPredictionReturnType::BAGGING_SCORE]);
        stdDev.append(predictionMaps.at(i)[VmafPredictionReturnType::STDDEV]);
        ci95LowScore.append(predictionMaps.at(i)[VmafPredictionReturnType::CI95_LOW]);
        ci95HighScore.append(predictionMaps.at(i)[VmafPredictionReturnType::CI95_HIGH]);
    }
    result.set_scores("bagging", baggingScore);
    result.set_scores("stddev", stdDev);
    result.set_scores("ci95_low", ci95LowScore);
    result.set_scores("ci95_high", ci95HighScore);
}

static const char VMAFOSS_DOC_VERSION[] = "1.3.7";

double RunVmaf(const char* fmt, int width, int height,
               int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data),
               void *user_data, const char *model_path, const char *log_path, const char *log_fmt,
               bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim,
               const char *pool_method, int n_thread, int n_subsample, bool enable_conf_interval)
{
    printf("Start calculating VMAF score...\n");

    if (width <= 0)
    {
        throw VmafException("Invalid width value (must be > 0)");
    }
    if (height <= 0)
    {
        throw VmafException("Invalid height value (must be > 0)");
    }
    if (n_thread < 0)
    {
        throw VmafException("Invalid n_thread value (must be >= 0)");
    }
    if (n_subsample <= 0)
    {
        throw VmafException("Invalid n_subsample value (must be > 0)");
    }

    Asset asset(width, height, fmt);
    std::unique_ptr<VmafQualityRunner> runner_ptr;
    if (enable_conf_interval)
    {
        runner_ptr = std::unique_ptr<BootstrapVmafQualityRunner>(new BootstrapVmafQualityRunner(model_path));
    }
    else
    {
        runner_ptr = std::unique_ptr<VmafQualityRunner>(new VmafQualityRunner(model_path));
    }

    Timer timer;
    timer.start();
    Result result = runner_ptr->run(asset, read_frame, user_data, disable_clip, enable_transform,
                               do_psnr, do_ssim, do_ms_ssim, n_thread, n_subsample);
    timer.stop();

    if (pool_method != NULL && (strcmp(pool_method, "min")==0))
    {
        result.setScoreAggregateMethod(ScoreAggregateMethod::MIN);
    }
    else if (pool_method != NULL && (strcmp(pool_method, "harmonic_mean")==0))
    {
        result.setScoreAggregateMethod(ScoreAggregateMethod::HARMONIC_MEAN);
    }
    else // mean or default
    {
        result.setScoreAggregateMethod(ScoreAggregateMethod::MEAN);
    }

    size_t num_frames_subsampled = result.get_scores("vmaf").size();
    double aggregate_vmaf = result.get_score("vmaf");
    double exec_fps = (double)num_frames_subsampled * n_subsample / (double)timer.elapsed();
    printf("Exec FPS: %f\n", exec_fps);

    if (pool_method)
    {
        printf("VMAF score (%s) = %f\n", pool_method, aggregate_vmaf);
    }
    else // default
    {
        printf("VMAF score = %f\n", aggregate_vmaf);
    }

    if (log_path != NULL && log_fmt !=NULL && (strcmp(log_fmt, "json")==0))
    {
        /* output to json */

        std::vector<std::string> result_keys = result.get_keys();
        double value;

        OTab params;
        params["model"] = _get_file_name(std::string(model_path));
        params["scaledWidth"] = width;
        params["scaledHeight"] = height;
        params["subsample"] = n_subsample;

        Arr metrics;
        for (size_t j=0; j<result_keys.size(); j++)
        {
            metrics.append(result_keys[j]);
        }

        Arr frames;
        for (size_t i_subsampled=0; i_subsampled<num_frames_subsampled; i_subsampled++)
        {
            OTab frame;
            frame["frameNum"] = i_subsampled * n_subsample;
            OTab metrics_scores;
            for (size_t j=0; j<result_keys.size(); j++)
            {
                value = result.get_scores(result_keys[j].c_str()).at(i_subsampled);
                value = _round_to_digit(value, 5);
                metrics_scores[result_keys[j].c_str()] = value;
            }
            frame["metrics"] = metrics_scores;
            frames.append(frame);
        }

        Val top = OTab();
        top["version"] = VMAFOSS_DOC_VERSION;
        top["params"] = params;
        top["metrics"] = metrics;
        top["frames"] = frames;

        std::ofstream log_file(log_path);
        JSONPrint(top, log_file, 0, true, 2);
        log_file.close();
    }
    else if (log_path != NULL)
    {
        /* output to xml */

        std::vector<std::string> result_keys = result.get_keys();
        pugi::xml_document xml;
        pugi::xml_node xml_root = xml.append_child("VMAF");
        xml_root.append_attribute("version") = VMAFOSS_DOC_VERSION;

        auto params_node = xml_root.append_child("params");
        params_node.append_attribute("model") = _get_file_name(std::string(model_path)).c_str();
        params_node.append_attribute("scaledWidth") = width;
        params_node.append_attribute("scaledHeight") = height;
        params_node.append_attribute("subsample") = n_subsample;

        auto info_node = xml_root.append_child("fyi");
        info_node.append_attribute("numOfFrames") = (int)num_frames_subsampled;
        info_node.append_attribute("aggregateVMAF") = aggregate_vmaf;
        info_node.append_attribute("execFps") = exec_fps;

        auto frames_node = xml_root.append_child("frames");
        for (size_t i_subsampled=0; i_subsampled<num_frames_subsampled; i_subsampled++)
        {
            auto node = frames_node.append_child("frame");
            node.append_attribute("frameNum") = (int)i_subsampled * n_subsample;
            for (size_t j=0; j<result_keys.size(); j++)
            {
                node.append_attribute(result_keys[j].c_str()) = result.get_scores(result_keys[j].c_str()).at(i_subsampled);
            }
        }

        xml.save_file(log_path);
    }

    return aggregate_vmaf;
}

inline double _round(double val)
{
    if( val < 0 ) return ceil(val - 0.5);
    return floor(val + 0.5);
}

inline double _round_to_digit(double val, int digit)
{
    size_t m = pow(10.0, digit);
    return _round(val * m) / m;
}

string _get_file_name(const std::string& s)
{
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      return(s.substr(i+1, s.length() - i));
   }
   return("");
}

