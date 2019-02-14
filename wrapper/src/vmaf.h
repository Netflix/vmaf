/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
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

#pragma once

#ifndef VMAF_H_
#define VMAF_H_

#include <vector>
#include <map>
#include <numeric>
#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vector>

#include "svm.h"
#include "chooseser.h"
#include "darray.h"
#include "libvmaf.h"

static const std::string BOOSTRAP_VMAF_MODEL_PREFIX = "vmaf_";

double RunVmaf(const char* fmt, int width, int height,
               int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data),
               void *user_data, const char *model_path, const char *log_path, const char *log_fmt,
               bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim,
               const char *pool_method, int n_thread, int n_subsample, bool enable_conf_interval);

class VmafException: public std::exception
{
public:
    explicit VmafException(const char *msg): msg(msg) {}
    virtual const char* what() const throw () { return msg.c_str(); }
private:
    std::string msg;
};

struct SvmDelete {
    void operator()(void *svm);
};

enum VmafPredictionReturnType
{
    SCORE,
    BAGGING_SCORE,
    STDDEV,
    CI95_LOW,
    CI95_HIGH,
    PLUS_DELTA,
    MINUS_DELTA
};

struct VmafPredictionStruct
{
    std::map<VmafPredictionReturnType, double> vmafPrediction;
    std::vector<double> vmafMultiModelPrediction;
};

class LibsvmNusvrTrainTestModel
{
public:
    LibsvmNusvrTrainTestModel(const char *model_path): model_path(model_path) {}
    Val feature_names, norm_type, slopes, intercepts, score_clip, score_transform;
    virtual void load_model();
    virtual VmafPredictionStruct predict(svm_node* nodes);
    void populate_and_normalize_nodes_at_frm(size_t i_frm,
            svm_node*& nodes, StatVector& adm2,
            StatVector& adm_scale0, StatVector& adm_scale1,
            StatVector& adm_scale2, StatVector& adm_scale3, StatVector& motion,
            StatVector& vif_scale0, StatVector& vif_scale1,
            StatVector& vif_scale2, StatVector& vif_scale3, StatVector& vif,
            StatVector& motion2);
    virtual ~LibsvmNusvrTrainTestModel() {}
protected:
    const char *model_path;
    std::unique_ptr<svm_model, SvmDelete> svm_model_ptr;
    void _read_and_assert_model(const char *model_path, Val& feature_names, Val& norm_type, Val& slopes,
            Val& intercepts, Val& score_clip, Val& score_transform);
    std::unique_ptr<svm_model, SvmDelete> _read_and_assert_svm_model(const char* libsvm_model_path);
    void _denormalize_prediction(double& prediction);

private:
    virtual void _assert_model_type(Val model_type);
};

class BootstrapLibsvmNusvrTrainTestModel: public LibsvmNusvrTrainTestModel {
public:
    BootstrapLibsvmNusvrTrainTestModel(const char *model_path): LibsvmNusvrTrainTestModel(model_path) {}
    virtual void load_model();
    virtual VmafPredictionStruct predict(svm_node* nodes);
    virtual ~BootstrapLibsvmNusvrTrainTestModel() {}
private:
    std::vector<std::unique_ptr<svm_model, SvmDelete>> bootstrap_svm_model_ptrs;
    std::string _get_model_i_filename(const char* model_path, int i_model);
    void _read_and_assert_model(const char *model_path, Val& feature_names, Val& norm_type, Val& slopes,
            Val& intercepts, Val& score_clip, Val& score_transform, int& numModels);
    virtual void _assert_model_type(Val model_type);
};

class VmafQualityRunner : public IVmafQualityRunner
{
public:
    VmafQualityRunner(const char *model_path): model_path(model_path) {}
    virtual Result run(Asset asset, int (*read_frame)(float *ref_data, float *main_data, float *temp_data,
               int stride, void *user_data), void *user_data, bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim, int n_thread, int n_subsample);
    virtual ~VmafQualityRunner() {}
protected:
    static void _transform_value(LibsvmNusvrTrainTestModel& model, double& prediction);
    static void _clip_value(LibsvmNusvrTrainTestModel& model, double& prediction);
    virtual void _set_prediction_result(
            std::vector<VmafPredictionStruct> predictionStructs,
            Result& result);
private:
    const char *model_path;
    static const int INIT_FRAMES = 1000;
    virtual std::unique_ptr<LibsvmNusvrTrainTestModel> _load_model(const char *model_path);
    virtual void _postproc_predict(VmafPredictionStruct& predictionStruct);
    virtual void _transform_score(LibsvmNusvrTrainTestModel& model, VmafPredictionStruct& predictionStruct);
    virtual void _clip_score(LibsvmNusvrTrainTestModel& model, VmafPredictionStruct& predictionStruct);
    virtual void _postproc_transform_clip(VmafPredictionStruct& predictionStruct);
    void _normalize_predict_denormalize_transform_clip(LibsvmNusvrTrainTestModel& model,
            size_t num_frms, StatVector& adm2,
            StatVector& adm_scale0, StatVector& adm_scale1,
            StatVector& adm_scale2, StatVector& adm_scale3, StatVector& motion,
            StatVector& vif_scale0, StatVector& vif_scale1,
            StatVector& vif_scale2, StatVector& vif_scale3, StatVector& vif,
            StatVector& motion2, bool enable_transform, bool disable_clip,
            std::vector<VmafPredictionStruct>& predictionStructs);
};

class BootstrapVmafQualityRunner: public VmafQualityRunner
{
public:
    BootstrapVmafQualityRunner(const char *model_path): VmafQualityRunner(model_path) {}
    virtual ~BootstrapVmafQualityRunner() {}
private:
    static constexpr double DELTA = 0.01;
    virtual std::unique_ptr<LibsvmNusvrTrainTestModel> _load_model(const char *model_path);
    virtual void _postproc_predict(VmafPredictionStruct& predictionStruct);
    virtual void _transform_score(LibsvmNusvrTrainTestModel& model, VmafPredictionStruct& predictionStruct);
    virtual void _clip_score(LibsvmNusvrTrainTestModel& model, VmafPredictionStruct& predictionStruct);
    virtual void _postproc_transform_clip(VmafPredictionStruct& predictionStruct);
    virtual void _set_prediction_result(
            std::vector<VmafPredictionStruct> predictionStructs,
            Result& result);
};

#endif /* VMAF_H_ */
