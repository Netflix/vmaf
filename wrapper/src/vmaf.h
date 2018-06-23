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

double RunVmaf(const char* fmt, int width, int height,
               int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data),
               void *user_data, const char *model_path, const char *log_path, const char *log_fmt,
               bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim,
               const char *pool_method, int n_thread, int n_subsample, bool enable_conf_interval);

class Asset
{
public:
    Asset(int w, int h, const char *fmt):
        w(w), h(h), fmt(fmt) {}
    Asset(int w, int h):
        w(w), h(h), fmt("yuv420p") {}
    int getWidth() { return w; }
    int getHeight() { return h; }
    const char* getFmt() { return fmt; }
private:
    const int w, h;
    const char *fmt;
};

class StatVector
{
public:
    StatVector() {}
    StatVector(std::vector<double> l): l(l) {}
    double mean()
    {
        _assert_size();
        double sum = 0.0;
        for (double e : l)
        {
            sum += e;
        }
        return sum / l.size();
    }
    double min()
    {
        _assert_size();
        double min_ = l[0];
        for (double e : l)
        {
            if (e < min_)
            {
                min_ = e;
            }
        }
        return min_;
    }
    double harmonic_mean()
    {
        _assert_size();
        double sum = 0.0;
        for (double e: l)
        {
            sum += 1.0 / (e + 1.0);
        }
        return 1.0 / (sum / l.size()) - 1.0;
    }
    double second_moment()
    {
        _assert_size();
        double sum = 0.0;
        for (double e : l)
        {
            sum += pow(e, 2);
        }
        return sum / l.size();
    }
    double percentile(double perc)
    {
        _assert_size();
        if (perc < 0.0) {
            perc = 0.0;
        }
        else if (perc > 100.0) {
            perc = 100.0;
        }
        std::vector<double> l(this->l);
        std::sort(l.begin(), l.end());
        double pos = perc * (this->l.size() - 1) / 100.0;
        int pos_left = (int)floor(pos);
        int pos_right = (int)ceil(pos);
        if (pos_left == pos_right) {
            return l[pos_left];
        }
        else {
            return l[pos_left] * (pos_right - pos) + l[pos_right] * (pos - pos_left);
        }

    }
    double var() { return second_moment() - pow(mean(), 2); }
    double std() { return sqrt(var()); }
    void append(double e) { l.push_back(e); }
    double at(size_t idx) { return l.at(idx); }
    size_t size() { return l.size(); }
private:
    std::vector<double> l;
    void _assert_size() {
        if (l.size() == 0) {
            throw std::runtime_error("StatVector size is 0.");
        }
    }
};

enum ScoreAggregateMethod
{
    MEAN,
    HARMONIC_MEAN,
    MIN
};

class Result
{
public:
    Result(): score_aggregate_method(ScoreAggregateMethod::MEAN) {}
    void set_scores(const std::string &key, const StatVector &scores) { d[key] = scores; }
    StatVector get_scores(const std::string &key) { return d[key]; }
    bool has_scores(const std::string &key) { return d.find(key) != d.end(); }
    double get_score(const std::string &key)
    {
        StatVector list = get_scores(key);
        if (score_aggregate_method == ScoreAggregateMethod::MIN)
        {
            return list.min();
        }
        else if (score_aggregate_method == ScoreAggregateMethod::HARMONIC_MEAN)
        {
            return list.harmonic_mean();
        }
        else // MEAN
        {
            return list.mean();
        }
    }
    std::vector<std::string> get_keys()
    {
        std::vector<std::string> v;
        for (std::map<std::string, StatVector>::iterator it = d.begin(); it != d.end(); ++it)
        {
            v.push_back(it->first);
        }
        return v;
    }
    void setScoreAggregateMethod(ScoreAggregateMethod scoreAggregateMethod)
    {
        score_aggregate_method = scoreAggregateMethod;
    }
private:
    std::map<std::string, StatVector> d;
    ScoreAggregateMethod score_aggregate_method;
};

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

class LibsvmNusvrTrainTestModel
{
public:
    LibsvmNusvrTrainTestModel(const char *model_path): model_path(model_path) {}
    Val feature_names, norm_type, slopes, intercepts, score_clip, score_transform;
    virtual void load_model();
    virtual std::map<VmafPredictionReturnType, double> predict(svm_node* nodes);
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
    virtual std::map<VmafPredictionReturnType, double> predict(svm_node* nodes);
    virtual ~BootstrapLibsvmNusvrTrainTestModel() {}
private:
    std::vector<std::unique_ptr<svm_model, SvmDelete>> bootstrap_svm_model_ptrs;
    std::string _get_model_i_filename(const char* model_path, int i_model);
    void _read_and_assert_model(const char *model_path, Val& feature_names, Val& norm_type, Val& slopes,
            Val& intercepts, Val& score_clip, Val& score_transform, int& numModels);
    virtual void _assert_model_type(Val model_type);
};

class VmafQualityRunner
{
public:
    VmafQualityRunner(const char *model_path): model_path(model_path) {}
    Result run(Asset asset, int (*read_frame)(float *ref_data, float *main_data, float *temp_data,
               int stride, void *user_data), void *user_data, bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim, int n_thread, int n_subsample);
    virtual ~VmafQualityRunner() {}
protected:
    static void _transform_value(LibsvmNusvrTrainTestModel& model, double& prediction);
    static void _clip_value(LibsvmNusvrTrainTestModel& model, double& prediction);
    virtual void _set_prediction_result(
            std::vector<std::map<VmafPredictionReturnType, double> > predictionMaps,
            Result& result);
private:
    const char *model_path;
    static const int INIT_FRAMES = 1000;
    virtual std::unique_ptr<LibsvmNusvrTrainTestModel> _load_model(const char *model_path);
    virtual void _postproc_predict(std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _transform_score(LibsvmNusvrTrainTestModel& model, std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _clip_score(LibsvmNusvrTrainTestModel& model, std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _postproc_transform_clip(std::map<VmafPredictionReturnType, double>& predictionMap);
    void _normalize_predict_denormalize_transform_clip(LibsvmNusvrTrainTestModel& model,
            size_t num_frms, StatVector& adm2,
            StatVector& adm_scale0, StatVector& adm_scale1,
            StatVector& adm_scale2, StatVector& adm_scale3, StatVector& motion,
            StatVector& vif_scale0, StatVector& vif_scale1,
            StatVector& vif_scale2, StatVector& vif_scale3, StatVector& vif,
            StatVector& motion2, bool enable_transform, bool disable_clip,
            std::vector<std::map<VmafPredictionReturnType, double>>& predictionMaps);
};

class BootstrapVmafQualityRunner: public VmafQualityRunner
{
public:
    BootstrapVmafQualityRunner(const char *model_path): VmafQualityRunner(model_path) {}
    virtual ~BootstrapVmafQualityRunner() {}
private:
    static constexpr double DELTA = 0.01;
    virtual std::unique_ptr<LibsvmNusvrTrainTestModel> _load_model(const char *model_path);
    virtual void _postproc_predict(std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _transform_score(LibsvmNusvrTrainTestModel& model, std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _clip_score(LibsvmNusvrTrainTestModel& model, std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _postproc_transform_clip(std::map<VmafPredictionReturnType, double>& predictionMap);
    virtual void _set_prediction_result(
            std::vector<std::map<VmafPredictionReturnType, double> > predictionMaps,
            Result& result);
};

#endif /* VMAF_H_ */
