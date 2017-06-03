/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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
#include <cstring>

double RunVmaf(const char* fmt, int width, int height,
               const char *ref_path, const char *dis_path, const char *model_path,
               const char *log_path, const char *log_fmt,
               bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim,
               const char *pool_method);

class Asset
{
public:
    Asset(int w, int h, const char *ref_path, const char *dis_path, const char *fmt):
        w(w), h(h), ref_path(ref_path), dis_path(dis_path), fmt(fmt) {}
    Asset(int w, int h, const char *ref_path, const char *dis_path):
        w(w), h(h), ref_path(ref_path), dis_path(dis_path), fmt("yuv420p") {}
    int getWidth() { return w; }
    int getHeight() { return h; }
    const char* getRefPath() { return ref_path; }
    const char* getDisPath() { return dis_path; }
    const char* getFmt() { return fmt; }
private:
    const int w, h;
    const char *ref_path, *dis_path, *fmt;
};

class StatVector
{
public:
    StatVector() {}
    StatVector(std::vector<double> l): l(l) {}
    double mean()
    {
        double sum = 0.0;
        for (double e : l)
        {
            sum += e;
        }
        return sum / l.size();
    }
    double min()
    {
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
        double sum = 0.0;
        for (double e: l)
        {
            sum += 1.0 / (e + 1.0);
        }
        return 1.0 / (sum / l.size()) - 1.0;
    }
    void append(double e) { l.push_back(e); }
    double at(size_t idx) { return l.at(idx); }
    size_t size() { return l.size(); }
private:
    std::vector<double> l;
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
    Result(): score_aggregate_method(MEAN) {}
    void set_scores(const std::string &key, const StatVector &scores) { d[key] = scores; }
    StatVector get_scores(const std::string &key) { return d[key]; }
    bool has_scores(const std::string &key) { return d.find(key) != d.end(); }
    double get_score(const std::string &key)
    {
        StatVector list = get_scores(key);
        if (score_aggregate_method == MIN)
        {
            return list.min();
        }
        else if (score_aggregate_method == HARMONIC_MEAN)
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

class VmafRunner
{
public:
    VmafRunner(const char *model_path): model_path(model_path)
    {
        /* follow the convention that if model_path is a/b.c, the
         * libsvm_model_path is always a/b.c.model */
        libsvm_model_path = new char[strlen(model_path) + 10];
        sprintf(libsvm_model_path, "%s.model", model_path);
    }
    ~VmafRunner() { delete[] libsvm_model_path; }
    Result run(Asset asset, bool disable_clip, bool enable_transform, bool do_psnr, bool do_ssim, bool do_ms_ssim);
private:
    const char *model_path;
    char *libsvm_model_path;
    static const int INIT_FRAMES = 1000;
};

#endif /* VMAF_H_ */
