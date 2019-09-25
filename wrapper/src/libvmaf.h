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

#ifndef LIBVMAF_H_
#define LIBVMAF_H_

#ifndef WINCE
#define TIME_TEST_ENABLE 		1 // 1: memory leak test enable 0: disable
#define MEM_LEAK_TEST_ENABLE 	0 // prints execution time in xml log when enabled.
#else
//For Windows memory leak test and execution time test cases are not handled.
#define TIME_TEST_ENABLE 0
#define MEM_LEAK_TEST_ENABLE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data),
				 void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip, int disable_avx, int enable_transform, int phone_model, int do_psnr,
				 int do_ssim, int do_ms_ssim, char *pool_method, int n_thread, int n_subsample, int enable_conf_interval);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <vector>
#include <cstring>
#include <map>
#include <memory>
#include <string>

class Asset
{
public:
    Asset(int w, int h, const char *fmt);
    Asset(int w, int h);
    int getWidth();
    int getHeight();
    const char* getFmt();
private:
    const int w, h;
    const char *fmt;
};

enum ScoreAggregateMethod
{
    MEAN,
    HARMONIC_MEAN,
    MINIMUM
};

class StatVector
{
public:
    StatVector();
    StatVector(std::vector<double> l);
    std::vector<double> getVector();
    double mean();
    double minimum();
    double harmonic_mean();
    double second_moment();
    double percentile(double perc);
    double var();
    double std();
    void append(double e);
    double at(size_t idx);
    size_t size();
private:
    std::vector<double> l;
    void _assert_size();
};


class Result
{
public:
    Result();
    void set_scores(const std::string &key, const StatVector &scores);
    StatVector get_scores(const std::string &key);
    bool has_scores(const std::string &key);
    double get_score(const std::string &key);
    std::vector<std::string> get_keys();
    void setScoreAggregateMethod(ScoreAggregateMethod scoreAggregateMethod);
private:
    std::map<std::string, StatVector> d;
    ScoreAggregateMethod score_aggregate_method;
};

class IVmafQualityRunner {
public:
    virtual Result run(Asset asset, int(*read_frame)(float *ref_data, float *main_data, float *temp_data,
        int stride, void *user_data), void *user_data, bool disable_clip, bool enable_transform,
        bool do_psnr, bool do_ssim, bool do_ms_ssim, int n_thread, int n_subsample) = 0;
    virtual ~IVmafQualityRunner() {}
};

class VmafQualityRunnerFactory {
public:
    static std::unique_ptr<IVmafQualityRunner> 
        createVmafQualityRunner(const char *model_path, bool enable_conf_interval);
};

class VmafException: public std::exception
{
public:
    explicit VmafException(const char *msg): msg(msg) {}
    virtual const char* what() const throw () { return msg.c_str(); }
private:
    std::string msg;
};

#endif

#endif /* _LIBVMAF_H */
