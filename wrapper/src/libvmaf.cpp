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

#include "libvmaf.h"
#include "vmaf.h"
#include <cstdio>
#include "cpu.h"
#include "jsonprint.h"
#include "jsonreader.h"

StatVector::StatVector() 
{
}

StatVector::StatVector(std::vector<double> l) : l(l) 
{
}

std::vector<double> StatVector::getVector()
{
    return l;
}

double StatVector::mean()
{
    _assert_size();
    double sum = 0.0;
    for (double e : l)
    {
        sum += e;
    }
    return sum / l.size();
}

double StatVector::minimum()
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

double StatVector::harmonic_mean()
{
    _assert_size();
    double sum = 0.0;
    for (double e : l)
    {
        sum += 1.0 / (e + 1.0);
    }
    return 1.0 / (sum / l.size()) - 1.0;
}

double StatVector::second_moment()
{
    _assert_size();
    double sum = 0.0;
    for (double e : l)
    {
        sum += pow(e, 2);
    }
    return sum / l.size();
}

double StatVector::percentile(double perc)
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

double StatVector::var()
{ 
    return second_moment() - pow(mean(), 2); 
}

double StatVector::std()
{ 
    return sqrt(var()); 
}

void StatVector::append(double e)
{ 
    l.push_back(e); 
}

double StatVector::at(size_t idx)
{ 
    return l.at(idx); 
}

size_t StatVector::size()
{ 
    return l.size(); 
}

void StatVector::_assert_size()
{
    if (l.size() == 0) {
        throw std::runtime_error("StatVector size is 0.");
    }
}

Result::Result() : score_aggregate_method(ScoreAggregateMethod::MEAN)
{
}

void Result::set_scores(const std::string &key, const StatVector &scores)
{
    d[key] = scores; 
}

StatVector Result::get_scores(const std::string &key)
{ 
    return d[key]; 
}

bool Result::has_scores(const std::string &key)
{
    return d.find(key) != d.end(); 
}

double Result::get_score(const std::string &key)
{
    StatVector list = get_scores(key);
    if (score_aggregate_method == ScoreAggregateMethod::MINIMUM)
    {
        return list.minimum();
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

std::vector<std::string> Result::get_keys()
{
    std::vector<std::string> v;
    for (std::map<std::string, StatVector>::iterator it = d.begin(); it != d.end(); ++it)
    {
        v.push_back(it->first);
    }
    return v;
}

unsigned int Result::get_num_frms()
{
    return num_frms;
}

void Result::set_num_frms(unsigned int num_frms)
{
    this->num_frms = num_frms;
}

void Result::setScoreAggregateMethod(ScoreAggregateMethod scoreAggregateMethod)
{
    score_aggregate_method = scoreAggregateMethod;
}

std::unique_ptr<IVmafQualityRunner> 
VmafQualityRunnerFactory::createVmafQualityRunner(VmafModel *vmaf_model_ptr) {
    std::unique_ptr<IVmafQualityRunner> runner_ptr;
    if (vmaf_model_ptr->vmaf_model_setting & VMAF_MODEL_SETTING_ENABLE_CONF_INTERVAL)
    {
        runner_ptr = std::unique_ptr<BootstrapVmafQualityRunner>(new BootstrapVmafQualityRunner(vmaf_model_ptr->path));
    }
    else
    {
        runner_ptr = std::unique_ptr<VmafQualityRunner>(new VmafQualityRunner(vmaf_model_ptr->path));
    }
    return runner_ptr;
}

extern "C" {

    enum vmaf_cpu cpu; // global

    int compute_vmaf(double* vmaf_score,
        int(*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data),
        int(*read_vmaf_picture)(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *user_data),
        void *user_data, VmafSettings *vmafSettings)
    {

        cpu = cpu_autodetect();

        if (vmafSettings->vmaf_feature_calculation_setting.disable_avx)
        {
            cpu = VMAF_CPU_NONE;
        }

        try {
            double score = RunVmaf(read_frame, read_vmaf_picture, user_data, vmafSettings);
            *vmaf_score = score;
            return 0;
        }
        catch (VmafException& e)
        {
            printf("Caught VmafException: %s\n", e.what());
            return -2;
        }
        catch (std::runtime_error& e)
        {
            printf("Caught runtime_error: %s\n", e.what());
            return -3;
        }
        catch (std::logic_error& e)
        {
            printf("Caught logic_error: %s\n", e.what());
            return -4;
        }
    }

    void replace_string_in_place(std::string& subject, const std::string& search,
                      const std::string& replace) {
size_t pos = 0;
while ((pos = subject.find(search, pos)) != std::string::npos) {
     subject.replace(pos, search.length(), replace);
     pos += replace.length();
}
}

    unsigned int get_additional_models(char *additional_model_paths, VmafModel *vmaf_model)
    {
        // read additional models, if any
        if (additional_model_paths != NULL) {

            std::string unknown_option_exception;
            std::string model_key, model_values;
            bool use_option;

            istringstream is(additional_model_paths);
            Val additional_model_path_val;
            Val inner_additional_model_path_val;

            ReadValFromJSONStream(is, additional_model_path_val);

            unsigned int additional_model_ind = 0;

            Tab tt = MakeTab(Tab(GetString(additional_model_path_val)));
            It kv_pair(tt);

            while (kv_pair()) {

                if (additional_model_ind + 1 > MAX_NUM_VMAF_MODELS)
                {
                    fprintf(stderr, "Error: at least %d models were passed in, but a maximum of %d are allowed.\n",
                        additional_model_ind + 1, MAX_NUM_VMAF_MODELS);
                    return -1;
                }

                // each model corresponds to a key-value pair
                // the value corresponds to a dictionary as well that we parse

                std::string name = GetString(kv_pair.key());

                vmaf_model[additional_model_ind + 1].name = (char*)malloc(name.length() + 1);

                if (!vmaf_model[additional_model_ind + 1].name)
                {
                    fprintf(stderr, "Malloc for additional model %d failed.\n", additional_model_ind);
                    return -1;
                }

                strcpy(vmaf_model[additional_model_ind + 1].name, name.c_str());

                vmaf_model[additional_model_ind + 1].vmaf_model_setting = VMAF_MODEL_SETTING_NONE;

                std::string path = "";

                model_values = GetString(kv_pair.value());

                // replace single quotes with double quotes and extra spaces added by parser
                replace_string_in_place(model_values, "'", "\"");
                replace_string_in_place(model_values, " ", "");

                istringstream inner_is(model_values.c_str());
                ReadValFromJSONStream(inner_is, inner_additional_model_path_val);

                Tab inner_tt = MakeTab(Tab(GetString(inner_additional_model_path_val)));
                It inner_kv_pair(inner_tt);

                while (inner_kv_pair()) {

                    use_option = !strcmp(GetString(inner_kv_pair.value()).c_str(), "1");
                    if (strcmp(GetString(inner_kv_pair.key()).c_str(), "model_path") == 0) {
                        path = GetString(inner_kv_pair.value());
                    }
                    else if ((strcmp(GetString(inner_kv_pair.key()).c_str(), "enable_transform") == 0) && use_option) {
                        vmaf_model[additional_model_ind + 1].vmaf_model_setting |= VMAF_MODEL_SETTING_ENABLE_TRANSFORM;
                    }
                    else if ((strcmp(GetString(inner_kv_pair.key()).c_str(), "enable_conf_interval") == 0) && use_option) {
                        vmaf_model[additional_model_ind + 1].vmaf_model_setting |= VMAF_MODEL_SETTING_ENABLE_CONF_INTERVAL;
                    }
                    else if ((strcmp(GetString(inner_kv_pair.key()).c_str(), "disable_clip") == 0) && use_option) {
                        vmaf_model[additional_model_ind + 1].vmaf_model_setting |= VMAF_MODEL_SETTING_DISABLE_CLIP;
                    }
                    else {
                        if ((strcmp(GetString(inner_kv_pair.key()).c_str(), "enable_transform") == 1) &&
                            (strcmp(GetString(inner_kv_pair.key()).c_str(), "enable_conf_interval") == 1) &&
                            (strcmp(GetString(inner_kv_pair.key()).c_str(), "disable_clip") == 1)) {
                            unknown_option_exception = "Additional model option " + GetString(inner_kv_pair.key()) + " is unknown.";
                            fprintf(stderr, "Error: %s.\n", unknown_option_exception.c_str());
                            return -1;
                        }
                    }

                }

                vmaf_model[additional_model_ind + 1].path = (char*)malloc(path.length() + 1);

                if (!vmaf_model[additional_model_ind + 1].path)
                {
                    fprintf(stderr, "Malloc for additional model path %d failed.\n", additional_model_ind);
                    return -1;
                }

                strcpy(vmaf_model[additional_model_ind + 1].path, path.c_str());
                additional_model_ind += 1;

            }

            return additional_model_ind;

        }
        else
        {
            return 0;
        }

    }

}
