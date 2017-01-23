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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>

#include "vmaf.h"
#include "darray.h"
#include "combo.h"
#include "svm.h"
#include "pugixml/pugixml.hpp"
#include "timer.h"
#include "chooseser.h"
#include "jsonprint.h"

#define VAL_EQUAL_STR(V,S) (Stringize((V)).compare((S))==0)
#define VAL_IS_LIST(V) ((V).tag=='n') /* check ocval.cc */
#define VAL_IS_NONE(V) ((V).tag=='Z') /* check ocval.cc */
#define VAL_IS_DICT(V) ((V).tag=='t') /* check ocval.cc */

inline double _round_to_digit(double val, int digit);
string _get_file_name(const std::string& s);

void SvmDelete::operator()(void *svm)
{
    svm_free_and_destroy_model((svm_model **)&svm);
}

void _read_and_assert_model(const char *model_path, Val& feature_names,
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

    if (!VAL_EQUAL_STR(model_type, "'LIBSVMNUSVR'"))
    {
        printf("Current vmafossexec only accepts model type LIBSVMNUSVR, "
                "but got %s\n", Stringize(model_type).c_str());
        throw VmafException("Incompatible model_type");
    }

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

Result VmafRunner::run(Asset asset, bool disable_clip, bool enable_transform,
                       bool do_psnr, bool do_ssim, bool do_ms_ssim)
{

#ifdef PRINT_PROGRESS
    printf("Read input model (pkl)...\n");
#endif

    Val feature_names, norm_type, slopes, intercepts, score_clip, score_transform;
    _read_and_assert_model(model_path, feature_names, norm_type, slopes, intercepts, score_clip, score_transform);

#ifdef PRINT_PROGRESS
    printf("Read input model (libsvm)...\n");
#endif

    std::unique_ptr<svm_model, SvmDelete> svm_model_ptr{svm_load_model(libsvm_model_path)};
    if (!svm_model_ptr)
    {
        printf("Error loading SVM model.\n");
        throw VmafException("Error loading SVM model");
    }

#ifdef PRINT_PROGRESS
    printf("Initialize storage arrays...\n");
#endif

    int w = asset.getWidth();
    int h = asset.getHeight();
    const char* ref_path = asset.getRefPath();
    const char* dis_path = asset.getDisPath();
    const char* fmt = asset.getFmt();
    char errmsg[1024];

    DArray adm_num_array,
           adm_den_array,
           adm_num_scale0_array,
           adm_den_scale0_array,
           adm_num_scale1_array,
           adm_den_scale1_array,
           adm_num_scale2_array,
           adm_den_scale2_array,
           adm_num_scale3_array,
           adm_den_scale3_array,
           motion_array,
           vif_num_scale0_array,
           vif_den_scale0_array,
           vif_num_scale1_array,
           vif_den_scale1_array,
           vif_num_scale2_array,
           vif_den_scale2_array,
           vif_num_scale3_array,
           vif_den_scale3_array,
           vif_array,
           psnr_array,
           ssim_array,
           ms_ssim_array;

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
    if (do_psnr)
    {
        psnr_array_ptr = &psnr_array;
    }
    else
    {
        psnr_array_ptr = NULL;
    }

    if (do_ssim)
    {
        ssim_array_ptr = &ssim_array;
    }
    else
    {
        ssim_array_ptr = NULL;
    }

    if (do_ms_ssim)
    {
        ms_ssim_array_ptr = &ms_ssim_array;
    }
    else
    {
        ms_ssim_array_ptr = NULL;
    }

#ifdef PRINT_PROGRESS
    printf("Extract atom features...\n");
#endif

    int ret = combo(ref_path, dis_path, w, h, fmt,
            &adm_num_array,
            &adm_den_array,
            &adm_num_scale0_array,
            &adm_den_scale0_array,
            &adm_num_scale1_array,
            &adm_den_scale1_array,
            &adm_num_scale2_array,
            &adm_den_scale2_array,
            &adm_num_scale3_array,
            &adm_den_scale3_array,
            &motion_array,
            &vif_num_scale0_array,
            &vif_den_scale0_array,
            &vif_num_scale1_array,
            &vif_den_scale1_array,
            &vif_num_scale2_array,
            &vif_den_scale2_array,
            &vif_num_scale3_array,
            &vif_den_scale3_array,
            &vif_array,
            psnr_array_ptr,
            ssim_array_ptr,
            ms_ssim_array_ptr,
            errmsg);
    if (ret)
    {
        throw VmafException(errmsg);
    }

    size_t num_frms = motion_array.used;
    bool num_frms_is_consistent =
               (adm_num_array.used == num_frms)
            && (adm_den_array.used == num_frms)
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
    if (psnr_array_ptr != NULL) { num_frms_is_consistent = num_frms_is_consistent && (psnr_array.used == num_frms); }
    if (ssim_array_ptr != NULL) { num_frms_is_consistent = num_frms_is_consistent && (ssim_array.used == num_frms); }
    if (ms_ssim_array_ptr != NULL) { num_frms_is_consistent = num_frms_is_consistent && (ms_ssim_array.used == num_frms); }
    if (!num_frms_is_consistent)
    {
        sprintf(errmsg, "Output feature vectors are of inconsistent dimensions: "
                "motion (%zu), adm_num (%zu), adm_den (%zu), vif_num_scale0 (%zu), "
                "vif_den_scale0 (%zu), vif_num_scale1 (%zu), vif_den_scale1 (%zu), "
                "vif_num_scale2 (%zu), vif_den_scale2 (%zu), vif_num_scale3 (%zu), "
                "vif_den_scale3 (%zu), vif (%zu), "
                "psnr (%zu), ssim (%zu), ms_ssim (%zu), "
                "adm_num_scale0 (%zu), adm_den_scale0 (%zu), adm_num_scale1 (%zu), "
                "adm_den_scale1 (%zu), adm_num_scale2 (%zu), adm_den_scale2 (%zu), "
                "adm_num_scale3 (%zu), adm_den_scale3 (%zu)",
                motion_array.used,
                adm_num_array.used,
                adm_den_array.used,
                vif_num_scale0_array.used,
                vif_den_scale0_array.used,
                vif_num_scale1_array.used,
                vif_den_scale1_array.used,
                vif_num_scale2_array.used,
                vif_den_scale2_array.used,
                vif_num_scale3_array.used,
                vif_num_scale3_array.used,
                vif_array.used,
                psnr_array.used,
                ssim_array.used,
                ms_ssim_array.used,
                adm_num_scale0_array.used,
                adm_den_scale0_array.used,
                adm_num_scale1_array.used,
                adm_den_scale1_array.used,
                adm_num_scale2_array.used,
                adm_den_scale2_array.used,
                adm_num_scale3_array.used,
                adm_num_scale3_array.used
                );

        throw VmafException(errmsg);
    }

#ifdef PRINT_PROGRESS
    printf("Generate final features (including derived atom features)...\n");
#endif

    double ADM2_CONSTANT = 1000.0;
    double ADM_SCALE_CONSTANT = 250.0;
    StatVector adm2, motion, vif_scale0, vif_scale1, vif_scale2, vif_scale3, vif, vmaf;
    StatVector adm_scale0, adm_scale1, adm_scale2, adm_scale3;
    StatVector psnr, ssim, ms_ssim;
    for (size_t i=0; i<num_frms; i++)
    {
        adm2.append((get_at(&adm_num_array, i) + ADM2_CONSTANT) / (get_at(&adm_den_array, i) + ADM2_CONSTANT));
        adm_scale0.append((get_at(&adm_num_scale0_array, i) + ADM_SCALE_CONSTANT) / (get_at(&adm_den_scale0_array, i) + ADM_SCALE_CONSTANT));
        adm_scale1.append((get_at(&adm_num_scale1_array, i) + ADM_SCALE_CONSTANT) / (get_at(&adm_den_scale1_array, i) + ADM_SCALE_CONSTANT));
        adm_scale2.append((get_at(&adm_num_scale2_array, i) + ADM_SCALE_CONSTANT) / (get_at(&adm_den_scale2_array, i) + ADM_SCALE_CONSTANT));
        adm_scale3.append((get_at(&adm_num_scale3_array, i) + ADM_SCALE_CONSTANT) / (get_at(&adm_den_scale3_array, i) + ADM_SCALE_CONSTANT));
        motion.append(get_at(&motion_array, i));
        vif_scale0.append(get_at(&vif_num_scale0_array, i) / get_at(&vif_den_scale0_array, i));
        vif_scale1.append(get_at(&vif_num_scale1_array, i) / get_at(&vif_den_scale1_array, i));
        vif_scale2.append(get_at(&vif_num_scale2_array, i) / get_at(&vif_den_scale2_array, i));
        vif_scale3.append(get_at(&vif_num_scale3_array, i) / get_at(&vif_den_scale3_array, i));
        vif.append(get_at(&vif_array, i));

        if (psnr_array_ptr != NULL) { psnr.append(get_at(&psnr_array, i)); }
        if (ssim_array_ptr != NULL) { ssim.append(get_at(&ssim_array, i)); }
        if (ms_ssim_array_ptr != NULL) { ms_ssim.append(get_at(&ms_ssim_array, i)); }
    }

#ifdef PRINT_PROGRESS
    printf("Normalize features, SVM regression, denormalize score, clip...\n");
#endif

    /* IMPORTANT: always allocate one more spot and put a -1 at the last one's
     * index, so that libsvm will stop looping when seeing the -1 !!!
     * see https://github.com/cjlin1/libsvm */
    svm_node nodes[feature_names.length() + 1];
    nodes[feature_names.length()].index = -1;

    for (size_t i=0; i<num_frms; i++)
    {

        if (VAL_EQUAL_STR(norm_type, "'linear_rescale'"))
        {
            for (size_t j=0; j<feature_names.length(); j++)
            {
                nodes[j].index = j + 1;
                if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm2_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * adm2.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale0_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * adm_scale0.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale1_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * adm_scale1.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale2_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * adm_scale2.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale3_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * adm_scale3.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_motion_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * motion.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale0_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * vif_scale0.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale1_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * vif_scale1.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale2_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * vif_scale2.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale3_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * vif_scale3.at(i) + double(intercepts[j + 1]);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_score'") == 0)
                    nodes[j].value = double(slopes[j + 1]) * vif.at(i) + double(intercepts[j + 1]);
                else
                {
                    printf("Unknown feature name: %s.\n", Stringize(feature_names[j]).c_str());
                    throw VmafException("Unknown feature name");
                }
            }
        }
        else
        {
            for (size_t j=0; j<feature_names.length(); j++)
            {
                nodes[j].index = j + 1;
                if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm2_score'") == 0)
                    nodes[j].value = adm2.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale0_score'") == 0)
                    nodes[j].value = adm_scale0.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale1_score'") == 0)
                    nodes[j].value = adm_scale1.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale2_score'") == 0)
                    nodes[j].value = adm_scale2.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale3_score'") == 0)
                    nodes[j].value = adm_scale3.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_motion_score'") == 0)
                    nodes[j].value = motion.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale0_score'") == 0)
                    nodes[j].value = vif_scale0.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale1_score'") == 0)
                    nodes[j].value = vif_scale1.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale2_score'") == 0)
                    nodes[j].value = vif_scale2.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale3_score'") == 0)
                    nodes[j].value = vif_scale3.at(i);
                else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_score'") == 0)
                    nodes[j].value = vif.at(i);
                else
                {
                    printf("Unknown feature name: %s.\n", Stringize(feature_names[j]).c_str());
                    throw VmafException("Unknown feature name");
                }
            }
        }

        /* feed to svm_predict */
        double prediction = svm_predict(svm_model_ptr.get(), nodes);

        if (VAL_EQUAL_STR(norm_type, "'linear_rescale'"))
        {
            /* denormalize */
            prediction = (prediction - double(intercepts[0])) / double(slopes[0]);
        }
        else
        {
            ;
        }

        /* score transform */
        if (enable_transform && !VAL_IS_NONE(score_transform))
        {
            double value = 0.0;

            /* quadratic transform */
            if (!VAL_IS_NONE(score_transform["p0"]))
            {
                value += double(score_transform["p0"]);
            }
            if (!VAL_IS_NONE(score_transform["p1"]))
            {
                value += double(score_transform["p1"]) * prediction;
            }
            if (!VAL_IS_NONE(score_transform["p2"]))
            {
                value += double(score_transform["p2"]) * prediction * prediction;
            }

            /* rectification */
            if (!VAL_IS_NONE(score_transform["out_lte_in"]) && VAL_EQUAL_STR(score_transform["out_lte_in"], "'true'"))
            {
                if (value > prediction)
                {
                    value = prediction;
                }
            }
            if (!VAL_IS_NONE(score_transform["out_gte_in"]) && VAL_EQUAL_STR(score_transform["out_gte_in"], "'true'"))
            {
                if (value < prediction)
                {
                    value = prediction;
                }
            }

            prediction = value;
        }

        /* score clip */
        if (!disable_clip && !VAL_IS_NONE(score_clip))
        {
            if (prediction < double(score_clip[0]))
            {
                prediction = double(score_clip[0]);
            }
            else if (prediction > double(score_clip[1]))
            {
                prediction = double(score_clip[1]);
            }
        }

        // printf("svm predict: %f, %f, %f, %f, %f, %f => %f\n",
        //        node[0].value, node[1].value, node[2].value,
        //        node[3].value, node[4].value, node[5].value, prediction);

#ifdef PRINT_PROGRESS
        printf("frame: %zu, ", i);
        printf("vmaf: %f, ", prediction);
        printf("adm2: %f, ", adm2.at(i));
        printf("adm_scale0: %f, ", adm_scale0.at(i));
        printf("adm_scale1: %f, ", adm_scale1.at(i));
        printf("adm_scale2: %f, ", adm_scale2.at(i));
        printf("adm_scale3: %f, ", adm_scale3.at(i));
        printf("motion: %f, ", motion.at(i));
        printf("vif_scale0: %f, ", vif_scale0.at(i));
        printf("vif_scale1: %f, ", vif_scale1.at(i));
        printf("vif_scale2: %f, ", vif_scale2.at(i));
        printf("vif_scale3: %f, ", vif_scale3.at(i));
        printf("vif: %f, ", vif.at(i));

        if (psnr_array_ptr != NULL) { printf("psnr: %f, ", psnr.at(i)); }
        if (ssim_array_ptr != NULL) { printf("ssim: %f, ", ssim.at(i)); }
        if (ms_ssim_array_ptr != NULL) { printf("ms_ssim: %f, ", ms_ssim.at(i)); }

        printf("\n");
#endif

        vmaf.append(prediction);
    }

    Result result{};
    result.set_scores("vmaf", vmaf);
    for (size_t j=0; j<feature_names.length(); j++)
    {
        if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm2_score'") == 0)
            result.set_scores("adm2", adm2);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale0_score'") == 0)
            result.set_scores("adm_scale0", adm_scale0);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale1_score'") == 0)
            result.set_scores("adm_scale1", adm_scale1);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale2_score'") == 0)
            result.set_scores("adm_scale2", adm_scale2);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_adm_scale3_score'") == 0)
            result.set_scores("adm_scale3", adm_scale3);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_motion_score'") == 0)
            result.set_scores("motion", motion);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale0_score'") == 0)
            result.set_scores("vif_scale0", vif_scale0);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale1_score'") == 0)
            result.set_scores("vif_scale1", vif_scale1);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale2_score'") == 0)
            result.set_scores("vif_scale2", vif_scale2);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_scale3_score'") == 0)
            result.set_scores("vif_scale3", vif_scale3);
        else if (strcmp(Stringize(feature_names[j]).c_str(), "'VMAF_feature_vif_score'") == 0)
            result.set_scores("vif", vif);
        else
        {
            printf("Unknown feature name: %s.\n", Stringize(feature_names[j]).c_str());
            throw VmafException("Unknown feature name");
        }
    }
    if (psnr_array_ptr != NULL) { result.set_scores("psnr", psnr); }
    if (ssim_array_ptr != NULL) { result.set_scores("ssim", ssim); }
    if (ms_ssim_array_ptr != NULL) { result.set_scores("ms_ssim", ms_ssim); }

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

// static const char VMAFOSS_XML_VERSION[] = "0.3.1";
//static const char VMAFOSS_XML_VERSION[] = "0.3.2"; // add vif, ssim and ms-ssim scores
//static const char VMAFOSS_XML_VERSION[] = "0.3.3"; // fix slopes and intercepts to match nflxtrain_vmafv3a.pkl
static const char VMAFOSS_XML_VERSION[] = "0.3.2"; // fix slopes and intercepts to match nflxall_vmafv4.pkl

double RunVmaf(const char* fmt, int width, int height,
               const char *ref_path, const char *dis_path, const char *model_path,
               const char *log_path, const char *log_fmt,
               bool disable_clip, bool enable_transform,
               bool do_psnr, bool do_ssim, bool do_ms_ssim,
               const char *pool_method)
{
    printf("Start calculating VMAF score...\n");

    Asset asset(width, height, ref_path, dis_path, fmt);
    VmafRunner runner{model_path};
    Timer timer;

    timer.start();
    Result result = runner.run(asset, disable_clip, enable_transform, do_psnr, do_ssim, do_ms_ssim);
    timer.stop();

    if (pool_method != NULL && (strcmp(pool_method, "min")==0))
    {
        result.setScoreAggregateMethod(MIN);
    }
    else if (pool_method != NULL && (strcmp(pool_method, "harmonic_mean")==0))
    {
        result.setScoreAggregateMethod(HARMONIC_MEAN);
    }
    else // mean or default
    {
        result.setScoreAggregateMethod(MEAN);
    }

    size_t num_frames = result.get_scores("vmaf").size();
    double aggregate_vmaf = result.get_score("vmaf");
    double exec_fps = (double)num_frames / (double)timer.elapsed();
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

        Arr metrics;
        for (size_t j=0; j<result_keys.size(); j++)
        {
            metrics.append(result_keys[j]);
        }

        Arr frames;
        for (size_t i=0; i<num_frames; i++)
        {
            OTab frame;
            frame["frameNum"] = i;
            OTab metrics_scores;
            for (size_t j=0; j<result_keys.size(); j++)
            {
                value = result.get_scores(result_keys[j].c_str()).at(i);
                value = _round_to_digit(value, 5);
                metrics_scores[result_keys[j].c_str()] = value;
            }
            frame["metrics"] = metrics_scores;
            frames.append(frame);
        }

        Val top = OTab();
        top["version"] = VMAFOSS_XML_VERSION;
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
        xml_root.append_attribute("version") = VMAFOSS_XML_VERSION;

        auto params_node = xml_root.append_child("params");
        params_node.append_attribute("model") = _get_file_name(std::string(model_path)).c_str();
        params_node.append_attribute("scaledWidth") = width;
        params_node.append_attribute("scaledHeight") = height;

        auto info_node = xml_root.append_child("fyi");
        info_node.append_attribute("numOfFrames") = (int)num_frames;
        info_node.append_attribute("aggregateVMAF") = aggregate_vmaf;
        info_node.append_attribute("execFps") = exec_fps;

        auto frames_node = xml_root.append_child("frames");
        for (size_t i=0; i<num_frames; i++)
        {
            auto node = frames_node.append_child("frame");
            node.append_attribute("frameNum") = (int)i;
            for (size_t j=0; j<result_keys.size(); j++)
            {
                node.append_attribute(result_keys[j].c_str()) = result.get_scores(result_keys[j].c_str()).at(i);
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

