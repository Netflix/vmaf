/**
 *
 *  Copyright 2016 Netflix, Inc.
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

#include "vmaf.h"
#include "darray.h"
#include "combo.h"
#include "svm.h"
#include "pugixml/pugixml.hpp"
#include "timer.h"
#include "chooseser.h"

#define VAL_EQUAL_STR(V,S) (Stringize((V)).compare((S))==0)
#define VAL_IS_LIST(V) ((V).tag=='n') /* check ocval.cc */
#define VAL_IS_NONE(V) ((V).tag=='Z') /* check ocval.cc */

void Result::set_scores(const std::string &key, const StatVector &scores)
{
	d[key] = scores;
}

StatVector Result::get_scores(const std::string &key)
{
	return d[key];
}

double Result::get_score(const std::string &key)
{
	StatVector list = get_scores(key);
	return list.mean();
}

void SvmDelete::operator()(void *svm)
{
	svm_free_model_content((svm_model *)svm);
}

void _read_and_assert_model(const char *model_path, Val& feature_names,
     Val& norm_type, Val& slopes, Val& intercepts, Val& score_clip)
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
    try
	{
        LoadValFromFile(model_path, model, SERIALIZE_P0);

        model_type = model["model_dict"]["model_type"];
        feature_names = model["model_dict"]["feature_names"];
        norm_type = model["model_dict"]["norm_type"];

        slopes = model["model_dict"]["slopes"];
        intercepts = model["model_dict"]["intercepts"];
        score_clip = model["model_dict"]["score_clip"];
    }
    catch (std::runtime_error& e)
    {
        printf("Input model at %s cannot be read successfully.\n", model_path);
        throw e;
    }

    if (!VAL_EQUAL_STR(model_type, "'LIBSVMNUSVR'"))
    {
        printf("Current vmafossexec only accepts model type LIBSVMNUSVR, "
                "but got %s\n", Stringize(model_type).c_str());
        throw std::runtime_error{"Incompatible model_type"};
    }

    if (!VAL_IS_LIST(feature_names))
    {
        printf("feature_names in model must be a list.\n");
        throw std::runtime_error{"Incompatible feature_names"};
    }

    if (!(VAL_EQUAL_STR(norm_type, "'none'") ||
          VAL_EQUAL_STR(norm_type, "'linear_rescale'")
          )
        )
    {
        printf("norm_type in model must be either 'none' or 'linear_rescale'.\n");
        throw std::runtime_error{"Incompatible norm_type"};
    }

    if ( VAL_EQUAL_STR(norm_type, "'linear_rescale'") &&
         (!VAL_IS_LIST(slopes) || !VAL_IS_LIST(intercepts))
        )
    {
        printf("if norm_type in model is 'linear_rescale', "
                "both slopes and intercepts must be a list.\n");
        throw std::runtime_error{"Incompatible slopes or intercepts"};
    }

    if (!(VAL_IS_NONE(score_clip) || VAL_IS_LIST(score_clip)))
    {
        printf("score_clip in model must be either None or list.\n");
        throw std::runtime_error{"Incompatible score_clip"};
    }
}

Result VmafRunner::run(Asset asset)
{

#ifdef PRINT_PROGRESS
	printf("Read input model (pkl)...\n");
#endif

    Val feature_names, norm_type, slopes, intercepts, score_clip;
    _read_and_assert_model(model_path, feature_names, norm_type, slopes, intercepts, score_clip);

#ifdef PRINT_PROGRESS
	printf("Read input model (libsvm)...\n");
#endif

    std::unique_ptr<svm_model, SvmDelete> svm_model_ptr{svm_load_model(libsvm_model_path)};
    if (!svm_model_ptr)
    {
        throw std::runtime_error{"error loading SVM model"};
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

	init_array(&adm_num_array, INIT_FRAMES);
	init_array(&adm_den_array, INIT_FRAMES);
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

#ifdef PRINT_PROGRESS
	printf("Extract atom features...\n");
#endif

	int ret = combo(ref_path, dis_path, w, h, fmt,
			&adm_num_array,
			&adm_den_array,
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
			&psnr_array,
			&ssim_array,
			&ms_ssim_array,
			errmsg);
	if (ret)
	{
        throw VmafException(errmsg);
	}

	size_t num_frms = motion_array.used;
	if (!(     adm_num_array.used == num_frms
			&& adm_den_array.used == num_frms
			&& vif_num_scale0_array.used == num_frms
			&& vif_den_scale0_array.used == num_frms
			&& vif_num_scale1_array.used == num_frms
			&& vif_den_scale1_array.used == num_frms
			&& vif_num_scale2_array.used == num_frms
			&& vif_den_scale2_array.used == num_frms
			&& vif_num_scale3_array.used == num_frms
			&& vif_den_scale3_array.used == num_frms
			&& vif_array.used == num_frms
			&& psnr_array.used == num_frms
			&& ssim_array.used == num_frms
			&& ms_ssim_array.used == num_frms
		))
	{
		sprintf(errmsg, "all2 outputs feature vectors of inconsistent dimensions: "
				"motion (%zu), adm_num (%zu), adm_den (%zu), vif_num_scale0 (%zu), "
				"vif_den_scale0 (%zu), vif_num_scale1 (%zu), vif_den_scale1 (%zu), "
				"vif_num_scale2 (%zu), vif_den_scale2 (%zu), vif_num_scale3 (%zu), "
				"vif_den_scale3 (%zu), vif (%zu), psnr (%zu), ssim (%zu), ms_ssim (%zu)",
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
				ms_ssim_array.used
				);

		throw VmafException(errmsg);
	}

#ifdef PRINT_PROGRESS
	printf("Generate final features (including derived atom features)...\n");
#endif

	StatVector adm2, motion, vif_scale0, vif_scale1, vif_scale2, vif_scale3, vif, score, psnr, ssim, ms_ssim;
	for (size_t i=0; i<num_frms; i++)
	{
		adm2.append((get_at(&adm_num_array, i) + 1000.0) / (get_at(&adm_den_array, i) + 1000.0));
		motion.append(get_at(&motion_array, i));
		vif_scale0.append(get_at(&vif_num_scale0_array, i) / get_at(&vif_den_scale0_array, i));
		vif_scale1.append(get_at(&vif_num_scale1_array, i) / get_at(&vif_den_scale1_array, i));
		vif_scale2.append(get_at(&vif_num_scale2_array, i) / get_at(&vif_den_scale2_array, i));
		vif_scale3.append(get_at(&vif_num_scale3_array, i) / get_at(&vif_den_scale3_array, i));
		vif.append(get_at(&vif_array, i));
		psnr.append(get_at(&psnr_array, i));
		ssim.append(get_at(&ssim_array, i));
		ms_ssim.append(get_at(&ms_ssim_array, i));
	}


#ifdef PRINT_PROGRESS
	printf("Normalize features, SVM regression, denormalize score, clip...\n");
#endif

	/* IMPORTANT: always allocate one more spot and put a -1 in node.index,
	 * so that libsvm will stop looping when seeing the -1 !!!
	 * see https://github.com/cjlin1/libsvm */
	svm_node nodes[6 + 1];

	for (size_t i=0; i<num_frms; i++)
	{

        if (VAL_EQUAL_STR(norm_type, "'linear_rescale'"))
        {
            /* 1. adm2 */
            nodes[0].index = 1;
            nodes[0].value = double(slopes[1]) * adm2.at(i) + double(intercepts[1]);

            /* 2. motion */
            nodes[1].index = 2;
            nodes[1].value = double(slopes[2]) * motion.at(i) + double(intercepts[2]);

            /* 3. vif_scale0 */
            nodes[2].index = 3;
            nodes[2].value = double(slopes[3]) * vif_scale0.at(i) + double(intercepts[3]);

            /* 4. vif_scale1 */
            nodes[3].index = 4;
            nodes[3].value = double(slopes[4]) * vif_scale1.at(i) + double(intercepts[4]);

            /* 5. vif_scale2 */
            nodes[4].index = 5;
            nodes[4].value = double(slopes[5]) * vif_scale2.at(i) + double(intercepts[5]);

            /* 6. vif_scale3 */
            nodes[5].index = 6;
            nodes[5].value = double(slopes[6]) * vif_scale3.at(i) + double(intercepts[6]);
        }
        else
        {
            /* 1. adm2 */
            nodes[0].index = 1;
            nodes[0].value = adm2.at(i);

            /* 2. motion */
            nodes[1].index = 2;
            nodes[1].value = motion.at(i);

            /* 3. vif_scale0 */
            nodes[2].index = 3;
            nodes[2].value = vif_scale0.at(i);

            /* 4. vif_scale1 */
            nodes[3].index = 4;
            nodes[3].value = vif_scale1.at(i);

            /* 5. vif_scale2 */
            nodes[4].index = 5;
            nodes[4].value = vif_scale2.at(i);

            /* 6. vif_scale3 */
            nodes[5].index = 6;
            nodes[5].value = vif_scale3.at(i);
        }

	    nodes[6].index = -1;

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

	    /* clip */
	    if (!VAL_IS_NONE(score_clip))
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
		printf("score: %f, ", prediction);
		printf("adm2: %f, ", adm2.at(i));
		printf("motion: %f, ", motion.at(i));
		printf("vif_scale0: %f, ", vif_scale0.at(i));
		printf("vif_scale1: %f, ", vif_scale1.at(i));
		printf("vif_scale2: %f, ", vif_scale2.at(i));
		printf("vif_scale3: %f, ", vif_scale3.at(i));
		printf("vif: %f, ", vif.at(i));
		printf("psnr: %f, ", psnr.at(i));
		printf("ssim: %f, ", ssim.at(i));
		printf("ms_ssim: %f, ", ms_ssim.at(i));
		printf("\n");
#endif

		score.append(prediction);
	}

	Result result{};
	result.set_scores("adm2", adm2);
	result.set_scores("motion", motion);
	result.set_scores("vif_scale0", vif_scale0);
	result.set_scores("vif_scale1", vif_scale1);
	result.set_scores("vif_scale2", vif_scale2);
	result.set_scores("vif_scale3", vif_scale3);
	result.set_scores("vif", vif);
	result.set_scores("score", score);
	result.set_scores("psnr", psnr);
	result.set_scores("ssim", ssim);
	result.set_scores("ms_ssim", ms_ssim);

	free_array(&adm_num_array);
	free_array(&adm_den_array);
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

double RunVmaf(int width, int height, const char *src, const char *dis, const char *model, const char *report)
{

	Asset asset(width, height, src, dis);
	VmafRunner runner{model};
	Timer timer;

	timer.start();
	Result result = runner.run(asset);
	timer.stop();

    size_t num_frames = result.get_scores("score").size();
    double aggregate_score = result.get_score("score");
    double processing_fps = (double)num_frames / (double)timer.elapsed();

    /* output to xml */
    pugi::xml_document xml;
    pugi::xml_node xml_root = xml.append_child("VMAFOSSCalculator");
    xml_root.append_attribute("version") = VMAFOSS_XML_VERSION;
    auto info_node = xml_root.append_child("Info");
    auto frames_node = xml_root.append_child("Frames");
    for (size_t i=0; i<num_frames; i++)
    {
    	auto node = frames_node.append_child("Frame");
        node.append_attribute("num") = (int)i;
        node.append_attribute("score") = result.get_scores("score").at(i);
        node.append_attribute("adm2") = result.get_scores("adm2").at(i);
        node.append_attribute("motion") = result.get_scores("motion").at(i);
        node.append_attribute("vif_scale0") = result.get_scores("vif_scale0").at(i);
        node.append_attribute("vif_scale1") = result.get_scores("vif_scale1").at(i);
        node.append_attribute("vif_scale2") = result.get_scores("vif_scale2").at(i);
        node.append_attribute("vif_scale3") = result.get_scores("vif_scale3").at(i);
        node.append_attribute("vif") = result.get_scores("vif").at(i);
        node.append_attribute("psnr") = result.get_scores("psnr").at(i);
        node.append_attribute("ssim") = result.get_scores("ssim").at(i);
        node.append_attribute("ms_ssim") = result.get_scores("ms_ssim").at(i);
    }
    info_node.append_attribute("numOfFrames") = (int)num_frames;
    info_node.append_attribute("aggregateScore") = aggregate_score;
    info_node.append_attribute("fps") = processing_fps;

    printf("Processing FPS: %f\n", processing_fps);

    if (report)
    {
    	xml.save_file(report);
    }

    return aggregate_score;
}
