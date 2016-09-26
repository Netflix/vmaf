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

int Asset::getWidth()
{
	return w;
}
int Asset::getHeight()
{
	return h;
}
const char* Asset::getRefPath()
{
	return ref_path;
}
const char* Asset::getDisPath()
{
	return dis_path;
}
const char* Asset::getFmt()
{
	return fmt;
}

double StatVector::mean()
{
	double sum = 0.0;
	for (double e : l)
	{
		sum += e;
	}
	return sum / l.size();
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

/* the first intercept/slope number is for the vmaf score, the rest 6 are for
 * adm2, motion, vif_scale0, vif_scale1, vif_scale2, vif_scale3 in order. The
  current slops and intercpets match nflxtrain_vmafv3a.pkl*/
static const double cINTERCEPTS[] = {-0.0181818181818, -1.97150772696, -0.0367108449062, -0.0820391009278, -0.285805849112, -0.48016754306, -0.776475692049};
static const double cSLOPES[] = {0.00945454545455, 2.97150772696, 0.0524677881284, 1.08203909458, 1.28580617639, 1.48016828489, 1.7764765384};
static const double cSCORE_CLIP[] = {0.0, 100.0};
const double* VmafRunner::INTERCEPTS = cINTERCEPTS;
const double* VmafRunner::SLOPES = cSLOPES;
const double* VmafRunner::SCORE_CLIP = cSCORE_CLIP;

Result VmafRunner::run(Asset asset)
{

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

    Val slopes, intercepts, score_clip;

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
    Val model, model_type, feature_names, norm_type;
    try
	{
//        LoadValFromFile(model_path, model, SERIALIZE_P0);

//        model_type = model["model_dict"]["model_type"];
//        feature_names = model["model_dict"]["feature_names"];
//        norm_type = model["model_dict"]["norm_type"];

//        slopes = model["model_dict"]["slopes"];
//        intercepts = model["model_dict"]["intercepts"];
//        score_clip = model["model_dict"]["score_clip"];
    }
    catch (std::runtime_error& e)
    {
        printf("Input model at %s cannot be read successfully.\n", model_path);
        throw e;
    }

    std::unique_ptr<svm_model, SvmDelete> svm_model_ptr{svm_load_model(libsvm_model_path)};
    if (!svm_model_ptr)
    {
        throw std::runtime_error{"error loading SVM model"};
    }

	for (size_t i=0; i<num_frms; i++)
	{
	    svm_node node[6];

	    /* 1. adm2 */
	    node[0].index = 1;
	    node[0].value = SLOPES[1] * adm2.at(i) + INTERCEPTS[1];

	    /* 2. motion */
	    node[1].index = 2;
	    node[1].value = SLOPES[2] * motion.at(i) + INTERCEPTS[2];

	    /* 3. vif_scale0 */
	    node[2].index = 3;
	    node[2].value = SLOPES[3] * vif_scale0.at(i) + INTERCEPTS[3];

	    /* 4. vif_scale1 */
	    node[3].index = 4;
	    node[3].value = SLOPES[4] * vif_scale1.at(i) + INTERCEPTS[4];

	    /* 5. vif_scale2 */
	    node[4].index = 5;
	    node[4].value = SLOPES[5] * vif_scale2.at(i) + INTERCEPTS[5];

	    /* 6. vif_scale3 */
	    node[5].index = 6;
	    node[5].value = SLOPES[6] * vif_scale3.at(i) + INTERCEPTS[6];

	    /* feed to svm_predict */
	    double prediction = svm_predict(svm_model_ptr.get(), node);

	    /* denormalize */
	    prediction = (prediction - INTERCEPTS[0]) / SLOPES[0];

	    /* clip */
	    if (prediction < SCORE_CLIP[0])
	    {
	    	prediction = SCORE_CLIP[0];
	    }
	    else if (prediction > SCORE_CLIP[1])
		{
	    	prediction = SCORE_CLIP[1];
		}

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
