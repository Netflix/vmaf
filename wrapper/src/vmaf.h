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

double RunVmaf(int width, int height, const char *src, const char *dis, const char *model, const char *report);

class Asset
{
public:
	Asset(int w, int h, const char *ref_path, const char *dis_path, const char *fmt):
		w(w), h(h), ref_path(ref_path), dis_path(dis_path), fmt(fmt) {}
	Asset(int w, int h, const char *ref_path, const char *dis_path):
		w(w), h(h), ref_path(ref_path), dis_path(dis_path), fmt("yuv420p") {}
	int getWidth();
	int getHeight();
	const char* getRefPath();
	const char* getDisPath();
	const char* getFmt();
private:
	const int w, h;
	const char *ref_path, *dis_path, *fmt;
};

class StatVector
{
public:
	StatVector() {}
	StatVector(std::vector<double> l): l(l) {}
	double mean();
	void append(double e);
	double at(size_t idx);
	size_t size();
private:
	std::vector<double> l;
};

class Result
{
public:
	Result() {}
	void set_scores(const std::string &key, const StatVector &scores);
	StatVector get_scores(const std::string &key);
	double get_score(const std::string &key);
private:
	std::map<std::string, StatVector> d;
};

class VmafException: public std::exception
{
public:
	explicit VmafException(const char *msg): msg(msg) {}
	virtual const char* what() const throw () { return msg.c_str(); }
private:
	std::string msg;
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
	Result run(Asset asset);
private:
	const char *model_path;
	char *libsvm_model_path;
	static const int INIT_FRAMES = 1000;
	static const double* INTERCEPTS;
	static const double* SLOPES;
	static const double* SCORE_CLIP;
};

struct SvmDelete {
    void operator()(void *svm);
};

#endif /* VMAF_H_ */
