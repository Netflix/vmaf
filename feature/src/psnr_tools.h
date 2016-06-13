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

#ifndef PSNR_TOOLS_H_
#define PSNR_TOOLS_H_

#include "psnr_options.h"

#ifdef PSNR_OPT_SINGLE_PRECISION
  typedef float number_t;

  #define read_image_b  read_image_b2s
  #define read_image_w  read_image_w2s

#else
  typedef double number_t;

	#define read_image_b  read_image_b2d
	#define read_image_w  read_image_w2d

#endif

int compute_psnr(const number_t *ref, const number_t *dis, int w, int h,
		int ref_stride, int dis_stride, double *score, double peak, double psnr_max);

#endif /* PSNR_TOOLS_H_ */
