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

#ifndef CONVOLUTION_INTERNAL_H_
#define CONVOLUTION_INTERNAL_H_

#include "macros.h"
#include <stdbool.h>

FORCE_INLINE inline float convolution_edge_s(bool horizontal, const float *filter, int filter_width, const float *src, int width, int height, int stride, int i, int j)
{
	int radius = filter_width / 2;

	float accum = 0;
	for (int k = 0; k < filter_width; ++k) {
		int i_tap = horizontal ? i : i - radius + k;
		int j_tap = horizontal ? j - radius + k : j;

		// Handle edges by mirroring.
		if (horizontal) {
			if (j_tap < 0)
				j_tap = -j_tap;
			else if (j_tap >= width)
				j_tap = width - (j_tap - width + 1);
		} else {
			if (i_tap < 0)
				i_tap = -i_tap;
			else if (i_tap >= height)
				i_tap = height - (i_tap - height + 1);
		}

		accum += filter[k] * src[i_tap * stride + j_tap];
	}
	return accum;
}

FORCE_INLINE inline double convolution_edge_d(bool horizontal, const double *filter, int filter_width, const double *src, int width, int height, int stride, int i, int j)
{
	int radius = filter_width / 2;

	double accum = 0;
	for (int k = 0; k < filter_width; ++k) {
		int i_tap = horizontal ? i : i - radius + k;
		int j_tap = horizontal ? j - radius + k : j;

		// Handle edges by mirroring.
		if (horizontal) {
			if (j_tap < 0)
				j_tap = -j_tap;
			else if (j_tap >= width)
				j_tap = width - (j_tap - width + 1);
		} else {
			if (i_tap < 0)
				i_tap = -i_tap;
			else if (i_tap >= height)
				i_tap = height - (i_tap - height + 1);
		}

		accum += filter[k] * src[i_tap * stride + j_tap];
	}
	return accum;
}

#endif // CONVOLUTION_INTERNAL_H_
