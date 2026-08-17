/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
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

/* Whole-sample symmetric ("reflect-101") boundary index; same semantics as
 * motion_mirror() in feature/integer_motion.h. Reflects repeatedly so any size >= 1
 * is safe; for in-contract offsets (|tap - i or j| within the filter's radius) at
 * most one reflection is taken, so behavior is bit-identical to the historical
 * single-bounce version for size >= radius+1. size <= 1 is special-cased: the
 * reflection period 2*(size-1) is 0 (or negative) and the loop would not
 * terminate. size is always >= 1 in the real call graph, so covering size <= 0
 * as well only makes the function total; it changes no reachable result. */
FORCE_INLINE int convolution_mirror(int tap, int size)
{
	if (size <= 1) return 0;
	while (tap < 0 || tap >= size) {
		if (tap < 0) tap = -tap;
		else tap = 2 * size - tap - 2;
	}
	return tap;
}

FORCE_INLINE float convolution_edge_s(bool horizontal, const float *filter, int filter_width, const float *src, int width, int height, int stride, int i, int j)
{
	int radius = filter_width / 2;

	float accum = 0;
	for (int k = 0; k < filter_width; ++k) {
		int i_tap = horizontal ? i : i - radius + k;
		int j_tap = horizontal ? j - radius + k : j;

		// Handle edges by mirroring.
		if (horizontal) {
			j_tap = convolution_mirror(j_tap, width);
		} else {
			i_tap = convolution_mirror(i_tap, height);
		}

		accum += filter[k] * src[i_tap * stride + j_tap];
	}
	return accum;
}

FORCE_INLINE float convolution_edge_sq_s(bool horizontal, const float *filter, int filter_width, const float *src, int width, int height, int stride, int i, int j)
{
	int radius = filter_width / 2;

	float accum = 0;
	float src_val;
	for (int k = 0; k < filter_width; ++k) {
		int i_tap = horizontal ? i : i - radius + k;
		int j_tap = horizontal ? j - radius + k : j;

		// Handle edges by mirroring.
		if (horizontal) {
			j_tap = convolution_mirror(j_tap, width);
		}
		else {
			i_tap = convolution_mirror(i_tap, height);
		}
		src_val = src[i_tap * stride + j_tap];
		accum += filter[k] * (src_val * src_val);
	}
	return accum;
}

FORCE_INLINE float convolution_edge_xy_s(bool horizontal, const float *filter, int filter_width, const float *src1, const float *src2, int width, int height, int stride1, int stride2, int i, int j)
{
	int radius = filter_width / 2;

	float accum = 0;
	float src_val1, src_val2;
	for (int k = 0; k < filter_width; ++k) {
		int i_tap = horizontal ? i : i - radius + k;
		int j_tap = horizontal ? j - radius + k : j;

		// Handle edges by mirroring.
		if (horizontal) {
			j_tap = convolution_mirror(j_tap, width);
		}
		else {
			i_tap = convolution_mirror(i_tap, height);
		}
		src_val1 = src1[i_tap * stride1 + j_tap];
		src_val2 = src2[i_tap * stride2 + j_tap];
		accum += filter[k] * (src_val1 * src_val2);
	}
	return accum;
}

#endif // CONVOLUTION_INTERNAL_H_
