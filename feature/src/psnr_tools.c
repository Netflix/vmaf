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

#include <math.h>

#include "psnr_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int compute_psnr(const number_t *ref, const number_t *dis, int w, int h,
		int ref_stride, int dis_stride, double *score, double peak, double psnr_max)
{

	double noise_ = 0;

	int ref_stride_ = ref_stride / sizeof(number_t);
	int dis_stride_ = dis_stride / sizeof(number_t);

	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			number_t ref_ = ref[i * ref_stride_ + j];
			number_t dis_ = dis[i * dis_stride_ + j];
			number_t diff = ref_ - dis_;
			noise_ += diff * diff;
		}
	}
	noise_ /= (w * h);

	double eps = 1e-10;
	*score = MIN(10 * log10(peak * peak / MAX(noise_, eps)), psnr_max);

	return 0;

}
