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

#ifndef OFFSET_H_
#define OFFSET_H_

/* Whether to use [0,255] or [-128,127] input pixel range. */
//#define OPT_RANGE_PIXEL_OFFSET 0
#define OPT_RANGE_PIXEL_OFFSET (-128)

int offset_image_s(float *buf, float off, int width, int height, int stride);

#endif /* OFFSET_H_ */
