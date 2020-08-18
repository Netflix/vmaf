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

#ifndef FEATURE_MOTION_H_
#define FEATURE_MOTION_H_

#include <stdint.h>

static const uint16_t filter[5] = { 3571, 16004, 26386, 16004, 3571 };
static const int filter_width = sizeof(filter) / sizeof(filter[0]);

#endif /* _FEATURE_MOTION_H_ */
