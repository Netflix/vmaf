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

#ifndef VIF_OPTIONS_H_
#define VIF_OPTIONS_H_

/* Whether to use an alternate arrangement of the core VIF statistic. */
#define VIF_OPT_ALTERNATE_STATISTIC

/* Whether to use an approximate implementation of log2 / log2f. */
#define VIF_OPT_FAST_LOG2

/* Whether to save intermedate results to files. */
/* #define VIF_OPT_DEBUG_DUMP */

/* Whether to keep the borders of the image after filtering. */
#define VIF_OPT_HANDLE_BORDERS

/* Enhancement gain imposed on vif, must be >= 1.0, where 1.0 means the gain is completely disabled */
#define DEFAULT_VIF_ENHN_GAIN_LIMIT (100.0)

/* Scaling factor for the Gaussian kernel, where 2.0 means multiplying the
 * standard deviation by 2 and expand the kernel length accordingly. */
#define DEFAULT_VIF_KERNELSCALE (1.0)

#endif /* VIF_OPTIONS_H_ */
