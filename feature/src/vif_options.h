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

/* Whether to use a 1-D formulation of the Gaussian filter. */
#define VIF_OPT_FILTER_1D

/* Whether to use single precision for computation. */
#define VIF_OPT_SINGLE_PRECISION
//#define VIF_OPT_DOUBLE_PRECISION

#endif /* VIF_OPTIONS_H_ */
