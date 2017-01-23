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

#ifndef ADM_OPTIONS_H_
#define ADM_OPTIONS_H_

/* Percentage of frame to discard on all 4 sides */
#define ADM_BORDER_FACTOR (0.1)

/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to save intermediate results to files. */
/* #define ADM_OPT_DEBUG_DUMP */

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

/* Whether to use single precision for computation. */
#define ADM_OPT_SINGLE_PRECISION
//#define ADM_OPT_DOUBLE_PRECISION

#endif /* ADM_OPTIONS_H_ */
