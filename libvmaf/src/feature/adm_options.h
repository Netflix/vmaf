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

/* Enhancement gain imposed on adm, must be >= 1.0, where 1.0 means the gain is completely disabled */
#define DEFAULT_ADM_ENHN_GAIN_LIMIT (100.0)

/* normalized viewing distance = viewing distance / ref display's physical height */
#define DEFAULT_ADM_NORM_VIEW_DIST (3.0)

/* reference display height in pixels */
#define DEFAULT_ADM_REF_DISPLAY_HEIGHT (1080)

/* Contrast sensitivity function */
enum ADM_CSF_MODE {
    ADM_CSF_MODE_WATSON97 = 0,
};

/* Default contrast sensitivity function */
#define DEFAULT_ADM_CSF_MODE (ADM_CSF_MODE_WATSON97)

#endif /* ADM_OPTIONS_H_ */
