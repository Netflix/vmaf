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

#ifndef ANSNR_OPTIONS_H_
#define ANSNR_OPTIONS_H_

/* Whether to use border replication instead of zero extension. */
/* #define ANSNR_OPT_BORDER_REPLICATE */

/* Whether to save intermediate results to files. */
/* #define ANSNR_OPT_DEBUG_DUMP */

/* Whether to use a 1-D approximation of filters. */
/* #define ANSNR_OPT_FILTER_1D */

/* Whether to normalize result by dividing against maximum ANSNR. */
/* #define ANSNR_OPT_NORMALIZE */

#endif /* ANSNR_OPTIONS_H_ */
