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

#ifndef ALL_OPTIONS_H_
#define ALL_OPTIONS_H_

/* Whether to use single precision for computation. Need to make sure that
 * ALL_OPT_SINGLE_PRECISION is consistent with MOTION_OPT_SINGLE_PRECISION,
 * ANSNR_OPT_SINGLE_PRECISION, ADM_OPT_SINGLE_PRECISION and
 * VIF_OPT_SINGLE_PRECISION, if all() is called.*/
#define ALL_OPT_SINGLE_PRECISION
//#define ALL_OPT_DOUBLE_PRECISION

#endif /* ALL_OPTIONS_H_ */
