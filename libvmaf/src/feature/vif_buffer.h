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

#ifndef VIF_BUFFER_H_
#define VIF_BUFFER_H_

typedef struct Intermediate_VifBuffer {
    //intermediate buffers used between horizontal and virtical pass for 1-D filters
    uint32_t *mu1;
    uint32_t *mu2;
    uint32_t *ref;
    uint32_t *dis;
    uint32_t *ref_dis;
    uint32_t *ref_convol;
    uint32_t *dis_convol;
} Intermediate_VifBuffer;

typedef struct VifBuffer {
    size_t stride;
    uint16_t *ref;
    uint16_t *dis;
    uint16_t *mu1;
    uint16_t *mu2;
    uint32_t *mu1_32;
    uint32_t *mu2_32;
    uint32_t *ref_sq;
    uint32_t *dis_sq;
    uint32_t *ref_dis;
    Intermediate_VifBuffer tmp;
} VifBuffer;

#endif /* VIF_BUFFER_H_ */
