/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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
#include "integer_vif_cuda.h"

#include "common.h"
#ifdef __cplusplus

extern "C" {
#endif

typedef struct filter_table_stuct {
  uint16_t filter[4][18];
} filter_table_stuct;

typedef struct filter_width_struct {
  int w[4];
} filter_width_struct;

typedef struct vif_accums {
  int64_t x;
  int64_t x2;
  int64_t num_x;
  int64_t num_log;
  int64_t den_log;
  int64_t num_non_log;
  int64_t den_non_log;
} vif_accums;

void filter1d_8(VifBufferCuda *buf, uint8_t* ref_in, uint8_t* dis_in, int w, int h, double vif_enhn_gain_limit,
                CUstream stream);

void filter1d_16(VifBufferCuda *buf, uint16_t* ref_in, uint16_t* dis_in, int w, int h, int scale, int bpc,
                 double vif_enhn_gain_limit, CUstream stream);

#ifdef __cplusplus
}
#endif