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

#ifndef DARRAY_H_
#define DARRAY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>

typedef struct
{
    double *array;
    size_t used;
    size_t size;
    pthread_mutex_t mutex;
} DArray;

void init_array(DArray *a, size_t init_size);
void insert_array(DArray *a, double e);
void insert_array_at(DArray *a, double e, int pos);
void free_array(DArray *a);
double get_at(DArray *a, int pos);

#ifdef __cplusplus
}
#endif

#endif /* DARRAY_H_ */
