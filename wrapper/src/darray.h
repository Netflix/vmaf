/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
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

#ifndef DARRAY_H_
#define DARRAY_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MULTI_THREADING
#include "pthread.h"
#endif

typedef struct
{
    double *array;
    size_t used;
    size_t size;
#ifdef MULTI_THREADING
    pthread_mutex_t mutex;
#endif
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
