/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
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

#include <stdlib.h>
#include "darray.h"

void init_array(DArray *a, size_t init_size)
{
    a->array = (double *)malloc(init_size * sizeof(double));
    a->used = 0;
    a->size = init_size;
#ifdef MULTI_THREADING
    pthread_mutex_init(&a->mutex, NULL);
#endif
}

void insert_array(DArray *a, double e)
{
#ifdef MULTI_THREADING
    pthread_mutex_lock(&a->mutex);
#endif
    if (a->used == a->size)
    {
        a->size *= 2;
        a->array = (double *)realloc(a->array, a->size * sizeof(double));
    }
    a->array[a->used++] = e;
#ifdef MULTI_THREADING
    pthread_mutex_unlock(&a->mutex);
#endif
}

void insert_array_at(DArray *a, double e, int pos)
{
#ifdef MULTI_THREADING
    pthread_mutex_lock(&a->mutex);
#endif
    if ((pos+1) > a->used)
    {
        a->used = pos+1;
    }

    while (a->used > a->size)
    {
        a->size *= 2;
        a->array = (double *)realloc(a->array, a->size * sizeof(double));
    }
    a->array[pos] = e;
#ifdef MULTI_THREADING
    pthread_mutex_unlock(&a->mutex);
#endif
}

double get_at(DArray *a, int pos)
{
    return a->array[pos];
}

void free_array(DArray *a)
{
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
#ifdef MULTI_THREADING
    pthread_mutex_destroy(&a->mutex);
#endif
}
