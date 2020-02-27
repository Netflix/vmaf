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

#ifndef __VMAF_THREAD_POOL_H__
#define __VMAF_THREAD_POOL_H__

#include <pthread.h>

typedef struct VmafThreadPool VmafThreadPool;

int vmaf_thread_pool_create(VmafThreadPool **tpool, unsigned n_threads);

int vmaf_thread_pool_enqueue(VmafThreadPool *pool, void (*func)(void *data),
                             void *data, size_t data_sz);

int vmaf_thread_pool_wait(VmafThreadPool *pool);

int vmaf_thread_pool_destroy(VmafThreadPool *tpool);

#endif /* __VMAF_THREAD_POOL_H__ */
