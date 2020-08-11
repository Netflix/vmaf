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

#include <stdint.h>

#include "test.h"
#include "thread_pool.h"

static void fn_a(void *data)
{
    (void) data;
    printf("thread ");
}

static void fn_b(void *data)
{
    (void) data;
    printf("pool ");
}

static void fn_c(void *data)
{
    (void) data;
    printf("test ");
}

typedef struct Fps {
    unsigned num, den;
} Fps;

static void fn_d(void *data)
{
    Fps *fps = data;
    printf("FPS: %d/%d ", fps->num, fps->den);
}

static char *test_thread_pool_create_enqueue_wait_and_destroy()
{
    int err;

    VmafThreadPool *pool;
    unsigned n_threads = 8;

    err = vmaf_thread_pool_create(&pool, n_threads);
    mu_assert("problem during vmaf_thread_pool_init", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_a, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_b, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_c, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    Fps fps = { 24, 1 };
    err = vmaf_thread_pool_enqueue(pool, fn_d, &fps, sizeof(fps));
    mu_assert("problem during vmaf_thread_pool_enqueue with data", !err);
    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem during vmaf_thread_pool_wait", !err);
    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem during vmaf_thread_pool_destroy", !err);

    printf("\n");

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_thread_pool_create_enqueue_wait_and_destroy);
    return NULL;
}
