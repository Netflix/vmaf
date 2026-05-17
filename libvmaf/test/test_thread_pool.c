/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include <errno.h>
#include <stdint.h>

#include "test.h"
#include "thread_pool.h"

static void fn_a(void *data, void **thread_data)
{
    (void)data;
    (void)thread_data;
    printf("thread ");
}

static void fn_b(void *data, void **thread_data)
{
    (void)data;
    (void)thread_data;
    printf("pool ");
}

static void fn_c(void *data, void **thread_data)
{
    (void)data;
    (void)thread_data;
    printf("test ");
}

typedef struct Fps {
    unsigned num, den;
} Fps;

static void fn_d(void *data, void **thread_data)
{
    (void)thread_data;
    Fps *fps = data;
    printf("FPS: %d/%d ", fps->num, fps->den);
}

static char *test_thread_pool_create_enqueue_wait_and_destroy()
{
    int err;

    VmafThreadPool *pool;
    VmafThreadPoolConfig cfg = {.n_threads = 8};

    err = vmaf_thread_pool_create(&pool, cfg);
    mu_assert("problem during vmaf_thread_pool_init", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_a, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_b, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    err = vmaf_thread_pool_enqueue(pool, fn_c, NULL, 0);
    mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    Fps fps = {24, 1};
    err = vmaf_thread_pool_enqueue(pool, fn_d, &fps, sizeof(fps));
    mu_assert("problem during vmaf_thread_pool_enqueue with data", !err);
    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem during vmaf_thread_pool_wait", !err);
    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem during vmaf_thread_pool_destroy", !err);

    printf("\n");

    return NULL;
}

/*
 * Regression test for audit finding #7: vmaf_thread_pool_create must
 * propagate errors from pthread_mutex_init / pthread_cond_init rather
 * than silently ignoring them (which would leave *pool pointing at a
 * partially-initialised struct and cause UB on the subsequent lock).
 *
 * We cannot inject ENOMEM from pthread_mutex_init in a pure unit test
 * without a libc wrapper; this test therefore covers the adjacent
 * failure paths that are directly triggerable:
 *
 *   - NULL pool pointer must return -EINVAL and not crash.
 *   - Zero n_threads must return -EINVAL.
 *   - Normal create + destroy must still work (ensures the checked-init
 *     path does not regress the happy path).
 */
static char *test_thread_pool_create_guards()
{
    int err;

    /* NULL pool pointer. */
    err = vmaf_thread_pool_create(NULL, (VmafThreadPoolConfig){.n_threads = 1});
    mu_assert("NULL pool arg must return -EINVAL", err == -EINVAL);

    /* Zero thread count. */
    VmafThreadPool *pool = NULL;
    err = vmaf_thread_pool_create(&pool, (VmafThreadPoolConfig){.n_threads = 0});
    mu_assert("zero n_threads must return -EINVAL", err == -EINVAL);
    mu_assert("pool must remain NULL on error", pool == NULL);

    /* Happy path: single thread, create and destroy. */
    err = vmaf_thread_pool_create(&pool, (VmafThreadPoolConfig){.n_threads = 1});
    mu_assert("single-thread create must succeed", !err);
    mu_assert("pool must be non-NULL on success", pool != NULL);
    err = vmaf_thread_pool_destroy(pool);
    mu_assert("destroy must succeed", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_thread_pool_create_enqueue_wait_and_destroy);
    mu_run_test(test_thread_pool_create_guards);
    return NULL;
}
