/**
 *
 *  Copyright 2026 Lusoris
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
#include <pthread.h>
#include <stdlib.h>

#include "test.h"

static int fail_create_at = -1, create_calls;
static int create_failed, invalid_detaches;
static int fail_malloc_at = -1, malloc_calls;

static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                                void *(*start)(void*), void *arg)
{
    if (create_calls++ == fail_create_at) {
        create_failed = 1;
        return EAGAIN;
    }
    return pthread_create(thread, attr, start, arg);
}

static int test_pthread_detach(pthread_t thread)
{
    /* A negative control must not detach the uninitialized pthread_t that
     * the old implementation passed here after a failed pthread_create. */
    if (create_failed) {
        invalid_detaches++;
        create_failed = 0;
        return 0;
    }
    return pthread_detach(thread);
}

static void *test_malloc(size_t size)
{
    return malloc_calls++ == fail_malloc_at ? NULL : malloc(size);
}

#define pthread_create test_pthread_create
#define pthread_detach test_pthread_detach
#define malloc test_malloc
#include "thread_pool.c"
#undef malloc
#undef pthread_detach
#undef pthread_create

static void mark_done(void *data, void **thread_data)
{
    (void)thread_data;
    int **done = data;
    **done = 1;
}

static char *check_create_failure(int failure_index)
{
    VmafThreadPool *pool = NULL;
    VmafThreadPoolConfig cfg = { .n_threads = 3 };
    create_calls = create_failed = invalid_detaches = 0;
    fail_create_at = failure_index;
    int err = vmaf_thread_pool_create(&pool, cfg);
    fail_create_at = -1;
    create_failed = 0;
    mu_assert("worker-creation failure not returned", err == -EAGAIN);
    mu_assert("failed creation returned a pool", !pool);
    mu_assert("detached a thread that was not created", !invalid_detaches);
    mu_assert("creation continued after failure", create_calls == failure_index + 1);

    cfg.n_threads = 1;
    mu_assert("retry creation failed", !vmaf_thread_pool_create(&pool, cfg));
    int completed = 0, *done = &completed;
    mu_assert("retry enqueue failed",
              !vmaf_thread_pool_enqueue(pool, mark_done, &done, sizeof(done)));
    mu_assert("retry wait failed", !vmaf_thread_pool_wait(pool));
    mu_assert("retry worker did not run", completed == 1);
    mu_assert("retry destroy failed", !vmaf_thread_pool_destroy(pool));
    return NULL;
}

static char *test_first_worker_failure(void)
{
    return check_create_failure(0);
}

static char *test_later_worker_failure(void)
{
    return check_create_failure(1);
}

static char *test_worker_array_allocation_failure(void)
{
    VmafThreadPool *pool = NULL;
    VmafThreadPoolConfig cfg = { .n_threads = 1 };
    malloc_calls = 0;
    fail_malloc_at = 1; /* The pool exists; its worker-array allocation fails. */
    int err = vmaf_thread_pool_create(&pool, cfg);
    fail_malloc_at = -1;
    mu_assert("worker allocation failure not returned", err == -ENOMEM);
    mu_assert("worker allocation failure left a dangling pool pointer", !pool);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_first_worker_failure);
    mu_run_test(test_later_worker_failure);
    mu_run_test(test_worker_array_allocation_failure);
    return NULL;
}
