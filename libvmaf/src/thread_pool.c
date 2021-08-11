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

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct VmafThreadPoolJob {
    void (*func)(void *data);
    void *data;
    struct VmafThreadPoolJob *next;
} VmafThreadPoolJob;

typedef struct VmafTreadPool {
    struct {
        pthread_mutex_t lock;
        pthread_cond_t empty;
        VmafThreadPoolJob *head, *tail;
    } queue;
    pthread_cond_t working;
    unsigned n_threads;
    unsigned n_working;
    bool stop;
} VmafThreadPool;

static VmafThreadPoolJob *vmaf_thread_pool_fetch_job(VmafThreadPool *pool)
{
    if (!pool) return NULL;
    if (!pool->queue.head) return NULL;

    VmafThreadPoolJob *job = pool->queue.head;
    if (!job->next) {
        pool->queue.head = NULL;
        pool->queue.tail = NULL;
    } else {
        pool->queue.head = job->next;
    }
    return job;
}

static void vmaf_thread_pool_job_destroy(VmafThreadPoolJob *job)
{
    if (!job) return;
    if (job->data) free(job->data);
    free(job);
}

static void *vmaf_thread_pool_runner(void *p)
{
    VmafThreadPool *pool = p;

    for (;;) {
        pthread_mutex_lock(&(pool->queue.lock));
        if (!pool->queue.head && !pool->stop)
            pthread_cond_wait(&(pool->queue.empty), &(pool->queue.lock));
        if (pool->stop) break;
        VmafThreadPoolJob *job = vmaf_thread_pool_fetch_job(pool);
        pool->n_working++;
        pthread_mutex_unlock(&(pool->queue.lock));
        if (job) {
            job->func(job->data);
            vmaf_thread_pool_job_destroy(job);
        }
        pthread_mutex_lock(&(pool->queue.lock));
        pool->n_working--;
        if (!pool->stop && pool->n_working == 0 && !pool->queue.head)
            pthread_cond_signal(&(pool->working));
        pthread_mutex_unlock(&(pool->queue.lock));
    }

    if (--(pool->n_threads) == 0)
        pthread_cond_signal(&(pool->working));

    pthread_mutex_unlock(&(pool->queue.lock));
    return NULL;
}

int vmaf_thread_pool_create(VmafThreadPool **pool, unsigned n_threads)
{
    if (!pool) return -EINVAL;
    if (!n_threads) return -EINVAL;

    VmafThreadPool *const p = *pool = malloc(sizeof(*p));
    if (!p) return -ENOMEM;
    memset(p, 0, sizeof(*p));
    p->n_threads = n_threads;

    pthread_mutex_init(&(p->queue.lock), NULL);
    pthread_cond_init(&(p->queue.empty), NULL);
    pthread_cond_init(&(p->working), NULL);

    for (unsigned i = 0; i < n_threads; i++) {
        pthread_t thread;
        pthread_create(&thread, NULL, vmaf_thread_pool_runner, p);
        pthread_detach(thread);
    }

    return 0;
}

int vmaf_thread_pool_enqueue(VmafThreadPool *pool, void (*func)(void *data),
                             void *data, size_t data_sz)
{
    if (!pool) return -EINVAL;
    if (!func) return -EINVAL;

    VmafThreadPoolJob *job = malloc(sizeof(*job));
    if (!job) return -ENOMEM;
    memset(job, 0, sizeof(*job));
    job->func = func;
    if (data) {
        job->data = malloc(data_sz);
        if (!job->data) goto free_job;
        memcpy(job->data, data, data_sz);
    }

    pthread_mutex_lock(&(pool->queue.lock));

    if (!pool->queue.head) {
        pool->queue.head = job;
        pool->queue.tail = pool->queue.head;
    } else {
        pool->queue.tail->next = job;
        pool->queue.tail = job;
    }

    pthread_cond_broadcast(&(pool->queue.empty));
    pthread_mutex_unlock(&(pool->queue.lock));

    return 0;

free_job:
    free(job);
    return -ENOMEM;
}

int vmaf_thread_pool_wait(VmafThreadPool *pool)
{
    if (!pool) return -EINVAL;

    pthread_mutex_lock(&(pool->queue.lock));
    while((!pool->stop && (pool->n_working || pool->queue.head)) ||
          (pool->stop && pool->n_threads))
        pthread_cond_wait(&(pool->working), &(pool->queue.lock));
    pthread_mutex_unlock(&(pool->queue.lock));
    return 0;
}
int vmaf_thread_pool_destroy(VmafThreadPool *pool)
{
    if (!pool) return -EINVAL;
    pthread_mutex_lock(&(pool->queue.lock));

    VmafThreadPoolJob *job = pool->queue.head;
    while (job) {
        VmafThreadPoolJob *next_job = job->next;
        vmaf_thread_pool_job_destroy(job);
        job = next_job;
    }

    pool->stop = true;
    pthread_cond_broadcast(&(pool->queue.empty));
    pthread_mutex_unlock(&(pool->queue.lock));
    vmaf_thread_pool_wait(pool);
    pthread_mutex_destroy(&(pool->queue.lock));
    pthread_cond_destroy(&(pool->queue.empty));
    pthread_cond_destroy(&(pool->working));

    free(pool);
    return 0;
}
