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
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "thread_pool.h"

/* Payload ≤ JOB_INLINE_DATA_SIZE lives inside the job struct itself,
 * avoiding a second malloc per enqueue. Sized to cover every
 * `data_sz` used by current callers (the CPU read_pictures stage
 * submits a 32-byte VmafPicture pair, MCP transports submit 48-byte
 * frame events — 64 bytes gives comfortable headroom). */
#define JOB_INLINE_DATA_SIZE 64

typedef struct VmafThreadPoolJob {
    void (*func)(void *data, void **thread_data);
    void *data;
    char inline_data[JOB_INLINE_DATA_SIZE];
    struct VmafThreadPoolJob *next;
} VmafThreadPoolJob;

typedef struct VmafThreadPoolWorker {
    struct VmafThreadPool *pool;
    void *data;
} VmafThreadPoolWorker;

typedef struct VmafThreadPool {
    struct {
        pthread_mutex_t lock;
        pthread_cond_t empty;
        VmafThreadPoolJob *head, *tail;
    } queue;
    pthread_cond_t working;
    /* n_threads: live worker count; decremented by each runner on exit (lock held).
     * n_workers_created: immutable after pool_create; used by destroy to iterate
     * the workers[] array without racing the decrement in runner. */
    unsigned n_threads;
    unsigned n_workers_created;
    unsigned n_working;
    bool stop;
    VmafThreadPoolWorker *workers;
    void (*thread_data_free)(void *thread_data);
    /* Recycled job objects for reuse, keyed off the pool's lock. */
    VmafThreadPoolJob *free_jobs;
} VmafThreadPool;

static VmafThreadPoolJob *vmaf_thread_pool_fetch_job(VmafThreadPool *pool)
{
    if (!pool)
        return NULL;
    if (!pool->queue.head)
        return NULL;

    VmafThreadPoolJob *job = pool->queue.head;
    if (!job->next) {
        pool->queue.head = NULL;
        pool->queue.tail = NULL;
    } else {
        pool->queue.head = job->next;
    }
    return job;
}

/* Drop a job's external data buffer (if it was heap-allocated rather
 * than living in job->inline_data). Caller owns the job slot itself. */
static void vmaf_thread_pool_job_clear_data(VmafThreadPoolJob *job)
{
    if (job->data && job->data != job->inline_data)
        free(job->data);
    job->data = NULL;
    job->func = NULL;
}

static void vmaf_thread_pool_job_destroy(VmafThreadPoolJob *job)
{
    if (!job)
        return;
    vmaf_thread_pool_job_clear_data(job);
    free(job);
}

/* Push `job` onto the pool's free list for later reuse by enqueue.
 * Caller must hold pool->queue.lock. */
static void vmaf_thread_pool_job_recycle(VmafThreadPool *pool, VmafThreadPoolJob *job)
{
    if (!job)
        return;
    vmaf_thread_pool_job_clear_data(job);
    job->next = pool->free_jobs;
    pool->free_jobs = job;
}

static void *vmaf_thread_pool_runner(void *p)
{
    VmafThreadPoolWorker *worker = p;
    VmafThreadPool *pool = worker->pool;

    for (;;) {
        pthread_mutex_lock(&(pool->queue.lock));
        if (!pool->queue.head && !pool->stop)
            pthread_cond_wait(&(pool->queue.empty), &(pool->queue.lock));
        if (pool->stop)
            break;
        VmafThreadPoolJob *job = vmaf_thread_pool_fetch_job(pool);
        pool->n_working++;
        pthread_mutex_unlock(&(pool->queue.lock));
        if (job) {
            job->func(job->data, &worker->data);
        }
        pthread_mutex_lock(&(pool->queue.lock));
        pool->n_working--;
        if (job)
            vmaf_thread_pool_job_recycle(pool, job);
        if (!pool->stop && pool->n_working == 0 && !pool->queue.head)
            pthread_cond_signal(&(pool->working));
        pthread_mutex_unlock(&(pool->queue.lock));
    }

    if (--(pool->n_threads) == 0)
        pthread_cond_signal(&(pool->working));

    pthread_mutex_unlock(&(pool->queue.lock));
    return NULL;
}

int vmaf_thread_pool_create(VmafThreadPool **pool, VmafThreadPoolConfig cfg)
{
    if (!pool)
        return -EINVAL;
    if (!cfg.n_threads)
        return -EINVAL;

    VmafThreadPool *const p = *pool = malloc(sizeof(*p));
    if (!p)
        return -ENOMEM;
    memset(p, 0, sizeof(*p));
    p->n_threads = cfg.n_threads;
    p->n_workers_created = cfg.n_threads; /* adjusted below on partial failure */
    p->thread_data_free = cfg.thread_data_free;

    p->workers = malloc(sizeof(*p->workers) * cfg.n_threads);
    if (!p->workers) {
        free(p);
        return -ENOMEM;
    }
    memset(p->workers, 0, sizeof(*p->workers) * cfg.n_threads);

    pthread_mutex_init(&(p->queue.lock), NULL);
    pthread_cond_init(&(p->queue.empty), NULL);
    pthread_cond_init(&(p->working), NULL);

    for (unsigned i = 0; i < cfg.n_threads; i++) {
        p->workers[i].pool = p;
        pthread_t thread;
        const int rc = pthread_create(&thread, NULL, vmaf_thread_pool_runner, &p->workers[i]);
        if (rc != 0) {
            /* Thread creation failed (e.g. EAGAIN under process-limit pressure).
             * Adjust n_threads downward so vmaf_thread_pool_wait does not hang
             * waiting for workers that never started.  If no threads at all
             * could be created, tear down and propagate the error. */
            p->n_threads = i;
            p->n_workers_created = i;
            if (i == 0) {
                pthread_mutex_destroy(&(p->queue.lock));
                pthread_cond_destroy(&(p->queue.empty));
                pthread_cond_destroy(&(p->working));
                free(p->workers);
                free(p);
                *pool = NULL;
                return -rc;
            }
            /* At least one thread started — pool is usable at reduced width.
             * Signal existing workers in case they are already waiting. */
            pthread_cond_broadcast(&(p->queue.empty));
            break;
        }
        pthread_detach(thread);
    }

    return 0;
}

int vmaf_thread_pool_enqueue(VmafThreadPool *pool, void (*func)(void *data, void **thread_data),
                             void *data, size_t data_sz)
{
    if (!pool)
        return -EINVAL;
    if (!func)
        return -EINVAL;

    pthread_mutex_lock(&(pool->queue.lock));

    /* Reuse a recycled slot if available, otherwise heap-allocate one.
     * The free list is mutex-protected so fetching and returning stays
     * coherent with the runner thread's recycle on job completion. */
    VmafThreadPoolJob *job = pool->free_jobs;
    if (job) {
        pool->free_jobs = job->next;
    } else {
        job = malloc(sizeof(*job));
        if (!job) {
            pthread_mutex_unlock(&(pool->queue.lock));
            return -ENOMEM;
        }
    }

    memset(job, 0, sizeof(*job));
    job->func = func;
    if (data) {
        if (data_sz <= JOB_INLINE_DATA_SIZE) {
            memcpy(job->inline_data, data, data_sz);
            job->data = job->inline_data;
        } else {
            job->data = malloc(data_sz);
            if (!job->data) {
                /* Return the job slot to the free list; releasing the
                 * heap copy would defeat the job-pool win on retry. */
                job->next = pool->free_jobs;
                pool->free_jobs = job;
                pthread_mutex_unlock(&(pool->queue.lock));
                return -ENOMEM;
            }
            memcpy(job->data, data, data_sz);
        }
    }

    if (!pool->queue.head) {
        pool->queue.head = job;
        pool->queue.tail = pool->queue.head;
    } else {
        pool->queue.tail->next = job;
        pool->queue.tail = job;
    }

    pthread_cond_signal(&(pool->queue.empty));
    pthread_mutex_unlock(&(pool->queue.lock));

    return 0;
}

int vmaf_thread_pool_wait(VmafThreadPool *pool)
{
    if (!pool)
        return -EINVAL;

    pthread_mutex_lock(&(pool->queue.lock));
    while ((!pool->stop && (pool->n_working || pool->queue.head)) ||
           (pool->stop && pool->n_threads))
        pthread_cond_wait(&(pool->working), &(pool->queue.lock));
    pthread_mutex_unlock(&(pool->queue.lock));
    return 0;
}
int vmaf_thread_pool_destroy(VmafThreadPool *pool)
{
    if (!pool)
        return -EINVAL;

    /* n_workers_created is written once at pool_create and never modified
     * afterwards, so it is safe to read without the lock.  Using n_threads
     * here would be a data race: runner threads decrement it under the lock
     * as they exit, which can race with this unsynchronised read. */
    const unsigned n_workers = pool->n_workers_created;

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

    /* Free every slot still on the recycle list. No lock needed —
     * vmaf_thread_pool_wait returns only after all workers exit. */
    job = pool->free_jobs;
    while (job) {
        VmafThreadPoolJob *next_job = job->next;
        free(job);
        job = next_job;
    }
    pool->free_jobs = NULL;

    if (pool->thread_data_free) {
        for (unsigned i = 0; i < n_workers; i++) {
            if (pool->workers[i].data)
                pool->thread_data_free(pool->workers[i].data);
        }
    }
    free(pool->workers);

    pthread_mutex_destroy(&(pool->queue.lock));
    pthread_cond_destroy(&(pool->queue.empty));
    pthread_cond_destroy(&(pool->working));

    free(pool);
    return 0;
}
