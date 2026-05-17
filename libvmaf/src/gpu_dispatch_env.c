/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Implementation of the once-snapshotted GPU dispatch env helper.
 *  See gpu_dispatch_env.h for the contract and ADR-0461 for rationale.
 *
 *  Design: a fixed-size table of (var_name, value) pairs protected by
 *  a single mutex.  On the first call for a given var_name the entry is
 *  populated under the lock; subsequent calls read the cached value
 *  lock-free after a simple pointer match.  The table holds at most
 *  GPU_DISPATCH_ENV_TABLE_CAP entries; 8 is generous for the current
 *  3 backends (CUDA, Vulkan, SYCL) + anticipated Metal.
 */
#include "gpu_dispatch_env.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#define GPU_DISPATCH_ENV_TABLE_CAP 8U

/* One row per distinct var_name. */
typedef struct {
    const char *var_name; /* points to the caller's string literal; NULL = free */
    const char *value;    /* strdup copy; NULL if var was unset at snapshot time */
} EnvRow;

static EnvRow g_rows[GPU_DISPATCH_ENV_TABLE_CAP];

#ifdef _WIN32
static CRITICAL_SECTION g_lock;
static INIT_ONCE g_lock_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK init_lock_cb(PINIT_ONCE once, PVOID param, PVOID *ctx)
{
    (void)once;
    (void)param;
    (void)ctx;
    InitializeCriticalSection(&g_lock);
    return TRUE;
}

static void lock_acquire(void)
{
    (void)InitOnceExecuteOnce(&g_lock_once, init_lock_cb, NULL, NULL);
    EnterCriticalSection(&g_lock);
}
static void lock_release(void)
{
    LeaveCriticalSection(&g_lock);
}
#else
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void lock_acquire(void)
{
    (void)pthread_mutex_lock(&g_lock);
}
static void lock_release(void)
{
    (void)pthread_mutex_unlock(&g_lock);
}
#endif

const char *vmaf_gpu_dispatch_env_get(const char *var_name)
{
    if (!var_name)
        return NULL;

    /* Fast path: look for an already-populated row without locking.
     * The pointer comparison on var_name is safe because callers pass
     * string literals that outlive the process. */
    for (unsigned i = 0U; i < GPU_DISPATCH_ENV_TABLE_CAP; i++) {
        if (g_rows[i].var_name == var_name)
            return g_rows[i].value;
    }

    /* Also check by string equality in case two TUs use different
     * pointer addresses for the same variable name. */
    for (unsigned i = 0U; i < GPU_DISPATCH_ENV_TABLE_CAP; i++) {
        if (g_rows[i].var_name && strcmp(g_rows[i].var_name, var_name) == 0)
            return g_rows[i].value;
    }

    /* Slow path: snapshot the variable under the lock. */
    lock_acquire();

    /* Re-check under lock to guard against a concurrent insert. */
    for (unsigned i = 0U; i < GPU_DISPATCH_ENV_TABLE_CAP; i++) {
        if (g_rows[i].var_name && strcmp(g_rows[i].var_name, var_name) == 0) {
            const char *cached = g_rows[i].value;
            lock_release();
            return cached;
        }
    }

    /* Find a free row. */
    EnvRow *row = NULL;
    for (unsigned i = 0U; i < GPU_DISPATCH_ENV_TABLE_CAP; i++) {
        if (!g_rows[i].var_name) {
            row = &g_rows[i];
            break;
        }
    }

    if (!row) {
        /* Table exhausted — fall back to a raw getenv.  Should never
         * happen in production (8 slots, at most 4 backends). */
        lock_release();
        /* NOLINT(concurrency-mt-unsafe): caller-contract bans concurrent
         * setenv before the first GPU frame; see ADR-0461. */
        return getenv(var_name); /* NOLINT(concurrency-mt-unsafe) */
    }

    /* Snapshot under the lock.  The caller-contract (ADR-0461) requires
     * that no other thread calls setenv("VMAF_*") concurrently with
     * getenv here; the lock serialises only multiple vmaf_gpu_dispatch_env_get
     * callers, not hypothetical concurrent setenv from user code. */
    /* NOLINT(concurrency-mt-unsafe): see above. */
    const char *val = getenv(var_name); /* NOLINT(concurrency-mt-unsafe) */
    row->var_name = var_name;
    if (val)
        row->value = strdup(val);
    lock_release();
    return row->value;
}
