/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA dispatch_strategy stub. Today every extractor returns
 *  DIRECT; this TU exists to expose the registry-aware decision
 *  surface so future graph-capture work for ADM (16 dispatches/
 *  frame) can land without touching the registration sites.
 *  See ADR-0181.
 */
#include "dispatch_strategy.h"
#include "../gpu_dispatch_parse.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/*
 * Cache the VMAF_CUDA_DISPATCH environment variable once, protected by
 * pthread_once, to avoid calling getenv() from multiple threads
 * simultaneously (concurrency-mt-unsafe per POSIX.1-2008 §2.2.2 if another
 * thread calls setenv/putenv/unsetenv concurrently).
 *
 * Contract: callers must set VMAF_CUDA_DISPATCH before the first CUDA frame
 * is submitted (i.e., before any call to vmaf_read_pictures on a CUDA
 * context). The snapshot is permanent; later setenv calls are not observed.
 *
 * NULL means the variable was unset at snapshot time.
 */
#ifdef _WIN32
static INIT_ONCE g_env_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t g_env_once = PTHREAD_ONCE_INIT;
#endif
static const char *g_env_disp = NULL;

#ifdef _WIN32
static BOOL CALLBACK cache_env_dispatch_w32(PINIT_ONCE once, PVOID param, PVOID *ctx)
{
    (void)once;
    (void)param;
    (void)ctx;
    const char *val = getenv("VMAF_CUDA_DISPATCH");
    if (val)
        g_env_disp = _strdup(val);
    return TRUE;
}
#endif

static void cache_env_dispatch(void)
{
    /* Serialised by pthread_once — only one thread reaches this function.
     * The remaining concurrency-mt-unsafe risk (concurrent setenv from another
     * thread racing the single getenv here) is caller-contract per
     * POSIX.1-2008 §2.2.2: "Conforming multi-threaded applications shall not
     * use setenv, unsetenv, or putenv while a call to getenv is in progress."
     * NOLINT(concurrency-mt-unsafe) — call is serialised by pthread_once;
     * simultaneous setenv is a caller-contract violation, not a library bug. */
    const char *val = getenv("VMAF_CUDA_DISPATCH"); // NOLINT(concurrency-mt-unsafe)
    if (val)
        g_env_disp = strdup(val); /* stable copy; outlives any subsequent setenv */
}

/* Strategy name table — index matches VmafCudaDispatchStrategy enum values:
 *   0 → VMAF_CUDA_DISPATCH_DIRECT
 *   1 → VMAF_CUDA_DISPATCH_GRAPH_CAPTURE
 * See ADR-0483. */
static const char *const k_cuda_strategy_names[] = {
    "direct", /* VMAF_CUDA_DISPATCH_DIRECT        = 0 */
    "graph",  /* VMAF_CUDA_DISPATCH_GRAPH_CAPTURE = 1 */
    NULL,
};

VmafCudaDispatchStrategy vmaf_cuda_select_strategy(const char *feature_name,
                                                   const VmafFeatureCharacteristics *chars,
                                                   unsigned frame_w, unsigned frame_h)
{
    (void)chars;
    (void)frame_w;
    (void)frame_h;

    /* Read VMAF_CUDA_DISPATCH at most once across all threads. POSIX uses
     * pthread_once for the acquire fence; Windows uses InitOnceExecuteOnce
     * via the dedicated `_w32` callback (pthread_once isn't available on
     * MSVC, so the MSVC+CUDA build can't link the POSIX variant). */
#ifdef _WIN32
    (void)InitOnceExecuteOnce(&g_env_once, cache_env_dispatch_w32, NULL, NULL);
#else
    (void)pthread_once(&g_env_once, cache_env_dispatch);
#endif

    int idx = (int)VMAF_CUDA_DISPATCH_DIRECT;
    if (vmaf_gpu_dispatch_parse_env(g_env_disp, feature_name, k_cuda_strategy_names, &idx))
        return (VmafCudaDispatchStrategy)idx;

    /* Stub default — DIRECT for every feature. CUDA graph capture
     * is a follow-up PR (see ADR-0181 § Consequences / follow-ups). */
    return VMAF_CUDA_DISPATCH_DIRECT;
}
