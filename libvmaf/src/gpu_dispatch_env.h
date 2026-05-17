/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Thread-safe, once-only snapshot of GPU dispatch env variables.
 *
 *  Problem: every GPU backend's dispatch_strategy module calls
 *  getenv("VMAF_<BACKEND>_DISPATCH") on each select_strategy()
 *  invocation. The POSIX.1-2008 §2.2.2 concurrency-mt-unsafe
 *  annotation on getenv means any concurrent setenv/putenv from
 *  another thread races with these calls.
 *
 *  CUDA dispatch_strategy.c (ADR-0181) already used pthread_once
 *  to snapshot the variable at first call. This header extracts
 *  that pattern into one reusable helper so Vulkan and SYCL (and
 *  any future backend) can adopt the same posture without copying
 *  the boilerplate. See ADR-0461.
 *
 *  Contract:
 *    • Call vmaf_gpu_dispatch_env_get("VMAF_FOO_DISPATCH") from any
 *      thread. The first caller snapshots the variable; subsequent
 *      callers receive the cached copy.
 *    • Set the env variable before the first GPU frame is submitted
 *      (i.e., before vmaf_read_pictures on a GPU context). Later
 *      setenv() calls are not observed — by design.
 *    • NULL is returned if the variable was unset at snapshot time.
 *    • Distinct var_name strings produce independent snapshots.
 *    • On Windows the POSIX pthread_once equivalent is
 *      InitOnceExecuteOnce; the shim is transparent to callers.
 */
#ifndef LIBVMAF_GPU_DISPATCH_ENV_H_
#define LIBVMAF_GPU_DISPATCH_ENV_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return a once-snapshotted copy of the environment variable
 * identified by @var_name.
 *
 * Thread-safe: internally uses pthread_once (POSIX) or
 * InitOnceExecuteOnce (Windows) so the getenv() call is serialised
 * across all threads.
 *
 * @param var_name  Name of the environment variable to snapshot
 *                  (e.g. "VMAF_VULKAN_DISPATCH"). Must be a string
 *                  literal or otherwise outlive the process; the
 *                  function uses the pointer as a lookup key.
 * @return          The snapshotted value, or NULL if the variable
 *                  was unset. The returned pointer is stable for the
 *                  process lifetime.
 */
const char *vmaf_gpu_dispatch_env_get(const char *var_name);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_GPU_DISPATCH_ENV_H_ */
