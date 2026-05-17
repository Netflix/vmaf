# ADR-0488: Shared once-snapshot helper for GPU dispatch env variables

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: gpu, cuda, vulkan, sycl, dispatch, threading, refactor, fork-local

## Context

Every GPU backend's `dispatch_strategy` module reads an env variable
(`VMAF_CUDA_DISPATCH`, `VMAF_VULKAN_DISPATCH`, `VMAF_SYCL_DISPATCH`) to allow
per-feature strategy overrides at runtime. The canonical way to call `getenv()`
from a multi-threaded library is to snapshot the value once before any threads
are launched and cache it — calling `getenv()` from concurrent threads is
POSIX.1-2008 §2.2.2 `concurrency-mt-unsafe` if another thread may be calling
`setenv()`/`putenv()` simultaneously.

The CUDA backend (`libvmaf/src/cuda/dispatch_strategy.c`, ADR-0181) already
used a bespoke `pthread_once` / `InitOnceExecuteOnce` block (~35 LOC) to
snapshot `VMAF_CUDA_DISPATCH`. The Vulkan and SYCL backends called `getenv()`
directly on every `select_strategy()` invocation, leaving the same latent
thread-safety exposure that the CUDA fix was meant to address. The dedup audit
(`dedup-audit-gpu-templates-2026-05-16.md`) identified this as opportunity 5:
extract the once-init pattern into a shared helper so all three backends adopt
the same posture without duplicating the boilerplate.

## Decision

Introduce `libvmaf/src/gpu_dispatch_env.{h,c}` with a single function:

```c
const char *vmaf_gpu_dispatch_env_get(const char *var_name);
```

The implementation uses a fixed-size table (8 slots, generously covering the
current 3 backends plus an anticipated 4th) protected by a single mutex. On
the first call for a given `var_name` the value is snapshotted under the lock
and cached permanently; subsequent calls return the cached pointer without
acquiring the lock (fast path: pointer equality; slow path: `strcmp`). The
file is compiled unconditionally into `libvmaf_sources` so every GPU backend
can include it without a Meson guard.

All three backends are updated to call `vmaf_gpu_dispatch_env_get()` instead
of bare `getenv()`. The CUDA backend's bespoke ~35-LOC `pthread_once` block is
removed; the shared helper subsumes it.

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Keep per-backend bespoke `pthread_once` | Rejected: Vulkan and SYCL still had no protection; CUDA's block would have to be cloned into each new backend. |
| X-macro / `#include`-with-define approach | Rejected: harder to lint-clean; the function call approach integrates naturally with clang-tidy. |
| Single global `pthread_once` per TU (replicate CUDA pattern in Vulkan/SYCL) | Rejected: that is exactly the cloning this ADR eliminates. |
| Lazy `getenv()` on every call (status quo for Vulkan/SYCL) | Rejected: latent thread-safety exposure per POSIX; the CUDA fix showed intent to address this. |

## Consequences

**Positive**
- The `concurrency-mt-unsafe` clang-tidy finding is closed for Vulkan and SYCL,
  matching the posture already established for CUDA.
- The CUDA-specific ~35-LOC bespoke once-init block is removed.
- Future backends (Metal, HIP dispatch override) adopt the pattern with a
  one-line call; no per-backend boilerplate needed.

**Neutral**
- `gpu_dispatch_env.c` is compiled on every build, including CPU-only builds
  where no GPU backend is active. The file is ~130 LOC and adds negligible
  compile time.
- The once-snapshot contract (set env before first GPU frame) was already
  established by ADR-0181; this ADR extends it to Vulkan and SYCL.

**No rebase impact** — this change is entirely within the fork-local
dispatch_strategy layer; no upstream Netflix code paths are modified.

## References

- Dedup audit: `.workingdir/dedup-audit-gpu-templates-2026-05-16.md` opportunity 5
- ADR-0181: `dispatch_strategy` module design (per-backend select_strategy registry)
- POSIX.1-2008 §2.2.2: `getenv` concurrency-mt-unsafe annotation
