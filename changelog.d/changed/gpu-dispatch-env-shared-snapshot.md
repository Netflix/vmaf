### Changed

- **GPU dispatch env thread safety** (ADR-0461): introduce
  `libvmaf/src/gpu_dispatch_env.{h,c}` — a single once-snapshot helper for
  `VMAF_*_DISPATCH` env variables. The Vulkan and SYCL `dispatch_strategy`
  modules now snapshot their respective env variables once at first call (via a
  mutex-protected table) instead of calling `getenv()` on every
  `select_strategy()` invocation. Closes the `concurrency-mt-unsafe` POSIX
  exposure that the CUDA backend already fixed with a bespoke `pthread_once`
  block; the CUDA block is replaced by the shared helper, removing ~35 LOC of
  duplicated boilerplate.
