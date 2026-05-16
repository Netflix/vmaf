**fix(cuda):** Replace bare `FIXME` and `TODO` markers in `libvmaf.c` CUDA paths
with structured `DEFERRED` comments that explain why the refactor is not yet done
(picture-callback async teardown API, host-pool perf profiling) and what would
unblock each item.  No functional change.
