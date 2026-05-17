## changed

- `float_vif` CPU extractor: hoist 10-plane VIF scratch buffer
  (`VIF_SCRATCH_BUF_CNT × plane_sz`) from per-frame `aligned_malloc` /
  `aligned_free` to `VifState` init/close lifecycle (ADR-0452). Eliminates
  ~79 MB/frame of allocator traffic at 1080p; ~6% wall-clock improvement
  per frame in single-threaded mode; super-linear savings in multi-threaded
  mode due to eliminated arena lock contention. Zero arithmetic change —
  bit-exact with pre-change output. `compute_vif` signature gains a
  `float *data_buf` parameter; callers are responsible for the allocation.
