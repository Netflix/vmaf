# Restore `rfe_hw_flags` per-frame bitmask cache (F2-B perf regression)

PR #1067 accidentally dropped the five edit sites from PR #1056 (`86e2498a8`) that
cached the `rfe_hw_flags()` result across frames. The per-frame O(n_extractors) linear
scan in `vmaf_read_pictures` was therefore re-introduced. This fix restores:

- `rfe_hw_flags_cache` and `rfe_hw_flags_dirty` fields on `VmafContext` (CUDA build only)
- dirty-flag initialisation in `vmaf_init`
- cache invalidation in `vmaf_use_feature`
- lazy-recompute read path in `vmaf_read_pictures`

ADR-0485.
