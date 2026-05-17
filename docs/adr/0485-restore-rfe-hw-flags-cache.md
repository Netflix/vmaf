# ADR-0485: Restore `rfe_hw_flags` per-frame bitmask cache after PR #1067 clobber

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `cuda`, `perf`, `bug`, `libvmaf`

## Context

PR #1056 (commit `86e2498a8`) introduced a lazy dirty-flag cache (`rfe_hw_flags_cache` /
`rfe_hw_flags_dirty`) on `VmafContext` to eliminate the O(n_extractors) linear scan of
`registered_feature_extractors` that `rfe_hw_flags()` performs on every call to
`vmaf_read_pictures`. The optimisation was tracked as perf-audit-pipeline-2026-05-16
finding F2-B.

PR #1067 (`c2a3c7e0f`) was a large squash-merge that included unrelated refactor work; it
rebased against a tree that already contained the struct fields and three write/read sites,
but the merge silently dropped all five of those lines, reverting `libvmaf.c` to the
pre-#1056 state. The regression restores the per-frame O(n) scan.

## Decision

Restore the five edit sites from `86e2498a8` verbatim:

1. `VmafContext` struct — two new fields under `#ifdef HAVE_CUDA`:
   `unsigned rfe_hw_flags_cache` and `bool rfe_hw_flags_dirty`.
2. `vmaf_init` — initialise `rfe_hw_flags_dirty = true` so the first frame triggers
   a recompute.
3. `vmaf_use_feature` — set `rfe_hw_flags_dirty = true` after a new extractor is
   appended, so the cache is invalidated correctly when the extractor set changes
   after frame-loop start.
4. `vmaf_read_pictures` — replace the unconditional `rfe_hw_flags()` call with a
   dirty-flag check; only recompute when dirty, then read from the cache.

All sites are guarded by `#ifdef HAVE_CUDA`; CPU-only and SYCL builds are unaffected.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Re-apply #1056 as a cherry-pick | Exact replay | Cherry-pick would conflict on the post-#1067 tree | Chosen as manual patch (identical result) |
| Leave as-is (accept regression) | No code change | Per-frame O(n) scan degrades throughput at n_extractors > ~4 | Perf regression, not acceptable |

## Consequences

- **Positive**: Restores the F2-B throughput gain; `rfe_hw_flags()` called once per
  unique extractor-set epoch instead of once per frame.
- **Negative**: None — purely additive cache with correct invalidation.
- **Neutral / follow-ups**: Consider a CI guard (e.g. a grep-based check) that
  verifies the cache fields are present in `libvmaf.c` to prevent silent future
  regressions.

## References

- PR #1056 (introduces the cache): commit `86e2498a8`
- PR #1067 (clobbers the cache): commit `c2a3c7e0f`
- Source: `req` — "PR #1067 clobbered #1056's F2-B perf optimization … restore the cache fields, the dirty-flag write paths, and the read-path that uses the cache instead of recomputing every frame."
