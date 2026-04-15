---
name: profile-hotpath
description: Profile a hot path (feature + backend) using the appropriate profiler (perf / ncu / Vtune / rocprof), produce flamegraph + top-N hot functions, and suggest concrete optimizations. Delegates to the perf-profiler agent.
---

# /profile-hotpath

## Invocation

```
/profile-hotpath <backend> <feature> [--resolution=576|640|720|1080|4k] [--frames=120]
```

## Behavior

Delegates the profiling + interpretation work to the `perf-profiler` agent (see
`.claude/agents/perf-profiler.md`), which:

1. Builds (if needed) with `--config=relwithdebinfo` for symbolicated profiles.
2. Selects the profiler for the backend.
3. Runs `build/tools/vmaf_bench` with fixed seed + frame count.
4. Produces a flamegraph under `build/profiles/<date>/flame.svg`.
5. Emits a top-10 hot-function list + recommendations.

## Notes

- The `perf-profiler` agent owns the actual interpretation — this skill just sets up
  the environment and invokes it.
- If the profiler isn't installed (`ncu` missing for CUDA, `vtune` for SYCL), the skill
  degrades to `perf record` as a fallback, with a clear note in the report.
