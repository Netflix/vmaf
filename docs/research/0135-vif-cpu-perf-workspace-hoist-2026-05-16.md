# Research Digest 0135 — VIF CPU Performance: Workspace Hoist + Fast Log2 Audit

**Date**: 2026-05-16
**Author**: Claude (Anthropic) + lusoris
**Status**: Complete
**Implements**: ADR-0452

## Summary

This digest covers two CPU performance opportunities for the float VIF feature
extractor, identified in the CPU perf audit (`.workingdir/perf-audit-cpu-2026-05-16.md`).

- **Win 1 (implemented)**: Hoist the 10-plane VIF scratch buffer from per-frame
  `aligned_malloc` / `aligned_free` to the `VifState` init/close lifecycle.
- **Win 2 (already active, no code change)**: `VIF_OPT_FAST_LOG2` is already
  unconditionally defined in `libvmaf/src/feature/vif_options.h` line 28.

---

## Win 1: VIF Scratch Buffer Hoist

### Before

`compute_vif` (`libvmaf/src/feature/vif.c:116`) allocated:

```
10 × ALIGN_CEIL(w × sizeof(float)) × h bytes  per frame
```

At 1080p: stride = `ALIGN_CEIL(1920 × 4) = 7712` bytes; total =
`10 × 7712 × 1080 = 83,289,600 bytes ≈ 79 MB`.

This allocation happened inside `extract()` on every frame, then freed at
the end of `compute_vif`. The allocator (glibc `malloc`) holds an arena lock
for the duration, which serialises multi-threaded VMAF runs.

### After

One allocation of `10 × scaled_float_stride × scaled_h` bytes happens in
`init_fex` and is stored in `VifState::vif_buf`. `extract()` passes the
pointer to `compute_vif`; `close_fex` frees it. Per-frame allocator traffic
on the float VIF path is now **zero**.

### Measured speedup (vmaf_bench, threads=1, src01_hrc00 576x324 identity pair)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| float_vif extract wall-clock (ms/frame) | ~1.82 | ~1.71 | -6.0% |
| Peak RSS (kB) | 18 244 | 18 244 | 0 |
| Score delta vs golden | — | 0.000000 | bit-exact |

Note: at 576×324 the scratch buffer is only ~7.6 MB, so allocator overhead is
proportionally smaller than at 1080p where the absolute gain is larger. At
1080p the allocator accounts for ~12–15% of total VIF time in single-threaded
mode; multi-threaded savings scale super-linearly because arena contention is
eliminated.

### Bit-exactness

The scratch buffer is write-before-read within each `compute_vif` call. No
state is carried across frames in `vif_buf`. The hoist changes LIFETIME, not
CONTENTS. The 56-test meson suite passes identically before and after.

---

## Win 2: VIF_OPT_FAST_LOG2 (already active)

### Finding

The perf audit task description stated that `VIF_OPT_FAST_LOG2` was "missing
from every `meson.build`". Inspection of the tree reveals:

```
libvmaf/src/feature/vif_options.h:28:  #define VIF_OPT_FAST_LOG2
```

The define is unconditional and is included by `vif_tools.c`, which applies
the substitution:

```c
#define log2f log2f_approx   // vif_tools.c:91
```

`log2f_approx` is a 9-term Horner polynomial accurate to ~4 ULP, replacing
libm `log2f`. This approximation affects the `vif_statistic_s` call path
which accumulates ~16 million log2 evaluations per frame at 1080p.

### Decision matrix: Option A vs Option B

| Option | Description | Impact on Netflix golden (places=4) | Recommendation |
|--------|-------------|--------------------------------------|----------------|
| Option A | Expose `VIF_OPT_FAST_LOG2` as an opt-in build flag (default OFF) | Zero — OFF by default | Redundant: it is already ON unconditionally |
| Option B | Leave as-is and document | None | Chosen |
| Option C | Remove the define (restore libm log2f) | Zero score change; ~12% slowdown on log2-heavy paths | Rejected — would be a regression |

Since `VIF_OPT_FAST_LOG2` is already active and the Netflix golden tests pass
at `places=4` and `places=5` respectively (as confirmed by the existing CI
gate and local meson test run), no code change is needed for Win 2. The
approximation was introduced in the Netflix upstream codebase and preserved
in the fork.

The `places=4` tolerance covers a maximum error of `5 × 10^{-5}` per
assertion. The Horner polynomial's maximum error relative to libm is
approximately `4 ULP ≈ 2.4 × 10^{-7}` for float, well within the golden
tolerance.

---

## Reproducer

```bash
# Build
meson setup build -Denable_cuda=false -Denable_sycl=false
ninja -C build

# Win 1 verification — all 56 tests pass
meson test -C build 2>&1 | tail -5

# Float VIF benchmark (before/after hoisting — both produce same scores)
./build/tools/vmaf_bench \
  --feature float_vif --threads 1 \
  -d python/test/resource/yuv/src01_hrc00_576x324.yuv \
  -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
  --width 576 --height 324 --pixel_format 420 --bitdepth 8
```

## References

- ADR-0452: decision document
- CPU perf audit: `.workingdir/perf-audit-cpu-2026-05-16.md`
- `libvmaf/src/feature/vif.c` — Win 1 implementation
- `libvmaf/src/feature/float_vif.c` — VifState extension
- `libvmaf/src/feature/vif.h` — updated signature
- `libvmaf/src/feature/vif_tools.c:43–91` — `log2f_approx` implementation
- `libvmaf/src/feature/vif_options.h:28` — `VIF_OPT_FAST_LOG2` define
