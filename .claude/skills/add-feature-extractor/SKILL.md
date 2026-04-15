---
name: add-feature-extractor
description: Scaffold a new feature extractor (e.g. a novel metric) with C source+header, registry entry, doc stub, and a smoke test. Does not produce a SIMD or GPU path — those come via /add-simd-path and /add-gpu-backend.
---

# /add-feature-extractor

## Invocation

```
/add-feature-extractor <name> [--type=full-reference|no-reference]
```

## Files created

| Path                                        | Purpose                                    |
|---------------------------------------------|--------------------------------------------|
| `libvmaf/src/feature/<name>.c`              | Scalar reference implementation            |
| `libvmaf/src/feature/<name>.h`              | Prototype                                  |
| `libvmaf/test/test_<name>.c`                | Smoke test (1 frame, fixed expected value) |
| `resource/doc/<name>.md`                    | Metric documentation (inputs, range, refs) |

## Files patched

- `libvmaf/src/feature/feature_extractor.c` — registry row.
- `libvmaf/src/feature/all.c` — `#include "<name>.h"` + array entry.
- `libvmaf/src/meson.build` — new source in the feature set.
- `libvmaf/test/meson.build` — register test.

## Template fills

- Extractor struct: `extract` pointer, name, type (FR/NR), features exposed.
- No SIMD / GPU variants wired — the feature first exists as a scalar implementation
  and is the reference against which SIMD/GPU paths are later tested.

## Guardrails

- Names collide-checked against the existing registry.
- The smoke test must actually pass (non-NaN, finite) against the sample YUVs in
  `python/test/resource/yuv/`.
