---
name: cross-backend-diff
description: Core debugging skill. Runs the same (ref, dist) through every enabled backend, reports per-frame per-feature ULP divergence, flags anything beyond configurable tolerance. Use after any SIMD / GPU change.
---

# /cross-backend-diff

## Invocation

```
/cross-backend-diff --ref=PATH --dist=PATH --width=W --height=H --pixfmt=420p
                    [--bitdepth=8] [--frames=48] [--tolerance-ulp=2]
                    [--backends=cpu,cuda,sycl,hip]
```

## Steps

1. Build every backend in `--backends` (reuses cached builds where possible).
2. Run `build/tools/vmaf` per backend with `--json --precision=17` for every feature
   of interest (default: adm, vif, motion, psnr, ssim).
3. Parse each JSON into `(backend, frame, feature) -> score` tables.
4. For every pair of backends, compute:
   - Absolute max diff
   - ULP distance (double bits XOR)
5. Report a table:
   ```
   Feature   Pair           MaxAbsDiff   MaxULP  WorstFrame   Verdict
   adm       cpu vs cuda    3.2e-16      1       17           OK
   vif       cpu vs sycl    4.1e-15      18      23           FAIL(>2)
   ...
   ```
6. Exit 0 if all ULPs ≤ `--tolerance-ulp`, else 1.

## Notes

- ULP > tolerance is a blocker. Either fix the reduction to use double accumulation
  (see `simd-reviewer`), or open a CODEOWNERS-approved exception.
- Default test clip: the Netflix normal pair (`src01_hrc00_576x324.yuv` ↔
  `src01_hrc01_576x324.yuv`). For checkerboard variants, use the two
  `checkerboard_1920_1080_10_3_*_0.yuv` pairs.
