---
name: validate-scores
description: Run identical (ref, dist) through all enabled backends and report per-backend score + pairwise ULP diffs. Use to verify bit-exactness of a new SIMD path, new GPU backend, or any hot-path change.
---

# /validate-scores

## Invocation

```
/validate-scores --ref=PATH --dist=PATH --width=W --height=H --pixfmt=420p --bitdepth=8
                 [--backends=cpu,cuda,sycl] [--precision=17]
```

## Steps

1. For each enabled backend in `--backends`, run:
   `build/tools/vmaf --reference REF --distorted DIST --width W --height H \
    --pixel_format PIXFMT --bitdepth BD --feature psnr --feature ssim --feature vif \
    --feature adm --feature motion --output /tmp/<backend>.json --json --precision=17`
2. Parse the resulting JSON; build a `(frame, feature) -> score` table per backend.
3. For every pair of backends, compute:
   - max absolute diff
   - max ULP distance (via `math.frexp` + integer reinterpretation)
4. Emit a report: per-feature worst case across all backend pairs.
5. Exit 0 if all pair-wise ULP ≤ 2 (bit-exact or within float reduction jitter);
   exit 1 otherwise.

## Notes

- Tolerance of 2 ULPs matches the SIMD-vs-scalar requirement in `docs/principles.md`.
- Higher tolerance is NOT permitted without CODEOWNERS approval; backend divergence
  must be explained and justified.
