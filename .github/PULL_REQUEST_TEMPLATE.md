<!-- Thanks for contributing! Please fill out the checklist so reviewers can ship this quickly. -->

## Summary

<!-- What does this PR change and why. 2-4 sentences. -->

## Type

<!-- Mark what applies. Conventional-commit compatible. -->

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement
- [ ] `refactor` — no behavior change
- [ ] `docs` — documentation only
- [ ] `test` — test-only
- [ ] `build` / `ci` — tooling / infra
- [ ] `port` — cherry-pick from upstream Netflix/vmaf
- [ ] `sycl` / `cuda` / `simd` — backend-specific

## Checklist

- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (the commit-msg hook enforces this).
- [ ] `make format && make lint` is green locally.
- [ ] Unit tests pass: `meson test -C build`.
- [ ] If I touched **any** SIMD/GPU code path, I ran `/cross-backend-diff` and the worst ULP is ≤ 2.
- [ ] If I touched a feature extractor with SIMD/GPU twins, I either updated every twin or listed the gap under "Known follow-ups" below.
- [ ] If I added a new `.c` / `.cpp` / `.cu` / `.h` / `.hpp`, it has the appropriate license header (see `CONTRIBUTING.md`).
- [ ] If this is a breaking change, the commit message uses `!` or `BREAKING CHANGE:` and the migration path is documented below.

## Netflix golden-data gate (D24)

<!-- CI runs the 3 Netflix CPU golden pairs (1 normal + 2 checkerboard) on every PR. -->

- [ ] I did **not** modify any `assertAlmostEqual(...)` score in the Netflix golden Python tests.
- [ ] If I believe a golden value must change, I have explained why below AND pinged @Lusoris for a CODEOWNERS exception.

## Cross-backend numerical results

<!-- Paste the /cross-backend-diff table if applicable. -->

```text
<feature> <cpu-vs-cuda-ULP> <cpu-vs-sycl-ULP> <cpu-vs-hip-ULP>
```

## Performance (if `perf` or `feat`)

<!-- Paste `/profile-hotpath` before/after, or benchmark output with % delta. -->

## Known follow-ups

<!-- Anything deliberately out of scope, linked issues, SIMD/GPU twins not updated. -->

## Breaking changes / migration

<!-- If any. Omit section otherwise. -->
