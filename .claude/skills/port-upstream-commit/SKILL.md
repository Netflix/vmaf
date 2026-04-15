---
name: port-upstream-commit
description: Cherry-pick a single upstream Netflix/vmaf commit onto the fork's master, auto-adapting for SIMD/GPU paths where the commit touches a feature we have multiple implementations of.
---

# /port-upstream-commit

## Invocation

```
/port-upstream-commit <sha> [--open-pr]
```

## Steps

1. `git fetch upstream`.
2. `git switch -c port/<sha-short> master`.
3. `git cherry-pick <sha>` (using `-x` so the commit message references upstream).
4. If conflicts: inspect; for conflicts in a file that has SIMD/GPU twins (e.g. edits
   to `float_adm.c` while we also have `x86/float_adm_avx2.c`, `x86/float_adm_avx512.c`,
   `arm64/float_adm_neon.c`, `cuda/adm_*.cu`, `sycl/integer_adm_sycl.cpp`), report all
   sibling files to the author so they can propagate the same change. Do NOT attempt
   automatic propagation — SIMD/GPU adaptations are not string-substitutions.
5. Run `/build-vmaf --backend=cpu` + `meson test -C build --suite=fast`.
6. Run `/cross-backend-diff` for the affected feature.
7. If `--open-pr`: `gh pr create` with title `port(upstream): <original subject>` and
   body including the upstream commit link, conflict summary, and propagation TODO.

## Guardrails

- Aborts if Netflix golden tests fail post-port.
- Never force-resolves conflicts — leaves them for human review.
- Links back to the upstream commit in the message (`(cherry picked from commit <sha>)`).
