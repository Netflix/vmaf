---
name: sync-upstream
description: Fetch upstream Netflix/vmaf master, create a sync/upstream-YYYYMMDD branch, merge upstream, auto-resolve trivial conflicts, and open a PR with the merge + surviving delta.
---

# /sync-upstream

## Invocation

```
/sync-upstream [--open-pr]
```

## Steps

1. `git fetch upstream`.
2. `git switch -c sync/upstream-$(date +%Y%m%d) master`.
3. `git merge upstream/master --no-ff`.
4. Conflict policy (matches D16):
   - Fork wins for: `.github/`, `README.md`, `CLAUDE.md`, `AGENTS.md`, `.claude/`,
     `Dockerfile*`, `libvmaf/meson_options.txt`, anything under `libvmaf/src/cuda/`,
     `libvmaf/src/sycl/`, `libvmaf/src/feature/{cuda,sycl}/`, `libvmaf/src/feature/x86/`,
     `libvmaf/src/feature/arm64/`.
   - Upstream wins for: feature metric code not touched by the fork — identify by
     checking `git log --follow origin/master -- <path>` for any fork commits.
   - Manual resolution required: `libvmaf/include/libvmaf/libvmaf.h`,
     `libvmaf/src/libvmaf.c`, `libvmaf/meson.build`, `libvmaf/tools/cli_parse.c`,
     `libvmaf/tools/vmaf.c`.
5. For manual conflicts, STOP and surface them with `file:line` context. Do NOT
   resolve.
6. On clean merge: `/build-vmaf --backend=cpu`, `meson test -C build`,
   `/cross-backend-diff` on the normal Netflix pair.
7. If `--open-pr`: `gh pr create` with title
   `chore(upstream): sync to upstream/master @ <sha>` and body including upstream
   commit count, conflict summary, and test results.

## Guardrails

- Refuses to run if working tree is dirty.
- Refuses to open a PR if the Netflix CPU golden tests fail after merge.
- Never `git push --force`.
