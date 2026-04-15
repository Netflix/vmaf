---
name: refresh-ffmpeg-patches
description: Rebase our ffmpeg-patches/ series onto the latest ffmpeg master (or a specified ref), resolve trivial conflicts, regenerate .patch files, and surface unresolved hunks for human attention.
---

# /refresh-ffmpeg-patches

Keeps our ffmpeg integration patches current with upstream ffmpeg. Run this whenever
`build-ffmpeg-with-vmaf` reports a patch apply failure, or periodically (monthly).

## Invocation

```
/refresh-ffmpeg-patches [--ffmpeg-ref=master|n7.0|<sha>] [--ffmpeg-dir=/tmp/ffmpeg]
                        [--branch=vmaf-fork-patches]
```

Defaults: `master` at HEAD, `/tmp/ffmpeg`, `vmaf-fork-patches` work branch.

## Steps

1. Clone or fetch ffmpeg into `--ffmpeg-dir`.
2. Checkout a clean work branch from `--ffmpeg-ref`: `git switch -c <branch>`.
3. Apply each `ffmpeg-patches/*.patch` via `git am --3way`.
4. If any `git am` fails:
   a. Capture the failing patch name + conflicting hunks to
      `/tmp/refresh-ffmpeg-report.md`.
   b. Attempt a 3-way merge with the current upstream file. If the result is trivially
      resolvable (new import added, no semantic overlap with our changes), resolve +
      `git am --continue`. Otherwise leave for human review and move on.
5. For patches that rebased cleanly, regenerate them as a single series via:
   `git format-patch --output-directory=$repo/ffmpeg-patches-new/ <base>..HEAD`.
6. Diff old vs new patch directory:
   - Any patch that is byte-identical: skip (no refresh needed).
   - Any patch that changed but still applies: overwrite in `ffmpeg-patches/`.
   - Any patch that failed: emit a clear diff + context snippet.
7. Summary output:
   - `<patches> total, <clean> clean, <refreshed> refreshed, <failed> needs-human`.
   - If `<failed>` > 0, list those by filename and point to the report.
8. Stop short of committing — print the exact `git add ffmpeg-patches/ && git commit`
   line the operator should run.

## Guardrails

- Does not touch `ffmpeg-patches/` until every patch has been processed.
- Does not commit. The human must inspect the diff and commit explicitly.
- Failure in one patch does not abort — process every patch, report all failures at
  end, so you get a full picture of upstream drift in one pass.

## When to use

- After a `/sync-upstream`-style pull from ffmpeg upstream.
- When `build-ffmpeg-with-vmaf` fails at step 4.
- On a monthly cadence via scheduled agent run, so drift is caught early.
