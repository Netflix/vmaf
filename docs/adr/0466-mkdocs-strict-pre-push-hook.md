# ADR-0466: mkdocs strict-mode pre-push hook

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: `docs`, `ci`, `git`, `hooks`, `mkdocs`

## Context

Audit slice G established that 38 of 50 master pushes failed the `docs.yml`
CI lane for trivial mkdocs strict-mode breakage: broken in-doc anchors, orphan
pages, and missing nav entries. The `docs.yml` lane (see ADR-0403 for the
strict-mode parameterisation) takes approximately 6 minutes per run on GitHub
Actions. Every failure required a corrective push and a second CI round-trip,
costing 12+ minutes of wall time per affected push.

The root cause is that no local check mirrors `mkdocs build --strict` before
the push reaches CI. Contributors (and agents) discover the failure only after
the CI cycle completes. The errors are mechanical — mkdocs emits the exact
filename, line, and anchor that caused the failure — so a local 5-second
pre-push run catches 100% of them before any network traffic.

## Decision

Add `scripts/git-hooks/pre-push-mkdocs-strict.sh` as a dedicated pre-push
gate. The script:

1. Detects whether the push touches `docs/` or `mkdocs.yml`; exits 0
   immediately if not (no latency penalty on non-docs pushes).
2. Skips silently when `mkdocs` is not on PATH (contributors who never
   touch docs are not required to install the docs toolchain).
3. Runs `mkdocs build --strict --quiet` against a temporary output directory.
4. On failure: prints the exact mkdocs error lines and exits 1, blocking
   the push with a human-readable message.
5. Provides a targeted bypass (`SKIP=mkdocs-strict git push`) in addition to
   the standard `git push --no-verify` escape hatch.

Wire-up via two complementary mechanisms:

- **Omnibus `scripts/git-hooks/pre-push`**: delegates to the new script so
  contributors using only the symlinked hook (installed via `make hooks-install`)
  pick it up automatically with no re-install step.
- **`.pre-commit-config.yaml` local hook** (`id: mkdocs-strict`,
  `stages: [pre-push]`, `files: '^(docs/|mkdocs\.yml)'`): allows `pre-commit
  run --hook-stage pre-push` to exercise the gate in isolation, consistent with
  the pattern established by `validate-pr-body` (ADR-0435).

`make hooks-install` requires no change: the existing symlink from
`.git/hooks/pre-push → scripts/git-hooks/pre-push` already covers the new
delegation call.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **A — Standalone pre-push hook + pre-commit wiring (chosen)** | 5 s local catch; zero CI round-trips for docs errors; no new dependencies beyond existing mkdocs toolchain | mkdocs must be installed in the active env; skips silently when absent | Best fit — identical pattern to ADR-0435's validate-pr-body hook |
| **B — `make docs-check` target only** | Zero-friction manual invocation | Requires contributor to remember; does not block a push; agents skip it | Does not eliminate the CI round-trip failure mode |
| **C — CI path-filter to skip docs.yml on non-docs pushes** | Reduces CI load | Removes the gate entirely for those pushes; errors only caught on docs-touching PRs | Weaker coverage; the gate should fire on every docs touch |
| **D — Promote mkdocs validation into `make lint`** | Always runs with the lint pass | `make lint` is slow (clang-tidy + semgrep); adding mkdocs makes it slower for C-only contributors | Disproportionate cost for a docs-only check |

## Consequences

- **Positive**: The 38/50 docs.yml failure rate drops to near-zero for
  contributors who have the docs toolchain installed. Broken anchors, orphan
  pages, and missing nav entries are caught in under 5 seconds before push.
- **Positive**: The hook is opt-out (SKIP=mkdocs-strict) rather than opt-in,
  so it activates for all contributors who run `make hooks-install` without
  any additional step.
- **Negative**: Contributors who have not installed `docs/requirements.txt`
  skip the check silently. CI remains the authoritative gate for those
  contributors.
- **Neutral**: `make hooks-install` does not need to be re-run by existing
  contributors — the delegation is in the omnibus `pre-push` script which the
  existing symlink already resolves.
- **Neutral**: The temporary site directory is cleaned up via `trap` on both
  success and failure paths; no build artefacts are left in the worktree.

## References

- Audit slice G: "38/50 master pushes failed docs.yml CI for trivial
  mkdocs strict-mode breakage" (per user direction 2026-05-15).
- ADR-0403: `docs/adr/0403-mkdocs-strict-gate-validation-policy.md` —
  parameterises the `--strict` categories this hook mirrors.
- ADR-0435: `docs/adr/0435-pr-body-pre-push-validation.md` — the
  validate-pr-body hook this follows as a pattern.
- ADR-0108: `docs/adr/0108-deep-dive-deliverables-rule.md`.
- `.github/workflows/docs.yml` — the CI lane this hook pre-empts locally.
- `scripts/git-hooks/pre-push-mkdocs-strict.sh` — implementation.
