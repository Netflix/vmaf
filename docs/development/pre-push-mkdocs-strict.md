# Pre-push mkdocs strict-mode gate

The fork runs `mkdocs build --strict` in CI (the `docs.yml` lane) on
every push that touches `docs/` or `mkdocs.yml`. Before this hook
landed, audit slice G found that 38 of 50 master pushes failed that
lane for trivial breakage (broken anchors, orphan pages, missing nav
entries). Each CI failure cost ~6 minutes. The pre-push hook catches
the same errors in ~5 seconds locally, before any network traffic.

See [ADR-0466](../adr/0466-mkdocs-strict-pre-push-hook.md) for the
decision record.

## Quick start

```bash
# Install all pre-push hooks (idempotent):
make hooks-install

# Run the mkdocs check manually (without pushing):
SKIP=mkdocs-strict git push --dry-run  # skips the hook
scripts/git-hooks/pre-push-mkdocs-strict.sh  # runs directly

# Install the docs toolchain if mkdocs is not on PATH:
pip install -r docs/requirements.txt
```

## How the hook works

1. **Scope check** — if the push does not touch `docs/` or
   `mkdocs.yml`, the hook exits 0 immediately (no latency on
   non-docs pushes).
2. **Toolchain check** — if `mkdocs` is not on PATH, the hook
   exits 0 with a one-line notice. Install via
   `pip install -r docs/requirements.txt`.
3. **Build** — runs `mkdocs build --strict --quiet` against a
   temporary `--site-dir`. The temporary directory is cleaned up
   on exit regardless of outcome.
4. **Result** — on success, a single confirmation line is printed.
   On failure, the exact mkdocs error lines are printed to stderr
   and the push is blocked with exit code 1.

## Bypassing the hook

| Situation | Command |
|---|---|
| Skip this hook only | `SKIP=mkdocs-strict git push` |
| Skip all pre-push hooks | `git push --no-verify` |

The targeted bypass (`SKIP=mkdocs-strict`) is preferred: it still
runs the PR-body deliverables gate and other pre-push hooks.

## Troubleshooting

### "mkdocs not found on PATH"

Install the docs toolchain:

```bash
pip install -r docs/requirements.txt
```

If you work only on C / Python / GPU code and never edit docs, the
hook will skip silently on every push — no action needed.

### "WARNING - Doc file … contains a link … but the target"

The exact anchor, page, or nav entry that broke is printed. Fix
the broken reference in the doc file and push again. Common causes:

- Renamed a heading → update all links that reference its anchor.
- Added a new page but forgot to add it to `mkdocs.yml` `nav:`.
- Moved a file → update any cross-links that point at the old path.

### "mkdocs build --strict" passes locally but fails in CI

The CI lane uses `docs/requirements.txt` pinned versions. Ensure
your local installation matches:

```bash
pip install -r docs/requirements.txt
```

If CI still fails after a local pass, check the CI log for the
exact warning — the `validation:` block in `mkdocs.yml`
(see [ADR-0403](../adr/0403-mkdocs-strict-gate-validation-policy.md))
governs which categories are `warn` vs. `info`.

## Wire-up details

The hook is registered in two places so all contributor workflows
are covered:

1. **`scripts/git-hooks/pre-push`** (the omnibus hook symlinked to
   `.git/hooks/pre-push` by `make hooks-install`) — delegates to
   `scripts/git-hooks/pre-push-mkdocs-strict.sh` at the end of its
   check sequence. Existing contributors do **not** need to re-run
   `make hooks-install`; the delegation call is picked up via the
   existing symlink.

2. **`.pre-commit-config.yaml`** — a `stages: [pre-push]` local
   hook (`id: mkdocs-strict`, `files: '^(docs/|mkdocs\.yml)'`) so
   `pre-commit run --hook-stage pre-push` exercises the gate in
   isolation. This is consistent with the `validate-pr-body` hook
   pattern (ADR-0435).
