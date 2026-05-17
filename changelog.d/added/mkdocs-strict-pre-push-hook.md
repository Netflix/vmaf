## Added

- **`scripts/git-hooks/pre-push-mkdocs-strict.sh` — mkdocs strict-mode pre-push gate
  ([ADR-0466](../docs/adr/0466-mkdocs-strict-pre-push-hook.md)).**
  Detects whether a push touches `docs/` or `mkdocs.yml`; skips immediately on
  non-docs pushes (no latency penalty). On docs-touching pushes, runs
  `mkdocs build --strict --quiet` against a temporary output directory and blocks
  the push with exact mkdocs error lines on failure. Catches broken anchors, orphan
  pages, and missing nav entries locally in ~5 seconds, eliminating the 38/50
  docs.yml CI failure rate identified in audit slice G (each CI round-trip cost
  ~6 minutes). Skips silently when `mkdocs` is not on PATH. Override via
  `SKIP=mkdocs-strict git push`. Wired into the omnibus
  `scripts/git-hooks/pre-push` (delegation call — no `make hooks-install`
  re-run needed by existing contributors) and into `.pre-commit-config.yaml`
  as a `stages: [pre-push]` local hook (`id: mkdocs-strict`).
