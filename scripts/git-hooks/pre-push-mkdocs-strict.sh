#!/usr/bin/env bash
# scripts/git-hooks/pre-push-mkdocs-strict.sh — pre-push mkdocs strict-mode gate.
#
# Detects whether the push touches docs/ or mkdocs.yml; skips entirely if it
# does not (non-docs pushes see no latency penalty). When docs are touched,
# runs `mkdocs build --strict --quiet` against a temporary output directory,
# prints the exact mkdocs error lines on failure, and exits non-zero to block
# the push.
#
# Rationale (ADR-0466): audit slice G found that 38 of 50 master pushes failed
# the docs.yml CI lane for trivial mkdocs strict-mode breakage (broken anchors,
# orphan pages, missing nav entries). Each CI round-trip costs ~6 minutes. A
# 5-second local pre-push check eliminates the entire failure mode.
#
# Bypass: SKIP=mkdocs-strict git push
#         (or the standard --no-verify flag which skips ALL pre-push checks)
#
# Requirements:
#   mkdocs (+ docs/requirements.txt) must be installed in the active Python env.
#   `pip install -r docs/requirements.txt` satisfies this. The hook skips
#   silently when mkdocs is not on PATH to avoid blocking contributors who
#   work only on non-docs code paths.
#
# See also: docs/development/pre-push-mkdocs-strict.md

set -euo pipefail

# ── Skip override ─────────────────────────────────────────────────────────────
if [[ "${SKIP:-}" == *mkdocs-strict* ]]; then
  echo "pre-push-mkdocs-strict: SKIP set — skipping mkdocs strict check." >&2
  exit 0
fi

# ── Repo root ─────────────────────────────────────────────────────────────────
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${repo_root}" ]; then
  exit 0
fi

# ── mkdocs availability ───────────────────────────────────────────────────────
if ! command -v mkdocs >/dev/null 2>&1; then
  echo "pre-push-mkdocs-strict: mkdocs not found on PATH — skipping." >&2
  echo "  Install: pip install -r ${repo_root}/docs/requirements.txt" >&2
  exit 0
fi

# ── Detect whether the push touches docs/ or mkdocs.yml ──────────────────────
# Read push arguments from stdin (format: <local-ref> <local-sha1> <remote-ref> <remote-sha1>)
# If stdin is not provided (e.g. direct invocation), fall back to diff vs origin/master.
touched_docs=0

if ! git rev-parse --verify origin/master >/dev/null 2>&1; then
  # No remote reference yet (first push) — run the check conservatively.
  touched_docs=1
else
  base="$(git merge-base origin/master HEAD 2>/dev/null || true)"
  if [ -n "${base}" ]; then
    if git diff --name-only "${base}..HEAD" 2>/dev/null |
      grep -qE '^(docs/|mkdocs\.yml)'; then
      touched_docs=1
    fi
  else
    # Cannot compute merge-base — run conservatively.
    touched_docs=1
  fi
fi

if [ "${touched_docs}" -eq 0 ]; then
  exit 0
fi

# ── Run mkdocs build --strict ─────────────────────────────────────────────────
tmp_site="$(mktemp -d)"
trap 'rm -rf "${tmp_site}"' EXIT

echo "pre-push-mkdocs-strict: docs/ touched — running mkdocs build --strict..." >&2

mkdocs_out="$(mktemp)"
trap 'rm -f "${mkdocs_out}"; rm -rf "${tmp_site}"' EXIT

set +e
mkdocs build --strict --quiet \
  --config-file "${repo_root}/mkdocs.yml" \
  --site-dir "${tmp_site}" \
  >"${mkdocs_out}" 2>&1
mkdocs_exit=$?
set -e

if [ "${mkdocs_exit}" -ne 0 ]; then
  cat >&2 <<'HEADER'

pre-push-mkdocs-strict: BLOCKED — mkdocs build --strict failed.
Fix the errors below before pushing (saves a ~6-minute CI round-trip):

HEADER
  cat "${mkdocs_out}" >&2
  cat >&2 <<'FOOTER'

Bypass (this check only): SKIP=mkdocs-strict git push
Bypass (all pre-push checks): git push --no-verify

See docs/development/pre-push-mkdocs-strict.md for troubleshooting.
ADR-0466: docs/adr/0457-mkdocs-strict-pre-push-hook.md
FOOTER
  exit 1
fi

echo "pre-push-mkdocs-strict: mkdocs build --strict passed." >&2
exit 0
