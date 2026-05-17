# ADR-0486: Extract `run_cmd` subprocess helper into `aiutils`

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris
- **Tags**: `ai`, `refactor`, `fork-local`

## Context

The dedup audit conducted on 2026-05-16
(`.workingdir/dedup-audit-python-helpers-2026-05-16.md`) identified six
groups of duplicated helper patterns across `ai/scripts/`. Groups 1–5 were
resolved by previous PRs (sha256, jsonl iteration, UTC timestamp, parquet
atomic write, and vmaftune corpus). Group 6 — subprocess execution wrappers
— remained unextracted.

Two `ai/scripts/` files carried inline subprocess boilerplate:

- `bvi_dvc_to_full_features.py`: a two-line `_run(cmd, **kw)` wrapper
  around `subprocess.run(cmd, check=True, **kw)` plus a direct
  `subprocess.run(..., capture_output=True, text=True)` call.
- `collect_gpu_calibration_data.py`: a `run_one(cmd)` function that called
  `subprocess.run(cmd, capture_output=True, text=True, check=False)` and
  mapped the return code to a `(int, str)` tuple.

Both patterns duplicate the same `capture_output=True, text=True` idiom and
both scripts import `subprocess` only for these calls.

## Decision

Add `aiutils.subprocess_utils.run_cmd()` to `ai/src/aiutils/` and export it
from `aiutils.__init__`. Replace the inline `subprocess` calls in both
scripts with `run_cmd`. The `run_one` domain wrapper in
`collect_gpu_calibration_data.py` is retained — its `(returncode, msg)`
return semantics are caller-specific — but its body delegates to `run_cmd`.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep inline (status quo) | Zero risk | LOC grows with each new script; pattern diverges over time | Defeats the aiutils consolidation strategy started in earlier PRs |
| Fully replace `run_one` | More LOC removed | `run_one`'s `(returncode, str)` return type is call-site contract; callers would need updating | Out of scope for a pure-dedup PR; `run_one` stays as a thin wrapper |

## Consequences

- **Positive**: `subprocess` is no longer imported directly in two scripts;
  the standard `capture=True` / `check=True` idiom has one canonical
  implementation with full docstring and type annotations.
- **Negative**: Scripts now depend on `aiutils`; already the case for every
  other script that uses `sha256` / `iter_jsonl` / etc.
- **Neutral**: `run_one` in `collect_gpu_calibration_data.py` becomes a
  one-liner; its public signature and callers are unchanged.

## References

- `.workingdir/dedup-audit-python-helpers-2026-05-16.md` Group 6
- [ADR-0480](0480-bootstrap-name-builder-dedup.md) — parallel dedup effort (C macros)
- PR #1076 — Groups 1+3 (sha256 + utc) for vmaftune corpus
