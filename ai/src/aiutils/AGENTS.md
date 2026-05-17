# aiutils — Shared AI Helpers

This package centralizes common utility patterns to reduce duplication across
`ai/scripts/` and downstream consumers.

## Invariants for new scripts

When writing a new script in `ai/scripts/`, follow these patterns:

1. **File hashing:** Import `sha256` from `aiutils.file_utils`, not a local `_sha256()`.
2. **UTC timestamps:** Use `now_iso_8601()` from `aiutils.time_utils` for ISO-8601
   second-precision UTC (not ad hoc `.isoformat()` calls).
3. **JSONL iteration:** Use `iter_jsonl()` from `aiutils.jsonl_utils` to read
   newline-delimited JSON, not inline generators.
4. **Atomic Parquet writes:** Use `write_parquet_atomic()` from `aiutils.parquet_utils`
   to safely write DataFrames with cleanup on failure.
5. **CLI setup:** (Opt-in) Logging and argument parser helpers are in `aiutils.cli_helpers`
   (see that module for factory functions). Not all scripts need consolidation yet;
   document custom patterns in your script's comments if you deviate.

## Module inventory

- `file_utils.py` — `sha256(path) -> str`
- `time_utils.py` — `now_iso_8601() -> str`
- `jsonl_utils.py` — `iter_jsonl(path) -> Iterator[tuple[int, dict]]`
- `parquet_utils.py` — `write_parquet_atomic(df, output, **kwargs) -> None`

## Future: CLI helpers (not yet extracted)

When CLI logging + argument parser duplication reaches a tipping point,
extract `cli_helpers.py` with factory functions for `ArgumentParser`
and `logging.basicConfig()`. Until then, each script handles its own CLI setup.
