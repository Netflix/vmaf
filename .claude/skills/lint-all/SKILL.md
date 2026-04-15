---
name: lint-all
description: Run clang-tidy, cppcheck, include-what-you-use, ruff, and semgrep in parallel; produce a single merged report.
---

# /lint-all

## Invocation

```
/lint-all [--fix] [--changed-only] [--severity=warning|error]
```

## Steps

1. Ensure `build/compile_commands.json` exists (run `meson setup build` if not —
   most linters need it).
2. Determine target file set:
   - Default: all C/C++/Python/shell sources tracked by git.
   - `--changed-only`: `git diff --name-only origin/master...HEAD`.
3. Run linters in parallel, each writing JSON to `build/lint/<tool>.json`:
   - `clang-tidy -p build --config-file=.clang-tidy <files>`
   - `cppcheck --enable=all --suppressions-list=.cppcheck-suppressions
     --project=build/compile_commands.json`
   - `include-what-you-use` (via `iwyu_tool.py -p build`)
   - `ruff check python/ ai/ scripts/`
   - `semgrep --config=.semgrep.yml`
4. Merge into one report (`build/lint/report.md`) grouped by file:line, sorted
   by severity. Each entry: `[tool] severity message (rule-id)`.
5. With `--fix`: apply the safe auto-fixes from `clang-tidy --fix-errors`,
   `ruff --fix`. Never auto-fix cppcheck / semgrep findings (too risky).
6. Exit 0 if no findings at or above `--severity` (default: error). Exit 1 otherwise.

## Guardrails

- Honors `.clang-tidy` `HeaderFilterRegex` so we never lint vendored headers.
- Skips files under `subprojects/`, `build/`, `libvmaf/test/data/`.
- semgrep runs in offline mode (no rule fetch over the network in CI).
