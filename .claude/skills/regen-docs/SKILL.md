---
name: regen-docs
description: Regenerate Doxygen + Sphinx docs, validate cross-references, surface stale or broken links.
---

# /regen-docs

## Invocation

```
/regen-docs [--strict] [--open]
```

## Steps

1. Verify tools: `doxygen --version`, `sphinx-build --version`. Bail with install
   hints if missing.
2. Run `doxygen Doxyfile` from `libvmaf/`. Capture warnings to
   `build/docs/doxygen-warnings.log`.
3. Run `sphinx-build -W -b html docs/ build/docs/html` (the `-W` promotes warnings
   to errors when `--strict`).
4. Validate cross-references:
   - Every `\ref` / `:ref:` target resolves.
   - No orphaned headings.
   - Code samples in docs still compile (extract via `sphinx.ext.doctest` where
     applicable).
5. Diff the generated HTML against the previous run (`git diff --no-index`); flag
   suspicious deletions (e.g. an entire API page disappearing).
6. If `--open`: open `build/docs/html/index.html` in the user's browser.
7. Print a summary: warning count, broken-link count, pages added/removed.

## Guardrails

- `--strict` fails the skill on any Doxygen warning. Use this mode in CI.
- Never edits doc sources — only regenerates output. If a header is missing
  documentation, surface it; don't auto-stub.
