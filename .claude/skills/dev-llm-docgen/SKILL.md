---
name: dev-llm-docgen
description: Draft a Doxygen @brief/@param/@return block for a named function in a C/C++/CUDA file via local LLM (Ollama).
---

# /dev-llm-docgen

## Invocation

```
/dev-llm-docgen <file> <symbol> [--model <name>]
```

## Steps

1. Verify `vmaf-dev-llm check` is green.
2. Verify the symbol exists in the file (simple grep for `<symbol>\\s*\\(`).
   If not, abort with a clear error.
3. Run `vmaf-dev-llm docgen --file <file> --symbol <symbol>` — capture
   stdout as the draft docblock.
4. Show the draft to the user and ask whether to:
   - **insert** — put it above the symbol's declaration (use `Edit` tool),
   - **copy** — print it verbatim for manual paste,
   - **regenerate** — re-run step 3.

   Do NOT insert automatically.

## Guardrails

- Never rewrites an existing docblock unless the user says `replace`.
- Rejects files under `subprojects/` (vendored upstream).
- Generated block must start with `/**` and end with `*/`; otherwise treat
  it as a bad response and regenerate.
- For files matching `*.cu` / `*.cuh`, uses `review_cuda.md` context to
  prefer CUDA-appropriate parameter descriptions (e.g., "pinned host
  pointer", "device-resident array").
