---
name: dev-llm-review
description: Run a local LLM code review via vmaf-dev-llm (Ollama-backed) against a specific file. Dev-time self-review pass before PR.
---

# /dev-llm-review

## Invocation

```
/dev-llm-review <path> [--model <name>]
```

## Steps

1. Verify the tool is installed: `vmaf-dev-llm --help` must exit 0.
   If not, instruct the user to `pip install -e dev-llm` and exit.
2. Verify Ollama is reachable: `vmaf-dev-llm check` must exit 0. If not,
   instruct the user to `ollama serve` and pull the default model:
   `ollama pull qwen2.5-coder:7b`.
3. Run `vmaf-dev-llm review --file <path>` (add `-m <model>` if provided).
4. Print the review output verbatim. Do not rewrite findings — the user
   should see exactly what the local model produced.

## Guardrails

- Never writes to the file being reviewed. Read-only.
- Does not send file contents to any cloud service — the backend defaults
  to local Ollama.
- If `path` is inside `subprojects/` (vendored upstream), refuse and
  suggest reviewing the fork's overlay instead.

## Output format

```
## Review of <path> (model: <name>)

<model output verbatim>
```
