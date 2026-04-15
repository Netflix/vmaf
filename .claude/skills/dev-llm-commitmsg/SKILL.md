---
name: dev-llm-commitmsg
description: Draft a Conventional-Commits message from the current `git diff --staged` using a local LLM (Ollama). User edits before commit.
---

# /dev-llm-commitmsg

## Invocation

```
/dev-llm-commitmsg [--model <name>]
```

## Steps

1. Verify `git diff --staged` has content. If empty, tell the user to
   stage changes first (`git add -p`) and exit.
2. Verify `vmaf-dev-llm check` is green (see `dev-llm-review` for the
   pre-flight).
3. Run `vmaf-dev-llm commitmsg` (add `-m <model>` if provided). Capture
   its stdout — that's the draft.
4. Present the draft and ask the user to:
   - **accept** — emit `git commit -m "<heredoc>"` using the draft verbatim,
   - **edit** — open `$EDITOR` with the draft and let the user refine,
   - **regenerate** — re-run step 3.

   Do NOT commit automatically — the user must confirm.

## Guardrails

- Refuses to run on `master` or a protected branch.
- Strips any `Co-Authored-By:` or `Signed-off-by:` lines from the draft —
  those are added by the pipeline.
- First line must match `^(feat|fix|perf|refactor|docs|test|build|ci|chore|revert)(\\([a-z0-9-]+\\))?: .+` before being accepted; otherwise regenerate.
