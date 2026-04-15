---
name: regen-snapshots
description: Regenerate fork-added test snapshot JSONs under testdata/ (scores_cpu_*.json, netflix_benchmark_results.json) after an intentional numerical change. Requires justification committed to the message.
---

# /regen-snapshots

## Invocation

```
/regen-snapshots --justification="<short rationale>" [--files=scores_cpu_640,scores_cpu_576,...]
                 [--backend=cpu|cuda|sycl|all]
```

## Steps

1. Refuse if `--justification` is missing or empty.
2. Build the requested backend (`/build-vmaf --backend=<backend>`).
3. For each snapshot in `--files` (default: all `testdata/scores_cpu_*.json` and
   `testdata/netflix_benchmark_results.json`):
   - Locate the matching regeneration script (`testdata/gen_cpu_golden.py`,
     `testdata/run_sycl_scores.py`, `testdata/benchmark_netflix.py`).
   - Run it, writing to a tempfile.
   - Diff old vs new; if identical, skip.
   - Otherwise, replace.
4. Emit a diff summary: file, frames affected, max delta observed.
5. Prepare a commit message draft:
   ```
   test(snapshots): regenerate <files>

   Justification: <justification>

   Affected: <summary>
   ```
   Print it; do NOT auto-commit.

## Guardrails

- Never touches `python/test/**` — those hold Netflix golden assertions (see §8 of
  CLAUDE.md). If the regeneration script tries to write there, ABORT.
- Max delta > 0.5 (absolute) without justification mentioning a model/feature rewrite
  aborts with an error.
