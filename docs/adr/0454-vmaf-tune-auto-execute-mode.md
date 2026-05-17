# ADR-0454: `vmaf-tune auto --execute` — Phase F real encode/score execution mode

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `vmaf-tune`, `phase-f`, `encode`, `score`, `cli`, `fork-local`

## Context

`vmaf-tune auto` (ADR-0325 / ADR-0364) runs the Phase F decision tree and emits a
deterministic JSON plan: one or more `(codec, preset, crf)` cells with predictor
estimates for VMAF and bitrate, and a `selected` flag marking the planner's chosen
winner (ADR-0428).  Until this ADR, plan emission was the terminal step — actual
FFmpeg encodes and libvmaf scores were left to the operator as a manual follow-up.
This made the `auto` subcommand useful for preview and pipeline composition but
prevented it from being a fully self-contained end-to-end tool.

The remaining Phase F work tracked in `.workingdir2/OPEN.md` identifies "real
encode/score execution mode" as the next concrete deliverable after pass 29's winner
selection.  Two design questions arise:

1. **Output format**: the planning dossier mentioned Parquet; the vmaf-tune package
   has zero mandatory dependencies (pyproject.toml `dependencies = []`).  Adding
   `pyarrow` as a mandatory dep for one output serialiser would break zero-dep
   installs.  JSONL is already used by `corpus.py` and understood by every downstream
   consumer that currently reads vmaf-tune output.
2. **Default behaviour**: plan-only must remain the default so existing CI and
   operator scripts are unaffected.

## Decision

We will add a `run_plan()` function in a new `vmaftune/executor.py` module that:

- Iterates the `selected` cell(s) from an `AutoPlan` (or all cells when
  `execute_all=True`).
- Drives FFmpeg via the existing `run_encode()` seam from `encode.py`.
- Scores each output with the libvmaf CLI via the existing `run_score()` seam from
  `score.py`.
- Writes one JSONL row per cell to `<runs_dir>/tune_results.jsonl`, appending so
  partial runs survive restarts.
- Exposes `encode_runner` and `score_runner` kwargs as test seams (same pattern as
  the rest of the harness).

The `vmaf-tune auto` CLI gains three new flags:

- `--execute` (store_true, default False) — enables execute mode; plan-only is
  unchanged when absent.
- `--runs-dir PATH` (default `runs/`) — destination for encoded files and
  `tune_results.jsonl`.
- `--execute-all` (store_true) — run every plan cell rather than only the selected
  winner.

Output format is JSONL (not Parquet) to preserve the zero-dependency invariant.  A
future optional `[execute]` extra can add `pyarrow` for operators who want Parquet
column-store output; that is out of scope for this ADR.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Parquet via `pyarrow` (mandatory dep) | Native columnar; best for downstream ML | Breaks zero-dep install; large binary dep | Zero-dep invariant is a deliberate design choice (pyproject.toml) |
| Parquet via `pyarrow` (optional extra) | Same columnar benefit; install-optional | Adds a new `[execute]` extra, more complex install matrix | Can be added later; not needed for Phase F unblock |
| CSV | Universally readable | No native null support; schema fragile with feature columns | JSONL handles variable feature sets (cambi-only vs full CANONICAL6) naturally |
| JSONL (chosen) | Zero new deps; consistent with corpus.py; null-safe | Not columnar; conversion step for ML consumers | Correct choice for v1; Parquet layer is additive |
| Embed in `auto.py` rather than a new module | Fewer files | `auto.py` is already large; mixing planning + execution concerns | Separation of concerns; `executor.py` is independently testable |

## Consequences

- **Positive**: `vmaf-tune auto --execute` is now a self-contained plan+run verb;
  operators get a JSONL file with encode size, encode time, VMAF score, and
  per-feature aggregates without any manual post-processing.
- **Positive**: The subprocess boundary is a clean test seam — `test_executor.py`
  achieves 100% path coverage without FFmpeg or the vmaf binary.
- **Positive**: `tune_results.jsonl` appends on each run, so partial runs and
  incremental re-runs do not overwrite previous results.
- **Negative**: JSONL is not columnar; downstream ML consumers that want efficient
  column reads need a one-off conversion (`polars.read_ndjson` / `pyarrow` table
  conversion).
- **Neutral**: `--execute` is off by default; existing `auto` callers are unaffected.
- **Neutral**: follow-up work can add saliency-aware per-shot execution (the other
  open Phase F item) as a second call to `run_plan` gated by the plan's saliency
  short-circuit metadata.

## References

- ADR-0325: `vmaf-tune auto` Phase F decision tree.
- ADR-0364: Phase F adaptive recipe and confidence-aware tuning.
- ADR-0428: Plan winner selection (`metadata.winner` + `cells[].selected`).
- `.workingdir2/OPEN.md` lines 184–188: Phase F remaining work (real encode/score
  execution).
- `req`: per user direction in agent task brief 2026-05-16, Phase F execute
  scaffolding.
