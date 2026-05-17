# ADR-0468: vmaf-tune executor — per-shot and saliency execution modes

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `vmaf-tune`, `executor`, `per-shot`, `saliency`, `phase-f`, `fork-local`

## Context

ADR-0454 introduced `vmaf-tune auto --execute`, which drives real FFmpeg encodes
and libvmaf scores for a planning-phase AutoPlan cell. That baseline covers the
"whole-clip, no ROI bias" case. Two capabilities that the planning phase already
supports needed equivalent execute-mode coverage:

1. **Per-shot scoring**: the Phase D planner (`per_shot.py`) can split a source
   into shot boundaries (via `vmaf-perShot` / TransNet V2) and recommend per-shot
   CRFs. Without an execute-mode counterpart, a caller can see the plan but cannot
   measure real per-shot VMAF without wiring up the plumbing themselves.

2. **Saliency-weighted encoding**: `saliency.py` / `saliency_aware_encode` applies
   per-codec ROI bias (x264 qpfile, x265 zones, SVT-AV1 qpmap, VVenC ROI CSV) to
   steer bits toward salient regions. The execute mode did not expose a saliency
   path, so callers had no single-call entry-point to encode-and-score with ROI
   bias active.

The two modes are logically independent: per-shot is about temporal segmentation;
saliency is about spatial bit allocation within a single encode.

## Decision

We extend `executor.py` with two new public functions:

- `run_plan_per_shot`: detects shot boundaries via `detect_shots` (falls back to a
  single-shot range when `vmaf-perShot` is absent), encodes and scores each shot
  segment independently using `frame_skip_ref` / `frame_cnt` to align the reference
  window, then reports a frame-length-weighted VMAF aggregate alongside per-shot rows
  in `tune_results_per_shot.jsonl`.

- `run_plan_saliency`: wraps `saliency_aware_encode` for each selected cell, records
  whether saliency actually ran (vs graceful fallback), and scores the output in the
  standard encode → score pipeline, writing rows to `tune_results_saliency.jsonl`.

Both functions keep the same test-seam pattern (`encode_runner`, `score_runner`,
`shot_runner`, `session_factory`) as the base `run_plan`, so they are fully testable
without any real binary.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Flags on `run_plan` (`--per-shot`, `--saliency`) | Single entry-point | Complex branching; callers cannot mix modes orthogonally | Separate entry-points are simpler and independently testable |
| New `executor_pershot.py` / `executor_saliency.py` modules | Clean separation | Extra import complexity; shared helpers would need a third module | Kept in `executor.py` — the shared dataclasses and `_log` helper stay local |
| Scene-change via FFmpeg `select=gt(scene,0.4)` | No extra binary | Much slower than TransNet V2; no frame-accurate boundary output | `detect_shots` already wraps `vmaf-perShot` with a tested fallback path |

## Consequences

- **Positive**: callers can measure real per-shot VMAF from a single call; the
  saliency-aware encode path is now end-to-end testable without custom wiring.
- **Negative**: per-shot execution runs N encodes + N score calls (N = shot count),
  which is proportionally slower than a single whole-clip run. This is expected and
  documented in `vmaf-tune.md`.
- **Neutral / follow-ups**: `run_plan_per_shot` inherits the `detect_shots` fallback
  behaviour — callers should check `shot_count` in the result row to distinguish
  real shot data from the single-shot sentinel.

## References

- ADR-0454: Phase F base execute mode.
- ADR-0222: `vmaf-perShot` C-side binary (TransNet V2 wrapper).
- ADR-0293: saliency-aware ROI encoding (`saliency_aware_encode`).
- Per user direction: Phase F follow-up item, 2026-05-16.
