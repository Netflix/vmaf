## vmaf-tune executor: per-shot and saliency execution modes (ADR-0468)

`vmaftune.executor` gains two new execution-mode entry-points that extend the Phase F
base execute mode (ADR-0454):

- **`run_plan_per_shot`**: detects shot boundaries via `vmaf-perShot` (TransNet V2),
  scores each segment independently with `frame_skip_ref`/`frame_cnt` reference
  alignment, and reports a frame-length-weighted VMAF aggregate in
  `tune_results_per_shot.jsonl`. Falls back to a single-shot range when
  `vmaf-perShot` is absent.

- **`run_plan_saliency`**: encodes each plan cell with saliency-aware ROI bias
  (`saliency_aware_encode`) and scores the result in the standard pipeline, writing
  rows including `saliency_available` to `tune_results_saliency.jsonl`. Gracefully
  falls back to a plain encode when onnxruntime or the model file is unavailable.

Both functions use the existing `encode_runner`/`score_runner` test-seam pattern and
require no new mandatory dependencies.
