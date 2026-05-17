- **DNN `vmaf_ort_run` — eliminate per-call heap allocation** (`libvmaf/src/dnn/ort_backend.c`):
  Replace five `calloc`/`free` pairs (input/output name and value pointer arrays,
  plus the input scratch buffer array) with fixed-size stack arrays bounded by
  `VMAF_ORT_MAX_IO = 8`. Removes five heap round-trips per `vmaf_ort_run` call.
  For a 300-frame clip scored with a 5-in/1-out FR regressor, this eliminates
  1500 heap alloc/free pairs with no correctness risk — the
  `n_inputs ≤ VMAF_ORT_MAX_IO` guard is enforced at runtime before the arrays
  are touched. (Audit finding F3-B from perf-audit-pipeline-2026-05-16.)
