## Fixed: `cambi` absent from cross-backend parity gate

`scripts/ci/cross_backend_parity_gate.py` listed 17 features in `FEATURE_METRICS` but
omitted `cambi`, even though `vmaf_fex_cambi_cuda`, `*_sycl`, and `*_vulkan` are all
registered in the extractor table (T7-36 / ADR-0210). The omission meant the
GPU-Parity Matrix Gate CI job (`tests-and-quality-gates.yml`) never exercised CAMBI
numerical parity against the CPU reference — a silent coverage gap.

**Fix:** added `"cambi": ("Cambi_feature_cambi_score",)` to `FEATURE_METRICS` and
`"cambi": 5e-5` to `FEATURE_TOLERANCE` (places=4 by construction per ADR-0210 §Precision
investigation — all GPU phases are integer arithmetic and the host residual runs the
unmodified CPU c-values code on byte-identical buffers). Added `cambi` to the
`--features` list in the `GPU-Parity Matrix Gate (lavapipe, T6-8)` workflow step.

The `cross_backend_vif_diff.py` per-feature lane already carried a `cambi` entry
(added with T7-36); this change brings `cross_backend_parity_gate.py` into alignment.
