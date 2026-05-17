## Added

- **`cambi` in cross-backend parity gate** (`scripts/ci/cross_backend_parity_gate.py`):
  adds `cambi` to the `FEATURE_METRICS` and `FEATURE_TOLERANCE` dicts with the metric
  name `Cambi_feature_cambi_score` and places=4 tolerance (5e-5). Closes coverage gap
  identified in audit-test-coverage-2026-05-16.md §5 — `cambi` has CUDA, SYCL, and
  Vulkan implementations but was entirely absent from parity testing. This gap allowed
  6 silent correctness bugs in `cambi_vulkan` v1 to slip through to PR #874.
