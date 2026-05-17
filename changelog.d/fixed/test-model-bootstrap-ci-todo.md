- Resolved two stale `TODO: add check for confidence interval` comments in
  `test_model.c`. Added `test_model_collection_bootstrap_type` which loads
  `vmaf_b_v0.6.3.json`, asserts the collection parsed as a bootstrap variant
  (`VMAF_MODEL_BOOTSTRAP_SVM_NUSVR`), and confirms `cnt > 1` — the structural
  preconditions for `VmafModelCollectionScore.bootstrap.ci.p95` scoring.
