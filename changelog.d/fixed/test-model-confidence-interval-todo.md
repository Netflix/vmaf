### Fixed

- `libvmaf/test/test_model.c`: resolved two stale `TODO: add check for confidence interval` markers
  in `test_model_check_default_behavior_unset_flags` and `test_model_check_default_behavior_set_flags`.
  Added `test_bootstrap_collection_structure` which loads `vmaf_b_v0.6.3.json` and asserts
  `mc->cnt == 20` and `mc->type == VMAF_MODEL_BOOTSTRAP_SVM_NUSVR`, providing the actual
  load-time bootstrap coverage the TODOs requested.  CI values are a scoring-time concept and
  are documented as such in the updated comments.
