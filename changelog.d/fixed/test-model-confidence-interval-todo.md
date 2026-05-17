**fix(test):** Replace two stale `TODO: add check for confidence interval` comments
in `test_model_check_default_behavior_unset_flags` and `_set_flags` with an
explanation that CI values are produced at scoring time, not load time.
Add `test_bootstrap_collection_structure`, which loads `vmaf_b_v0.6.3.json`
and asserts `mc->cnt == 20` and `mc->type == VMAF_MODEL_BOOTSTRAP_SVM_NUSVR`,
closing the load-time bootstrap coverage gap the TODOs requested.
