Restore `adm_p_norm` explanatory comment in `integer_adm.c` clobbered by
PR #1088 squash-merge. Replaces the open-ended `TODO` with a structured
note explaining that `adm_p_norm=3.0f` is fixed by the Netflix
training-data contract and cannot become a runtime parameter without a
full model retrain.
