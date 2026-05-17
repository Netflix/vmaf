Remove two stale `//TODO: dedupe` markers from `predict.c` and `libvmaf.c`.
Both functions (`bootstrap_append_named_scores` and `vmaf_score_pooled_model_collection`)
were already refactored into their current forms; the comments referred to
a deduplication opportunity that no longer applies to the current code structure.
