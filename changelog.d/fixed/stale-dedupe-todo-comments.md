Remove two stale `//TODO: dedupe` comments from `predict.c` and `libvmaf.c`
that pointed at each other. The deduplication was completed in PR #1067
(ADR-0480, `bootstrap_names.h` shared header); replace each marker with a
brief cross-reference comment so reviewers can trace the shared constants.
