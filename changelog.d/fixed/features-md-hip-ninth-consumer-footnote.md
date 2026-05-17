Restore `integer_ms_ssim_hip` (ninth HIP consumer, ADR-0285, batch-5) to
footnote 5 in `docs/metrics/features.md`. The entry was added by PR #1084
but clobbered when PR #1088 merged from a stale base; the code fix in
PR #1123 restored `feature_extractor.c` but did not restore the doc line.
