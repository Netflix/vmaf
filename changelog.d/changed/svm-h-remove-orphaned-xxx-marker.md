## Changed: remove orphaned `/* XXX */` marker from vendored `svm.h`

`libvmaf/src/svm.h:101` carried an empty `/* XXX */` comment inherited from the
upstream libsvm vendored source with no explanation or tracking issue. The two
adjacent comment lines (`/* 1 if svm_model is created by svm_load_model*/` and
`/* 0 if svm_model is created by svm_train */`) are merged into a single two-line
comment to clarify the `free_sv` semantics. No logic changes; SVM is deprecated and
the header is consumed only via `libvmaf/src/predict.c`.

Identified in the 2026-05-15 code-quality audit (AUDIT-2026-05-15.md §8).
