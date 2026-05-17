**perf(dnn):** Replace five per-call `calloc`/`free` pairs in `vmaf_ort_run` with
fixed-size stack arrays (`VMAF_ORT_MAX_IO = 8`). Eliminates heap allocation on
the multi-input inference hot path (audit item F3-B).
