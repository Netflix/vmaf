**MCP server: cache ORT `InferenceSession` instances across calls**

`eval_model_on_split` and `compare_models` previously created a new
`ort.InferenceSession` on every call, costing 20–200 ms per invocation
even when the model file had not changed.  A module-level LRU cache (max
4 sessions, keyed on `(resolved_path, mtime)`) now reuses sessions for
the lifetime of the server process.  Cache invalidation is automatic when
the model file is replaced on disk.  Repeated calls with the same model
now incur only the inference cost, not the session-init overhead.
