### test(libvmaf): cover vmaf_model_collection_load_from_path and vmaf_model_collection_feature_overload

Add three test cases to `test_model_collection_api.c` covering the two
public C API symbols that had zero unit-test coverage:

- `vmaf_model_collection_load_from_path`: happy path (loads
  `vmaf_b_v0.6.3.json` via filesystem path) and error path (non-existent
  path returns non-zero).
- `vmaf_model_collection_feature_overload`: NULL-pointer guard (NULL
  `model_collection` argument returns non-zero without crashing).

Wire `test_model_collection_api` into `libvmaf/test/meson.build` under
`suite: fast` with `JSON_MODEL_PATH` injected via `c_args` so the
path-based load test resolves the model file at build time, matching the
pattern used by `test_model`.  No YUV frames are required.
