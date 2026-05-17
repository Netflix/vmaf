**fix(model):** Convert `vmaf_tiny_v1_medium.onnx` from external-data layout to
single-file inline. The model was saved referencing `mlp_medium_final.onnx.data`
but only `vmaf_tiny_v1_medium.onnx.data` was committed, causing
`ort.InferenceSession()` to raise `FileNotFoundError` at load time. The model is
11 KB after inlining; no external data file is needed.
