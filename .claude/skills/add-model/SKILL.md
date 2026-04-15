---
name: add-model
description: Register a new VMAF model (.json / .pkl / .onnx) with the build, validate its schema, and add a loader smoke test.
---

# /add-model

## Invocation

```
/add-model <path> [--type=classical-json|bootstrap-pkl|tiny-onnx] [--install]
```

## Steps

1. Validate the file exists + has an allowed extension.
2. For `.json`: parse; require top-level keys `model_type`, `feature_names`, `score_clip`,
   `model` (LibsvmNuSvr / BootstrapLibsvmNuSvr / Onnx). Reject on unknown keys.
3. For `.pkl`: use `python/vmaf/tools/check_pkl_model.py` (if present) to validate
   the pickle — refuse arbitrary-code-execution pickles (see SECURITY.md).
4. For `.onnx`: require opset ≥ 17, run `onnx.checker.check_model(...)`, verify
   input/output shapes match the declared `model_type`.
5. Copy to `model/` (or `model/tiny/` for `.onnx`) — never overwrite.
6. Patch `libvmaf/src/meson.build` or `model/meson.build` to add the file to the
   install set if `--install` is passed.
7. Add a loader test in `libvmaf/test/test_model.c` that loads the model, asserts
   basic metadata is read, and unloads cleanly.
8. Emit a summary: model name, type, install target, test added.

## Guardrails

- `.pkl` models are loaded in a sandboxed Python subprocess with `restrictedpython` or
  equivalent; never trusted to execute arbitrary code at load time.
- `.onnx` models run through `onnxruntime.InferenceSession` with the operator allowlist
  specified in `libvmaf/src/dnn/allowed_ops.txt`.
