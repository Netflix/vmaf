# libvmaf DNN runtime

C-side integration for running tiny ONNX models inside libvmaf — the runtime
consumed by the `--tiny-model` CLI flag and the ffmpeg `libvmaf` filter's
`tiny_model=` option.

## Files

| File | Role |
| --- | --- |
| [dnn_api.c](dnn_api.c) | Public `libvmaf/dnn.h` entry points |
| [ort_backend.{c,h}](ort_backend.c) | ONNX Runtime C-API session wrapper + EP select |
| [model_loader.{c,h}](model_loader.c) | File sniff, size cap, sidecar JSON parse |
| [op_allowlist.{c,h}](op_allowlist.c) | Allowlist of permitted ONNX op types |
| [tensor_io.{c,h}](tensor_io.c) | luma ↔ F32/F16 tensor conversion + normalization |

## Public API

Declared in [`libvmaf/include/libvmaf/dnn.h`](../../include/libvmaf/dnn.h):

```c
int vmaf_dnn_available(void);

int vmaf_use_tiny_model(VmafContext *ctx,
                        const char *onnx_path,
                        const VmafDnnConfig *cfg);
```

`VmafDnnConfig` selects an execution provider (`VMAF_DNN_DEVICE_{AUTO,CPU,CUDA,OPENVINO,ROCM}`),
intra-op thread count, and an fp16-io hint. AUTO picks the best EP compiled
into ORT.

## Build

Meson feature option `enable_dnn` — `auto` (default), `enabled`, `disabled`.

```bash
meson setup build -Denable_dnn=enabled     # require ORT, fail build if missing
meson setup build -Denable_dnn=auto        # use ORT if present, stub otherwise
meson setup build -Denable_dnn=disabled    # never link ORT
```

When ORT is missing or disabled, the public symbols still exist but return
`-ENOSYS`; `vmaf_dnn_available()` returns 0.

## Security

- Size cap (`VMAF_MAX_MODEL_BYTES`, default 50 MB).
- `op_allowlist.c` rejects graphs with ops outside a narrow set (conv / gemm /
  elementwise / pool / norm / common activations).
- Sidecar JSON (`<name>.json` next to `<name>.onnx`) records kind, opset,
  normalization, and expected output range for runtime sanity checks.
- Cosign keyless signatures on release artefacts — see [SECURITY.md](../../../SECURITY.md).

## Shipped models

[`model/tiny/`](../../../model/tiny/) — each `.onnx` has a matching `.json`
written by `vmaf-train register`.
