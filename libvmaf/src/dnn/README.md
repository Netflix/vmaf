# libvmaf DNN runtime

C-side integration for running tiny ONNX models inside libvmaf.

## Status

Scaffolding only. Header + loader stub are in place; the feature extractor
hookup is gated behind `-Denable_dnn=true` (default **off**) and is wired
into meson once a first shippable model lands under
[`model/tiny/`](../../../model/tiny/).

## Design

- Runtime: **ONNX Runtime** (C API), linked dynamically.
- Execution providers: CPU (baseline), CUDA (if built with `-Denable_cuda=true`),
  DirectML / CoreML / SYCL (via TensorRT EP or oneDNN EP) once validated.
- Model artefacts: shipped under `model/tiny/*.onnx` and discoverable via
  the same `--model version=<name>` flag used for the stock SVR models.
- Determinism: pinned opset (17), pinned EP config per backend, ULP drift
  across EPs bounded by the same cross-backend gate as the classical path.

## API sketch

```c
#include "dnn/vmaf_dnn.h"

VmafDnnSession *sess = NULL;
int err = vmaf_dnn_open(&sess, "model/tiny/vmaf_tiny_v0.onnx",
                        VMAF_DNN_PROVIDER_CPU);
/* feed features → score */
vmaf_dnn_infer(sess, feat, n_feat, &score);
vmaf_dnn_close(sess);
```

## License

BSD-3-Clause-Plus-Patent.
