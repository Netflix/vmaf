# Tiny AI — inference

Three consumer surfaces share one runtime: `vmaf` CLI, libvmaf C API, and
ffmpeg filters. All three funnel through
[`libvmaf/src/dnn/ort_backend.c`](../../libvmaf/src/dnn/ort_backend.c).

## Prerequisites

- libvmaf built with `-Denable_dnn=enabled` (or `auto` with ONNX Runtime
  discoverable via `pkg-config`).
- ONNX Runtime ≥ 1.20 — on Ubuntu 24.04, install via:

  ```bash
  scripts/dev-setup/ubuntu.sh --with-onnxruntime
  ```

  Other distros have the same flag.
- A `.onnx` model + sidecar `.json` pair under `model/tiny/` or anywhere
  else — the CLI flag accepts an absolute path.

Verify at runtime:

```bash
vmaf --help | grep -- '--tiny-model'   # must list the flag
vmaf --tiny-model /missing.onnx 2>&1   # should print a clear error,
                                       # not "option not found"
```

## Surface 1 — the `vmaf` CLI

```bash
# C1 — drop-in augmentation of the classic SVM.
vmaf -r ref.yuv -d dis.yuv -w 1920 -h 1080 -p 420 -b 8 \
     -m version=vmaf_v0.6.1 \
     --tiny-model model/tiny/vmaf_tiny_fr_v1.onnx \
     --tiny-device cuda

# C2 — no-reference.
vmaf -d dis.yuv -w 1920 -h 1080 -p 420 -b 8 \
     --tiny-model model/tiny/vmaf_nr_mobilenet_v1.onnx \
     --no-reference
```

New flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--tiny-model PATH` | — | ONNX model path; sidecar JSON at `${PATH%.onnx}.json`. |
| `--tiny-device STR` | `auto` | `auto` \| `cpu` \| `cuda` \| `openvino` \| `rocm`. |
| `--tiny-threads N` | `0` | CPU EP intra-op threads; 0 = ORT default. |
| `--tiny-fp16` | off | Request fp16 I/O when the EP supports it. |
| `--no-reference` | off | Skip reference loading; only valid with an NR tiny model. |

Output JSON gains a `tiny_model` block alongside `pooled_metrics`:

```json
{
  "pooled_metrics": { "vmaf": { "mean": 91.23... } },
  "tiny_model": {
    "name": "vmaf_tiny_fr_v1",
    "kind": "fr",
    "device": "cuda",
    "mean": 90.8...,
    "per_frame": [...]
  }
}
```

## Surface 2 — the libvmaf C API

```c
#include <libvmaf/libvmaf.h>
#include <libvmaf/dnn.h>

VmafContext *ctx;
vmaf_init(&ctx, (VmafConfiguration){ /* ... */ });

if (!vmaf_dnn_available()) {
    fprintf(stderr, "libvmaf built without --enable_dnn; rebuild.\n");
    return 1;
}

VmafDnnConfig dnn_cfg = {
    .device       = VMAF_DNN_DEVICE_CUDA,
    .device_index = 0,
    .threads      = 0,
    .fp16_io      = false,
};
int err = vmaf_use_tiny_model(ctx, "/models/vmaf_tiny_fr_v1.onnx", &dnn_cfg);
if (err < 0) { /* handle -errno */ }

/* … feed frames as usual; tiny-model scores appear in the same
     per-frame collector the built-in SVM uses. */
```

The sidecar JSON is discovered automatically at
`${onnx_path%.onnx}.json`. Its `kind` field (`fr` / `nr`) tells libvmaf
whether to expect a reference.

## Surface 3 — ffmpeg filters

Apply `ffmpeg-patches/*.patch` against a pinned FFmpeg SHA (see
[`ffmpeg-patches/test/build-and-run.sh`](../../ffmpeg-patches/test/build-and-run.sh))
then:

```bash
# C1 / C2 scoring through vf_libvmaf.
ffmpeg -i ref.mp4 -i dis.mp4 \
    -lavfi "[0:v][1:v]libvmaf=tiny_model=/models/vmaf_tiny_fr_v1.onnx:tiny_device=cuda" \
    -f null -

# C3 learned pre-filter.
ffmpeg -i in.mp4 \
    -vf "vmaf_pre=model=/models/filter_denoise_residual_v1.onnx:device=cuda" \
    out.mp4
```

## Execution-provider matrix

| Backend flag | ORT EP | Notes |
| --- | --- | --- |
| `--tiny-device cpu` | CPUExecutionProvider | Always available. |
| `--tiny-device cuda` | CUDAExecutionProvider | Requires CUDA-enabled ORT; shares context with libvmaf-cuda. |
| `--tiny-device openvino` | OpenVINOExecutionProvider | Covers Intel GPU / SYCL / oneAPI. |
| `--tiny-device rocm` | ROCmExecutionProvider | Wired, untested until HIP backend lands. |
| `--tiny-device auto` | best available | Probes CUDA → OpenVINO → ROCm → CPU. |

## Expected cross-device variance

Running the same `.onnx` on two different EPs produces near-identical
scores:

- CPU vs CUDA (FP32): within **1e-4**.
- CPU vs CUDA (FP16 via `--tiny-fp16`): within **1e-2**.

CI exercises CPU-only; GPU parity is checked manually on the dev workstation
for now (planned: self-hosted runner).
