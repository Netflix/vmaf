# Tiny AI — overview

The **Tiny AI** surface lets you ship small, specialized perceptual-quality
models alongside the classic VMAF SVM without introducing a second ML
runtime or giving up libvmaf's C-only deployment story. The whole feature
is gated on `-Denable_dnn=auto|enabled|disabled` (default `auto`) and
consumed through a single public header, `libvmaf/dnn.h`.

## The four capabilities

| # | Name | Shape | What you do with it |
| --- | --- | --- | --- |
| **C1** | FR regressor | feature-vector → MOS | Replace / augment the upstream `vmaf_v0.6.1` SVM with a lightweight MLP trained on your own reference-dataset. |
| **C2** | NR metric | frame → MOS | Predict quality without a reference (live encodes, consumer telemetry). |
| **C3** | Learned filter | frame → frame | Denoise / deblock / sharpen before encoding to squeeze out VMAF/PSNR budget. |
| **C4** | LLM dev helpers | repo-time only | Review, commit-msg drafting, docgen — never linked into libvmaf. |

C1/C2/C3 share **one** runtime (ONNX Runtime C API). C4 is independent and
lives in [`dev-llm/`](../../dev-llm/).

## How the pieces fit together

```
       ┌─────────────────────────┐
       │        ai/              │  torch + lightning + typer
       │  (train / export / reg) │  → ONNX
       └────────────┬────────────┘
                    │  .onnx + sidecar .json
                    ▼
       ┌─────────────────────────┐
       │  model/tiny/ (git-lfs)  │  committed tiny models
       └────────────┬────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────┐
   │     libvmaf/src/dnn/  (C, ORT)       │   one runtime
   │     vmaf_use_tiny_model(...)         │   shared by all surfaces
   └──────────────┬────────────┬──────────┘
                  │            │
                  ▼            ▼
           vmaf CLI       ffmpeg vf_libvmaf  +  vf_vmaf_pre
```

**Key invariant.** Training lives in Python and depends on PyTorch +
Lightning. Runtime lives in C and depends only on ONNX Runtime. The
boundary is the `.onnx` + sidecar JSON pair on disk.

## When to reach for Tiny AI

| You want to… | Use | Read |
| --- | --- | --- |
| Beat the upstream SVM's PLCC on your own MOS data | C1 | [training.md](training.md) |
| Score VMAF without a reference | C2 | [inference.md](inference.md) |
| Pre-filter frames before encoding | C3 | [inference.md](inference.md) |
| Compare a new model's PLCC/SROCC/RMSE to the SVM baseline | — | [benchmarks.md](benchmarks.md) |
| Understand the operator allowlist + signature model | — | [security.md](security.md) |

## Decisions

The full design trail is in [`.workingdir2/phases/03-framework/3k-tiny-ai.md`](../../.workingdir2/phases/03-framework/3k-tiny-ai.md)
(D20–D23). This doc set is the user-facing distillation.
