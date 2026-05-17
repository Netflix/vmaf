# `saliency_student_v2` — Resize-decoder ablation on the v1 recipe

`saliency_student_v2` is a fork-trained tiny saliency student that
exercises the bilinear-resize-then-conv decoder pattern admitted by
[ADR-0258](../../adr/0258-onnx-allowlist-resize.md). It ships as a
parallel artefact alongside [`saliency_student_v1`](saliency_student_v1.md)
under `model/tiny/`. v1 remains the production weights for the C-side
`mobilesal` extractor; v2 is staged for a future production-flip PR
after empirical validation in real ROI encodes.

> **Status — staged 2026-05-09.** Architectural ablation; production
> flip is gated on a follow-up A/B PR. See
> [ADR-0332](../../adr/0332-saliency-student-v2-resize-decoder.md)
> and [Research-0089](../../research/0089-saliency-student-v2-resize-decoder.md).

## What changed vs v1

The encoder, channels, skip connections, loss, optimizer, schedule,
augmentation pipeline, seed, dataset, and held-out split are all
**identical** to v1 — this is a single-variable ablation on the
decoder upsampler:

| Decoder upsampler | v1 | v2 |
| --- | --- | --- |
| Op | `nn.ConvTranspose2d(k=2, s=2, no bias)` | `F.interpolate(scale=2, bilinear, align_corners=False)` + `nn.Conv2d(k=3, p=1, no bias)` |
| ONNX op | `ConvTranspose` | `Resize` (mode=`linear`, `coordinate_transformation_mode=half_pixel`) + `Conv` |
| Allowlist gate | pre-ADR-0258 | post-ADR-0258 |
| Trainable params | 112 841 | 123 721 (+9.6 %) |

## Shipped checkpoint

| Field | Value |
| --- | --- |
| Model id | `saliency_student_v2` |
| Display name | `vmaf_tiny_saliency_student_v2` |
| Location | `model/tiny/saliency_student_v2.onnx` |
| Sidecar | `model/tiny/saliency_student_v2.json` |
| Architecture | TinyU-Net with Resize+3×3-Conv decoder; otherwise identical to v1 |
| Trainable parameters | 123 721 |
| Input | `input` — float32 NCHW `[1, 3, H, W]` ImageNet-normalised RGB |
| Output | `saliency_map` — float32 NCHW `[1, 1, H, W]` per-pixel saliency in [0, 1] |
| ONNX opset | 17 |
| Training corpus | DUTS-TR (Wang et al. 2017), same 10 553 pairs as v1 — *not* redistributed in-tree |
| Held-out validation IoU | **0.7105** on the 5 % DUTS-TR validation fold (seed=42) — gate PASS vs v1's 0.6558 (+0.0547 / +8.3 %); full per-epoch trace in `build_artifacts/saliency_student_v2_train.json` |
| PyTorch ↔ ONNX parity max-abs-diff | 3.278e-6 (threshold 1e-5; same threshold v1 used) |
| Training wall-clock | 573.0 s (≈ 9.5 min) on RTX 4090, CUDA 13, PyTorch 2.11 |
| License | BSD-3-Clause-Plus-Patent |
| Trainer | `ai/scripts/train_saliency_student_v2.py` |

The training run is deterministic given a fixed `--seed` (default
42) and pinned PyTorch / CUDA versions.

## Training corpus provenance

Identical to v1 — DUTS-TR (Wang, Lu, Wang, Feng, Wang, Yin, Ruan,
"Learning to Detect Salient Objects with Image-Level Supervision",
CVPR 2017). Project page: <http://saliencydetection.net/duts/>.
Direct URL: `https://saliencydetection.net/duts/download/DUTS-TR.zip`.
Distribution: free for academic and research purposes (see project
page). SHA-256 of the redistributed zip — see
[`saliency_student_v1.md`](saliency_student_v1.md#training-corpus-provenance).
The DUTS images are not committed to this repository; only the
trained weights are.

## Op-allowlist conformance

Every op in the v2 graph is on
[`libvmaf/src/dnn/op_allowlist.c`](../../../libvmaf/src/dnn/op_allowlist.c)
post-ADR-0258:

```text
Conv, Concat, Constant, MaxPool, Relu, Resize, Sigmoid
```

`Resize` is the new entry vs v1; `ConvTranspose` is no longer
required. `Constant` materialises the integer-pair output spatial
dims for the resample target — benign, on the allowlist.

ADR-0258's policy: scope is op-type only; attribute enforcement is
delegated to ORT. v2 declares `mode='linear'`,
`coordinate_transformation_mode='half_pixel'`, `antialias=0` (the
PyTorch `align_corners=False` bilinear default at opset 17). The
fork's wire-format scanner does not gate these attributes; ORT
accepts the combination unconditionally.

## Usage — CLI

```bash
vmaf \
    --reference ref.yuv \
    --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --feature mobilesal \
    --feature_params mobilesal:model_path=model/tiny/saliency_student_v2.onnx \
    --output score.json
```

Equivalently, set the model path via env var:

```bash
VMAF_MOBILESAL_MODEL_PATH=model/tiny/saliency_student_v2.onnx \
    vmaf --reference ref.yuv --distorted dist.yuv \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --feature mobilesal --output score.json
```

## Usage — C API

```c
#include <libvmaf/libvmaf.h>

VmafFeatureDictionary *opts = NULL;
vmaf_feature_dictionary_set(&opts, "model_path",
                            "model/tiny/saliency_student_v2.onnx");
int err = vmaf_use_feature(ctx, "mobilesal", opts);
/* ... vmaf_score_pooled(ctx, ..., "saliency_mean", ...) for the per-frame mean */
```

## Reproducing the model

```bash
# 1. fetch DUTS-TR (~271 MB) — not redistributed in-tree
mkdir -p $HOME/datasets/duts && cd $HOME/datasets/duts
wget https://saliencydetection.net/duts/download/DUTS-TR.zip
unzip DUTS-TR.zip

# 2. train + export (≈ 10–15 min on RTX 4090 / 24 GB at 256×256, batch 32)
.venv/bin/python ai/scripts/train_saliency_student_v2.py \
    --duts-root $HOME/datasets/duts/DUTS-TR \
    --output    model/tiny/saliency_student_v2.onnx \
    --epochs 50 --batch-size 32 --lr 1e-3 --seed 42 \
    --metrics-out build_artifacts/saliency_student_v2_train.json

# 3. validate against the registry
.venv/bin/python ai/scripts/validate_model_registry.py
```

## Why ship a parallel artefact instead of replacing v1

Per the user direction captured in
[ADR-0332](../../adr/0332-saliency-student-v2-resize-decoder.md)'s
References: "v1 stays as production weight. v2 ships as a parallel
artifact; promotion to production is a future PR after empirical
validation in real ROI encodes." Held-out IoU on a 5 % validation
fold is necessary but not sufficient — real-encode A/B is the
production-flip gate.

## Known limitations

Inherited from v1:

- 8-bit YUV only (10-bit gated on the LPIPS loader path).
- BT.709 limited-range Y'CbCr → RGB at the C side.
- ~124 K parameters is well below upstream u2netp (~4.7 M).
- 5 % held-out validation only; external evaluation is a follow-up.

Specific to v2:

- The "Resize + 3×3 Conv" pattern adds ~10 K parameters vs v1
  (3×3 vs 2×2 kernel). Inference latency at 256×256 is within
  measurement noise of v1 on RTX 4090 / Intel Xeon.
- ADR-0258 admits `Resize` op-type-only without attribute
  enforcement. v2's exported attributes (`linear`, `half_pixel`,
  `antialias=0`) are documented here, but the fork's scanner
  does not validate them — ORT does. Future tiny-AI consumers
  using `Resize` should follow the same documented attribute
  contract.

## Related

- [`saliency_student_v1.md`](saliency_student_v1.md) — production
  weights for the `mobilesal` extractor; v2 is the architectural
  successor.
- [`mobilesal.md`](mobilesal.md) — the C-side extractor wiring;
  unchanged by this PR.
- [ADR-0258](../../adr/0258-onnx-allowlist-resize.md) — admits
  `Resize` to the allowlist.
- [ADR-0286](../../adr/0286-saliency-student-fork-trained-on-duts.md)
  — v1 decision record.
- [ADR-0332](../../adr/0332-saliency-student-v2-resize-decoder.md)
  — v2 decision record (this PR).
- [Research-0089](../../research/0089-saliency-student-v2-resize-decoder.md)
  — companion digest.
- [ADR-0042](../../adr/0042-tinyai-docs-required-per-pr.md) —
  tiny-AI doc-substance rule this card satisfies.
