# `saliency_student_v1` — fork-trained tiny saliency student

`saliency_student_v1` is a small, fork-trained image-saliency model
that replaces the smoke-only `mobilesal_placeholder_v0` checkpoint as
the production weights for the C-side `mobilesal` feature extractor.
It runs through the same `feature_mobilesal.c` extractor as before
(no C changes, no public-API changes); the only delta is the ONNX
weights it loads.

> **Status — shipped 2026-05-03.** Supersedes
> `mobilesal_placeholder_v0` (which remains in the registry with
> `smoke: true` for legacy reasons). See
> [ADR-0286](../../adr/0286-saliency-student-fork-trained-on-duts.md)
> and [Research-0054](../../research/0062-saliency-student-from-scratch-on-duts.md).

## What the output means

The C-side extractor still reduces the per-pixel saliency map to a
single scalar feature `saliency_mean`, one value per frame. With
`saliency_student_v1` weights this is now a content-dependent signal
rather than a content-independent constant.

| Value | Interpretation |
| --- | --- |
| **~0.0** | Flat / featureless content; no salient subject |
| **~0.1 – 0.3** | Typical natural-content frame |
| **~0.4 – 0.6** | Foreground subject occupies a sizeable fraction of the frame |
| **~0.7+** | Subject dominates; mostly-salient content |
| **1.0** | Saturated (every pixel maxed) — usually a sign of model misuse |

Saliency-mean is **not** a quality score on its own — it is a content
descriptor. Downstream consumers correlate `saliency_mean` against
existing metric features (e.g. `vmaf`, `lpips`, `psnr`) to study how
foreground-vs-background distortion affects subjective quality.

## Shipped checkpoint

| Field | Value |
| --- | --- |
| Model id | `saliency_student_v1` |
| Display name | `vmaf_tiny_saliency_student_v1` |
| Location | `model/tiny/saliency_student_v1.onnx` |
| Sidecar | `model/tiny/saliency_student_v1.json` |
| Architecture | TinyU-Net — 3 down + 3 up encoder/decoder with skip connections; ConvTranspose stride-2 upsampling; sigmoid head |
| Trainable parameters | 112 841 |
| Input | `input` — float32 NCHW `[1, 3, H, W]` ImageNet-normalised RGB |
| Output | `saliency_map` — float32 NCHW `[1, 1, H, W]` per-pixel saliency in [0, 1] |
| ONNX opset | 17 |
| Training corpus | DUTS-TR (Wang et al. 2017) — 10 553 RGB images + binary saliency masks; *not* redistributed in-tree |
| Held-out validation IoU | recorded in `build_artifacts/saliency_student_v1_train.json` (`best_val_iou`); ship gate is ≥ 0.5 |
| License | BSD-3-Clause-Plus-Patent (the trained weights are wholly fork-owned; DUTS images are training input only and not bundled) |
| Exporter | `ai/scripts/train_saliency_student.py` |

The training run is deterministic given a fixed `--seed` (default 42)
and pinned PyTorch / CUDA versions. The ONNX exporter writes the
graph with `do_constant_folding=True` and `training=TrainingMode.EVAL`
so BatchNorm folds into the preceding Conv weights.

## Training corpus provenance

| Field | Value |
| --- | --- |
| Dataset | DUTS-TR (training split of DUTS — "Detect Usage and Track Saliency") |
| Authors | Wang, Lu, Wang, Feng, Wang, Yin, Ruan |
| Citation | "Learning to Detect Salient Objects with Image-Level Supervision", CVPR 2017 |
| Project page | <http://saliencydetection.net/duts/> |
| Direct URL | `https://saliencydetection.net/duts/download/DUTS-TR.zip` |
| Distribution licence | "Free for academic and research purposes" (see project page) |
| Last-Modified header (at training time) | 2025-03-10 |
| Content-Length | 270 997 309 bytes (~271 MB) |
| SHA-256 | `ce61e023c8f59d022b4d46981cf16813b83d089242e6489a45630d83962ea058` |
| Pair count | 10 553 (image + binary mask, 1:1) |

**Acknowledgement.** This fork's `saliency_student_v1` weights were
trained on DUTS-TR. We thank the DUTS authors for distributing the
dataset under permissive academic-research terms. The DUTS images
themselves are deliberately *not* committed to this repository; only
the trained weights are.

## Input / output contract

The C extractor binds tensors by name; `saliency_student_v1` declares
exactly the names the existing `feature_mobilesal.c` already binds to,
so the new file is a true drop-in.

```text
inputs:
  input         float32[1, 3, H, W]   ImageNet-normalised RGB, NCHW
outputs:
  saliency_map  float32[1, 1, H, W]   per-pixel saliency in [0, 1]
```

`H` and `W` are dynamic in the ONNX — the model is fully convolutional
so inference at native resolution is unrestricted (subject to the
rule of thumb that DUTS-TR was trained at 256×256 random crops; very
small inputs may lose quality). ImageNet normalisation
(mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) is
applied in the C side via the shared
`vmaf_tensor_from_rgb_imagenet()` helper, identical to LPIPS's
wiring.

## Op-allowlist

Every op in the graph is on `libvmaf/src/dnn/op_allowlist.c`:
`Conv`, `BatchNormalization` (folded into `Conv` at export by
constant folding), `Relu`, `MaxPool`, `ConvTranspose`, `Concat`,
`Sigmoid`. `Resize` is *not* used — the decoder upsamples with
`ConvTranspose` (stride 2) so the graph loads clean against vanilla
origin/master with no allowlist patch in this PR.

## Usage — CLI

```bash
vmaf \
    --reference ref.yuv \
    --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --feature mobilesal \
    --feature_params mobilesal:model_path=model/tiny/saliency_student_v1.onnx \
    --output score.json
```

The output JSON gains a per-frame `saliency_mean` column alongside any
other features requested in the same run.

Equivalently, set the model path via env var:

```bash
VMAF_MOBILESAL_MODEL_PATH=model/tiny/saliency_student_v1.onnx \
    vmaf --reference ref.yuv --distorted dist.yuv \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --feature mobilesal --output score.json
```

## Usage — C API

```c
#include <libvmaf/libvmaf.h>

VmafFeatureDictionary *opts = NULL;
vmaf_feature_dictionary_set(&opts, "model_path",
                            "model/tiny/saliency_student_v1.onnx");
int err = vmaf_use_feature(ctx, "mobilesal", opts);
/* ... vmaf_score_pooled(ctx, ..., "saliency_mean", ...) for the per-frame mean */
```

## Reproducing the model

```bash
# 1. fetch DUTS-TR (~271 MB) — not redistributed in-tree
mkdir -p $HOME/datasets/duts && cd $HOME/datasets/duts
wget https://saliencydetection.net/duts/download/DUTS-TR.zip
unzip DUTS-TR.zip

# 2. train + export (≈ 10 min on RTX 4090 / 24 GB at 256×256, batch 32)
.venv/bin/python ai/scripts/train_saliency_student.py \
    --duts-root $HOME/datasets/duts/DUTS-TR \
    --output    model/tiny/saliency_student_v1.onnx \
    --epochs 50 --batch-size 32 --lr 1e-3 --seed 42 \
    --metrics-out build_artifacts/saliency_student_v1_train.json

# 3. validate against the registry
.venv/bin/python ai/scripts/validate_model_registry.py
```

The training script is deterministic given the seed and the pinned
PyTorch / CUDA versions; re-runs reproduce the val-IoU curve to
within float-rounding noise.

## Known limitations

Inherited from the C-side extractor (see
[`mobilesal.md`](mobilesal.md)):

- **Bit depth**: 8-bit YUV only. 10-bit support is gated on the same
  loader path landing for LPIPS.
- **Pixel format**: `YUV420P`, `YUV422P`, `YUV444P` accepted;
  `YUV400P` rejected.
- **Colour space**: BT.709 limited-range Y'CbCr → RGB at the C side.
  BT.2020 / full-range is approximate.

Specific to `saliency_student_v1`:

- **Capacity**: ~113 K parameters is well below upstream u2netp
  (~4.7 M). Absolute IoU on external test sets (DUTS-TE, ECSSD) is
  expected to be below SOTA. v1 is a useful baseline, not a
  state-of-the-art SOD model.
- **Training corpus diversity**: DUTS-TR is the de-facto standard
  single-dataset SOD corpus but skews toward natural-photo content.
  Synthetic / animated / heavily-graphic content may saturate the
  saliency map. Multi-dataset training is a future
  `saliency_student_v2` follow-up.
- **In-loop validation only**: v1 reports validation IoU on a 5 %
  hold-out of DUTS-TR. External evaluation against DUTS-TE / ECSSD is
  a follow-up.

## Related

- [`mobilesal.md`](mobilesal.md) — original placeholder + extractor
  wiring; superseded by this card for production use, retained for
  history.
- [ADR-0218](../../adr/0218-mobilesal-saliency-extractor.md) —
  original `mobilesal` extractor design.
- [ADR-0257](../../adr/0257-mobilesal-real-weights-deferred.md) —
  the upstream-MobileSal deferral that this PR partly unblocks.
- [ADR-0286](../../adr/0286-saliency-student-fork-trained-on-duts.md)
  — the decision record for this model.
- [Research-0054](../../research/0062-saliency-student-from-scratch-on-duts.md)
  — dataset, architecture, and recipe digest.
- [ADR-0042](../../adr/0042-tinyai-docs-required-per-pr.md) —
  tiny-AI doc-substance rule this card satisfies.
