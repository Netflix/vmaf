# `vmaf-tune --saliency-aware`

`vmaf-tune recommend-saliency` runs a single saliency-aware encode. It
materialises a saliency sidecar from the shipped saliency model,
translates that sidecar into codec-specific ROI/QP controls, and then
dispatches the encode through the normal codec-adapter path.

The implementation lives in `tools/vmaf-tune/src/vmaftune/saliency.py`
and is wired through `tools/vmaf-tune/src/vmaftune/cli.py`.

## Quick Start

```shell
vmaf-tune recommend-saliency \
    --src ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --encoder libx264 \
    --preset medium --crf 23 \
    --saliency-offset -3 \
    --output roi.mp4
```

## What Happens

1. `load_saliency_sidecar()` loads an existing sidecar or runs the
   saliency model to create one.
2. `build_roi_plan()` converts frame-level saliency into encoder ROI
   controls.
3. `run_saliency_encode()` dispatches the codec-specific encode.
4. Unsupported ROI encoders fall back to a plain encode with a warning
   rather than failing the whole run.

Supported saliency ROI encoders are:

| Encoder | ROI channel |
| --- | --- |
| `libx264` | `-x264-params qpfile=...` |
| `libaom-av1` | patched FFmpeg `-qpfile ...` bridge |
| `libx265` | `-x265-params zones=...` |
| `libsvtav1` | `-svtav1-params qp-file=...` |
| `libvvenc` | `-vvenc-params ROIFile=...` |

The shipped default model is documented in
[`saliency_student_v1.md`](../ai/models/saliency_student_v1.md).

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--src PATH` | — | Source clip. |
| `--width / --height` | — | Source geometry. |
| `--pix-fmt` | `yuv420p` | Source pixel format. |
| `--framerate` | `24.0` | Source framerate. |
| `--duration` | `0.0` | Source duration. |
| `--encoder` | `libx264` | Codec adapter. |
| `--preset` | `medium` | Codec preset. |
| `--crf` | `23` | Base quality before ROI offsets. |
| `--saliency-offset` | `-3` | QP/quality offset applied to salient regions. |
| `--saliency-model PATH` | shipped model | Override saliency ONNX path. |
| `--saliency-aggregator` | `mean` | Temporal reducer for sampled per-frame saliency masks. One of `mean`, `ema`, `max`, `motion-weighted`. See [Temporal aggregation](#temporal-aggregation) below. |
| `--saliency-ema-alpha` | `0.6` | Current-frame weight when `--saliency-aggregator=ema`. Range `(0, 1]`; higher values weight recent frames more heavily. |
| `--ffmpeg-bin` | `ffmpeg` | FFmpeg binary. |
| `--output PATH` | — | Encoded output. |

## Temporal aggregation

`--saliency-aggregator` controls how the per-frame saliency masks produced
by `saliency_student_v1` are reduced to the single ROI pattern applied to
the encode pass.

| Aggregator | Behaviour | Use when |
| --- | --- | --- |
| `mean` | Per-pixel arithmetic mean across all sampled frames. Preserves the historical implementation. | Default, stable clips, and baseline comparisons. |
| `ema` | Exponential moving average; `--saliency-ema-alpha` is the weight of the current frame. Older frames decay geometrically. | Clips with scene changes or motion bursts where the most-recent frames dominate the salient region. |
| `max` | Per-pixel maximum over all sampled masks. | Missing a briefly salient object is worse than over-protecting background; conservative choice for sports or highlight reels. |
| `motion-weighted` | Weighted mean where each sampled frame is weighted by its luma delta from the previous sampled frame. Still frames contribute less than high-motion frames. | Motion-heavy clips where foreground objects define the perceptually important regions. |

All four reducers use the same `saliency_student_v1` ONNX weights and the
same downstream QP-offset mapping, so changing the aggregator does not
change the model contract or the encoder sidecar format. The default
(`mean`) matches pre-ADR-0396 behaviour and is suitable for most clips.

## See Also

- [`vmaf-tune.md`](vmaf-tune.md) — base tool.
- [`vmaf-tune-ffmpeg.md`](vmaf-tune-ffmpeg.md) — FFmpeg integration
  recipe.
- [`vmaf-roi-score.md`](vmaf-roi-score.md) — saliency-weighted scoring.
- [ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md) — design
  decision.
