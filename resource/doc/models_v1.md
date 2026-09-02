# Models (v1)

As of June 2026, this repository ships a new generation of VMAF models, referred to as **VMAF v1**. These models have demonstrated better accuracy compared to the previous set of models (VMAF v0). For more information please refer to the [tech blog](https://medium.com/netflix-techblog/vmaf-v1-good-is-not-good-enough-60d7e4244ea8). For the previous generation of models, see [models_v0.md](models_v0.md).

The v1 models live under `model/vmaf_v1.0.16/` and are selected via the `--model` option, in the same way as the v0 models.

## Overview of VMAF v1 models

VMAF v1 supports the following models:

- **Standard 1080p Model:** This model is calibrated for 1080p video viewed at a standard 3H distance. It uses an operating range of [0, 100].
- **Phone Model:** Derived by setting the normalized viewing distance to 5H (based on experimental data), this model adjusts the DLM, AIM, and chroma feature calculations to reflect reduced artifact visibility on smaller screens viewed from a greater relative distance. It retains the standard [0, 100] range.
- **4K Model:** We release two v1 4K models:
  - A **1.5H variant**, based on a discerning 4K@1.5H viewing condition. This variant is conceptually similar to its v0 4K counterpart and operates on a [0, 100] range. For most users, this variant is the default choice.
  - A **3H variant**, based on a consumer-like 4K@3H viewing condition. This variant operates on a [0, 110] range, which helps to quantify the additional perceptual benefit of 4K resolution over 1080p when both are viewed at 3H.

VMAF v1 should ideally be applied at 10-bit precision for SDR, which helps more accurately capture the presence of banding. Even if the encoded video is 8-bit, VMAF can still be measured at 10 bits by appropriately preprocessing both video inputs.

## Which model file should I use?

| Scenario                       | Display | Normalized viewing distance | Model file                                                   | Score range |
| ------------------------------ | ------- | --------------------------- | ------------------------------------------------------------ | ----------- |
| Standard 1080p                 | 1080p   | 3H                          | `model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json` | [0, 100]    |
| Phone                          | 1080p   | 5H                          | `model/vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json` | [0, 100]    |
| 4K default                     | 2160p   | 1.5H                        | `model/vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json` | [0, 100]    |
| 4K consumer TV                 | 2160p   | 3H               | `model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json` | [0, 110]    |

Example invocation for the default 1080p 3H model, using the `vmaf` command-line tool from `libvmaf`:

```shell script
./build/tools/vmaf \
    --reference ref.yuv \
    --distorted dis.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 10 \
    --model path=model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json \
    --output output.xml
```

### Specifying encode-side parameters

If the encode-side width, height, and bit depth of the distorted video are known (i.e., the dimensions and bit depth at which the video was actually encoded, before any rescaling for display), they can be passed to the CAMBI feature used by VMAF as additional `--model` parameters. For example, if we have a 1280x720 8-bit encoded video that is displayed on a 1080p display, we measure VMAF at a 1920x1080 resolution as follows:

```shell script
./build/tools/vmaf \
    --reference ref.yuv \
    --distorted dis.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 10 \
    --model path=model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json:cambi.enc_width=1280:cambi.enc_height=720:cambi.enc_bitdepth=8 \
    --output output.xml
```

These overrides are merged into the model's CAMBI feature options, so the CAMBI instance evaluated by VMAF is the one that takes them into account — no separate CAMBI instance is registered. When `enc_width`/`enc_height`/`enc_bitdepth` are not provided, CAMBI falls back to the input width/height and bitdepth.

## High-frame-rate (HFR) content

For high-frame-rate (HFR) content, VMAF v1 provides `_hfr` model variants under `model/vmaf_v1.0.16_hfr/`. The model-selection logic above carries over identically — just substitute the `_hfr` directory and filename.

In the context of these models, HFR refers to frame rates roughly double the common 24/30 fps cases (e.g., 60 fps); the `_hfr` variants are calibrated for the ~50/60 fps regime ("HFR" can refer to higher frame rates in some applications). Compared to the standard models, they use a wider, five-frame temporal motion window (differencing over frames i-2, i, i+2) with moving-average smoothing, which reduces the quality under-prediction that v0 showed at higher frame rates without inflating scores from the denser inter-frame signal.

HFR handling remains an area of active improvement — the current variants don't fully capture the perceptual impact of high frame rates, and we expect to refine it in future releases.
