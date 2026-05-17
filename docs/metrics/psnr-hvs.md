# PSNR-HVS

PSNR-HVS (Peak Signal-to-Noise Ratio - Human Visual System) extends traditional
PSNR with contrast sensitivity function (CSF) weighting applied in the DCT
domain, making it more sensitive to perceptually significant distortions.

## Variants

| Extractor name | Algorithm | Options |
|---|---|---|
| `psnr_hvs` | CSF-weighted DCT-domain PSNR | `enable_chroma` |

## `psnr_hvs` extractor

The extractor computes PSNR in the 8x8 DCT domain with per-coefficient
weighting derived from a human visual system contrast sensitivity model
(Ponomarenko et al.). It is the extractor invoked when VMAF model JSON files
reference `"psnr_hvs"`.

### Output features

| Feature name | Description | Condition |
|---|---|---|
| `psnr_hvs` | HVS-weighted PSNR on the luma (Y) plane | Always |
| `psnr_hvs_cb` | HVS-weighted PSNR on the Cb (U) plane | `enable_chroma=true` only |
| `psnr_hvs_cr` | HVS-weighted PSNR on the Cr (V) plane | `enable_chroma=true` only |

## Options

- `enable_chroma` (bool, default `false`): emit per-plane `_cb` and `_cr` scores in addition to luma. YUV400P sources are always luma-only.

### How to run

```bash
# Luma-only PSNR-HVS (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature psnr_hvs --output /dev/stdout

# Per-channel PSNR-HVS (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'psnr_hvs:enable_chroma=true' --output /dev/stdout
```

## See also

- [Features](features.md) - full feature extractor reference
