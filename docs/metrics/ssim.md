# SSIM

SSIM (Structural Similarity Index Measure) quantifies perceptual image quality
by comparing luminance, contrast, and structure between a reference and a
distorted frame.

## Variant

The fork ships one CPU SSIM extractor:

| Extractor name | Algorithm | Options |
|---|---|---|
| `integer_ssim` | Integer fixed-point implementation | `enable_chroma` |

## `integer_ssim` extractor

The extractor uses an integer fixed-point computation compatible with the
upstream Netflix reference. It is the extractor invoked when VMAF model JSON
files reference `"integer_ssim"`.

### Output features

| Feature name | Description | Condition |
|---|---|---|
| `integer_ssim` | SSIM on the luma (Y) plane | Always |
| `integer_ssim_cb` | SSIM on the Cb (U) chroma plane | `enable_chroma=true` only |
| `integer_ssim_cr` | SSIM on the Cr (V) chroma plane | `enable_chroma=true` only |

## Options

- `enable_chroma` (bool, default `false`): emit per-plane `_cb` and `_cr` scores in addition to luma. YUV400P sources are always luma-only.

### How to run

```bash
# Luma-only SSIM (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature integer_ssim --output /dev/stdout

# Per-channel SSIM (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'integer_ssim:enable_chroma=true' --output /dev/stdout
```

## See also

- [MS-SSIM](ms-ssim.md) - multi-scale structural similarity
- [SSIMULACRA2](ssimulacra2.md) - perceptually tuned alternative
