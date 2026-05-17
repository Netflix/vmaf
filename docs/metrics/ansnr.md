# ANSNR

ANSNR (Average Noise-to-Signal Ratio) measures the ratio of distortion energy
to signal energy averaged over the frame, providing a complementary view to
PSNR for characterizing codec noise behavior.

## Variant

| Extractor name | Algorithm | Options |
|---|---|---|
| `float_ansnr` | Floating-point implementation | `enable_chroma` |

## `float_ansnr` extractor

The extractor computes the noise-to-signal ratio in the spatial domain using
floating-point arithmetic. It is the extractor invoked when VMAF model JSON
files reference `"float_ansnr"`.

### Output features

| Feature name | Description | Condition |
|---|---|---|
| `float_ansnr` | ANSNR on the luma (Y) plane | Always |
| `float_ansnr_cb` | ANSNR on the Cb (U) chroma plane | `enable_chroma=true` only |
| `float_ansnr_cr` | ANSNR on the Cr (V) chroma plane | `enable_chroma=true` only |

## Options

- `enable_chroma` (bool, default `false`): emit per-plane `_cb` and `_cr` scores in addition to luma. YUV400P sources are always luma-only.

### How to run

```bash
# Luma-only ANSNR (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature float_ansnr --output /dev/stdout

# Per-channel ANSNR (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'float_ansnr:enable_chroma=true' --output /dev/stdout
```

## See also

- [Features](features.md) - full feature extractor reference
