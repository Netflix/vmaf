# VIF

VIF (Visual Information Fidelity) measures how much of the reference image's
visual information is preserved in the distorted image, using a statistical
model of natural scenes and human visual system sensitivity.

## Variant

| Extractor name | Algorithm | Options |
| --- | --- | --- |
| `integer_vif` | Integer fixed-point multi-scale implementation | `enable_chroma`, `vif_enhn_gain_limit`, `vif_skip_scale0` |

## `integer_vif` extractor

The extractor uses an integer fixed-point implementation of the VIF algorithm
across multiple spatial scales. It is the extractor invoked when VMAF model
JSON files reference `"integer_vif"`.

### Output features

| Feature name | Description | Condition |
| --- | --- | --- |
| `integer_vif` | VIF on the luma (Y) plane | Always |
| `integer_vif_cb` | VIF on the Cb (U) chroma plane | `enable_chroma=true` only |
| `integer_vif_cr` | VIF on the Cr (V) chroma plane | `enable_chroma=true` only |
| `integer_vif_scale0` | VIF at scale 0 (luma) | Always |
| `integer_vif_scale1` | VIF at scale 1 (luma) | Always |
| `integer_vif_scale2` | VIF at scale 2 (luma) | Always |
| `integer_vif_scale3` | VIF at scale 3 (luma) | Always |

## Options

| Option | Alias | Type | Default | Range | Effect |
| --- | --- | --- | --- | --- | --- |
| `enable_chroma` | — | bool | `false` | — | Emit per-plane `_cb` / `_cr` scores in addition to luma. YUV400P sources are always luma-only. |
| `vif_enhn_gain_limit` | `egl` | double | `100.0` | `1.0–100.0` | Cap the per-pixel enhancement-gain ratio so over-sharpened content cannot saturate the score. Set to `1.0` to disable enhancement gain entirely. |
| `vif_skip_scale0` | `ssclz` | bool | `false` | — | Skip scale-0 (finest pyramid level) calculations; scale-0 outputs are forced to `0.0` and excluded from the fused score. Matches the CPU path for GPU backends. |
| `debug` | — | bool | `false` | — | Emit additional per-scale numerator/denominator debug metrics. |

### How to run

```bash
# Luma-only VIF (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature integer_vif --output /dev/stdout

# Per-channel VIF (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'integer_vif:enable_chroma=true' --output /dev/stdout

# Skip scale-0 (GPU-parity mode)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'integer_vif:vif_skip_scale0=true' --output /dev/stdout
```

## See also

- [Features](features.md) - full feature extractor reference
