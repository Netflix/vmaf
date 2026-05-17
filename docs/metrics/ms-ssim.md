# MS-SSIM

MS-SSIM (Multi-Scale Structural Similarity Index Measure) extends SSIM to a
multi-resolution pyramid, providing a perceptual similarity metric that is
robust to viewing distance and display resolution variation. Each scale
captures structural information at a different spatial frequency.

## Variant

The fork ships one CPU MS-SSIM extractor:

| Extractor name | Algorithm | Options |
|---|---|---|
| `float_ms_ssim` | Floating-point IQA library, 5-scale Gaussian pyramid | `enable_lcs`, `enable_db`, `clip_db`, `enable_chroma` |

GPU twins (`float_ms_ssim_cuda`, `float_ms_ssim_sycl`, `float_ms_ssim_vulkan`)
currently expose only `enable_lcs`; `enable_chroma` support for GPU backends
is a planned follow-up.

## `float_ms_ssim` extractor

The extractor uses the IQA library's Gaussian-window floating-point
implementation with a 5-scale Laplacian pyramid (Wang et al. 2004). It is the
extractor invoked when VMAF model JSON files reference `"float_ms_ssim"`.

The minimum supported input resolution is 176x176. Smaller inputs cause the
5-level pyramid to fall below the 11-tap Gaussian kernel footprint and are
rejected with an error at init time (Netflix#1414 / ADR-0153).

### Output features

| Feature name | Description | Condition |
|---|---|---|
| `float_ms_ssim` | MS-SSIM on the luma (Y) plane | Always |
| `float_ms_ssim_cb` | MS-SSIM on the Cb (U) chroma plane | `enable_chroma=true` only |
| `float_ms_ssim_cr` | MS-SSIM on the Cr (V) chroma plane | `enable_chroma=true` only |
| `float_ms_ssim_l_scale0-4` | Per-scale luminance component | `enable_lcs=true`, luma only |
| `float_ms_ssim_c_scale0-4` | Per-scale contrast component | `enable_lcs=true`, luma only |
| `float_ms_ssim_s_scale0-4` | Per-scale structure component | `enable_lcs=true`, luma only |

## Options

- `enable_chroma` (bool, default `false`): emit per-plane `_cb` and `_cr` scores in addition to luma. YUV400P sources are always luma-only.
- `enable_lcs` (bool, default `false`): emit per-scale luminance, contrast, and structure intermediate components for the luma plane.
- `enable_db` (bool, default `false`): report the luma MS-SSIM score as dB (`-10 * log10(1 - score)`).
- `clip_db` (bool, default `false`): clip the dB score at the theoretical peak. Only meaningful when `enable_db=true`.

### How to run

```bash
# Luma-only MS-SSIM (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature float_ms_ssim --output /dev/stdout

# Per-channel MS-SSIM (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'float_ms_ssim:enable_chroma=true' --output /dev/stdout
```

## See also

- [SSIM](ssim.md) - single-scale structural similarity
- [SSIMULACRA2](ssimulacra2.md) - perceptually tuned alternative
