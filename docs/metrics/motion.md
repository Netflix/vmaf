# Motion

Motion measures temporal activity in the **reference** stream by computing the
mean absolute difference (MAD) between consecutive Gaussian-blurred luma frames.
It is used as a VMAF model feature to weight distortion scores — scenes with high
motion are treated differently from static content.

Three extractor variants are registered:

| Extractor name   | Algorithm                      | Temporal? |
|------------------|-------------------------------|-----------|
| `motion`         | Integer fixed-point (Motion2) | Yes       |
| `motion_v2`      | Integer pipelined (Motion2 v2)| Yes       |
| `float_motion`   | Floating-point (Motion2)      | Yes       |

---

## `motion` extractor (integer fixed-point)

Registered name: `motion` (`VmafFeatureExtractor vmaf_fex_integer_motion`).

Applies a separable 5-tap Gaussian filter to each reference luma frame, keeps a
circular buffer of up to five blurred frames, and emits the minimum SAD across the
two-frame (or five-frame) temporal window.

### Output features

| Feature name                             | Description                                     | Condition         |
|------------------------------------------|-------------------------------------------------|-------------------|
| `VMAF_integer_feature_motion2_score`     | Motion2 score (shipped VMAF model input)        | Always            |
| `VMAF_integer_feature_motion3_score`     | Perceptually blended motion score               | Always            |
| `VMAF_integer_feature_motion_score`      | Raw (unfixed) motion score for back-compat      | `debug=true` only |

Frame 0 always emits `motion2_score = 0.0`.

### Output range

`[0, motion_max_val]`. Zero for a frozen reference; larger values indicate more
temporal activity. No inherent upper bound — clamped to `motion_max_val` (default
10 000).

### Options

| Option                    | Alias    | Type   | Default   | Range         | Effect                                                                 |
|---------------------------|----------|--------|-----------|---------------|------------------------------------------------------------------------|
| `debug`                   | —        | bool   | `true`    | —             | Emit `motion_score` (legacy unfixed variant) alongside `motion2_score` |
| `motion_force_zero`       | `force_0`| bool   | `false`   | —             | Override all emitted scores to `0.0`; used for deterministic fixtures  |
| `motion_fps_weight`       | `mfw`    | double | `1.0`     | `0.0–5.0`     | Multiplicative FPS-aware correction applied before clamping            |
| `motion_blend_factor`     | `mbf`    | double | `1.0`     | `0.0–1.0`     | Blend factor for `motion3_score`                                       |
| `motion_blend_offset`     | `mbo`    | double | `40.0`    | `0.0–1000.0`  | Score offset at which blending begins for `motion3_score`              |
| `motion_max_val`          | `mmxv`   | double | `10000.0` | `0.0–10000.0` | Upper clamp applied to emitted scores                                  |
| `motion_five_frame_window`| `mffw`   | bool   | `false`   | —             | Use a five-frame SAD window instead of three-frame (CPU only)          |
| `motion_moving_average`   | `mma`    | bool   | `false`   | —             | Apply a two-frame moving average to `motion3_score`                    |

### Backend coverage

| Backend        | Status    | Notes                                         |
|----------------|-----------|-----------------------------------------------|
| Scalar C       | Supported | Reference implementation                      |
| AVX2           | Supported | `x86/motion_avx2.c`                           |
| AVX-512        | Supported | `x86/motion_avx512.c`                         |
| NEON (AArch64) | Supported | `arm64/motion_neon.c`                         |
| CUDA           | Supported | `feature/cuda/integer_motion_cuda.c`          |
| SYCL           | Supported | `feature/sycl/integer_motion_sycl.cpp`        |
| Vulkan         | Supported | `feature/vulkan/integer_motion_vulkan.c`      |
| HIP            | Supported | `feature/hip/integer_motion_hip.c`            |
| Metal          | Supported | `feature/metal/integer_motion_metal.mm`       |

All GPU backends emit `motion2_score` and `motion3_score` in 3-frame window mode.
The 5-frame window (`motion_five_frame_window=true`) and `motion_moving_average`
are CPU-only; GPU paths return `-ENOTSUP` at `init()` when these are set.

### How to run

```bash
# Integer motion (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature motion --output /dev/stdout

# With FPS weight
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'motion:motion_fps_weight=0.8' \
    --output /dev/stdout
```

---

## `motion_v2` extractor (pipelined integer)

Registered name: `motion_v2` (`VmafFeatureExtractor vmaf_fex_integer_motion_v2`).

A pipelined re-implementation that exploits the linearity of the blur kernel:
`SAD(blur(f[N-1]), blur(f[N])) == sum(|blur(f[N-1] - f[N])|)`. The frame
difference, blur, and absolute-sum are fused into a single row-at-a-time
pipeline requiring only one scratch row. Per-frame blurred-state storage is
eliminated.

### Output features

| Feature name                                  | Description                                       | Condition  |
|-----------------------------------------------|---------------------------------------------------|------------|
| `VMAF_integer_feature_motion_v2_sad_score`    | Per-frame sum of absolute blurred differences     | Always     |
| `VMAF_integer_feature_motion2_v2_score`       | Motion2-equivalent smoothed score                 | Always     |

### Output range

`[0, motion_max_val]`. Same units as `motion`.

### Options

| Option               | Alias     | Type   | Default   | Range         | Effect                                    |
|----------------------|-----------|--------|-----------|---------------|-------------------------------------------|
| `motion_force_zero`  | `force_0` | bool   | `false`   | —             | Override all scores to `0.0`              |
| `motion_fps_weight`  | `mfw`     | double | `1.0`     | `0.0–5.0`     | FPS-aware multiplicative correction       |
| `motion_blend_factor`| `mbf`     | double | `1.0`     | `0.0–1.0`     | Blend factor for motion3-style score      |
| `motion_blend_offset`| `mbo`     | double | `40.0`    | `0.0–1000.0`  | Blend offset                              |
| `motion_max_val`     | `mmxv`    | double | `10000.0` | `0.0–10000.0` | Upper clamp                               |
| `motion_five_frame_window` | `mffw` | bool | `false` | —             | Five-frame SAD window                     |
| `motion_moving_average` | `mma` | bool   | `false`   | —             | Two-frame moving average                  |

### Backend coverage

| Backend        | Status    | Notes                                                                  |
|----------------|-----------|------------------------------------------------------------------------|
| Scalar C       | Supported | Reference implementation                                               |
| AVX2           | Supported | `x86/motion_v2_avx2.c`                                                 |
| AVX-512        | Supported | `x86/motion_v2_avx512.c`                                               |
| NEON (AArch64) | Supported | `arm64/motion_v2_neon.c` (ADR-0145, bit-exact)                         |
| CUDA           | Supported | `feature/cuda/integer_motion_v2_cuda.c`                                |
| SYCL           | Supported | `feature/sycl/integer_motion_v2_sycl.cpp`                              |
| Vulkan         | Supported | `feature/vulkan/motion_v2_vulkan.c` (ADR-0193, bit-exact on 8/10 bpc) |
| HIP            | Supported | `feature/hip/integer_motion_v2_hip.c`                                  |
| Metal          | Supported | `feature/metal/integer_motion_v2_metal.mm`                             |

All GPU kernels are **bit-exact** vs the scalar CPU reference on 8-bit and 10-bit
inputs (max_abs_diff = 0.0 across the cross-backend gate fixture per ADR-0193).

---

## `float_motion` extractor (floating-point)

Registered name: `float_motion` (`VmafFeatureExtractor vmaf_fex_float_motion`).

Floating-point twin of `motion` using `float` arithmetic throughout. Provides
additional options for chroma channels and a half-resolution scale-1 SAD term.

### Output features

| Feature name                          | Description                                | Condition         |
|---------------------------------------|--------------------------------------------|-------------------|
| `VMAF_feature_motion2_score`          | Motion2 score                              | Always            |
| `VMAF_feature_motion3_score`          | Perceptually blended score                 | Always            |
| `VMAF_feature_motion_score`           | Raw (unfixed) motion score                 | `debug=true` only |

### Output range

`[0, motion_max_val]`. Same semantics as `motion`.

### Options

| Option               | Alias     | Type   | Default   | Range         | Effect                                                                    |
|----------------------|-----------|--------|-----------|---------------|---------------------------------------------------------------------------|
| `debug`              | —         | bool   | `true`    | —             | Emit `motion_score` alongside `motion2_score`                             |
| `motion_force_zero`  | `force_0` | bool   | `false`   | —             | Override all scores to `0.0`                                              |
| `motion_fps_weight`  | `mfw`     | double | `1.0`     | `0.0–5.0`     | FPS-aware multiplicative correction                                       |
| `motion_blend_factor`| `mbf`     | double | `1.0`     | `0.0–1.0`     | Blend factor for `motion3_score`                                          |
| `motion_blend_offset`| `mbo`     | double | `40.0`    | `0.0–1000.0`  | Blend offset for `motion3_score`                                          |
| `motion_add_scale1`  | —         | bool   | `false`   | —             | Add half-resolution SAD term on top of full-resolution Y-plane SAD        |
| `motion_add_uv`      | —         | bool   | `false`   | —             | Sum U and V plane SADs into the score (CPU only)                          |
| `motion_filter_size` | —         | int    | `5`       | `1, 3, 5`     | Gaussian blur kernel size; `5` = original Motion2, `3` = cheaper variant  |
| `motion_max_val`     | `mmxv`    | double | `10000.0` | `0.0–10000.0` | Upper clamp applied to emitted scores                                     |

`motion_add_scale1`, `motion_add_uv`, `motion_filter_size`, and `motion_max_val`
were ported from Netflix/vmaf commit `b949cebf`. With default settings the output
is bit-identical to the pre-port baseline on the Y-plane SIMD fast path.
`motion_add_uv=true` is CPU-only; GPU paths score the Y plane only.

### Backend coverage

| Backend        | Status    | Notes                                              |
|----------------|-----------|----------------------------------------------------|
| Scalar C       | Supported | Reference implementation                           |
| AVX2           | Supported | `x86/float_motion_avx2.c`                          |
| AVX-512        | Supported | `x86/float_motion_avx512.c`                        |
| NEON (AArch64) | Supported | `arm64/float_motion_neon.c`                        |
| CUDA           | Supported | `feature/cuda/float_motion_cuda.c` (ADR-0196)      |
| SYCL           | Supported | `feature/sycl/float_motion_sycl.cpp` (ADR-0196)   |
| Vulkan         | Supported | `feature/vulkan/float_motion_vulkan.c` (ADR-0196)  |
| HIP            | Supported | `feature/hip/float_motion_hip.c` (ADR-0273)        |
| Metal          | Supported | `feature/metal/float_motion_metal.mm`              |

Empirical GPU parity: max_abs_diff <= 3e-6 (8-bit, 48 frames) across CUDA,
SYCL, and Vulkan backends (ADR-0196).

### How to run

```bash
# Float motion (default options)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature float_motion --output /dev/stdout

# With 3-tap filter
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'float_motion:motion_filter_size=3' \
    --output /dev/stdout
```

---

## Input format constraints

All three extractors:

- Accept YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.
- Operate on the Y (luma) plane only by default. Chroma is only included when
  `motion_add_uv=true` is set on `float_motion`, and only for formats other
  than YUV 4:0:0.
- Require a minimum frame size of 3x3 pixels (5-tap Gaussian minimum
  dimension = filter_radius + 1 = 3). Smaller frames are rejected with `-EINVAL`
  at `init()`.
- Are temporal extractors: frame 0 always emits `0.0` for all motion scores.

## See also

- [Features](features.md) — full feature extractor reference table
- [ADR-0193](../adr/0193-motion-v2-vulkan.md) — `motion_v2` Vulkan kernel
- [ADR-0196](../adr/0196-float-motion-gpu.md) — `float_motion` GPU kernels
- [ADR-0219](../adr/0219-motion3-gpu-coverage.md) — `motion3` GPU coverage
