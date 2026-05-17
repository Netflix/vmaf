# HIP (AMD ROCm) compute backend

> **Status (2026-05-17):** 20 of 23 feature extractors now have real device
> kernels; three legacy-API stubs remain (`adm_hip`, `vif_hip`, `motion_hip`).
>
> | Extractor | Feature name | Added in |
> | --- | --- | --- |
> | `integer_psnr_hip` | `psnr_hip` | ADR-0241 |
> | `float_psnr_hip` | `float_psnr_hip` | ADR-0254 |
> | `ciede_hip` (now `integer_ciede_hip`) | `ciede_hip` | ADR-0259 / PR #1016 |
> | `float_moment_hip` | `float_moment_hip` | ADR-0260 |
> | `float_ansnr_hip` | `float_ansnr_hip` | ADR-0266 |
> | `integer_motion_v2_hip` | `motion_v2_hip` | ADR-0267 |
> | `float_motion_hip` | `float_motion_hip` | ADR-0373 |
> | `float_ssim_hip` | `float_ssim_hip` | ADR-0375 |
> | `float_vif_hip` | `float_vif_hip` | ADR-0379 |
> | `integer_psnr_hvs_hip` | `psnr_hvs_hip` | PR #995 |
> | `integer_cambi_hip` | `cambi_hip` | PR #996 |
> | `ssimulacra2_hip` | `ssimulacra2_hip` | PR #1000 |
> | `integer_vif_hip` | `integer_vif_hip` | PR #1001 |
> | `integer_motion_hip` | `integer_motion_hip` | PR #1004 |
> | `integer_adm_hip` | `integer_adm_hip` | PR #1007 |
> | `integer_ciede_hip` | `ciede_hip` (alias) | PR #1016 |
> | `integer_moment_hip` | `integer_moment_hip` | PR #1017 |
> | `integer_ms_ssim_hip` | `ms_ssim_hip` | ADR-0285 / PR #1013 |
> | `integer_ssim_hip` | `integer_ssim_hip` | PR #999 |
> | `float_adm_hip` | `float_adm_hip` | ADR-0468 / PR #1024 |
>
> All 20 real kernels require `enable_hip=true` + `enable_hipcc=true`.
> Without `enable_hipcc`, the scaffold `-ENOSYS` posture is preserved.
> The three stubs (`adm_hip`, `vif_hip`, `motion_hip`) use an older
> `_init/_run/_destroy` API shape that predates the HSACO kernel template;
> they remain at `-ENOSYS` pending an API redesign.

## Building

```bash
meson setup build -Denable_cuda=false -Denable_sycl=false \
                  -Denable_hip=true -Denable_hipcc=true
ninja -C build
meson test -C build
```

`enable_hipcc=false` (the default) compiles the HIP C host runtime but
skips the `hipcc`-compiled kernel objects; every extractor returns
`-ENOSYS` at `init()`. Set both flags to `true` to compile and link the
real device kernels.

The scaffold has **zero hard runtime dependencies** — no ROCm SDK,
no `hipcc`, no `amdhip64`. The Meson build files include an optional
`dependency('hip-lang', required: false)` probe so a host that already
has ROCm installed will see the dependency resolve; the scaffold compiles
cleanly without it.

## Runtime

When built with HIP and device kernels, the backend is available for
explicit opt-in:

```bash
./build/tools/vmaf --feature psnr_hip --reference ref.yuv ...
./build/tools/vmaf --feature float_ansnr_hip --reference ref.yuv ...
./build/tools/vmaf --feature integer_vif_hip --reference ref.yuv ...
./build/tools/vmaf --feature integer_adm_hip:adm_skip_scale0=true --reference ref.yuv ...
```

FFmpeg backend selector: `hip_device=N` (patch `0011-libvmaf-wire-hip-backend-selector.patch`
in `ffmpeg-patches/`; see [ADR-0380](../../adr/0380-ffmpeg-hip-backend-selector.md)).

## Source layout

```text
libvmaf/src/hip/                  # HIP runtime (common, picture_hip, dispatch_strategy)
libvmaf/src/feature/hip/          # per-feature kernels
  integer_psnr_hip.c              # uint64 atomic-SSE warp-64 __shfl_down
  float_psnr_hip.c                # float (ref-dis)^2 reduction per block
  float_ansnr_hip.c               # (sig, noise) per-block float partials
  float_motion_hip.c              # 5x5 Gaussian blur + per-block float SAD
  float_moment_hip.c              # four uint64 atomic accumulator kernel
  float_ssim_hip.c                # two-pass separable 11-tap Gaussian kernel
  float_vif_hip.c                 # multi-scale VIF float pipeline
  float_adm_hip.c                 # ADM float pipeline (ADR-0468)
  ciede_hip.c                     # legacy alias for integer_ciede_hip
  integer_ciede_hip.c             # YUV->Lab, CIEDE2000 dE, warp-64 shfl_down
  integer_motion_v2_hip.c         # raw-pixel ping-pong, 5-tap Gaussian diff
  integer_motion_hip.c            # 5-tap Gaussian blur + warp-reduced SAD
  integer_moment_hip.c            # four uint64 atomic accumulator (integer)
  integer_psnr_hvs_hip.c          # PSNR-HVS frequency-weighted distortion
  integer_ssim_hip.c              # two-pass separable Gaussian + SSIM combine
  integer_ms_ssim_hip.c           # multi-scale SSIM (5 scales, biorthogonal LPF)
  integer_adm_hip.c               # ADM DWT2 + CSF + CM + decouple pipeline
  integer_vif_hip.c               # multi-scale VIF integer pyramid
  integer_cambi_hip.c             # CAMBI banding detection
  ssimulacra2_hip.c               # SSIMULACRA2 (host YUV->XYB + GPU IIR blur)
  adm_hip.c                       # stub — returns -ENOSYS (legacy API)
  vif_hip.c                       # stub — returns -ENOSYS (legacy API)
  motion_hip.c                    # stub — returns -ENOSYS (legacy API)
```

## Kernel notes

- **`integer_psnr_hip`** — uint64 atomic-SSE kernel, warp-64 `__shfl_down`
  reduction. Emits `psnr_y`.
- **`float_psnr_hip`** — float (ref-dis)² reduction per block. Emits `float_psnr`.
- **`float_ansnr_hip`** — per-block (sig, noise) float-partial kernel, 3×3 ref +
  5×5 dis filter with shared-memory mirror-padded tile. Emits `float_ansnr` +
  `float_anpsnr`.
- **`float_motion_hip`** — temporal extractor. 5×5 separable Gaussian blur +
  per-block float SAD partials, blur ping-pong (`blur[2]`), first-frame
  `compute_sad=0` short-circuit, motion2 tail emission in `flush()`. Emits
  `VMAF_feature_motion_score` + `VMAF_feature_motion2_score`.
- **`float_moment_hip`** — four uint64 atomic accumulator kernel (ref1st,
  dis1st, ref2nd, dis2nd), warp-64 two-uint32-shuffle reduction. Host divides
  by w×h. Emits four `float_moment_*` features.
- **`float_ssim_hip`** — two-pass separable 11-tap Gaussian kernel. Pass 1
  (horiz): five intermediate float buffers over (W-10)×H. Pass 2 (vert + SSIM
  combine): per-block float partial sum over (W-10)×(H-10). Host accumulates in
  double. Emits `float_ssim`.
- **`integer_ciede_hip`** — HtoD copies of all 6 YUV planes (ref + dis Y/U/V),
  per-pixel YUV→Lab conversion, CIEDE2000 ΔE accumulation per block, host log10
  transform. Emits `ciede2000`. Warp-64 `__shfl_down` without mask.
- **`integer_motion_v2_hip`** — temporal extractor. Raw-pixel ping-pong (`pix[2]`),
  separable 5-tap Gaussian diff filter with arithmetic right-shift (critical for
  bit-exactness vs CPU — see ADR-0138/0139 and PR #587 AVX2 srlv_epi64 regression),
  single int64 atomic SAD accumulator, host-side `min(cur, next)` fold in `flush()`.
  Emits `VMAF_integer_feature_motion_v2_sad_score` +
  `VMAF_integer_feature_motion2_v2_score`.
- **`integer_motion_hip`** — 5-tap Gaussian blur + warp-reduced SAD, ping-pong
  frame buffer; mirrors `integer_motion_cuda.c` call-graph. Emits
  `VMAF_feature_motion2_score`.
- **`integer_psnr_hvs_hip`** — frequency-weighted distortion per 8×8 block,
  porting the CUDA twin. Emits `psnr_hvs` + per-channel variants.
- **`integer_ssim_hip`** — two-pass separable 11-tap Gaussian SSIM, GCN/RDNA
  warp-size-64 adaptation. Emits `integer_ssim`.
- **`integer_ms_ssim_hip`** — multi-scale SSIM over 5 pyramid levels; 9-tap
  biorthogonal LPF decimation + separable 11-tap Gaussian per scale. Emits
  `float_ms_ssim`. Per ADR-0285.
- **`integer_adm_hip`** — full ADM DWT2 + CSF + CM + decouple pipeline (five
  kernel files). Mirrors `integer_adm_cuda.c`. Emits `adm2` + per-scale values.
- **`integer_vif_hip`** — multi-scale VIF integer pyramid; respects
  `vif_skip_scale0` (PR #1063) and `vif_enhn_gain_limit`. Emits `vif_scale0..3`.
- **`integer_cambi_hip`** — CAMBI banding detection; full HIP port per PR #996
  (ADR-0345 Phase 3). Emits `cambi`.
- **`ssimulacra2_hip`** — host-side YUV→XYB + GPU IIR blur + host double-precision
  combine; mirrors the CUDA twin. Emits `ssimulacra2`.
- **`float_adm_hip`** — ADM float pipeline, ninth kernel-template consumer
  (ADR-0468). Mirrors `float_adm_cuda.c`. Emits `float_adm2`.
- **`float_vif_hip`** — multi-scale VIF float pipeline; respects
  `vif_skip_scale0` (PR #1180). Emits `float_vif_scale0..3`.

## Remaining stubs

`adm_hip`, `vif_hip`, and `motion_hip` use the older `_init/_run/_destroy` API
shape that requires a separate `VmafFeatureExtractor` redesign before promotion.
Each returns `-ENOSYS` at `init()`. Tracked in
[docs/state.md](../../state.md).

## Caveats

- `enable_hip` is `boolean` defaulting to **false**. `enable_hipcc` (also
  `boolean`, default **false**) controls whether `hipcc`-compiled kernel objects
  are linked. Both must be `true` for real GPU computation.
- HIP runtime types (`hipDevice_t`, `hipStream_t`) cross the public ABI as
  `uintptr_t`. This keeps `libvmaf_hip.h` free of `<hip/hip_runtime.h>`,
  mirroring the pattern Vulkan adopted in ADR-0184.
- No CI runner with a real AMD GPU exists on GitHub-hosted infrastructure.
  The CI compile lane (`Build — Ubuntu HIP`) runs with `-Denable_hip=true`
  but `-Denable_hipcc=false`, so kernels are not compiled or exercised on CI.

## References

- [ADR-0212](../../adr/0212-hip-backend-scaffold.md) — the original scaffold.
- [ADR-0241](../../adr/0241-hip-first-consumer-psnr.md) — first consumer (`integer_psnr_hip`).
- [ADR-0254](../../adr/0254-hip-second-consumer-float-psnr.md) —
  second consumer (`float_psnr_hip`).
- [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) — third consumer.
- [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) —
  fourth consumer (`float_moment_hip`).
- [ADR-0266](../../adr/0266-hip-fifth-consumer-float-ansnr.md) —
  fifth consumer (`float_ansnr_hip`).
- [ADR-0267](../../adr/0267-hip-sixth-consumer-motion-v2.md) —
  sixth consumer (`motion_v2_hip`).
- [ADR-0372](../../adr/0372-hip-batch1-runtime-kernels.md) — batch-1 kernels.
- [ADR-0373](../../adr/0373-hip-batch2-runtime-kernels.md) — batch-2 kernels.
- [ADR-0375](../../adr/0375-hip-batch3-runtime-kernels.md) — batch-3 kernels.
- [ADR-0377](../../adr/0377-hip-batch4-runtime-kernels.md) — batch-4 kernels.
- [ADR-0379](../../adr/0379-hip-float-vif.md) — `float_vif_hip`.
- [ADR-0380](../../adr/0380-ffmpeg-hip-backend-selector.md) — FFmpeg selector.
- [ADR-0468](../../adr/0468-hip-float-adm.md) — `float_adm_hip`.
- [Research-0033](../../research/0033-hip-applicability.md) —
  AMD market-share + ROCm Linux maturity survey.
