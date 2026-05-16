# HIP (AMD ROCm) compute backend

> **Status (2026-05-10, batch-4 complete):** the host-side HIP runtime is
> wired (T7-10b). Eight of eleven feature extractors now have real device
> kernels; three stubs remain pending an ADM/VIF API redesign:
>
> | Extractor | Feature name | Governing ADR |
> |---|---|---|
> | `float_psnr_hip` | `float_psnr_hip` | ADR-0254 |
> | `integer_psnr_hip` | `psnr_hip` | ADR-0372 |
> | `float_ansnr_hip` | `float_ansnr_hip` | ADR-0372 |
> | `float_motion_hip` | `float_motion_hip` | ADR-0373 |
> | `float_moment_hip` | `float_moment_hip` | ADR-0375 |
> | `float_ssim_hip` | `float_ssim_hip` | ADR-0375 |
> | `ciede_hip` | `ciede_hip` | ADR-0377 |
> | `integer_motion_v2_hip` | `motion_v2_hip` | ADR-0377 |
>
> All eight require `enable_hip=true` + `enable_hipcc=true`.
> Without `enable_hipcc`, the scaffold `-ENOSYS` posture is preserved.
> The remaining three extractors (`adm_hip`, `vif_hip`, `integer_motion_hip`)
> remain at `-ENOSYS` pending an ADM/VIF `_init/_run/_destroy` API redesign.
> See ADR-0372 (batch-1), ADR-0373 (batch-2), ADR-0375 (batch-3),
> ADR-0377 (batch-4) for rationale.

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
```

FFmpeg backend selector: `hip_device=N` (patch `0011-libvmaf-wire-hip-backend-selector.patch`
in `ffmpeg-patches/`; see [ADR-0380](../../adr/0380-ffmpeg-hip-backend-selector.md)).

## Source layout

```text
libvmaf/src/hip/                  # HIP runtime (common, picture_hip, dispatch_strategy)
libvmaf/src/feature/hip/          # per-feature kernels
  float_psnr_hip.c                # float (ref-dis)^2 reduction per block
  integer_psnr_hip.c              # uint64 atomic-SSE warp-64 __shfl_down
  float_ansnr_hip.c               # (sig, noise) per-block float partials
  float_motion_hip.c              # 5×5 Gaussian blur + per-block float SAD
  float_moment_hip.c              # four uint64 atomic accumulator kernel
  float_ssim_hip.c                # two-pass separable 11-tap Gaussian kernel
  ciede_hip.c                     # YUV→Lab, CIEDE2000 ΔE, warp-64 shfl_down
  integer_motion_v2_hip.c         # raw-pixel ping-pong, 5-tap Gaussian diff
  adm_hip.c                       # stub — returns -ENOSYS
  vif_hip.c                       # stub — returns -ENOSYS
  motion_hip.c                    # stub — returns -ENOSYS
```

## Kernel notes

- **`float_psnr_hip`** — float (ref-dis)² reduction per block. Emits `float_psnr`.
- **`integer_psnr_hip`** — uint64 atomic-SSE kernel, warp-64 `__shfl_down`
  reduction. Emits `psnr_y`.
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
  double. Emits `float_ssim`. v1: `scale=1` only.
- **`ciede_hip`** — HtoD copies of all 6 YUV planes (ref + dis Y/U/V), per-pixel
  YUV→Lab conversion, CIEDE2000 ΔE accumulation per block, host log10 transform.
  Emits `ciede2000`. Warp-64 `__shfl_down` without mask.
- **`integer_motion_v2_hip`** — temporal extractor. Raw-pixel ping-pong (`pix[2]`),
  separable 5-tap Gaussian diff filter with arithmetic right-shift (critical for
  bit-exactness vs CPU — see ADR-0138/0139 and PR #587 AVX2 srlv_epi64 regression),
  single int64 atomic SAD accumulator, host-side `min(cur, next)` fold in `flush()`.
  Emits `VMAF_integer_feature_motion_v2_sad_score` +
  `VMAF_integer_feature_motion2_v2_score`.

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
- [ADR-0254](../../adr/0254-hip-second-consumer-float-psnr.md) — second consumer (`float_psnr_hip`).
- [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) — third consumer (`ciede_hip`).
- [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) — fourth consumer (`float_moment_hip`).
- [ADR-0266](../../adr/0266-hip-fifth-consumer-float-ansnr.md) — fifth consumer (`float_ansnr_hip`).
- [ADR-0267](../../adr/0267-hip-sixth-consumer-motion-v2.md) — sixth consumer (`motion_v2_hip`).
- [ADR-0372](../../adr/0372-hip-batch1-runtime-kernels.md) — batch-1 runtime kernels.
- [ADR-0373](../../adr/0373-hip-batch2-runtime-kernels.md) — batch-2 runtime kernels.
- [ADR-0375](../../adr/0375-hip-batch3-runtime-kernels.md) — batch-3 runtime kernels.
- [ADR-0377](../../adr/0377-hip-batch4-runtime-kernels.md) — batch-4 runtime kernels.
- [ADR-0380](../../adr/0380-ffmpeg-hip-backend-selector.md) — FFmpeg HIP backend selector.
- [Research-0033](../../research/0033-hip-applicability.md) — AMD market-share + ROCm Linux maturity.
