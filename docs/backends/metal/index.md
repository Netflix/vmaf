# Metal (Apple Silicon) compute backend

> **Status: runtime + first kernel batch.** The Metal backend now has a
> real Apple-Silicon runtime, shared-memory `MTLBuffer` picture storage,
> metallib embedding, and eight wired feature extractors:
> `float_ansnr_metal`, `float_moment_metal`, `float_motion_metal`,
> `float_psnr_metal`, `float_ssim_metal`, `integer_motion_metal`,
> `integer_psnr_metal`, and `motion_v2_metal`.
>
> The dispatch support predicate recognises both those extractor names
> and their provided feature keys (`psnr_y`, `psnr_cb`, `psnr_cr`,
> `float_ms_ssim`, `motion2_v2_score`, etc.). Remaining metrics such
> as VIF, ADM, CIEDE, CAMBI, and SSIMULACRA2 are still future kernel
> ports.
>
> Governing ADRs:
> [ADR-0361](../../adr/0361-metal-compute-backend.md),
> [ADR-0420](../../adr/0420-metal-backend-runtime-t8-1b.md), and
> [ADR-0421](../../adr/0421-metal-first-kernel-motion-v2.md).

## Why Metal

Apple Silicon (M1+) is the perf story for Apple-platform users. The
fork's existing Apple-Silicon coverage is the NEON SIMD CPU path
(per [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md) and the
wider NEON twin matrix); this backend adds the GPU compute path that
NEON cannot reach.

Three properties make a native Metal backend worth shipping:

1. **Unified memory.** `MTLBuffer` allocations created with
   `MTLResourceStorageModeShared` are zero-copy across CPU↔GPU; the
   submit-side H2D / D2H staging the CUDA / HIP / Vulkan backends
   spend the bulk of their complexity on collapses to host stores
   and direct `[buffer contents]` reads.
2. **First-party Apple compute API.** OpenCL is deprecated since
   macOS 10.14 (2018) and receives no driver updates; Vulkan reaches
   the GPU only through MoltenVK's translation layer (Vulkan command
   buffer → Metal command buffer rewrite) which adds per-dispatch
   overhead. Metal is the supported user-space surface.
3. **No PCIe boundary.** GPU and CPU share the same DRAM with cache
   coherence; the runtime PR can keep the previous-frame ref Y
   plane in one shared buffer rather than ping-ponging two device
   allocations the way the HIP twin does.

See [ADR-0361 §Context](../../adr/0361-metal-compute-backend.md#context)
for the full reasoning and rejected alternatives (MoltenVK, oneAPI,
OpenCL, Swift-based runtime).

## Apple Silicon only

The runtime PR (T8-1b) gates device selection on
`MTLGPUFamily.Apple7` (M1 and later) via
`-[id<MTLDevice> supportsFamily:]`. Intel Macs and non-macOS hosts
surface as `-ENODEV` from `vmaf_metal_state_init`. Reasoning: Apple
discontinued Intel-Mac GPU parity, and the unified-memory zero-copy
story does not apply on Intel-Mac discrete GPUs (Radeon Pro / Vega)
which sit behind PCIe. See
[ADR-0361 §Apple Silicon-only](../../adr/0361-metal-compute-backend.md#apple-silicon-only-apple-gpu-family-7-reject-intel-mac).

## Build

On macOS:

```bash
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke
```

`-Denable_metal=auto` (the default) auto-resolves to enabled on
`host_machine.system() == 'darwin'` and disabled elsewhere.
`-Denable_metal=disabled` suppresses the auto-probe even on macOS.
`-Denable_metal=enabled` forces the Metal frameworks to be linked;
on non-macOS hosts the meson `dependency('Metal')` probe fails the
setup step with a clear missing-framework error.

The backend has zero hard runtime dependencies on non-macOS hosts
because the Metal subdirectory is not entered there unless
`-Denable_metal=enabled` is forced. On macOS the `dependency('Metal')`
/ `dependency('IOSurface')` probes resolve to the system frameworks;
`MetalKit` is optional.

## Runtime layer

The runtime layer uses Objective-C++ `.mm` TUs under ARC and keeps
Metal object handles opaque at C boundaries as `void *` / `uintptr_t`.
`vmaf_metal_context_new` creates an Apple-Family-7+ `id<MTLDevice>`
and `id<MTLCommandQueue>`, `picture_metal.mm` allocates shared
`MTLBuffer` storage, and `kernel_template.mm` wraps per-feature
command-buffer lifecycle and readback waits.

Kernel sources are Metal Shading Language (`.metal`) compiled to
`.air` and linked into a `default.metallib` with `xcrun metal` /
`xcrun metallib`. The metallib is embedded into the libvmaf binary's
`__TEXT,__metallib` section and loaded by the Obj-C++ host dispatch
files.

## Rollout sequence

1. **T8-1 (scaffold PR + batch-1)** — public header, `src/metal`
   tree, first consumer registrations, `enable_metal` Meson option,
   smoke test, and macOS CI lane.
2. **T8-1b (runtime PR)** — `MTLCreateSystemDefaultDevice` /
   `id<MTLCommandQueue>` / `id<MTLBuffer>` lifecycle. Runtime entry
   points return `0` on a real Apple Silicon device and `-ENODEV` on
   Intel Mac or non-Apple-Family-7 GPUs.
3. **T8-1c…T8-1j (first kernel batch)** — `motion_v2`, float/integer
   PSNR, float moment, float ANSNR, float/integer motion, and
   float SSIM/MS-SSIM host dispatch + MSL kernels.
4. **T8-1k+** — remaining kernels (VIF, ADM, CIEDE, CAMBI,
   SSIMULACRA2, etc.) follow as their own PRs gated by the `places=4`
   cross-backend-diff lane (per [ADR-0214](../../adr/0214-gpu-parity-ci-gate.md)).
5. **`enable_metal` default flip** from `auto` to `enabled`: only
   after the kernel matrix proves bit-exactness via the `places=4`
   cross-backend gate (mirrors the `enable_vulkan` and `enable_hip`
   roadmaps).

## Feature extractor options

### `float_ssim_metal`

`float_ssim_metal` now reaches full option parity with the CPU `float_ssim`
extractor (ADR-0484):

- `enable_lcs` (bool, default `false`) — emit per-frame luminance
  (`float_ssim_l`), contrast (`float_ssim_c`), and structure (`float_ssim_s`)
  sub-scores alongside the composite SSIM score.  When enabled, the
  `float_ssim_vert_combine` kernel accumulates three additional per-WG partial
  sums (L, C, S) in a single threadgroup reduction pass — no extra dispatch.
- `enable_db` (bool, default `false`) — convert the SSIM score to decibels:
  `-10·log10(1 − SSIM)`.  Applied host-side after the partial-sum reduction.
- `clip_db` (bool, default `false`) — clamp the dB output to a finite maximum
  derived from frame dimensions and bit depth.  Mirrors the CPU helper exactly.
- `scale` (int, default `0` = auto-detect) — decimation scale factor.
  v1 supports scale=1 only; `scale=0` on frames where auto-detect would choose
  `scale>1` returns `-EINVAL` at init time with a log message.

Usage example:

```bash
vmaf --feature float_ssim_metal:enable_lcs=true:enable_db=true \
     --reference ref.yuv --distorted dist.yuv ...
```

## Coordination with NEON

The Metal backend targets the GPU on Apple Silicon. The NEON SIMD
twin matrix (per [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md))
stays the CPU-side path on the same hardware. The two are
complementary:

- Small / latency-sensitive runs land on NEON via the existing CPU
  dispatch (no GPU command-buffer setup overhead).
- Large / throughput-bound runs land on Metal when one of the shipped
  Metal feature kernels is requested; the GPU's parallelism + unified
  memory eliminate both the
  CPU-bound bottleneck and the H2D / D2H staging cost.

Backend selection follows the standard libvmaf precedence (see
[../index.md](../index.md) §Runtime selection): GPU paths win when
available, CPU SIMD wins otherwise.

## Verification

The macOS CI lane `Build — macOS Metal` is the ground-truth gate; it
runs on every PR with `-Denable_metal=enabled` and exercises the smoke
test plus the currently wired kernel batch. Linux-host dev sessions
cannot reproduce the lane locally because `Metal.framework` only exists
on macOS hosts.

Reviewers verifying locally on a Mac:

```bash
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke
```

## References

- [ADR-0361](../../adr/0361-metal-compute-backend.md) — original
  audit-first Metal backend ADR.
- [ADR-0212](../../adr/0212-hip-backend-scaffold.md) — HIP scaffold
  precedent (T7-10).
- [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) — Vulkan
  scaffold precedent (T5-1) — the original audit-first GPU-backend
  pattern.
- [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md) — motion_v2
  NEON twin on Apple Silicon CPU.
- [ADR-0214](../../adr/0214-gpu-parity-ci-gate.md) — `places=4`
  cross-backend gate; the runtime PR's incoming numerics gate.
- Apple Developer documentation — Metal-cpp,
  <https://developer.apple.com/metal/cpp/> (accessed 2026-05-09).
