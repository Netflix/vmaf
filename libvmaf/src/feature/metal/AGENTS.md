# `libvmaf/src/feature/metal/` — Metal feature-kernel directory

## Purpose

Contains one `.mm` (Objective-C++ host dispatch) + one `.metal` (Metal
Shading Language device kernel) pair per feature extractor in the Metal
GPU backend, plus the per-T8-1 scaffold `.c` stubs that were superseded.

Only `.mm` + `.metal` pairs are functional. The `.c` stubs (e.g.
`float_psnr_metal.c`) are replaced by their `.mm` counterparts once
a real kernel lands; they are removed from `metal_sources` in
`libvmaf/src/metal/meson.build` when the conversion happens.

## Rebase-sensitive invariants

- **Per-WG float/uint partials — no atomics**: Apple MSL does not
  expose `atomic_ulong` (`atomic_fetch_add_explicit` for `ulong`
  silently compiles but fails on device — confirmed CI run 25685703780
  / job 75408804495). All Metal kernels use a per-threadgroup
  `float`/`uint` partials array indexed by
  `bid.y * grid_groups.x + bid.x`, reduced on the host in `double`.
  Do not introduce `atomic_ulong` or `atomic_fetch_add_explicit`
  for 64-bit types.

- **`simd_sum` reduction**: MSL `simd_sum()` is the standard two-level
  reduction primitive. All kernels use:
  1. `simd_sum(per_thread_val)` → lane 0 of each SIMD group writes to
     a `threadgroup float simd_partials[8]` array.
  2. Thread 0 (`lid == 0`) sums the `simd_count` SIMD-group partials
     into the global `partials[bid.y * grid_groups.x + bid.x]` slot.

- **8×16 threadgroup / 20×20 shared tile (radius-2 kernels)**:
  `integer_motion_v2`, `float_ansnr`, `float_motion`, `integer_motion`,
  and `float_ssim` all use a 16×16 threadgroup with a 20×20 shared
  tile (4-element halo radius-2). Tile pitch is 21 (not 20) to avoid
  bank conflicts on Apple GPU 32-bank threadgroup memory.

- **Per-WG partials buffer**: each `.mm` allocates a Shared-storage
  `MTLBuffer` sized `ceil(W/16) * ceil(H/16)` float (or uint)
  elements, one per threadgroup. For `float_moment`, the buffer holds
  4 floats per threadgroup (interleaved ref1st/dis1st/ref2nd/dis2nd).

- **Bridge-retained PSO slots**: each `.mm` stores
  `MTLComputePipelineState` handles as `void *` under
  `__bridge_retained` cast (one per bpc variant). `close_fex_metal`
  must release them via `__bridge_transfer` to avoid leaks.

- **`float_moment` feature name correction**: the T8-1 scaffold
  `float_moment_metal.c` erroneously listed `{"float_moment1",
  "float_moment2", "float_std", NULL}` as `provided_features`. The
  correct names (matching CPU, CUDA, HIP, SYCL, Vulkan) are
  `{"float_moment_ref1st", "float_moment_dis1st",
  "float_moment_ref2nd", "float_moment_dis2nd", NULL}`. The `.mm`
  conversion uses the correct names; the `.c` file is removed from
  `metal_sources` on merge.

## Kernel files

| File                               | Status      | Feature(s)                                                              |
|------------------------------------|-------------|-------------------------------------------------------------------------|
| `integer_motion_v2.metal`          | Done (T8-1c) | `VMAF_integer_feature_motion_v2_sad_score`, `motion2_v2_score`         |
| `integer_motion_v2_metal.mm`       | Done (T8-1c) | host dispatch                                                           |
| `float_psnr.metal`                 | Done (T8-1d) | `float_psnr`                                                            |
| `float_psnr_metal.mm`              | Done (T8-1d) | host dispatch                                                           |
| `float_moment.metal`               | Done (T8-1e) | `float_moment_ref1st`, `float_moment_dis1st`, `float_moment_ref2nd`, `float_moment_dis2nd` |
| `float_moment_metal.mm`            | Done (T8-1e) | host dispatch (fixes provided_features)                                 |
| `float_ansnr.metal`                | Done (T8-1f) | `float_ansnr`                                                           |
| `float_ansnr_metal.mm`             | Done (T8-1f) | host dispatch                                                           |
| `integer_psnr.metal`               | Done (T8-1g) | `psnr_y`, `psnr_cb`, `psnr_cr`                                          |
| `integer_psnr_metal.mm`            | Done (T8-1g) | host dispatch                                                           |
| `float_motion.metal`               | Done (T8-1h) | `float_motion`                                                          |
| `float_motion_metal.mm`            | Done (T8-1h) | host dispatch                                                           |
| `integer_motion.metal`             | Done (T8-1i) | `VMAF_integer_feature_motion_y_score`, `motion2_score`, `motion3_score` |
| `integer_motion_metal.mm`          | Done (T8-1i) | host dispatch                                                           |
| `float_ssim.metal`                 | Done (T8-1j) | `float_ssim`, `float_ms_ssim`                                           |
| `float_ssim_metal.mm`              | Done (T8-1j) | host dispatch                                                           |
| `float_psnr_metal.c`               | Scaffold     | replaced by float_psnr_metal.mm                                         |
| `float_moment_metal.c`             | Scaffold     | replaced by float_moment_metal.mm                                       |
| `float_ansnr_metal.c`              | Scaffold     | replaced by float_ansnr_metal.mm                                        |
| `integer_psnr_metal.c`             | Scaffold     | replaced by integer_psnr_metal.mm                                       |
| `float_motion_metal.c`             | Scaffold     | replaced by float_motion_metal.mm                                       |
| `integer_motion_metal.c`           | Scaffold     | replaced by integer_motion_metal.mm                                     |
| `float_ssim_metal.c`               | Scaffold     | replaced by float_ssim_metal.mm                                         |
| `float_ms_ssim.metal`             | Done (T8-2b) | `float_ms_ssim` — 5-scale pyramid, Wang weights                        |
| `float_ms_ssim_metal.mm`          | Done (T8-2b) | host dispatch (ADR-0490)                                                |

## Rebase-sensitive invariants (motion_fps_weight)

- **`motion_fps_weight` cross-backend parity** — see the canonical
  invariant note in [`../cuda/AGENTS.md`](../cuda/AGENTS.md).
  `integer_motion_v2_metal.mm` and `float_motion_metal.mm` both carry
  the `motion_fps_weight` option and apply it identically to the
  CUDA / SYCL / Vulkan / HIP twins: `motion_v2` applies the weight in
  `flush()` to both scores before the min; `float_motion` applies it
  in `collect()` (index >= 2, to both `w_cur` and `w_prev` before the
  min) and in `flush()` (scaled tail emission). When
  `motion_fps_weight = 1.0` (default) the arithmetic is a no-op and
  the `places=4` gate must pass. Any future change to the weight
  application math must span all motion-family GPU twins in the same PR.

## Governing ADRs

- [ADR-0490](../../../../docs/adr/0490-float-ms-ssim-metal-port.md) — T8-2b: float_ms_ssim_metal port
- [ADR-0421](../../../../docs/adr/0421-metal-first-kernel-motion-v2.md) — T8-1c through T8-1k batch specification
- [ADR-0420](../../../../docs/adr/0420-metal-backend-runtime-t8-1b.md) — runtime (T8-1b), prerequisite
- [ADR-0361](../../../../docs/adr/0361-metal-compute-backend.md) — scaffold (T8-1), origin
- [ADR-0214](../../../../docs/adr/0214-gpu-parity-ci-gate.md) — `places=4` cross-backend parity gate
