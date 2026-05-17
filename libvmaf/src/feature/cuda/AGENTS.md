# AGENTS.md — libvmaf/src/feature/cuda

Orientation for agents working on per-feature CUDA kernels (host
glue + `.cu` device code). Parent: [../AGENTS.md](../AGENTS.md). The
backend runtime (context, stream, picture-pool) lives one level up
in [`../../cuda/AGENTS.md`](../../cuda/AGENTS.md).

## Scope

```text
feature/cuda/
  <feature>_cuda.{c,h}        # host glue: registration, submit/collect, kernel-template wiring
  <feature>/                  # subdirectory of `.cu` device code (where the host glue is non-trivial)
    *.cu                      # CUDA kernel TUs (compiled with nvcc)
    *.cuh                     # device-side helpers (included from .cu only)
```

Examples: `integer_psnr_cuda.c` is a single-file consumer using the
kernel-template flat shape; `integer_adm/` is a multi-`.cu` consumer
because ADM splits across DWT2 + decouple + CSF + CM passes.

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md) +
  [../../AGENTS.md](../../AGENTS.md) +
  [`../../cuda/AGENTS.md`](../../cuda/AGENTS.md)).
- **Wholly-new fork files use the dual Netflix + Lusoris/Claude
  copyright header** per [ADR-0025](../../../../docs/adr/0025-copyright-handling-dual-notice.md).
  Many TUs here predate the dual-notice rule and carry only the
  Netflix header (with NVIDIA contributor lines on the `integer_adm/`
  CUDA kernels) — that is correct for upstream-mirrored files; do
  not retro-fit.
- **`#include` order** mirrors the SYCL / Vulkan twins:
  `feature_collector.h` / `feature_extractor.h` first, then
  `cuda/integer_<feature>_cuda.h`, then `cuda_helper.cuh` /
  `kernel_template.h`. Don't shuffle.
- **fmaf contraction is OFF for precision-critical kernels.** The
  parent build line passes `--fmad=false` to `nvcc` for feature
  TUs that participate in cross-backend gates with `places=4`.
  Removing it drifts `float_adm_cuda` /  `ssimulacra2_cuda` past
  the gate (mirror of the SYCL `-fp-model=precise` and Vulkan
  GLSL `precise` / `NoContraction` rules). On rebase: keep the
  flag.

## Twin-update rules

Every TU in this directory has at least one cross-backend twin.
A change to one twin **must** ship with the matching change(s) in
the same PR:

| Feature | Twins |
| --- | --- |
| **psnr** | `integer_psnr_cuda.c` ↔ `../sycl/integer_psnr_sycl.cpp` ↔ `../vulkan/psnr_vulkan.c` (+ `psnr.comp`) ↔ `../hip/integer_psnr_hip.c` |
| **ciede** | `integer_ciede_cuda.c` ↔ `../sycl/integer_ciede_sycl.cpp` ↔ `../vulkan/ciede_vulkan.c` (+ `ciede.comp`) ↔ `../hip/ciede_hip.c` |
| **moment** | `integer_moment_cuda.c` ↔ `../sycl/integer_moment_sycl.cpp` ↔ `../vulkan/moment_vulkan.c` (+ `moment.comp`) ↔ `../hip/float_moment_hip.c` |
| **motion** | `integer_motion_cuda.c` ↔ `../sycl/integer_motion_sycl.cpp` ↔ `../vulkan/motion_vulkan.c` (+ `motion.comp`) |
| **motion_v2** | `integer_motion_v2_cuda.c` ↔ `../sycl/integer_motion_v2_sycl.cpp` ↔ `../vulkan/motion_v2_vulkan.c` (+ `motion_v2.comp`) ↔ `../hip/integer_motion_v2_hip.c` |
| **vif (integer)** | `integer_vif_cuda.c` (+ `integer_vif/filter1d.cu`) ↔ `../sycl/integer_vif_sycl.cpp` ↔ `../vulkan/vif_vulkan.c` (+ `vif.comp`) |
| **adm (integer)** | `integer_adm_cuda.c` (+ `integer_adm/*.cu`) ↔ `../sycl/integer_adm_sycl.cpp` ↔ `../vulkan/adm_vulkan.c` (+ `adm.comp`) |
| **ssim** | `integer_ssim_cuda.c` ↔ `../sycl/integer_ssim_sycl.cpp` ↔ `../vulkan/ssim_vulkan.c` (+ `ssim.comp`) |
| **ms_ssim** | `integer_ms_ssim_cuda.c` ↔ `../sycl/integer_ms_ssim_sycl.cpp` ↔ `../vulkan/ms_ssim_vulkan.c` (+ `ms_ssim.comp`) |
| **psnr_hvs** | `integer_psnr_hvs_cuda.c` ↔ `../sycl/integer_psnr_hvs_sycl.cpp` ↔ `../vulkan/psnr_hvs_vulkan.c` (+ `psnr_hvs.comp`) |
| **ssimulacra2** | `ssimulacra2_cuda.c` (+ `ssimulacra2/*.cu`) ↔ `../sycl/ssimulacra2_sycl.cpp` ↔ `../vulkan/ssimulacra2_vulkan.c` (+ `ssimulacra2_*.comp`) |
| **float_*** | `float_adm_cuda.c` / `float_ansnr_cuda.c` / `float_motion_cuda.c` / `float_psnr_cuda.c` / `float_vif_cuda.c` ↔ matching `../sycl/float_*_sycl.cpp` ↔ `../vulkan/float_*_vulkan.c` ↔ partial `../hip/float_*_hip.c` |
| **cambi** | `integer_cambi_cuda.c` (+ `integer_cambi/cambi_score.cu`) ↔ `../vulkan/cambi_vulkan.c` (+ `cambi_*.comp`) — Strategy II hybrid twin. SYCL twin pending (T3-15b). |

The full GPU twin matrix is governed by the GPU long-tail batches:
[ADR-0182](../../../../docs/adr/0182-gpu-long-tail-batch-1.md) (psnr /
ciede / moment), [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md)
(ssim / ms_ssim / psnr_hvs), [ADR-0192](../../../../docs/adr/0192-gpu-long-tail-batch-3.md)
(motion_v2 / float_ansnr / float-twins / ssimulacra2 / cambi).

## Rebase-sensitive invariants

- **`integer_ms_ssim_cuda.c::extract_metrics_*` honours the
  `enable_lcs` GPU contract** (ADR-0243). Emits 15 extra metrics
  (`float_ms_ssim_{l,c,s}_scale{0..4}`) when `enable_lcs=true`,
  matching the CPU `float_ms_ssim` extractor metric-wise (all
  `l_scale*` first, then `c_*`, then `s_*`). Renaming or
  reordering breaks the public API surface and the cross-backend
  parity gate. See [../../AGENTS.md §"MS-SSIM `enable_lcs` GPU
  contract"](../../AGENTS.md).

- **`integer_motion_cuda.c::motion3_postprocess_*` honours the
  motion3 GPU contract** (ADR-0219). Applies CPU's host-side
  post-process to motion2 with no device-side state. Two
  invariants flow: (1) `motion_five_frame_window=true` returns
  `-ENOTSUP` at `init()`; (2) any change to `motion_blend()` /
  `motion_max_val` / moving-average must mirror across the three
  GPU motion twins in the same PR. See [../../AGENTS.md §"motion3_score
  GPU contract"](../../AGENTS.md).

- **`integer_motion_cuda.c::submit_fex_cuda` runs the SAD
  `cuMemsetD8Async` on `pic_stream`, NOT on `s->str`** (ADR-0358).
  The kernel atomic-adds into the same single-int64 buffer on
  `pic_stream`; both streams are `CU_STREAM_NON_BLOCKING` and have
  no event linking them, so co-locating the memset on the same
  stream as the kernel is the only thing that orders them. The
  matching pattern lives at `integer_motion_v2_cuda.c:188`. Any
  rebase or follow-up that reverts the memset onto a separate
  stream silently re-introduces the cross-stream race.

- **`integer_motion_cuda.c::collect_fex_cuda` and `flush_fex_cuda`
  emit `motion2_score = MIN(score * motion_fps_weight, motion_max_val)`,
  NOT the raw `min(prev, cur)` SAD score** (ADR-0358). Mirrors the
  CPU reference at `integer_motion.c:563`. The
  `motion3_postprocess_cuda` moving-average guard reads
  `frame_index > 2` (NOT `> 1`) because `frame_index` is
  pre-incremented in `collect()` before the helper runs. Tripped
  by non-default `motion_fps_weight ≠ 1.0` /
  `motion_moving_average = true`.

- **`integer_ms_ssim_cuda.c` and `integer_ssim_cuda.c` pass
  `channel=0` to `picture_copy()`** per the upstream
  d3647c73 prerequisite port. If a future upstream commit
  evolves the signature further, update these call sites in
  lockstep with the upstream-mirror callers (`float_*` series).
  See [../../AGENTS.md §"`picture_copy()` carries a `channel`
  parameter"](../../AGENTS.md).

- **`integer_psnr_hvs_cuda.c` participates in the engine-scope CUDA
  drain batch.** Its `submit_fex_cuda` queues all three plane partial
  DtoH copies on `s->lc.str`, records `s->lc.finished` via
  `vmaf_cuda_kernel_submit_post_record`, and registers the lifecycle
  with `drain_batch`. Its `collect_fex_cuda` must call
  `vmaf_cuda_kernel_collect_wait` before reading `h_partials[]`;
  reintroducing raw `cuMemcpyDtoHAsync` + `cuStreamSynchronize` in
  collect reopens T-GPU-OPT-3's per-frame sync stall. The scheduling
  change is CUDA-only and does not require SYCL / Vulkan twin edits
  because it does not alter kernel math or emitted metrics.

- **`integer_psnr_hvs_cuda.c` honours `enable_chroma` option parity** (mirrors
  ADR-0453 on the psnr_hvs surface). The `enable_chroma` option (default
  `false`) clamps `n_planes` to 1 in `init_fex_cuda` when set to `false`,
  and YUV400P sources always force `n_planes=1` regardless of the option.
  All plane loops (`upload_frame`, `launch_plane_kernels`,
  `enqueue_partials_readback`, `collect_fex_cuda`, `close_fex_cuda`) iterate
  over `s->n_planes`, not the compile-time constant `PSNR_HVS_NUM_PLANES`.
  The Vulkan and SYCL twins do not yet carry this option; add it there in
  lockstep if the combined-score formula diverges. The `collect_fex_cuda`
  combined-score path emits luma dB only when `n_planes == 1`.

- **`integer_psnr_hvs/psnr_hvs_score.cu` parallelises only the integer
  DCT passes.** The first eight CUDA threads perform the two 8-point
  DCT passes over shared memory; all float means, variance, masking,
  and final masked-error accumulation remain thread-0 serial in CPU
  scan order. Do not move the float reductions into warp or block
  reductions without a new numeric-contract ADR and a refreshed
  cross-backend tolerance row.

- **`integer_cambi_cuda.c` + `integer_cambi/cambi_score.cu` are
  Strategy II hybrid** (ADR-0360 / T3-15a). The GPU kernels
  (`cambi_spatial_mask_kernel`, `cambi_decimate_kernel`,
  `cambi_filter_mode_kernel`) are bit-exact w.r.t. the CPU
  implementation. The host residual calls `vmaf_cambi_calculate_c_values`
  and `vmaf_cambi_spatial_pooling` via `cambi_internal.h`. If upstream
  Netflix refactors `cambi.c` and renames those entry points,
  `cambi_internal.h` **and** `cambi_vulkan.c` must be updated in the
  same PR. Never remove the `cuStreamSynchronize` calls inside
  `submit_fex_cuda` — they guard the DtoH coherency for the host
  residual. `places=4` gate is load-bearing; do not loosen it.

- **`cuLaunchKernel` `kernelParams[]` must point to the device-pointer
  VALUE, not to a `VmafCudaBuffer` struct** (Issue #857 / fix PR). The
  dispatch helpers in `integer_cambi_cuda.c` (`dispatch_mask`,
  `dispatch_decimate`, `dispatch_filter_mode`) pass `&buf->data`
  (address of the `CUdeviceptr` field) to `cuLaunchKernel`. Passing
  `(void *)buf` (address of the struct) makes the driver read
  `buf->size` (a host byte count) as a device pointer, causing an
  immediate GPU invalid-address fault (SIGSEGV/SIGBUS on the host).
  The same invariant applies to every CUDA feature extractor that
  allocates device-side flat buffers via `vmaf_cuda_buffer_alloc`
  and passes them directly to `cuLaunchKernel`: always use
  `&buf->data`, never `(void *)buf`. Device-pointer arithmetic must
  also be performed on the `CUdeviceptr` integer type directly —
  avoid casting through `uint8_t *` (UB even though it round-trips
  on x86-64 today).

- **`integer_psnr_cuda.c` honours `enable_chroma` option parity** (ADR-0453).
  The `enable_chroma` option (default `true`) clamps `n_planes` to 1 in
  `init_fex_cuda` when set to `false`, matching CPU
  `integer_psnr.c::init`'s behaviour. The clamp runs after the
  `pix_fmt == YUV400P` guard so that YUV400 sources are always luma-only
  regardless of the option. On rebase: if upstream Netflix adds an
  `enable_chroma` option to the CPU path that behaves differently from the
  fork's GPU guard, audit both and keep the GPU clamp semantically
  equivalent. The SYCL and Vulkan twins carry the identical guard and must
  move in lockstep with any change to this one. The cross-backend parity
  gate at `places=4` covers both `enable_chroma=true` (default) and
  `enable_chroma=false` paths.

- **Host-side preprocessing in CUDA feature extractor `submit` callbacks
  must download GPU→host first.** Pictures passed to a CUDA extractor's
  `submit()` have device pointers in `data[]`; the host cannot read them
  directly. Use `vmaf_cuda_picture_download_async` followed by
  `cuStreamSynchronize` on the picture's private stream (obtained via
  `vmaf_cuda_picture_get_stream`) before passing the picture to any
  host-side function that dereferences `data[]`. The CAMBI extractor
  (`integer_cambi_cuda.c::submit_fex_cuda`) is the canonical example
  of this pattern (Issue #857 fix). All other CUDA extractors in this
  directory currently keep preprocessing on the GPU and are not affected,
  but the rule applies to any future extractor that mixes GPU input
  pictures with host-side preprocessing.

- **`integer_adm_cuda.c` must NOT include `feature/adm_options.h`
  directly.** `DEFAULT_ADM_NOISE_WEIGHT`, `DEFAULT_ADM_CSF_SCALE`,
  `DEFAULT_ADM_CSF_DIAG_SCALE`, and the full 4-member
  `enum ADM_CSF_MODE` arrive transitively via
  `cuda/integer_adm_cuda.h` → `feature/integer_adm.h`.  A direct
  include reintroduces the 2-member `enum ADM_CSF_MODE` from
  `adm_options.h` and causes a redeclaration error.

- **`integer_adm_cuda.c` / `float_adm_cuda.c` expose three ADM
  tuning parameters** (`adm_csf_scale`, `adm_csf_diag_scale`,
  `noise_weight`) with the same defaults as the CPU path (PR #731).
  If upstream Netflix adds or renames these parameters in
  `integer_adm.c` / `float_adm.c`, the CUDA twins must be updated
  in the same PR.

- **`motion_fps_weight` is a cross-backend parity parameter** — all
  motion-family GPU twins must expose `motion_fps_weight` in their
  `VmafOption options[]` table and apply it identically: for
  `integer_motion_v2_*` (flush-based motion2), the weight scales both
  `score_cur` and `score_next` before the min in `flush()`; for
  `float_motion_*` (collect-based motion2), the weight scales both
  `prev_motion_score` and `motion_score` before the min in `collect()`
  (for index >= 2) and scales `prev_motion_score` alone in `flush()`.
  When `motion_fps_weight = 1.0` (default) the arithmetic is a
  no-op and the `places=4` cross-backend gate must continue to pass.
  If the application math ever changes in the CPU reference
  (`integer_motion_v2.c` / `float_motion.c`), all GPU twins must be
  updated in the same PR. Twins in scope: `integer_motion_v2_cuda.c`,
  `integer_motion_v2_sycl.cpp`, `motion_v2_vulkan.c`,
  `integer_motion_v2_hip.c`, `integer_motion_v2_metal.mm`,
  `float_motion_cuda.c`, `float_motion_sycl.cpp`,
  `float_motion_vulkan.c`, `float_motion_hip.c`,
  `float_motion_metal.mm`. PR #863 initially wired this option.

- **`integer_adm/adm_cm.cu` (and the rest of the `integer_adm/`
  subdirectory) carries an NVIDIA copyright line** alongside the
  Netflix one. This is upstream-mirror — keep both headers
  verbatim on rebase.

- **`kernel_template.h` mirror with HIP** (ADR-0241). The CUDA
  `cuda/kernel_template.h` (one level up) and HIP
  `../hip/kernel_template.h` move in lockstep. Any change to
  the CUDA template's struct fields, helper signatures, or
  semantics requires a paired HIP change in the same PR.
  Consumers of the template (`integer_psnr_cuda.c` and
  follow-on `integer_ciede_cuda.c` / `integer_moment_cuda.c` /
  ...) lock the HIP twins call-graph-for-call-graph; see
  [`../../hip/AGENTS.md`](../../hip/AGENTS.md) for the full
  consumer list.

## Build

CUDA feature TUs compile only when `meson setup -Denable_cuda=true`.
The `enable_cuda` umbrella flag gates inclusion via
`#if HAVE_CUDA` blocks in `feature/feature_extractor.c`.

## Governing ADRs

- [ADR-0182](../../../../docs/adr/0182-gpu-long-tail-batch-1.md) +
  [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0192](../../../../docs/adr/0192-gpu-long-tail-batch-3.md) —
  GPU long-tail batches. Every CUDA feature kernel here corresponds
  to a row in one of these.
- [ADR-0214](../../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.
- [ADR-0219](../../../../docs/adr/0219-motion3-gpu-contract.md) —
  motion3 GPU contract.
- [ADR-0241](../../../../docs/adr/0241-hip-first-consumer-psnr.md) —
  kernel-template mirror between CUDA and HIP.
- [ADR-0243](../../../../docs/adr/0243-enable-lcs-gpu.md) — MS-SSIM
  `enable_lcs` GPU contract.
- [ADR-0246](../../../../docs/adr/0246-cuda-kernel-template-feature.md) —
  per-feature CUDA kernel-template scaffolding.
- [ADR-0360](../../../../docs/adr/0360-cambi-cuda.md) —
  CAMBI CUDA port (Strategy II hybrid, T3-15a).
