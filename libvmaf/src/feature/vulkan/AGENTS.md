# AGENTS.md — libvmaf/src/feature/vulkan

Orientation for agents working on per-feature Vulkan host-glue (`.c`
TUs that drive compute shaders). Parent: [../AGENTS.md](../AGENTS.md).
The backend runtime (instance, device, VMA, picture, image-import)
lives one level up in
[`../../vulkan/AGENTS.md`](../../vulkan/AGENTS.md). The GLSL compute
shaders live one level deeper in
[`shaders/AGENTS.md`](shaders/AGENTS.md).

## Scope

```text
feature/vulkan/
  <feature>_vulkan.c              # host glue: VkPipeline + VkDescriptorSet creation + push-constant assembly + per-frame submit
  shaders/<feature>*.comp         # GLSL compute shader (separate AGENTS.md — see below)
```

The build-time `spv_embed.py` helper one level up converts each
`.comp` to a static C array embedded in the host TU. Adding a new
shader to the build means adding it to
[`../../vulkan/meson.build`](../../vulkan/meson.build) AND including
its embedded SPV in the matching `<feature>_vulkan.c`.

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md) +
  [../../AGENTS.md](../../AGENTS.md) +
  [`../../vulkan/AGENTS.md`](../../vulkan/AGENTS.md)).
- **Wholly-new fork files use the dual Netflix + Lusoris/Claude
  copyright header** per [ADR-0025](../../../../docs/adr/0025-copyright-handling-dual-notice.md).
  Every TU here is fork-original.
- **Host glue mirrors the GLSL contract.** Push-constant struct
  layout, descriptor-binding indices, and dispatch dimensions are
  encoded *both* in the host C and the shader. Drift between them
  = silent miscompute or a Vulkan validation-layer error at
  dispatch time.
- **Volk-remapped `vk*` symbols** — every Vulkan call here resolves
  through the `vmaf_priv_vk*` rename per
  [ADR-0185](../../../../docs/adr/0185-vulkan-hide-volk-symbols.md).
  Don't introduce a raw `vkCreate*` reference; the force-included
  remap header rewrites it at the preprocessor level.

## Twin-update rules

Every TU in this directory has CUDA + SYCL twins (and increasingly
HIP). The complete cross-backend table lives in
[`../cuda/AGENTS.md`](../cuda/AGENTS.md); changes to a Vulkan host
TU **must** ship with matching changes to:

1. The matching GLSL shader in [`shaders/`](shaders/AGENTS.md) if
   the kernel semantics change.
2. The CUDA / SYCL / HIP twins if the cross-backend score
   contract changes.

The cross-backend parity gate at `places=4` (or relaxed per the
calibration table at `scripts/ci/gpu_ulp_calibration.yaml`,
ADR-0234) catches drift but only after a full GPU run.

## Rebase-sensitive invariants

- **`psnr_vulkan.c` chroma plane loop and `enable_chroma` option**
  ([ADR-0216](../../../../docs/adr/0216-vulkan-chroma-psnr.md) /
  [ADR-0453](../../../../docs/adr/0453-psnr-enable-chroma-gpu-parity.md)).
  Carries `ref_in[3] / dis_in[3] / se_partials[3]` arrays in
  `PsnrVulkanState` (Y / Cb / Cr) and dispatches the same
  `psnr.comp` shader once per active plane in a single command
  buffer. Rebases that "simplify" this loop back to a single luma
  dispatch silently regress `psnr_cb` / `psnr_cr` to CPU
  fall-through and break the cross-backend parity gate. The
  descriptor pool is sized for 12 sets (4 frames in flight × 3
  planes) — do **not** shrink without re-checking lavapipe
  behaviour under `frames-in-flight > 1`. `n_planes` is clamped to
  1 in two cases: (1) `pix_fmt == YUV400P` (chroma absent); (2)
  `enable_chroma == false` (caller opted out). The latter is the
  ADR-0453 addition. The `enable_chroma` option carries
  `default_val.b = true`; do **not** flip the default. See
  [`../../AGENTS.md §"Vulkan PSNR chroma contract"`](../../AGENTS.md).

- **`ms_ssim_vulkan.c` honours the `enable_lcs` GPU contract**
  (ADR-0243). Emits 15 extra metrics
  (`float_ms_ssim_{l,c,s}_scale{0..4}`) when `enable_lcs=true`.
  Mirrors the CPU + CUDA + SYCL twins metric-wise. See
  [../../AGENTS.md §"MS-SSIM `enable_lcs` GPU contract"](../../AGENTS.md).

- **`motion_vulkan.c::motion3_postprocess_*` honours the motion3
  GPU contract** (ADR-0219). `motion_five_frame_window=true`
  returns `-ENOTSUP` at `init()`. Any Netflix upstream sync that
  touches `motion_blend()` mirrors across the three GPU motion
  twins in the same PR. See [../../AGENTS.md §"motion3_score GPU
  contract"](../../AGENTS.md).

- **`motion_v2_vulkan.c` mirror divergence** (ADR-0193). The
  Vulkan kernel uses an **edge-replicating** mirror that diverges
  from `motion.comp`'s non-replicating mirror — load-bearing per
  the underlying CPU code path. Do **not** unify the two mirror
  shapes on rebase.

- **`cambi_vulkan.c` is hybrid host/GPU**
  ([ADR-0205](../../../../docs/adr/0205-cambi-gpu-feasibility.md) +
  [ADR-0210](../../../../docs/adr/0210-cambi-vulkan-integration.md)).
  The Vulkan kernel offloads only the embarrassingly-parallel phases
  (preprocessing scaffold + derivative + 7×7 SAT spatial mask + 2×
  decimate + 3-tap mode filter); the precision-sensitive
  `calculate_c_values` sliding-histogram pass + top-K spatial pooling
  stay on the host. The host residual call site
  (`cambi_vk_extract` → `vmaf_cambi_calculate_c_values`) is
  intentionally the same C as CPU `cambi.c::calculate_c_values`.
  Strategy III (fully-on-GPU c-values) is documented in
  [research digest 0020](../../../../docs/research/0020-cambi-gpu-strategies.md)
  but **deferred** — do not attempt it inside v1.

- **`ssimulacra2_vulkan.c` calls into the SIMD host-path TUs**
  (ADR-0252). The pyramid-layout XYB conversion + 2x2 downsample
  are dispatched to `ssimulacra2_host_avx2.c` /
  `ssimulacra2_host_neon.c`, which carry the same ADR-0161
  bit-exactness contract as their CPU-extractor siblings. **On
  rebase**: if the scalar `ss2v_host_linear_rgb_to_xyb` or
  `ss2v_downsample_2x2` arithmetic order in this TU changes, the
  SIMD twins and their `test_host_xyb` / `test_host_downsample`
  scalar references must be updated in lockstep. See
  [../AGENTS.md §"SSIMULACRA 2 Vulkan host-path SIMD"](../AGENTS.md).

- **`v1: scale=1 only` constraint on ssim/ms_ssim/cuda twins**
  applies on Vulkan too (ADR-0188 / 0189 / 0190). Auto-decimation
  is a v2 follow-up; do not silently enable it on rebase.

- **`adm_vulkan.c` / `float_adm_vulkan.c` expose three ADM tuning
  parameters** (`adm_csf_scale`, `adm_csf_diag_scale`, `noise_weight`)
  with the same defaults as the CPU path (PR #731). If upstream Netflix
  adds or renames these parameters, the Vulkan twins must be updated in
  the same PR.

- **`motion_fps_weight` cross-backend parity** — see the canonical
  invariant note in [`../cuda/AGENTS.md`](../cuda/AGENTS.md).
  `motion_v2_vulkan.c` and `float_motion_vulkan.c` both carry the
  `motion_fps_weight` option and apply it in `flush()` / `extract()`
  exactly as documented there. Any future change to the weight
  application math must span all motion-family GPU twins in the same PR.

- **`adm_vulkan.c` integer fast-path gated on CSF-scale defaults.**
  The hard-coded `i_rfactor` fast-path for `3.0 * 1080` default
  viewing geometry is gated by:
  `bool csf_default = (fabs(s->adm_csf_scale - 1.0) < 1e-9) &&
  (fabs(s->adm_csf_diag_scale - 1.0) < 1e-9)`.
  Removing or loosening this guard produces wrong rfactors when
  non-default CSF scales are passed. Update the guard if the
  fast-path formula changes.

- **`vif_vulkan.c` / `adm_vulkan.c` / `motion_vulkan.c` two-level GPU
  reduction** (ADR-0356 / T-GPU-PERF-VK-3). Each of these three
  kernels now runs a *second* compute dispatch per frame (vif_reduce.comp,
  adm_reduce.comp, motion_reduce.comp) that reduces the per-WG
  accumulator SSBO to a tiny fixed-size output buffer. Key invariants:
  - The `VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT |
    VK_ACCESS_SHADER_WRITE_BIT` memory barrier between the per-WG
    dispatch and the reducer dispatch is **load-bearing** — removing it
    reintroduces a write-after-write hazard on the accumulator SSBO and a
    read-before-write hazard on `reduced_accum`.
  - The reducer shaders require `VK_EXT_shader_atomic_int64`
    (`shaderBufferInt64Atomics`). Do not port these shaders to a platform
    that doesn't advertise this feature without adding a fallback (e.g.
    a one-WG serial path).
  - `vmaf_vulkan_buffer_invalidate()` must be called on `reduced_accum`
    / `reduced_sad` after the fence wait and before the host reads the
    result. Omitting it is a coherency bug on non-coherent heaps (common
    on discrete GPU without ReBAR).
  - `reduced_accum` / `reduced_sad` must be zeroed (host `memset` +
    `flush`) **each frame** because the reducer uses `atomicAdd` to
    accumulate from multiple WGs into a single output. The per-WG
    `accum` buffer also continues to be zeroed each frame.
  - Bit-exactness: int64 reduction is order-independent (two's-complement
    addition is commutative and associative). The GPU reducer produces
    the same totals as the removed host loop regardless of WG execution
    order. This is the load-bearing claim that drives the places=4 gate.
  - The `MAX_SUBGROUPS = 256` constant in the reducer shaders matches
    `local_size_x = 256`. Changing the WG size requires updating both.

## Build

Vulkan feature TUs compile only when `meson setup -Denable_vulkan=true`.
The umbrella flag pulls in `dependency('vulkan')` + volk + glslc + VMA.

- **Submit-pool destroy-before-pipeline ordering** ([ADR-0256](../../../../docs/adr/0256-vulkan-submit-pool-template.md) /
  [ADR-0354](../../../../docs/adr/0354-vulkan-submit-pool-pr-c-four-kernels.md)).
  In every extractor that uses `VmafVulkanKernelSubmitPool`,
  `vmaf_vulkan_kernel_submit_pool_destroy()` MUST be called **before**
  any `vmaf_vulkan_kernel_pipeline_destroy()` call in `close_fex()`.
  Reversing the order frees the pipeline's descriptor pool + command
  pool before the submit pool drains its command buffers — undefined
  behaviour. The invariant applies to all 13 migrated extractors (PR-A,
  PR-B, PR-C). Extractors in scope for PR-C:
  `cambi_vulkan.c`, `ssimulacra2_vulkan.c`, `float_ansnr_vulkan.c`,
  `moment_vulkan.c`.

## Governing ADRs

- [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0189](../../../../docs/adr/0189-ssim-vulkan.md) +
  [ADR-0190](../../../../docs/adr/0190-ms-ssim-vulkan.md) — ssim /
  ms_ssim Vulkan; 11-tap Gaussian baked into GLSL byte-for-byte from
  `iqa/ssim_tools.h::g_gaussian_window_h`.
- [ADR-0193](../../../../docs/adr/0193-motion-v2-vulkan.md) —
  `motion_v2` Vulkan kernel; edge-replicating mirror.
- [ADR-0205](../../../../docs/adr/0205-cambi-gpu-feasibility.md) +
  [ADR-0210](../../../../docs/adr/0210-cambi-vulkan-integration.md) —
  cambi Vulkan integration (Strategy II hybrid).
- [ADR-0214](../../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.
- [ADR-0216](../../../../docs/adr/0216-vulkan-chroma-psnr.md) —
  PSNR chroma Vulkan contract.
- [ADR-0219](../../../../docs/adr/0219-motion3-gpu-contract.md) —
  motion3 GPU contract.
- [ADR-0243](../../../../docs/adr/0243-enable-lcs-gpu.md) — MS-SSIM
  `enable_lcs` GPU contract.
- [ADR-0252](../../../../docs/adr/0252-ssimulacra2-host-xyb-simd.md) —
  SSIMULACRA 2 Vulkan host-path SIMD.
- [ADR-0256](../../../../docs/adr/0256-vulkan-submit-pool.md) —
  submit pool design (`VmafVulkanKernelSubmitPool`).
- [ADR-0356](../../../../docs/adr/0356-vulkan-two-level-gpu-reduction.md) —
  Two-level GPU reduction for VIF / ADM / motion accumulators (T-GPU-PERF-VK-3).
- [ADR-0357](../../../../docs/adr/0357-vulkan-readback-alloc-flag.md) —
  UPLOAD vs READBACK VMA allocation flag separation.
- [ADR-0353](../../../../docs/adr/0353-vulkan-submit-pool-pr-b-six-kernels.md) —
  PR-B six-kernel submit-pool migration.

## Submit-pool ordering invariant (ADR-0256 / ADR-0353)

Kernels that use `VmafVulkanKernelSubmitPool` **must** destroy the pool
**before** calling `vmaf_vulkan_kernel_pipeline_destroy`. This is because
`vmaf_vulkan_kernel_pipeline_destroy` internally calls `vkDeviceWaitIdle`
and then destroys the descriptor pool; if the submit pool still holds a
pending fence at that point, cleanup will race and produce a validation
error.

The required `close_fex()` tear-down order for all migrated kernels is:

```c
vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
// ... any other per-bundle pool destroys ...
vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);
```

For `ms_ssim_vulkan.c` specifically:
```c
vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_decimate);
vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_ssim);
vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_ssim);
```

Pre-allocated descriptor sets (allocated via `vmaf_vulkan_kernel_descriptor_sets_alloc`)
are freed implicitly when the descriptor pool is destroyed inside
`vmaf_vulkan_kernel_pipeline_destroy`. **Do not** call `vkFreeDescriptorSets`
on pre-allocated sets — it is a double-free.

**Migrated kernels (PR-A + PR-B)**: `adm`, `motion`, `psnr` (PR-A /
PR #563), `ssim`, `ciede`, `ms_ssim`, `motion_v2`, `float_psnr`,
`float_motion` (PR-B / ADR-0353). Remaining legacy kernels (`ansnr`,
`vif`, `ssimulacra2`, `cambi`) are deferred to PR-C.

## Buffer allocation invariant (ADR-0357)

Feature kernel files in this directory must allocate buffers with the correct
VMA classification:

- **UPLOAD buffers** (CPU writes pixel data or LUT, then calls
  `vmaf_vulkan_buffer_flush`, then GPU reads): use `vmaf_vulkan_buffer_alloc`.
  Examples: `ref_in`, `dis_in`, `log2_lut`, `div_lookup`, `csf_f`, `src_ref`,
  `src_dis`, `ref_lin`, `dis_lin`, `raw_in_buf`, blur intermediate scratch,
  pyramid downscale buffers.

- **READBACK buffers** (GPU writes per-workgroup accumulator or partial-sum
  slots, CPU reduces them post-fence): use `vmaf_vulkan_buffer_alloc_readback`
  and call `vmaf_vulkan_buffer_invalidate(ctx, buf)` after the fence-wait,
  before calling `vmaf_vulkan_buffer_host(buf)`.  Examples: `accum`, `partials`,
  `sad_partials`, `sums`, `se_partials`, `l/c/s_partials`, `num/den_partials`,
  `sig/noise_partials`, `mu1`, `mu2`, `s11`, `s22`, `s12`.

Using `alloc` for a readback buffer silently incurs 4–8× slower CPU reads on
discrete GPUs (uncached BAR heap instead of HOST_CACHED).  Missing the
`invalidate` call makes the CPU see stale data on non-coherent heaps.  Both
bugs are silent (no crash, wrong throughput or wrong result respectively).

See [ADR-0357](../../../../docs/adr/0357-vulkan-readback-alloc-flag.md).
