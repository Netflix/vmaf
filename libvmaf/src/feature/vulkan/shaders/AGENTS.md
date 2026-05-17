# AGENTS.md — libvmaf/src/feature/vulkan/shaders

Orientation for agents working on the GLSL compute shaders that back
the Vulkan feature kernels. Parent: [../AGENTS.md](../AGENTS.md). The
host glue that drives these shaders lives in the parent directory
(`<feature>_vulkan.c`).

## Scope

```text
shaders/
  <feature>.comp                # GLSL 450 compute shader (one TU per kernel)
  ssimulacra2_*.comp            # SSIMULACRA 2 multi-shader pyramid (blur, mul, ssim, xyb)
  cambi_{preprocess,derivative,mask_dp,decimate,filter_mode}.comp  # cambi GPU phases (ADR-0210)
```

Compiled to SPV at build time by `glslc` and embedded into the matching
`<feature>_vulkan.c` via the `spv_embed.py` helper one level up
([`../../vulkan/spv_embed.py`](../../../vulkan/spv_embed.py)).

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md) +
  [../../../vulkan/AGENTS.md](../../../vulkan/AGENTS.md)).
- **Wholly-new fork files use the dual Lusoris/Claude (Anthropic)
  copyright header**. Every shader here is fork-original.
- **GLSL FMA contraction is OFF for precision-critical kernels.**
  Compile shaders with `-O0` or use `precise` / `NoContraction`
  decorations on the load-bearing accumulators. This matches the
  CUDA `--fmad=false` and SYCL `-fp-model=precise` rules. Whenever
  you add a new GPU twin, run the cross-backend gate at the
  contracted `places` precision target before declaring success.
- **`precise` is NOT a substitute for the cross-backend gate on
  NVIDIA at API ≥ 1.4.** Driver 595.71 has been observed to drift
  on `vif.comp` scale-2 (45/48 mismatches, max abs `1.527e-02`)
  under a 1.4 bump *despite* every load-bearing FP op being
  correctly decorated with `OpDecorate ... NoContraction` (verified
  at the SPIR-V `OpFDiv` / `OpFMul` / `OpFSub` ID level). See
  ADR-0264
  ([`0264-vulkan-1-4-bump-blocked-on-fp-contraction.md`](../../../../../docs/adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md)),
  [ADR-0269](../../../../../docs/adr/0269-vif-ciede-precise-step-a.md),
  and Research-0056
  ([`0056-vif-ciede-precise-step-a-implementation.md`](../../../../../docs/research/0056-vif-ciede-precise-step-a-implementation.md)).
  The `apiVersion` sites in
  [`../../../vulkan/common.c`](../../../vulkan/common.c) and
  [`../../../vulkan/vma_impl.cpp`](../../../vulkan/vma_impl.cpp)
  remain pinned at `VK_API_VERSION_1_3` until Step B's gating
  conditions in ADR-0264 are met.

## Twin-update rules

Every shader in this directory has a CPU + CUDA + SYCL twin (and
increasingly HIP). The complete cross-backend table lives in
[`../../cuda/AGENTS.md`](../../cuda/AGENTS.md). A change to a
shader's kernel semantics **must** ship with matching changes to:

1. The matching `<feature>_vulkan.c` host TU in the parent
   directory if push-constant layout, descriptor bindings, or
   dispatch dimensions change.
2. The CUDA + SYCL twins (and HIP twin if it exists) if the
   cross-backend score contract changes.

## Rebase-sensitive invariants

- **`ssim.comp` + `ms_ssim.comp` 11-tap Gaussian table is baked
  byte-for-byte from `../../iqa/ssim_tools.h::g_gaussian_window_h`**
  (ADR-0189 / 0190). Any change to the CPU 11-tap table requires
  a paired GLSL update; the SPV is reproducible only when the
  literal taps match. **On rebase**: re-extract the table from
  the CPU header before declaring victory.

- **`motion.comp` vs `motion_v2.comp` mirror semantics diverge by
  design** (ADR-0193). `motion.comp` uses non-replicating mirror;
  `motion_v2.comp` uses **edge-replicating** mirror. Do **not**
  unify them on rebase — the underlying CPU code paths require
  the divergence.

- **`vif.comp` g/sv_sq computation is in `double` — do NOT revert to
  `float` or `precise float`** (ADR-0492). The VIF gain factor `g` and
  residual variance `sv_sq` are computed in double precision via
  `GL_EXT_shader_explicit_arithmetic_types_float64` to match the CPU
  reference (`integer_vif.c` double path). Reverting to `precise float`
  reduces back to ~7 ULP/px fp32 bias, accumulating to ~2×10⁻⁴ VMAF
  delta and failing the ADR-0214 places=4 gate. The `precise` qualifier
  on these variables was removed in this PR; do not re-add it.

- **`vif.comp` `precise` decorations are conservatively scoped**
  (ADR-0264). Widening `precise` past the empirically-determined
  set onto helper functions or onto auxiliary axes makes the
  cross-backend gate strictly worse on NVIDIA. The shader carries
  inline comments recording this empirical bound; do **not** widen
  the scope without re-measuring against the actual NVIDIA lane.

- **`ciede.comp` `precise` scope is empirically optimal**
  (ADR-0269). Widening `precise` into the helper functions
  (`get_h_prime`, `get_upcase_t`, `get_r_sub_t`,
  `srgb_to_linear`, `xyz_to_lab_map`) or onto the Lab axes makes
  the cross-backend gate strictly worse on NVIDIA (5/48 → 46/48
  mismatches). Same instruction as `vif.comp`: keep the scope
  conservative; re-measure before widening.

- **`float_vif.comp` strict-mode compilation and `precise` qualifiers**
  (ADR-0381 / PR #718). `float_vif.comp` is in the
  `psnr_hvs_strict_shaders` list in `meson.build` — it compiles with
  `glslc -O0`, not `-O`. Removing it from that list (e.g. to speed up
  builds) restores the SPIR-V optimizer's FMA-contraction and
  reassociation on the vertical-pass inner loop (`a_xx += c_k * ref_v *
  ref_v`) and on the sigma variance expressions
  (`sigma1_sq = xx - mu1*mu1`). At scales 2 and 3 the local variance is
  very small, so the contraction-induced catastrophic cancellation pushes
  nearly all pixels into the unconditional low-sigma branch, saturating
  the per-scale score to ~1.0 and inflating VMAF by ~+1.07. The
  `precise` qualifiers on the accumulator variables and sigma expressions
  provide defence-in-depth against driver-side contraction (Vulkan 1.4
  NVIDIA + newer MoltenVK may contract after SPIR-V emission). Do **not**
  remove either safeguard without re-running the per-scale VIF parity
  gate (`places=4`) on every CI hardware lane.

- **`cambi_*.comp` is the GPU phases of a hybrid host/GPU port**
  ([ADR-0210](../../../../../docs/adr/0210-cambi-vulkan-integration.md)).
  The five shaders here implement the embarrassingly-parallel
  phases (preprocessing, derivative, 7×7 SAT spatial mask, 2×
  decimate, 3-tap mode filter). The precision-sensitive
  `calculate_c_values` + top-K pooling do **not** have shaders
  — they stay on the host. **Do not** add a `cambi_c_values.comp`
  without an ADR amendment per the deferred Strategy III in
  [research digest 0020](../../../../../docs/research/0020-cambi-gpu-strategies.md).

- **`ssimulacra2_{blur,mul,ssim,xyb}.comp` shader split** mirrors
  the CPU + CUDA pipeline (ADR-0162 / 0163 / 0252). The shaders
  are dispatched in sequence by `../ssimulacra2_vulkan.c`; the
  XYB conversion + 2×2 downsample paths are also exercised by
  the SIMD host-path TUs (`../../x86/ssimulacra2_host_avx2.c` /
  `../../arm64/ssimulacra2_host_neon.c`) which carry the same
  ADR-0161 bit-exactness contract. Re-grouping summation order
  in any ssimulacra2 shader drifts ~1 ULP and breaks the
  cross-backend gate.

## Build

Shaders compile only when `meson setup -Denable_vulkan=true`. The
build line invokes `glslc --target-env=vulkan1.3` per shader; precision-sensitive shaders (`psnr_hvs.comp`, `float_vif.comp`, `ssimulacra2_{blur,ssim,xyb}.comp`) compile with `-O0` to disable SPIR-V-optimizer reassociation/FMA-contraction (see `psnr_hvs_strict_shaders` list in `meson.build`); all others compile with `-O`
(see `../../vulkan/meson.build` for the per-shader rule).

## Governing ADRs

- [ADR-0188](../../../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0189](../../../../../docs/adr/0189-ssim-vulkan.md) +
  [ADR-0190](../../../../../docs/adr/0190-ms-ssim-vulkan.md) — ssim /
  ms_ssim Vulkan kernels.
- [ADR-0193](../../../../../docs/adr/0193-motion-v2-vulkan.md) —
  motion_v2 edge-replicating mirror divergence.
- [ADR-0210](../../../../../docs/adr/0210-cambi-vulkan-integration.md) —
  cambi Vulkan integration (Strategy II hybrid).
- [ADR-0214](../../../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.
- [ADR-0216](../../../../../docs/adr/0216-vulkan-chroma-psnr.md) —
  psnr chroma Vulkan contract.
- ADR-0264
  ([`0264-vulkan-1-4-bump-blocked-on-fp-contraction.md`](../../../../../docs/adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md))
  and
  [ADR-0269](../../../../../docs/adr/0269-vif-ciede-precise-step-a.md)
  — conservative `precise` scoping on NVIDIA.
