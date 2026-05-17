# ADR-0468: Introduce integer_adm_vulkan.c as canonical Vulkan integer ADM extractor

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `vulkan`, `build`, `feature-extractor`

## Context

The existing `adm_vulkan.c` implements the full 4-scale integer ADM pipeline on
Vulkan and exports `vmaf_fex_integer_adm_vulkan`. However, the file name and
the extractor `.name` field (`"adm_vulkan"`) are inconsistent with the naming
convention used by every other integer-path GPU extractor in the fork:
`integer_vif_vulkan.c`, `integer_adm_cuda.c`, `integer_adm_sycl.cpp`, etc.

The inconsistency was introduced when the Vulkan ADM extractor was first
scaffolded (ADR-0178) before the `integer_*` naming convention was formalised
for GPU backends. It creates friction when searching for the extractor by name
and when cross-referencing the CUDA port.

## Decision

We will introduce `integer_adm_vulkan.c` as the canonical file, with:

- GLSL compute shaders `integer_adm.comp` / `integer_adm_reduce.comp` (content-
  identical to `adm.comp` / `adm_reduce.comp` with updated headers naming the
  canonical pair).
- The extractor `.name` field set to `"integer_adm_vulkan"`.
- The exported C symbol remaining `vmaf_fex_integer_adm_vulkan` (the symbol
  `feature_extractor.c` already references; no link change).

The legacy `adm_vulkan.c` is retained in the build as a compatibility shim
(exporting `vmaf_fex_integer_adm_vulkan_legacy` with `.name = "adm_vulkan"`),
ensuring no ABI/API breakage for callers that look up the extractor by string
name through `vmaf_feature_extractor_by_name()`.

## Alternatives considered

Doing a pure rename (delete `adm_vulkan.c`, replace with `integer_adm_vulkan.c`
keeping the `.name = "adm_vulkan"` field) was rejected because the PR must
update the `.name` to match the file name for consistency. Keeping the old file
as a legacy shim is a lower-risk one-PR approach: the model dispatch tables
continue to function unmodified since the C symbol name is unchanged.

Renaming the shaders in-place (`adm.comp` → `integer_adm.comp`) and deleting
the old `.comp` files was considered but rejected — the legacy `adm_vulkan.c`
shim still links `adm_spv.h` / `adm_reduce_spv.h` so the old shaders must be
retained until the shim is removed in a follow-up cleanup PR.

## Consequences

- **Positive**: `integer_adm_vulkan.c` is now consistent with `integer_vif_vulkan.c`
  and the CUDA / SYCL twins. `grep integer_adm` finds all integer ADM GPU paths.
- **Negative**: two `.comp` shaders now exist for the same algorithm
  (`adm.comp` + `integer_adm.comp`). The legacy pair will be removed in a
  follow-up when `adm_vulkan.c` is deleted.
- **Neutral / follow-ups**: a follow-up PR should delete `adm_vulkan.c`,
  `adm.comp`, `adm_reduce.comp`, and update any remaining `.name = "adm_vulkan"`
  string lookups to `"integer_adm_vulkan"`.

## References

- ADR-0178: original Vulkan ADM kernel scaffold.
- ADR-0350: two-level GPU reduction (`adm_reduce.comp`).
- Source: `req` — user directed: "Port CUDA integer_adm_cuda.c to Vulkan.
  Create integer_adm_vulkan.c + GLSL compute shaders. Wire + register."
