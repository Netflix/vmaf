# ADR-0483: Extract shared `vmaf_gpu_dispatch_parse_env` tokenizer

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `cuda`, `sycl`, `vulkan`, `refactor`, `dedup`

## Context

Each GPU backend (CUDA, SYCL, Vulkan) parses its dispatch env variable
(`VMAF_CUDA_DISPATCH`, `VMAF_SYCL_DISPATCH`, `VMAF_VULKAN_DISPATCH`) with an
identical 34-LOC tokenizer loop — `parse_per_feature_override` — that was
duplicated verbatim across three translation units. The only differences were
the output-enum type and the strategy-name strings ("graph"/"direct" for
CUDA/SYCL, "primary"/"reuse" for Vulkan). Any bug fix or grammar extension
had to be applied three times, and the copies had already begun to drift
slightly in comment quality.

The dedup was identified in the GPU-template dedup audit
(`dedup-audit-gpu-templates-2026-05-16.md`, opportunity 1) with an estimated
68 LOC saving and zero risk to bit-exactness (the function is pure env-variable
parsing with no effect on kernel-launch order or accumulator state).

## Decision

Extract a single `static inline` function `vmaf_gpu_dispatch_parse_env` into a
new header `libvmaf/src/gpu_dispatch_parse.h`. The function takes a
NULL-terminated `strategy_names` array instead of backend-specific enum types,
returning an integer index that each backend maps to its own enum. The header is
C89-compatible and free of backend-specific includes, so it can be included from
both C (CUDA, Vulkan) and C++ (SYCL) translation units without complications.

Each backend's `dispatch_strategy` TU replaces the removed function with a
`static const char *const k_<backend>_strategy_names[]` table and a single call
to `vmaf_gpu_dispatch_parse_env`.

## Alternatives considered

Keeping the three copies separately was rejected because the function is
provably identical (diff output: 0 lines of semantic difference) and will grow
as more strategies are added.

An X-macro approach was considered but rejected: it is harder to lint-clean and
harder for future contributors to read than a plain function + table.

A shared `.c` compilation unit (not header-only) was considered but rejected
because it would require adding a new meson source to every backend's link set
— unnecessary overhead for a 30-line inline that needs no state.

## Consequences

- **Positive**: The tokenizer loop lives in exactly one place. Future grammar
  extensions (new strategy names, whitespace rules) require a single change.
  The 68 LOC of duplicated code is removed; the replacement is ~20 LOC of
  tables across three files + one ~90-LOC header.
- **Negative**: None material. The `static inline` is emitted once per TU, but
  the function is only called once per `select_strategy` invocation (not a hot
  path), so the duplication at the object-code level is immaterial.
- **Neutral / follow-ups**: Future backends (HIP, Metal) that add dispatch env
  parsing should use `vmaf_gpu_dispatch_parse_env` from day one rather than
  duplicating the loop.

## References

- `.workingdir/dedup-audit-gpu-templates-2026-05-16.md` — opportunity 1
- ADR-0181 (feature-characteristics registry — where `dispatch_strategy` was first added)
- PR that implements this: see `refactor/gpu-dispatch-parse-dedup` branch
