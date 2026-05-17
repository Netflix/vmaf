# ADR-0486: Codify the three-function GPU backend context-API contract in docs

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `docs`, `gpu`, `hip`, `metal`, `vulkan`, `cuda`, `api`, `fork-local`

## Context

Every GPU backend in libvmaf exposes the same three-function API surface:
`vmaf_<backend>_context_new`, `vmaf_<backend>_context_destroy`, and
`vmaf_<backend>_device_count`.  The Vulkan, HIP, and Metal backends all define
this triple in their respective `common.h` headers, but no single place states
the expected signatures, error-return contract, or the rationale for the shape.
Without a written contract, each new backend must reverse-engineer the pattern
from existing headers; invariants (e.g., `context_destroy(NULL)` is a no-op;
`device_count` returns `-ENODEV` not `0` on discovery failure) drift silently.

The dedup audit (`.workingdir/dedup-audit-gpu-templates-2026-05-16.md`,
opportunity 4) identified this as a zero-LOC-savings but non-zero-risk gap:
the three headers are structurally identical, the implementations are
necessarily different, and the right action is documentation rather than code
unification.

## Decision

Add `docs/backends/context-api-contract.md` that states the three-function
shape, the POSIX errno return contract, the CUDA-deviation exemption, the
opaque-handle-accessor pattern, and a new-backend checklist.  No source files
are modified; the doc alone closes the drift risk.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Shared C header with function-pointer table | Single machine-readable contract | Requires every backend to register a vtable; breaks header purity (backend headers stay free of cross-backend deps) | Over-engineering for a shape that has only ~4 implementations |
| Comment block in each `common.h` referencing a contract | Keeps the contract co-located with the declaration | 60-line comment must be kept in sync across 4+ files — the exact drift problem we are solving | |
| `docs/backends/context-api-contract.md` (chosen) | One place, no source change, low merge-conflict surface | Docs can drift from code if not maintained | Lowest overhead; checklist section prompts new-backend authors to verify |

## Consequences

- **Positive**: New-backend authors have a single reference for signatures and
  error contracts.  Reviewers can point to the doc when a new backend deviates
  from the contract.
- **Negative**: The doc can drift from the headers if a backend changes its
  signature without updating the doc.  Mitigated by the PR-template
  ffmpeg-patches checkbox (any public-header change triggers a patch review,
  which surfaces the contract doc for a sync check).
- **Neutral / follow-ups**: The CUDA deviation (upstream naming) is explicitly
  exempted and documented; no follow-up rename needed.

## References

- Dedup audit: `.workingdir/dedup-audit-gpu-templates-2026-05-16.md` §4
  "Context API surface shape".
- HIP context API: `libvmaf/src/hip/common.h` (ADR-0212).
- Metal context API: `libvmaf/src/metal/common.h` (ADR-0361).
- Vulkan context API: `libvmaf/src/vulkan/vulkan_common.h` (ADR-0175).
- CUDA deviation: `libvmaf/src/cuda/common.h` (upstream Netflix API).
- Related: ADR-0108 (deep-dive deliverables rule).
