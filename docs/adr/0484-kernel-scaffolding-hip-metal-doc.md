# ADR-0484: Extend kernel-scaffolding.md with HIP and Metal lifecycle contract

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: docs, hip, metal, gpu, fork-local

## Context

`docs/backends/kernel-scaffolding.md` was introduced by ADR-0246 to document
the CUDA and Vulkan kernel lifecycle templates. It mentioned the HIP template
in the file list but contained no HIP section, and omitted Metal entirely.

The GPU-template dedup audit (2026-05-16, `.workingdir/dedup-audit-gpu-templates-2026-05-16.md`)
identified that the lifecycle struct comment blocks in `hip/kernel_template.h`
and `metal/kernel_template.h` are 60-70% identical to the corresponding CUDA
description, and that each header independently documents the same four-phase
lifecycle contract (init / submit / collect / close). Without a single
authoritative source, the three separate header comment blocks drift
independently as each backend matures.

The recommended action was to extract the shared contract description into the
existing doc, rather than duplicating it across three `.h` headers, so future
backend authors have a single reference and header comments can remain brief.

## Decision

We will extend `docs/backends/kernel-scaffolding.md` to add:

- A **HIP template** section documenting `VmafHipKernelLifecycle`,
  `VmafHipKernelReadback`, and the six lifecycle helpers, including a
  differences table comparing it to the CUDA template.
- A **Metal template** section documenting `VmafMetalKernelLifecycle`,
  `VmafMetalKernelBuffer`, and the six lifecycle helpers, including a
  differences table comparing it to the HIP template.
- A **Lifecycle contract** section stating the four-phase contract
  (init / submit / collect / close) that all four backends share, so the
  shared invariant lives in one place rather than being restated in each
  header's comment block.
- Updates to the **Why per-backend** and **Why helper functions** rationale
  sections to cover HIP and Metal's out-of-line posture.

No changes to C source files, no meson changes, no new link-time symbols.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep separate header comment blocks | No doc debt | Invariant text drifts independently across three `.h` files | Defeats the purpose of having a shared doc |
| Shared `.inc` file included by all three headers | Single source for struct comments | Preprocessor abuse; `.inc` files are harder to read standalone and not friendly to clangd | The doc is the right level of abstraction for contract prose |
| Cross-backend unified struct (`uintptr_t`-only base) | One C type | Loses backend-specific field names (`str` vs `cmd_queue`); breaks consumer code that reads fields by name | Handle-type differences are load-bearing per ADR-0246 |

## Consequences

- **Positive**: New backend authors read one doc to understand the lifecycle
  contract instead of three headers. The four-phase init/submit/collect/close
  contract is stated once, unambiguously.
- **Negative**: The doc must be kept in sync when a backend adds a new helper
  (e.g., `vmaf_hip_kernel_submit_post_record` was added in a later PR). This
  is a lighter burden than keeping three header comment blocks in sync.
- **Neutral**: The individual headers retain brief per-function comments;
  the doc is the authoritative narrative description.

## References

- [ADR-0246](0246-gpu-kernel-template.md) — CUDA + Vulkan kernel template origin.
- [ADR-0241](0241-hip-first-consumer-psnr.md) — HIP kernel template introduction.
- [ADR-0361](0361-metal-compute-backend.md) — Metal kernel template introduction.
- `.workingdir/dedup-audit-gpu-templates-2026-05-16.md` — audit item 2:
  "Lifecycle struct + comment duplication; ~40–50 LOC saveable; recommended:
  shared doc + `.inc` file."
- req: "implement: extract duplicated code/text into a shared header/helper."
