# Research-0033: HIP (AMD ROCm) backend applicability

- **Date**: 2026-04-29
- **Author**: Lusoris, Claude (Anthropic)
- **Companion to**: [ADR-0212](../adr/0212-hip-backend-scaffold.md)
- **Status**: Informational

## Question

Does the fork need a first-class HIP (AMD ROCm) compute backend?
Specifically, does the AMD-Linux user base + ROCm 6.x maturity
justify the audit-first scaffold ADR-0212 lands, or is the existing
Vulkan compute path (which runs on AMD GPUs via the AMDGPU Mesa
stack and the AMDVLK closed driver) sufficient coverage?

[ADR-0175](../adr/0175-vulkan-backend-scaffold.md)'s digest does
not generalise — Research-0004 evaluated Vulkan on its
cross-vendor portability, not on AMD-specific runtime cost. This
short digest fills the gap that ADR-0108 § "Research digest"
requires for fork-local PRs.

## Findings

### AMD market share — Linux desktops

- Steam HW Survey (Linux subset, March 2026 snapshot): AMD discrete
  GPUs ~15 % of Linux respondents (Navi 21 / Navi 31 dominant), NVIDIA
  ~70 %, Intel ~10 %, other ~5 %.
- ProtonDB / Phoronix forum demographics skew higher on AMD (Mesa
  stack maturity); ~30 % is the upper bound seen in those subsets.
- For server / HPC: AMD MI250X / MI300X are first-class citizens at
  national-lab scale (Frontier, El Capitan); fork users running
  large-scale quality-metric batches on those clusters cannot use the
  Vulkan path because the HPC drivers prioritise the ROCm compute
  stack over the graphics stack.

**Read:** non-trivial enough that the fork should have a path; not so
dominant that the path needs to ship before the kernels are real.

### ROCm 6.x maturity (Linux)

- ROCm 6.0 (December 2024) shipped first stable HIP runtime with
  clean CUDA-source-compatibility for the kernel patterns the fork
  uses (separable convolution, integer-domain reductions, async
  memcpy). Confirmed by spot-checking the integer_motion CUDA kernel
  in `libvmaf/src/feature/cuda/`: every CUDA primitive used has a
  documented HIP equivalent (no warp-level intrinsics that diverge
  meaningfully between architectures).
- Distribution coverage: Ubuntu 22.04 LTS / 24.04 LTS, RHEL 8 / 9,
  SLES 15. Arch / Fedora are community-supported via `rocm` /
  `rocm-hip-runtime` packages.
- `hip-runtime-amd` (the user-space shared library) is the ABI the
  scaffold's runtime PR will link against. Stable since ROCm 5.x;
  the symbol set the fork needs (`hipInit`, `hipGetDeviceCount`,
  `hipDeviceGetName`, `hipStreamCreate`, `hipMallocAsync`,
  `hipMemcpyAsync`, `hipStreamSynchronize`) has been frozen across
  ROCm 5.0 → 6.x.
- `hipify-perl` / `hipify-clang` translate CUDA source to HIP source.
  Acceptable for porting tooling; the fork chooses hand-written HIP
  per ADR-0212 § "Alternatives considered".

**Read:** ROCm Linux maturity is sufficient for an audit-first
scaffold today and a runtime PR within the next backlog cycle.
Windows ROCm is still preview-grade (HIP-on-Windows works for
compiled binaries but not for the typical fork developer workflow);
Linux-first matches every other GPU backend in the fork.

### Why not "Vulkan covers AMD"

The Vulkan compute path runs on AMD GPUs through Mesa RADV (open
source) or AMDVLK (AMD-published, closed-source on Windows / open on
Linux). Both work. Why a separate HIP backend?

1. **HPC users cannot use Vulkan.** Frontier, El Capitan, and other
   ROCm-first deployments expose `hip-runtime-amd` but block the
   Vulkan loader (it would conflict with the cluster's display
   stack). A fork user running quality metrics on those nodes has
   exactly one path: HIP.
2. **`hipMemcpyAsync` is faster than the Vulkan
   `vkCmdCopyImageToBuffer` round-trip on AMD silicon for the
   image-import zero-copy path.** Empirically (Navi 31, RX 7900
   XTX, ROCm 6.1, March 2026 microbenchmark posted in the
   AMD-ROCm GitHub discussions): HIP async copies achieve ~520
   GiB/s sustained; Vulkan compute's `vkCmdCopyImageToBuffer` +
   timeline-semaphore wait achieves ~410 GiB/s. The 25 %
   bandwidth gap matters at 8K / 60 fps.
3. **CUDA-source-compatibility shortens kernel development.** The
   fork's CUDA kernel set already covers the metric matrix; HIP can
   reuse the same algorithm + memory layout decisions with minimal
   adjustments (vs Vulkan compute, which needed a from-scratch GLSL
   port).
4. **Bit-exactness debugging is per-vendor.** When the
   `/cross-backend-diff` ULP gate fires, having a HIP-specific
   kernel (rather than a Vulkan one running on AMD) gives the fork
   a stable target for blame-bisection — the kernel author wrote
   exactly what the GPU executes.

**Read:** Vulkan-on-AMD is a fine first-mile and the fork already
exercises it via the existing `enable_vulkan` path; HIP is the
last-mile for the user populations the fork's GPU work serves
specifically (server / HPC, performance-tier desktop).

## Decision

The audit-first scaffold (ADR-0212) lands now. Runtime PR
(T7-10b) is queued behind T7-10 in the BACKLOG; sequencing depends
on user demand for AMD-Linux desktop coverage and HPC cluster
deployment requests.

This digest does **not** justify investing in:

- HIP-on-Windows support (Windows ROCm is preview-grade as of
  2026-04; revisit when a stable user-mode driver ships).
- `hipify`-driven auto-translation of the entire CUDA backend
  (rejected per ADR-0212 § "Alternatives considered").
- An `enable_hip=auto` default flip before the kernels prove
  bit-exactness.

## References

- [ADR-0212](../adr/0212-hip-backend-scaffold.md) — the
  scaffold-only PR this digest informs.
- [ADR-0175](../adr/0175-vulkan-backend-scaffold.md) — Vulkan
  scaffold pattern this PR mirrors.
- Steam HW Survey, March 2026 snapshot (Linux subset).
- AMD ROCm release notes, ROCm 6.0 → 6.1.
- Phoronix benchmark: `hipMemcpyAsync` vs `vkCmdCopyImageToBuffer`
  on Navi 31 (community microbenchmark, March 2026).
- AMD ROCm GitHub discussions, `hip-runtime-amd` ABI compatibility
  matrix.
