# ADR-0212: HIP (AMD ROCm) compute backend — scaffold-only audit-first PR (T7-10)

- **Status**: Accepted
- **Date**: 2026-04-29
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: gpu, hip, rocm, amd, scaffold, audit-first, fork-local

## Context

The fork's GPU portfolio currently covers NVIDIA (CUDA), Intel
(SYCL / oneAPI), and software / cross-vendor (Vulkan compute) compute
paths. The matrix has a load-bearing gap: AMD GPUs running ROCm have no
first-class backend. AMD's discrete-GPU share on Linux desktops is
non-trivial (Steam HW Survey 2026 ~15 % among Linux respondents — see
[Research-0033](../research/0033-hip-applicability.md)) and ROCm 6.x
ships a stable `hip-runtime-amd` user-space that mirrors CUDA's API
surface closely.

Backlog item **T7-10** queued the HIP backend behind the Vulkan
scaffold + runtime work. The Vulkan T5-1 → T5-1b → T5-1c sequence has
now validated the audit-first split end-to-end (per
[ADR-0175](0175-vulkan-backend-scaffold.md) +
[ADR-0176](0176-vulkan-vif-cross-backend-gate.md) +
[ADR-0193](0193-motion-v2-vulkan.md)): land static surfaces in one
focused PR, then runtime + kernels in follow-up PRs against a stable
base. T7-10 reproduces that pattern for HIP.

This ADR is the audit-first companion. Same shape as ADR-0175 for
Vulkan, ADR-0173 for the PTQ harness, ADR-0167 for doc-drift
enforcement: ship the **static surfaces** (header, build wiring,
kernel stubs, smoke, docs) in a focused PR so the runtime PRs that
follow have a stable base to land on.

## Decision

### Land scaffold only — no ROCm SDK at build time

The PR creates:

- Public header
  [`libvmaf/include/libvmaf/libvmaf_hip.h`](../../libvmaf/include/libvmaf/libvmaf_hip.h):
  declares `VmafHipState`, `VmafHipConfiguration`,
  `vmaf_hip_state_init` / `_import_state` / `_state_free`,
  `vmaf_hip_list_devices`, `vmaf_hip_available`. Mirrors the CUDA +
  Vulkan + SYCL pattern.
- Backend tree under
  [`libvmaf/src/hip/`](../../libvmaf/src/hip/) — `common.{c,h}`,
  `picture_hip.{c,h}`, `dispatch_strategy.{c,h}`, `meson.build`. All
  entry points return `-ENOSYS` or do-nothing.
- Kernel stubs at
  [`libvmaf/src/feature/hip/`](../../libvmaf/src/feature/hip/) —
  `adm_hip.c`, `vif_hip.c`, `motion_hip.c`. `_init` / `_run` entry
  points return `-ENOSYS` until kernels arrive.
- New `enable_hip` boolean option (default **false**) in
  [`libvmaf/meson_options.txt`](../../libvmaf/meson_options.txt).
- Conditional `subdir('hip')` in
  [`libvmaf/src/meson.build`](../../libvmaf/src/meson.build);
  `hip_sources` + `hip_deps` threaded through
  `libvmaf_feature_static_lib` alongside the existing CUDA / SYCL /
  Vulkan / DNN aggregations.
- Smoke test
  [`libvmaf/test/test_hip_smoke.c`](../../libvmaf/test/test_hip_smoke.c)
  with 9 sub-tests pinning the scaffold contract — the four internal
  context-lifecycle checks plus one `-ENOSYS` (or NULL-safe `-EINVAL`)
  expectation per public C-API entry point. Wired in
  [`libvmaf/test/meson.build`](../../libvmaf/test/meson.build) under
  `if get_option('enable_hip') == true`.
- New CI matrix row "Build — Ubuntu HIP (T7-10 scaffold)" in
  [`libvmaf-build-matrix.yml`](../../.github/workflows/libvmaf-build-matrix.yml)
  that compiles with `-Denable_hip=true`. No ROCm SDK is installed on
  the runner; the scaffold compiles cleanly without it.
- New docs at
  [`docs/backends/hip/overview.md`](../backends/hip/overview.md) and
  the index row in
  [`docs/backends/index.md`](../backends/index.md) flipped from
  "planned" to "scaffold only".

### Zero hard runtime dependencies for the scaffold

The scaffold has no required `dependency('hip-lang')`, no `hipcc`, no
`amdhip64`. The `meson.build` includes an optional probe
(`required: false`) so a host that already has ROCm installed will see
the dependency resolve; the scaffold compiles cleanly without it.
Adding the hard linkage is the responsibility of the first kernel PR
(T7-10b runtime). Reasoning: the scaffold's CI run validates "the
build wiring + meson dispatch + test harness work end-to-end"; landing
build deps before any kernel uses them gates the scaffold's own CI
green-light on a ROCm SDK that no code touches.

### Default `enable_hip` to `false`, type `boolean`

The option is `boolean` defaulting to **false**, matching the convention
used by `enable_cuda` and `enable_sycl`. Vulkan uses `feature` (`auto` /
`enabled` / `disabled`) for parity with `enable_dnn`; HIP follows the
GPU-vendor-pair convention instead so the AMD / NVIDIA / Intel triad
stays uniform — operators flipping between the three GPU vendors see
the same syntax (`-Denable_<vendor>=true|false`).

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Land scaffold + runtime + first kernel in one PR | Single round of review, the kernel is exercised against real HIP from the start | Too large to review in one pass; splits the trust boundary between "the scaffold compiles + smoke-tests" and "this kernel produces correct numbers" — different review skills, different CI gates | Audit-first separation per the same pattern as ADR-0175 / ADR-0173 |
| Default `enable_hip` to `auto` / `enabled` (feature option) | Builds with ROCm installed pick HIP up automatically; matches the Vulkan convention | Silent flip on a CI host that happens to have ROCm packages; consumer mode mismatch (a build claims HIP support but `vmaf_hip_available` returns 1 even though every other call is `-ENOSYS`) | Boolean default-off keeps the scaffold opt-in and matches `enable_cuda` / `enable_sycl`; uniform GPU-vendor flag syntax |
| Skip the scaffold; auto-translate the existing CUDA backend with `hipify-perl` / `hipify-clang` | Free initial coverage of every CUDA kernel; no manual port | `hipify` produces source that diverges from idiomatic HIP for non-trivial kernels (warp-level intrinsics, async memcpy, cooperative groups); the fork's CUDA backend uses several CUDA-12 features (`cudaGraphConditional`, NVTX) that have no clean HIP equivalent; bit-exactness vs CPU + CUDA + SYCL + Vulkan would still need a per-kernel audit | Reject — a hand-written HIP backend gives the fork a known-good codebase under fork license; `hipify` is fine as a porting *tool* but not as a production source generator. The runtime PR may still selectively use it as a starting point per kernel |
| Separate `libvmaf_hip.so` (out-of-tree backend, similar to a plug-in) | No risk of HIP-specific code paths bloating the default `libvmaf.so`; AMD-specific testing isolated | Adds a new ABI boundary the fork has to maintain, conflicts with existing in-tree CUDA / SYCL / Vulkan precedent (all in-tree behind `-Denable_<backend>=true`); breaks `/cross-backend-diff` ergonomics (would need to dlopen the HIP backend) | Reject — in-tree matches the fork's existing GPU backend topology and the `meson_options.txt` opt-in pattern keeps default builds free of HIP code |
| AMD-only path vs `hipify`-CUDA-via-translation-layer | Either choice covers AMD; hipify reuses CUDA's existing kernel set | Translation layers (`HIPCC` ↔ `nvcc`) bring tooling-version pin pain; AMD-only keeps the runtime story crisp | AMD-only (hand-written) — see `hipify` row above; the runtime PR uses CUDA kernels as *reference* but ports each one explicitly so the resulting kernel reads as native HIP |
| ROCm runtime vs HIP runtime (use AMD's lower-level `ROCr` API) | Skips the HIP veneer; matches what the kernel team would call directly on AMD | HIP is the supported user-space layer; `ROCr` is a kernel-driver-adjacent surface AMD documents primarily for runtime maintainers; portability story (HIP also targets CUDA via `__HIP_PLATFORM_NVIDIA__`) is lost | HIP — the fork's audience is researchers running quality metrics, not GPU runtime developers; HIP gives one source tree that compiles against AMD ROCm and (optionally) NVIDIA CUDA |
| Skip the smoke test | Saves writing 9 sub-tests | Pre-existing pattern: ADR-0175 explicitly justified the Vulkan smoke as cheap (~5 ms) bit-rot insurance; the same logic applies. A future PR that accidentally enables a code path (e.g. by linking against a real `dependency('hip-lang')` without flipping the kernel-bodies) would trip the smoke test rather than landing silently broken | Reject — smoke test stays |

## Consequences

**Positive:**

- Header surface lands without committing to runtime details. Future
  HIP-targeting consumers (third-party tools, MCP surfaces) can compile
  against the API today; calls fail cleanly with `-ENOSYS` until the
  runtime arrives.
- Build matrix gains a new lane that compiles the scaffold every PR —
  bit-rot is caught immediately.
- The `/add-gpu-backend` skill is exercised on a third backend (after
  Vulkan); the scaffold serves as proof that the abstraction layer is
  clean enough to reproduce, satisfying T7-10's gating condition (see
  the BACKLOG row).
- AMD users on Linux see a clear "this is the path forward" entry in
  `docs/backends/index.md` even before kernels exist, with a concrete
  `-Denable_hip=true` build flag they can compile against.

**Negative:**

- Six new C / H source files (1 public header, 4 implementation, 1
  test) + 1 ADR + 1 doc + 1 research digest with no functional code
  yet. Acceptable for an audit-first PR; the runtime PR will swap the
  bodies in place.
- `vmaf_hip_available()` returns `1` when built with `-Denable_hip=true`
  regardless of whether the kernels are real. A future PR could flip
  `vmaf_hip_available` to return `0` until the runtime PR lands, but
  that would break the scaffold's smoke test contract; for now the
  function honestly reports "the build was opted in", and operators
  read the docs for status. Same convention as Vulkan T5-1.
- No FFmpeg patch in this PR. The fork's `ffmpeg-patches/` series
  doesn't currently consume the HIP API surface (no `hip_device`
  filter option, no AVHWDeviceContext wiring); the runtime PR will add
  the filter option once `vmaf_hip_state_init` actually works. CLAUDE
  §12 r14 only requires patch updates when an existing patch already
  consumes the surface — see `docs/rebase-notes.md` entry for T7-10.

**Neutral / follow-ups:**

- Runtime PR (T7-10b) needs ROCm CI bring-up. GitHub-hosted runners
  do not currently expose AMD GPUs; the runtime PR's smoke test will
  skip the device-resident assertions when `vmaf_hip_device_count() == 0`,
  matching the Vulkan-on-lavapipe-less-CI pattern.
- ADR-0212 will be referenced by every subsequent T7-10X kernel PR for
  the rebase-invariant story.

## Tests

- `libvmaf/test/test_hip_smoke.c` (9 sub-tests, all pass locally):
  - `test_context_new_returns_zeroed_struct`
  - `test_context_new_rejects_null_out`
  - `test_context_destroy_null_is_noop`
  - `test_device_count_scaffold_returns_zero`
  - `test_available_reports_build_flag`
  - `test_state_init_returns_enosys`
  - `test_import_state_returns_enosys`
  - `test_state_free_null_is_noop`
  - `test_list_devices_returns_enosys`
- New CI lane: `Build — Ubuntu HIP (T7-10 scaffold)` in the libvmaf
  build matrix. Compiles with `-Denable_hip=true` and runs the smoke
  test (no ROCm SDK / no AMD GPU needed; every assertion checks the
  contract path).

## What lands next (rough sequence)

1. **Runtime PR (T7-10b)**: `hipInit` / `hipGetDeviceCount` /
   `hipDeviceGetName` probe; `hipStreamCreate` per state;
   `vmaf_hip_state_init` returns `0` on a real device. The smoke
   contract flips from "`-ENOSYS` everywhere" to "device_count >= 0,
   state_init succeeds when devices >= 1, skip when none".
2. **VIF kernel PR (T7-10c)**: first feature on the HIP compute path
   (same pathfinder choice as Vulkan T5-1b). Bit-exact-vs-CPU
   validation via `/cross-backend-diff`.
3. **ADM + motion + long-tail kernels**: parity with the CPU + CUDA +
   SYCL + Vulkan matrix.
4. **CI ROCm runner** (post-runtime): when GitHub Actions exposes AMD
   GPU runners, replace the build-only matrix row with a real
   smoke-execution row.
5. **`enable_hip` default flip** to `auto` / `true`: only after the
   matrix proves bit-exactness (mirrors the `enable_vulkan` flip
   roadmap in ADR-0175).

## References

- [ADR-0175](0175-vulkan-backend-scaffold.md) — the Vulkan scaffold
  precedent this ADR mirrors. Same audit-first split.
- [ADR-0176](0176-vulkan-vif-cross-backend-gate.md) — the Vulkan T5-1b
  cross-backend gate + state-level API that validated the
  second-GPU-backend pattern, gating T7-10 per the BACKLOG row.
- [ADR-0173](0173-ptq-int8-audit-impl.md) — the same audit-first
  pattern applied to the PTQ harness.
- [Research-0033](../research/0033-hip-applicability.md) — AMD market
  share + ROCm Linux maturity check.
- [`/add-gpu-backend`](../../.claude/skills/add-gpu-backend/SKILL.md) —
  the skill that produced the initial scaffold.
- [BACKLOG T7-10](../../.workingdir2/BACKLOG.md) — backlog row.
- `req` — user direction in T7-10 implementation prompt
  (paraphrased): "HIP (AMD) GPU backend scaffold (audit-first,
  mirrors Vulkan T5-1)".

### Status update 2026-05-09: CI lane enabled

The "Build — Ubuntu HIP" CI lane now installs `rocm-hip-runtime-dev`
from the official AMD apt repo at
`https://repo.radeon.com/rocm/apt/7.2.3` (Ubuntu 24.04 / `noble`)
and compiles + links the runtime PR against `amdhip64`. The lane is
promoted from advisory to required (added to the
`required-aggregator.yml` allow-list). Wall-clock cost: ~3-5 min
extra per HIP-lane run (apt-get + ~200 MB download); acceptable
because HIP is opt-in (`-Denable_hip=true`) and the lane only runs
on its own matrix row. Smoke tests run on the runner without an AMD
GPU — `vmaf_hip_device_count() == 0` short-circuits device-resident
assertions per the contract pinned in `test_hip_smoke.c`. Lane
renamed from "T7-10 scaffold" to "T7-10b runtime" to reflect that
the runtime PR (#499) lands on top of this CI bring-up.

### Status update 2026-05-08: T7-10b runtime landed

The T7-10b runtime PR landed against a host with a working ROCm
7.2.x install and a `gfx1036` AMD GPU visible to `hipGetDeviceCount`.
It flips:

- `libvmaf/src/hip/kernel_template.c` — every helper now wraps a
  real HIP runtime call (`hipStreamCreateWithFlags`,
  `hipEventCreateWithFlags`, `hipMalloc`, `hipHostMalloc`,
  `hipMemsetAsync`, `hipStreamWaitEvent`, `hipStreamSynchronize`,
  `hipStreamDestroy`, `hipEventDestroy`, `hipFree`, `hipHostFree`).
  Rolls back on partial-failure paths.
- `libvmaf/src/hip/common.c` — `vmaf_hip_device_count`,
  `vmaf_hip_state_init`, `vmaf_hip_state_free`, and
  `vmaf_hip_list_devices` now invoke real HIP runtime calls.
  `vmaf_hip_import_state` stays at `-ENOSYS` pending T7-10c (the
  first-feature-kernel PR wires the dispatch hookup).
- `libvmaf/src/hip/meson.build` — flips `dependency('hip-lang')`
  from `required: false` to a hard linkage. Falls back to
  `cc.find_library('amdhip64', dirs: hip_search_paths)` rooted at
  `/opt/rocm/lib` (and `HIP_PATH` if set) because the ROCm 7.x
  package ships no `hip-lang.pc` and the cmake config's clangrt
  expectation breaks under meson's CMake probe. The fallback path
  attaches `-D__HIP_PLATFORM_AMD__=1` and the ROCm include dir as
  a system include.
- `libvmaf/test/test_hip_smoke.c` — the four kernel-template
  helpers + `vmaf_hip_state_init` + `vmaf_hip_list_devices` now
  pin the runtime contract (success on a host with `>=1` HIP
  device, `-ENODEV` when none). Adds a pinned-host → device →
  pinned-host `hipMemcpy` round-trip with a sentinel byte pattern
  to guard the memory-pool path from regression.

The `enable_hip` build option remains default-off and now
hard-requires the ROCm runtime when on. The next planned PR
(T7-10c) lands the first feature kernel (likely VIF, mirroring
the Vulkan T5-1b pathfinder choice) and flips
`vmaf_hip_import_state` from `-ENOSYS` to a real
`VmafContext`-side dispatch hookup.
