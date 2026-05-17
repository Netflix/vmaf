# Research digests

Iteration-time research notes for the lusoris vmaf fork. Each digest
captures **what was investigated and why** for a fork-local
workstream — source links, alternatives weighed, prior art, dead ends.

These are *not* ADRs:

- An [ADR](../adr/) records a *decision* and its alternatives at the
  moment it was made. The body is frozen once Accepted.
- A research digest records the *learning* behind that decision (and
  the iterations that followed). It can be amended as new evidence
  arrives, the same way a lab notebook is.

A typical workstream has one ADR (the decision) and one research
digest (the supporting investigation). Some PRs reuse an existing
digest by linking; that is fine.

## When to write one

Required by [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md) on
every fork-local PR that makes a non-trivial design choice. PRs
without a design choice (e.g., a one-line bug fix in fork-added code)
state "no research digest needed: trivial" in the PR description and
skip the file. Reuse over duplication: if the workstream already has
a digest, link it from the new PR instead of starting a parallel one.

## Format

Each file is named `NNNN-kebab-case-topic.md` with a 4-digit
zero-padded ID assigned in commit order. The structure mirrors
[`0000-template.md`](0000-template.md):

```markdown
# Research-NNNN: <short, descriptive title>

- **Status**: Active | Superseded by Research-MMMM | Archived
- **Workstream**: <ADR-NNNN, ADR-MMMM, ...>
- **Last updated**: YYYY-MM-DD

## Question         — what was the unknown going in
## Sources          — papers, upstream docs, Netflix issues, prior PRs
## Findings         — what was learned, with citations
## Alternatives explored — what didn't work and why
## Open questions   — what is still unknown
## Related          — ADRs, PRs, issues
```

Conventions:

- IDs are assigned in commit order and never reused.
- Digests are *amendable* — update the `Last updated` date when you
  add findings. To replace one entirely, add `Status: Superseded by
  Research-MMMM` and write a new file.
- Cite sources inline with `[link text](URL)` so readers can verify.
- Keep one digest per workstream, not per PR. Cross-link from the PR
  description.

## Index

| ID | Title | Status | Workstream |
| --- | --- | --- | --- |
| [0001](0001-bisect-model-quality-cache.md) | Cache shape for `bisect-model-quality` nightly | Active | [ADR-0109](../adr/0109-nightly-bisect-model-quality.md) |
| [0002](0002-automated-rule-enforcement.md) | Automating process-ADR enforcement (0100 / 0105 / 0106 / 0108) | Active | [ADR-0124](../adr/0124-automated-rule-enforcement.md) |
| [0003](0003-ssimulacra2-port-sourcing.md) | SSIMULACRA 2 port source selection + upstream-drift strategy | Active | [ADR-0126](../adr/0126-ssimulacra2-extractor.md) |
| [0004](0004-vulkan-backend-design.md) | Vulkan compute backend — loader, shader language, allocator, DMABUF import | Active | [ADR-0127](../adr/0127-vulkan-compute-backend.md) |
| [0005](0005-embedded-mcp-transport.md) | Embedded MCP in libvmaf — threading, JSON library, SSE server, Power-of-10 fit | Active | [ADR-0128](../adr/0128-embedded-mcp-in-libvmaf.md) |
| [0006](0006-tinyai-ptq-accuracy-targets.md) | Tiny-AI PTQ int8 — accuracy targets, ORT API comparison, calibration sourcing | Active | [ADR-0129](../adr/0129-tinyai-ptq-quantization.md) |
| [0007](0007-ssimulacra2-scalar-port.md) | SSIMULACRA 2 scalar port — YUV handling, blur deviation, snapshot tooling | Active | [ADR-0126](../adr/0126-ssimulacra2-extractor.md), [ADR-0130](../adr/0130-ssimulacra2-scalar-implementation.md) |
| [0008](0008-ms-ssim-decimate-simd.md) | MS-SSIM decimate SIMD — FLOP accounting, summation order, bit-exactness | Active | [ADR-0125](../adr/0125-ms-ssim-decimate-simd.md) |
| [0010](0010-speed-netflix-upstream-direction.md) | Is Netflix about to ship a SpEED-driven VMAF successor? (informational) | Active | — |
| [0011](0011-iqa-convolve-avx2.md) | `_iqa_convolve` AVX2 — bit-exactness via `__m256d`, kernel invariants, Amdahl | Active | [ADR-0138](../adr/0138-iqa-convolve-avx2-bitexact-double.md) |
| [0012](0012-ssim-simd-bitexact.md) | SSIM SIMD bit-exactness to scalar — where the ULP drifted | Active | [ADR-0139](../adr/0139-ssim-simd-bitexact-double.md) |
| [0013](0013-simd-dx-framework.md) | SIMD DX framework — audit + NEON bit-exactness port | Active | [ADR-0140](../adr/0140-simd-dx-framework.md) |
| [0014](0014-psnr-hvs-neon.md) | `psnr_hvs` NEON sister port — half-wide split strategy, aarch64 gotchas, QEMU verification limits | Active | [ADR-0160](../adr/0160-psnr-hvs-neon-bitexact.md) |
| [0015](0015-ssimulacra2-simd.md) | SSIMULACRA 2 AVX2 + AVX-512 + NEON — per-lane `cbrtf`, left-to-right summation, 2×2 downsample deinterleave | Active | [ADR-0161](../adr/0161-ssimulacra2-simd-bitexact.md) |
| [0016](0016-ssimulacra2-iir-blur-simd.md) | SSIMULACRA 2 IIR blur SIMD — row-batching with gather (horizontal), column-SIMD (vertical), bit-exact to scalar | Active | [ADR-0162](../adr/0162-ssimulacra2-iir-blur-simd.md) |
| [0017](0017-ssimulacra2-ptlr-simd.md) | SSIMULACRA 2 `picture_to_linear_rgb` SIMD — per-lane scalar reads, SIMD matmul, per-lane scalar `powf` | Active | [ADR-0163](../adr/0163-ssimulacra2-ptlr-simd.md) |
| [0018](0018-ssimulacra2-snapshot-gate.md) | SSIMULACRA 2 snapshot-JSON regression gate — why fork self-consistency beats libjxl/Pacidus cross-check at this scope | Active | [ADR-0164](../adr/0164-ssimulacra2-snapshot-gate.md) |
| [0031](0031-intel-ai-pc-applicability.md) | Intel AI-PC NPU + EP applicability to tiny-AI / `dnn/` — verdict: defer NPU; iGPU already covered by OpenVINO EP | Active | — (backlog T7-9) |
| [0046](0046-vmaf-tiny-v3-mlp-medium-evaluation.md) | `vmaf_tiny_v3` (mlp_medium 6→32→16→1, 769 params) vs v2 (mlp_small 257 params): 4-corpus parquet, identical recipe; Netflix LOSO mean PLCC 0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021 (+0.0008 mean, -29 % std). Decision matrix + per-fold table; ship-alongside-v2 recommendation. | Active | [ADR-0241](../adr/0241-vmaf-tiny-v3-mlp-medium.md) |
| [0048](0048-vmaf-tiny-v4-mlp-large-evaluation.md) | `vmaf_tiny_v4` (mlp_large, 3 073 params) — does the architecture ladder saturate? Verdict: yes, +0.0001 mean PLCC vs v3 (below 1 std). Ladder stops at v4. | Active | [ADR-0242](../adr/0242-vmaf-tiny-v4-mlp-large.md) |
| [0053](0054-iqa-convolve-block-widen-attempt.md) | `iqa_convolve` block-of-N tap widen — failed-attempt post-mortem; per-tap widen is load-bearing for bit-exactness, block-of-4 reorder mismatches scalar on 27.67 % of pixels (10 M Monte Carlo) | Active | [ADR-0138](../adr/0138-iqa-convolve-avx2-bitexact-double.md) |
| [0054](0056-vif-ciede-precise-step-a-implementation.md) | `precise` decoration audit on `vif.comp` + `ciede.comp` — Step A of the Vulkan 1.4 bump path. ciede improves 19× (42/48 → 5/48 mismatches at NVIDIA driver 595.71); vif decorated correctly but the 1.4 regression is *not* in the tagged float ops. Step B stays blocked. | Active | [ADR-0269](../adr/0269-vif-ciede-precise-step-a.md), [ADR-0264](../adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md) |
| [0055](0055-ciede-vulkan-nvidia-f32-f64-root-cause.md) | Root-causes the residual 5/48 NVIDIA-Vulkan ciede2000 places=4 mismatch (1.78× threshold, max abs `8.9e-05`) deferred from PR #346. Triangulates double-CPU vs experimental float-CPU vs NVIDIA-Vulkan: f32-CPU matches NVIDIA-GPU to ~6e-7 on the 5 failing high-ΔE frames. Conclusion: structural f32/f64 colour-space-chain precision gap, not a driver fast-math bug. Mitigations rejected; documented as fork debt. | Active | [ADR-0273](../adr/0273-ciede-vulkan-nvidia-f32-f64-precision-gap.md) |
| [0085](0085-vendor-neutral-vvc-encode-landscape.md) | Vendor-neutral VVC (H.266) GPU encode landscape — survey of NVENC (Ada+ silicon only), AMD AMF / Intel QSV (decode-only in 2026), `VK_KHR_video_encode_h266` (unratified), HIP / SYCL ports of VVenC (3–6 eng-month effort), NN-VC tools via ONNXRuntime EPs (vendor-neutral today), ZLUDA (rejected). Cost / risk / value matrix + three-tier rollout recommendation feeding ADR-0315. | Active | [ADR-0315](../adr/0315-vendor-neutral-vvc-encode-strategy.md) |
| [0086](0086-usage-doc-coverage-audit-2026-05-08.md) | Usage-doc coverage audit against ADR-shipped surfaces — 255 ADRs scanned, 46 GOOD / 31 BACKFILL / 178 N/A; identifies 5 highest-leverage gaps (vmaf-tune codec adapters, `--score-backend`, `--cache`, Vulkan image import, HDR + sample-clip) for full backfill in this PR; remaining 26 land as ADR-cited stubs. | Active | [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md), [ADR-0167](../adr/0167-doc-drift-enforcement.md) |
| [0090](0090-phase-a-promotion-audit-2026-05-08.md) | Phase-A-promotion audit (2026-05-08) — repo-wide scan for surfaces still flagged "Phase A only / scaffold-only / Phase B pending" whose follow-up wiring hasn't shipped. 5 production-blocking promotions (HDR not actually wired into `iter_rows`; 15 of 17 codec adapters bypass `ffmpeg_codec_args`; `vmaf-tune fast` has no CLI subcommand; embedded MCP and HIP runtimes still `-ENOSYS`), 12 cosmetic doc-drift items, 9 ADRs ready for `Proposed→Accepted`. Recommended sprint plan + sibling-agent coordination notes. | Active | [ADR-0237](../adr/0237-quality-aware-encode-automation.md), [ADR-0261](../adr/0261-vmaf-tune-hdr-aware.md), [ADR-0276](../adr/0276-vmaf-tune-fast-path.md), [ADR-0209](../adr/0209-embedded-mcp-runtime.md) |
| [0091](0091-partial-integration-audit-2026-05-08.md) | End-to-end integration audit of every shipped libvmaf feature extractor against an 8-rung ladder (CPU → backends → SIMD → corpus → trainer → predictor → docs → tests). 22 extractors inventoried; **0 score 8/8**. Engine rungs (1-3) mostly green; learning rungs (4-6) red across the board because `CORPUS_ROW_KEYS` captures only `vmaf_score` and `ShotFeatures` accepts no libvmaf metric outputs. Surprise findings: `vmaf_fex_ssim` (integer SSIM) is defined but never registered — dead symbol since CPU registration list ships without it. Top-5 promotions ranked by AI-stack ROI. | Active | — |
| [0126](0126-vmaf-tune-hdr-dispatch-coverage-2026-05-15.md) | `vmaf-tune` HDR dispatch coverage — widens the central `hdr_codec_args()` table for AV1 NVENC, HEVC/AV1 QSV, HEVC/AV1 AMF, HEVC VideoToolbox, and libaom while keeping private SEI flags limited to verified families. | Active | [ADR-0300](../adr/0300-vmaf-tune-hdr-aware.md) |
| [0135](0135-chug-extraction-perf-2026-05-16.md) | CHUG/K150K extractor I/O cost breakdown and Win 1 + Win 2 optimisations — per-clip cost audit from perf-audit §6; replaces O(N²) parquet flush with at-end-only write via JSONL staging; adds ffprobe skip from CHUG sidecar geometry; decision matrix for in-memory vs streaming vs DuckDB; projected wall-time savings for 5992-clip CHUG run. | Active | — (perf-audit-pipeline-2026-05-16.md §6) |
| [0136](0136-hdr-ugc-dataset-license-audit-2026-05-15.md) | HDR/UGC dataset license + access audit (2026-05-15) — evaluates 13 candidate corpora from Audit Slice C.7; 6 datasets classified ACTIONABLE-NOW (Beyond8Bits, HDRSDR-VQA, LIVE HDR Database, IPI-MobileHDRVQA, HDR-VDC, CHUG already active); 5 BLOCKED on access or license; 1 BLOCKED on infrastructure. HDRSDR-VQA's 6-display pairwise design surfaces the new panel/display-aware workstream scoped in ADR-0459. | Active | [ADR-0459](../adr/0459-vmaftune-panel-aware-recommendations.md) |

| [0053](0053-post-merge-cpu-profile-2026-05-03.md) | Post-merge CPU profile 2026-05-03 — perf top-10 after PRs #310–#321; surfaces 3 new opt targets (convolve widen, SSIM double reduction, VIF gather elimination) | Active | — |
| [0081](0081-fr-regressor-v2-ensemble-real-corpus-methodology.md) | Real-corpus retrain methodology for the `fr_regressor_v2` deep ensemble — corpus-size sufficiency (9 ref + 70 dis @ `.workingdir2/netflix/`), 9-fold LOSO sizing inherited from the deterministic ADR-0291 baseline, seed-diversity hyperparameters, and the `Seeking_25fps` weak-fold diagnostic for HOLD-on-spread cases. | Active | [ADR-0309](../adr/0309-fr-regressor-v2-ensemble-real-corpus-retrain.md) |
| [0089](0089-vulkan-vif-fp-residual-bisect-2026-05-08.md) | CPU `double` vs Vulkan `float` stage bisect on the residual NVIDIA-Vulkan `integer_vif_scale2` 45/48-frame `places=4` mismatch at API 1.4 (T-VK-VIF-1.4-RESIDUAL). Static SPIR-V re-verification confirms only 5 FP-arithmetic ops in `vif.comp` and all 5 are `NoContraction`-decorated post-PR #346 — SPIR-V mitigation surface is exhausted. SYCL counter-example (same f32 contract, passes the gate) rules out a pure f32-vs-f64 class issue. Localised root cause: NVIDIA `shaderFloatControls2`-v2 codegen flip at API 1.4 on a non-IEEE-bound default (reciprocal-multiply, fast-rsq) outside the SPIR-V declarable surface. Phase-2 shader fix not warranted; recommends per-stage NVIDIA dynamic dump or `places=3` override ADR. | Active | [ADR-0264](../adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md), [ADR-0269](../adr/0269-vif-ciede-precise-step-a.md) |
| [0090](0090-upstream-skip-bucket-triage-2026-05-08.md) | Per-commit triage of the 41 upstream commits binned `SKIP-doc-or-format` by the `/sync-upstream` Pass-2 heuristic on 2026-05-08. Splits into 5 PORT_NOW (motion_v2 mirroring bugfix, motion_v2 option cluster + `prev_prev_ref` API, two cambi internals), 18 PORT_LATER (python/test MyTestCase migration, blocked on agent-E worktree), 4 DEFER_INDEFINITELY, 1 PORTED_SILENTLY (`662fb9ce` semaphores → fork commit `e5a52e74`), 12 MERGE_BOUNDARY. Surfaces the riskiest item: the python/test mass-port is +5 600 LOC and crosses Netflix-golden assertions in `feature_extractor_test.py`. | Active | — (companion to Research-0089) |
| [0135](0135-vulkan-dispatch-overhead-2026-05-15.md) | Vulkan dispatch overhead characterization — T7-18: startup dominated by uncached `vkCreateComputePipelines`; per-frame fence/submit overhead ruled out; pipeline-cache fix recommended | Active | T7-18 |
| [0091](0091-cambi-cuda-integration.md) | CAMBI CUDA integration trade-offs (T3-15a): per-thread 49-read vs shared-memory SAT for the spatial-mask kernel; synchronous vs async ring-buffer DtoH for the 5-scale pipeline; `host_pinned` slot reuse for score storage; two compile-time bugs found and fixed (`cuMemcpyDtoH` arg order, `VMAF_FEATURE_DISPATCH_SEQUENTIAL` non-existence). Predecessor: Research-0032 (Vulkan twin). | Active | [ADR-0360](../adr/0360-cambi-cuda.md) |
<!-- Backfill entries for older workstreams land here as their authors
     revisit the corresponding code. -->

*(Index seeded by [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md)'s
adoption PR; backfilled digests for the existing major workstreams
will be added as their authors revisit the corresponding code.)*
| [0135](0135-cambi-cuda-smem-tile-2026-05-16.md) | CAMBI CUDA spatial-mask SLM tile -- design analysis: img-tile correctness bug, 26x read reduction via direct zd_tile load, bank-conflict accepted at uint8 row access | Active | [ADR-0464](../adr/0464-cambi-cuda-smem-tile.md) |
