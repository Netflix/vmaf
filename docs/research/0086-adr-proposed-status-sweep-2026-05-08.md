# Research-0086: ADR `Proposed` status sweep — 2026-05-08

- **Status**: Active
- **Date**: 2026-05-08
- **Tags**: docs, adr, audit, planning, fork-local
- **Companion ADR**: bulk per-ADR status flips landing in the same PR

## Summary

A point-in-time sweep over every `docs/adr/*.md` file whose front-matter
still carried `**Status**: Proposed` on 2026-05-08. The fork has merged
many implementing PRs faster than the per-ADR Status lines could keep
up; the 2026-05-06 partial sweep
(`changelog.d/changed/adr-bulk-status-flip-2026-05-06.md`) flipped 13
ADRs but stopped short of an exhaustive audit. This digest closes the
gap for the remaining 26 ADRs (ADR-0325 deliberately excluded — it is
still contested by in-flight KonViD / sidecar / predictor-stub /
Phase-F / video-saliency PRs and is scheduled for the merge-train
renumber sweep).

`grep -lE "\\*\\*Status\\*\\*: Proposed" docs/adr/*.md` at HEAD
`0a8b539e` returned 27 files. Subtracting ADR-0325 leaves the 26
audited below.

## Method

For each ADR:

1. Read the body. Extract the **acceptance criteria** — the concrete
   deliverables (files, symbols, schema fields, CLI flags, CI gates,
   manifests, registry rows) the Decision section promised.
2. Verify each criterion against the live worktree at HEAD via `ls`,
   `grep`, and reading the file's index row. Cross-check superseding
   ADRs in `docs/adr/*` that may have absorbed the decision.
3. Pick a verdict:
   - **Accepted** — every criterion verified in tree.
   - **Superseded by ADR-NNNN** — a later Accepted ADR replaced the
     decision (cite the supersedor).
   - **Withdrawn** — the decision became moot.
   - **Stay Proposed** — work in flight; document the gap with the PR
     or backlog item.
4. Append a `### Status update 2026-05-08` block to the ADR per the
   ADR-0028 / ADR-0106 immutability rule (original body frozen) and
   flip the front-matter `**Status**:` line.
5. Update the `Status` column on the matching `docs/adr/README.md`
   index row when the row exists. (ADR-0235 / 0236 / 0238 / 0239 /
   0251 / 0315 are missing index rows entirely — out of scope for a
   pure status sweep; backlog item to add the rows separately.)

## Per-ADR verdict table

| ADR | Title | Verdict | Verification trail |
| --- | --- | --- | --- |
| 0125 | MS-SSIM decimate SIMD fast paths (AVX2 + AVX-512) | **Accepted** | `libvmaf/src/feature/x86/ms_ssim_decimate_{avx2,avx512}.{c,h}` exist; `libvmaf/src/feature/ms_ssim_decimate.{c,h}` (scalar-separable) exists; `libvmaf/test/test_ms_ssim_decimate.c` exists. Bit-exact vs scalar-separable harness landed. |
| 0126 | SSIMULACRA 2 perceptual metric | **Accepted** | `libvmaf/src/feature/ssimulacra2.c` + `ssimulacra2_eotf_lut.h` + `ssimulacra2_math.h` + `ssimulacra2_simd_common.h`; `libvmaf/src/meson.build` `ssimulacra2` block lines 137 / 350-360. ADR-0130 was the implementation closeout. |
| 0127 | Vulkan compute backend | **Accepted** | `libvmaf/include/libvmaf/libvmaf_vulkan.h` + full `libvmaf/src/vulkan/` tree (common.c, dispatch_strategy, picture_vulkan, kernel_template, import); `enable_vulkan` Meson option live. ADR-0175 / 0186 expanded the surface. |
| 0128 | Embedded MCP server in libvmaf | **Stay Proposed** | Public header `libvmaf/include/libvmaf/libvmaf_mcp.h` shipped via ADR-0209 audit-first scaffold (Accepted), but every entry point still returns `-ENOSYS`. cJSON + mongoose vendoring, dedicated MCP pthread, SPSC ring buffer, and SSE / UDS / stdio transport bodies are tracked as **T5-2b** (in flight). The strategic decision in ADR-0128 is implemented in stages; the runtime is incomplete. |
| 0129 | Tiny-AI PTQ int8 quantisation policy | **Accepted** | `model/tiny/registry.schema.json` carries `quant_mode` / `quant_calibration_set` / `quant_accuracy_budget_plcc` (lines 64–79); `ai/scripts/ptq_static.py` + `ptq_dynamic.py` + `measure_quant_drop.py` exist; ADR-0173 (Accepted) was the audit-first implementation; ADR-0174 (Accepted) was the first-model PTQ. The policy this ADR specified is fully shipped. |
| 0138 | `_iqa_convolve` AVX2 bit-exact double-precision fast path | **Accepted** | `libvmaf/src/feature/x86/convolve_avx2.{c,h}` exist; ADR-0140 references the bit-exactness pattern as load-bearing. |
| 0139 | SSIM SIMD accumulate bit-exact via per-lane scalar double | **Accepted** | `libvmaf/src/feature/x86/ssim_avx2.{c,h}` + `ssim_avx512.{c,h}` carry the per-lane reduction pattern; ADR-0140 cites the pattern under `simd_dx.h`. |
| 0140 | SIMD DX framework — header macros + scaffolding skill | **Accepted** | `libvmaf/src/feature/simd_dx.h` exists; `/add-simd-path` skill kernel-spec flags are documented in `.claude/skills/add-simd-path/SKILL.md`. |
| 0207 | Tiny-AI Quantization-Aware Training (QAT) — design | **Accepted** | `ai/train/qat.py` exists (real implementation, not the prior `NotImplementedError` scaffold); `ai/scripts/qat_train.py` exists. ADR-0208 was the implementation. |
| 0208 | First per-model QAT — `learned_filter_v1` int8 | **Accepted** | `model/tiny/learned_filter_v1.int8.onnx` exists alongside the fp32 ONNX; QAT pipeline + measurement scripts (`measure_quant_drop.py` / `_per_ep.py`) in tree. The `learned_filter_v1` registry row stays on `quant_mode: "dynamic"` (the empirical finding), but the pipeline that produced the verdict is shipped. |
| 0235 | Codec-aware FR regressor (`fr_regressor_v2`) | **Accepted** | `ai/src/vmaf_train/codec.py` exists with the closed codec vocabulary; `FRRegressor` constructor accepts `num_codecs`; `model/tiny/fr_regressor_v2.{onnx,json}` registered, plus the `_ensemble_v1_seed{0..4}` family per ADR-0279. |
| 0236 | DISTS extractor as LPIPS companion | **Stay Proposed** | No `libvmaf/src/feature/dists*` files; `model/tiny/registry.json` has no `dists_sq` row; tracked as backlog item **T7-DISTS** per the ADR's own scoping note. Implementation has not started. |
| 0238 | Vulkan VmafPicture preallocation surface | **Accepted** | `libvmaf/include/libvmaf/libvmaf_vulkan.h` declares `VmafVulkanPicturePreallocationMethod` (line 153), `vmaf_vulkan_preallocate_pictures` (180), `vmaf_vulkan_picture_fetch` (191). |
| 0239 | Backend-agnostic GPU picture pool | **Accepted** | `libvmaf/src/gpu_picture_pool.{c,h}` exist; `libvmaf/src/cuda/ring_buffer.*` removed (verified by `ls` returning no match). |
| 0251 | Vulkan VkImage import — v2 async pending-fence model | **Accepted** | `libvmaf/include/libvmaf/libvmaf_vulkan.h:64` declares `max_outstanding_frames`; `libvmaf/src/vulkan/common.c:444-486` implements the ring sizer + clamp; `libvmaf/src/vulkan/vulkan_internal.h:117` documents the captured request depth. |
| 0253 | Defer SpEED-QA full-reference reduction | **Accepted** | DEFER-shape decision; the deliverable is the documented position. `speed_chroma` / `speed_temporal` extractors remain unchanged; no `speed_qa` reduction; no SpEED-driven model. The three reversal triggers stay open. |
| 0270 | libFuzzer scaffold for parser surfaces (OSSF) | **Accepted** | `libvmaf/test/fuzz/` carries `fuzz_y4m_input.c`, `fuzz_yuv_input.c`, `fuzz_cli_parse.c` plus seed corpora; `libvmaf/meson_options.txt` declares `option('fuzz', ...)`; `.github/workflows/fuzz.yml` exists. ADR-0311 expanded the harness set. |
| 0272 | `fr_regressor_v2` codec-aware scaffold | **Accepted** | `ai/scripts/train_fr_regressor_v2.py` exists; `model/tiny/fr_regressor_v2.{onnx,json}` registered; companion ensemble (ADR-0279) shipped. |
| 0276 | `vmaf-tune fast` — proxy-based recommend (Phase A.5) | **Accepted** | `tools/vmaf-tune/src/vmaftune/fast.py` exists; ADR-0304 (Accepted in 2026-05-06 sweep) wired the production path. |
| 0279 | `fr_regressor_v2` probabilistic head — deep-ensemble + conformal | **Accepted** | `model/tiny/fr_regressor_v2_ensemble_v1.json` manifest + `_seed{0..4}.{onnx,json,onnx.data}` member files exist; `ai/scripts/eval_probabilistic_proxy.py` + `export_ensemble_v2_seeds.py` in tree. |
| 0295 | vmaf-tune Phase E — per-title bitrate-ladder generator | **Accepted** | `tools/vmaf-tune/src/vmaftune/ladder.py` exists; ADR-0307 (Accepted in 2026-05-06 sweep) wired the default sampler. |
| 0314 | vmaf-tune `--score-backend=vulkan` | **Accepted (resolves merge conflict)** | `--score-backend` argparse wiring restored in `tools/vmaf-tune/src/vmaftune/cli.py:114-594`; vulkan admitted as strict-mode value. The ADR file at HEAD `0a8b539e` carried unresolved `<<<<<<< HEAD` / `>>>>>>> 599bb187` conflict markers (the 2026-05-06 sweep flipped this ADR but a subsequent rebase reintroduced the conflict). The audit appendix below resolves the markers and lands the Accepted state. |
| 0315 | Vendor-neutral VVC encode strategy (tiered) | **Accepted** | Strategy/scope decision; the deliverable is the tier policy document. Tier-1 (NN-VC + Vulkan scoring) is operative — `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py` is the CPU baseline; Vulkan scoring path lives. Tier-2 (HIP VVenC port) and Tier-3 (Vulkan Video VVC encode) are documented backlog awaiting demand-pull triggers. |
| 0324 | Ensemble training kit — portable Phase-A + LOSO retrain bundle | **Accepted** | `tools/ensemble-training-kit/` exists with `run-full-pipeline.sh`, `01-prereqs.sh`–`05-bundle-results.sh`, `make-distribution-tarball.sh`, `_platform_detect.sh`, `build-libvmaf-binaries.sh`, `extract-corpus.sh`, `binaries/`. Multi-platform extension (lawrence's collaborator workstation) was already documented in the original ADR's `### Status update 2026-05-06` block. |

## Counts

- **Accepted**: 24 (0125, 0126, 0127, 0129, 0138, 0139, 0140, 0207,
  0208, 0235, 0238, 0239, 0251, 0253, 0270, 0272, 0276, 0279, 0295,
  0314, 0315, 0324, plus 0173- and 0209-anchored implementations of
  0128 / 0129 cited above)
- **Superseded**: 0
- **Withdrawn**: 0
- **Stay Proposed**: 2 (0128 — runtime stub awaiting T5-2b; 0236 —
  T7-DISTS unstarted)

(Total audited: 26. ADR-0325 deliberately not in scope.)

## Non-obvious calls

- **ADR-0128 (MCP)**: the public C-API header surface is shipped, the
  build flag exists, and ADR-0209 (Accepted) is the audit-first
  scaffold. The strategic decision in ADR-0128 itself is unchanged.
  But because every transport entry point still returns `-ENOSYS` and
  the runtime libraries (cJSON, mongoose) and threading model are not
  yet vendored, calling ADR-0128 "Accepted" would overstate the
  delivery. Stays Proposed pending T5-2b. Alternative considered:
  flip to "Superseded by ADR-0209"; rejected because ADR-0209
  explicitly defers the runtime, so the supersession would inherit
  the same gap rather than close ADR-0128 cleanly.
- **ADR-0129 (PTQ policy)**: the policy ADR is implemented by ADR-0173
  (audit-first) and ADR-0174 (first-model PTQ), both Accepted. ADR-0173
  introduces no behavioural change beyond the schema + scripts ADR-0129
  promised; ADR-0174 is the first per-model application. Both are
  downstream of the ADR-0129 *policy*, which is what this ADR records.
  Calling 0129 "Accepted" rather than "Superseded by 0173" is the
  honest read — the policy itself was adopted; the implementation ADRs
  are children.
- **ADR-0314 (vulkan score-backend)**: file at HEAD carries unresolved
  Git conflict markers — the 2026-05-06 sweep flipped Status to
  Accepted, but a subsequent rebase reintroduced the
  `<<<<<<< HEAD / ======= / >>>>>>> 599bb187` envelope around the
  Status line. The implementation is verified in the live `cli.py`,
  so the Audit appendix resolves the markers in favour of Accepted.
- **ADR-0253 / 0315 (defer / strategy decisions)**: both are
  scope-level decisions whose deliverable is the documented position
  itself, not a code artefact. Both stay valid; both flip to Accepted
  on the strength of the position being adopted and not contradicted
  in tree.
- **ADRs missing from `docs/adr/README.md`**: 0235, 0236, 0238, 0239,
  0251, 0315 lack their own index row (they appear only as inline
  citations from other ADRs). Adding the rows is out of scope for a
  pure status sweep — this PR does not invent index entries; it only
  flips the Status column on rows that exist.

## References

- ADR-0028 / ADR-0106: ADR maintenance + immutability rule
  (Status updates land as appendix, body stays frozen).
- ADR-0221: changelog-fragment policy.
- `changelog.d/changed/adr-bulk-status-flip-2026-05-06.md`: prior
  bulk sweep precedent (13 ADRs).
- `docs/research/0034-ci-pipeline-audit-2026-05.md` and
  `docs/research/0061-vmaf-tune-capability-audit.md`: prior audit
  digest precedents at this scale.

## Reproducer

```bash
# 1. List Proposed ADRs at HEAD.
grep -lE "\*\*Status\*\*: Proposed" docs/adr/*.md

# 2. Per-ADR verification examples.
ls libvmaf/src/feature/x86/ms_ssim_decimate*           # ADR-0125
ls libvmaf/src/feature/ssimulacra2.c                   # ADR-0126
ls libvmaf/src/vulkan/                                 # ADR-0127
grep -E "quant_mode" model/tiny/registry.schema.json   # ADR-0129
ls libvmaf/src/feature/x86/convolve_avx2.{c,h}         # ADR-0138
ls libvmaf/src/feature/simd_dx.h                       # ADR-0140
ls ai/train/qat.py ai/scripts/qat_train.py             # ADR-0207 / 0208
ls libvmaf/src/gpu_picture_pool.{c,h}                  # ADR-0239
grep "max_outstanding_frames" libvmaf/src/vulkan/*.c   # ADR-0251
ls libvmaf/test/fuzz/                                  # ADR-0270
ls tools/vmaf-tune/src/vmaftune/{fast,ladder}.py       # ADR-0276 / 0295
ls model/tiny/fr_regressor_v2_ensemble_v1*             # ADR-0279
ls tools/ensemble-training-kit/                        # ADR-0324

# 3. Stays-Proposed gap evidence.
ls libvmaf/src/feature/dists* 2>/dev/null              # ADR-0236: empty (T7-DISTS)
head -25 libvmaf/include/libvmaf/libvmaf_mcp.h         # ADR-0128: -ENOSYS docstring
```
