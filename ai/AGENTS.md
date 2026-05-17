# AGENTS.md — ai/

Orientation for agents working on the tiny-AI **training** side. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Python package for training, exporting, and registering tiny-AI
checkpoints that are then consumed by [libvmaf/src/dnn/](../libvmaf/src/dnn/AGENTS.md)
at runtime. Stack: PyTorch + Lightning → ONNX.

```text
ai/
  pyproject.toml   # package metadata (training-only deps)
  src/             # vmaf-train CLI + model defs + dataset loaders
  tests/           # pytest unit tests
  configs/         # dataset manifests + training recipes
  lpips_export.py  # re-export richzhang/PerceptualSimilarity → ONNX
```

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Boundary is `.onnx` + sidecar JSON on disk.** Training lives here,
  runtime lives in `libvmaf/src/dnn/`, and the two communicate only through
  files in `model/tiny/`. No imports cross this boundary.
- **Every shipped `.onnx` has a registry entry** in
  [`../model/tiny/registry.json`](../model/tiny/) with sha256, upstream
  source, license, and opset. See
  [ADR-0039](../docs/adr/0039-onnx-runtime-op-walk-registry.md).
- **ONNX opset**: export requests opset 17 but torch dynamo may emit 18
  (downconvert sometimes fails in `onnx.version_converter`). Record the
  emitted opset in the registry sidecar rather than failing the export.
- **ImageNet normalisation lives in the graph**, not in the C helper — for
  any ImageNet-family model, absorb the inverse ImageNet transform into
  the exported graph so the C side uses the shared
  `vmaf_tensor_from_rgb_imagenet()` helper unchanged. See
  [ADR-0041](../docs/adr/0041-lpips-sq-extractor.md).
- **Roundtrip-validate** every export against `onnxruntime` to atol=1e-5
  before committing. See [ADR-0021](../docs/adr/0021-training-stack-pytorch-lightning.md).
- **Docs**: every new model or training recipe ships a page under
  `docs/ai/` in the same PR. See
  [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md).
- **Bisect-cache fixture is content-stable** — `ai/testdata/bisect/`
  is the deterministic default for the nightly `bisect-model-quality`
  workflow. Regenerate the committed synthetic cache via
  `python ai/scripts/build_bisect_cache.py` with seeds
  `FEATURE_SEED=20260418` / `MODEL_SEED=20260419`. The same script can
  materialise a real DMOS/MOS-aligned parquet via
  `--source-features` + optional `--target-column`; that path must
  preserve the canonical-six feature order and still normalise the
  output target column to `mos`. CI runs the script with `--check`. As
  of ADR-0262 the parquet leg of the check uses logical
  `pyarrow.Table.equals` content comparison (schema + row count +
  values), tolerating writer-version-string drift in the `created_by`
  parquet header — but ONNX still compares byte-for-byte via
  `filecmp.cmp(shallow=False)`, which means ONNX-side determinism must
  stay intact. **Do not** remove the
  `model.producer_name = "vmaf-train.bisect-cache"`,
  `model.producer_version = "1"`, or `model.ir_version = 9` pins in
  `_save_linear_fr`: those three lines are what stabilises ONNX bytes
  across `onnx` minor versions. See
  [ADR-0262](../docs/adr/0262-bisect-cache-logical-comparison.md) +
  [ADR-0109](../docs/adr/0109-nightly-bisect-model-quality.md) +
  [Research-0001](../docs/research/0001-bisect-model-quality-cache.md).

## Governing ADRs

- [ADR-0020](../docs/adr/0020-tinyai-four-capabilities.md) — four capabilities (C1–C4).
- [ADR-0021](../docs/adr/0021-training-stack-pytorch-lightning.md) — PyTorch + Lightning training stack.
- [ADR-0023](../docs/adr/0023-tinyai-user-surfaces.md) — `vmaf-train` CLI as one of four surfaces.
- [ADR-0036](../docs/adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope (LPIPS, MobileSal, TransNet V2, …).
- [ADR-0039](../docs/adr/0039-onnx-runtime-op-walk-registry.md) — runtime op-allowlist + registry schema.
- [ADR-0041](../docs/adr/0041-lpips-sq-extractor.md) — LPIPS export pattern (ImageNet-in-graph).
- [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md) — doc-substance rule.
- [ADR-0218](../docs/adr/0218-mobilesal-saliency-extractor.md) — MobileSal saliency extractor (T6-2a) ships a smoke-only synthetic ONNX placeholder under `model/tiny/mobilesal.onnx`; the C extractor binds tensors by name (`input` → `saliency_map`) so a real upstream MobileSal export drops in without C changes. Saliency-weighted FR features and the `tools/vmaf-roi` CTU sidecar are the T6-2b follow-up — do not bundle them into the T6-2a surface.
- [ADR-0286](../docs/adr/0286-saliency-student-fork-trained-on-duts.md) — `saliency_student_v1` is the fork-trained tiny U-Net that replaces `mobilesal_placeholder_v0` as the production weights for the `mobilesal` extractor. Tensor-name contract (`input`, `saliency_map`) and NCHW shapes are unchanged from ADR-0218, so any future weights swap (multi-dataset student, distilled u2netp, …) drops in without C changes. v1's decoder uses `ConvTranspose` for stride-2 upsampling because `Resize` was not on the allowlist at v1's training time; that constraint was lifted by [ADR-0258](../docs/adr/0258-onnx-allowlist-resize.md). DUTS-TR images are *not* committed in-tree; only the trained `.onnx` + sidecar are. Trainer at [`scripts/train_saliency_student.py`](scripts/train_saliency_student.py); reproducer in [`docs/ai/models/saliency_student_v1.md`](../docs/ai/models/saliency_student_v1.md).
- [ADR-0332](../docs/adr/0332-saliency-student-v2-resize-decoder.md) — `saliency_student_v2` is the Resize-decoder ablation on the v1 recipe. The decoder upsampler swaps to `F.interpolate(scale=2, bilinear, align_corners=False)` + `nn.Conv2d(k=3)` (ONNX `Resize` mode=`linear`, `coordinate_transformation_mode=half_pixel` per ADR-0258); every other architectural decision is held identical to v1 so the ablation is single-variable. v2 ships as a parallel artefact under `model/tiny/saliency_student_v2.onnx` — **v1 stays as the production weights for the `mobilesal` extractor** until a follow-up PR validates v2 in real ROI encodes. The v1 trainer (`train_saliency_student.py`) and v2 trainer (`train_saliency_student_v2.py`) MUST stay byte-identical outside the `_ResizeConv` / `nn.ConvTranspose2d` swap and the model-class name; any v1 recipe change that diverges from v2 (or vice versa) destroys the clean-ablation property. Future `saliency_student_v3` (multi-dataset / larger student per ADR-0286 backlog) is a new ADR, not a fork of v2.
- [ADR-0396](../docs/adr/0396-video-saliency-extension.md) — video-saliency follow-ups are evaluated at encoder block granularity before model promotion. `ai/scripts/eval_saliency_per_mb.py` is the measurement harness: it pairs predicted/ground-truth masks by stem, reduces each mask to fixed block means, thresholds blocks, and reports macro/micro IoU. Keep this script dependency-light (`numpy` only; `.npy` + PGM masks) so it remains usable in training sandboxes without image I/O stacks.
- [ADR-0109](../docs/adr/0109-nightly-bisect-model-quality.md) — nightly bisect workflow + synthetic placeholder cache.
- [ADR-0235](../docs/adr/0235-codec-aware-fr-regressor.md) — codec-aware FR regressor (`fr_regressor_v2`). `CODEC_VOCAB` in [`src/vmaf_train/codec.py`](src/vmaf_train/codec.py) is **closed and order-stable** — the index of each codec is the one-hot column index baked into trained ONNX. Adding a codec appends to the tuple and bumps `CODEC_VOCAB_VERSION`; reordering silently invalidates every shipped `fr_regressor_v2_*.onnx`. `FRRegressor(num_codecs=0)` must remain the v1 single-input contract — flipping the default would break every existing `model/tiny/fr_regressor_v1.onnx` consumer. Feature-dump scripts emit a `codec` column tagged at the call site (BVI-DVC: `"x264"`, Netflix Public: `"unknown"`); never silently default to a codec that doesn't match what the script actually encoded.
- [ADR-0305](../docs/adr/0305-encoder-knob-space-pareto-analysis.md) — **knob-sweep corpus invariant.** The 12,636-cell sweep at `runs/phase_a/full_grid/comprehensive.jsonl` (gitignored, locally generated) is the source of truth for `tools/vmaf-tune/codec_adapters/*` recipe defaults. Pareto frontiers are stratified per `(source, codec, rc_mode)` slice — never collapsed to a global hull (companion [Research-0063](../docs/research/0063-encoder-knob-space-cq-vs-vbr-stratification.md) shows the global-hull failure mode regresses NVENC h264/hevc by ~4 VMAF at cq=30). **Recipes that regress vs the bare encoder at matched bitrate within the same slice MUST NOT ship as adapter defaults.** The regression-detection check lives in `ai/scripts/analyze_knob_sweep.py` (`detect_recipe_regressions(...)`) and is exercised by `ai/tests/test_knob_sweep_analysis.py::test_recipe_regression_detection`; new codec adapter PRs cite the per-(codec, rc_mode) hull row from `reports/summary.md` (or "no hull entry yet — bare default") in their PR description. Methodology + scaffolded findings: [Research-0077](../docs/research/0077-encoder-knob-space-pareto-frontiers.md).

## Netflix-corpus training prep (ADR-0242 / ADR-0203)

The top-level [`ai/data/`](data/) and [`ai/train/`](train/) packages
(distinct from the `vmaf_train` package under `src/`) host the
runnable Netflix-corpus prep stack:

- [`ai/data/netflix_loader.py`](data/netflix_loader.py) — pair distorted
  YUVs with their ref by parsing the Netflix ladder filename
  convention. `iter_pairs(data_root, *, sources=, max_pairs=,
  assume_dims=)` is the only public surface.
- [`ai/data/feature_extractor.py`](data/feature_extractor.py) — wraps
  the libvmaf CLI in JSON mode. Defaults to `build/tools/vmaf`; honours
  `$VMAF_BIN`. Raises `RuntimeError` with explicit build instructions
  on missing binary.
- [`ai/data/scores.py`](data/scores.py) — `vmaf_v0.6.1` distillation
  scores (per-frame + pooled). Honours `$VMAF_MODEL_PATH`.
- [`ai/train/dataset.py`](train/dataset.py) — `NetflixFrameDataset`
  with explicit `payload_provider=` + `assume_dims=` injection points
  for unit tests.
- [`ai/train/eval.py`](train/eval.py) — PLCC / SROCC / KROCC / RMSE +
  latency. Either `onnx_path=` or `predictions=` (exactly one).
- [`ai/train/train.py`](train/train.py) — CLI entry point. Runs
  standalone (`python ai/train/train.py …`) or as a module
  (`python -m ai.train.train`); both forms work because the script
  fixes `sys.path` when `__package__` is empty.

**Rebase-sensitive invariants** (track when upstream Netflix/vmaf adds
its own training surface):

- The `iter_pairs` filename regex is fork-specific. If upstream adds a
  loader with a different ladder convention, do NOT merge them — keep
  ours under `ai/data/` and theirs under whatever path they pick.
- The per-clip JSON cache schema (`{features:{feature_names,
  per_frame, n_frames}, scores:{per_frame, pooled}}`) is consumed by
  both the dataset and any downstream consumer. Bumping the schema
  must invalidate `$VMAF_TINY_AI_CACHE` (or version-tag the path).
- The smoke command `python ai/train/train.py --epochs 0
  --assume-dims 16x16` MUST stay runnable without a built `vmaf`
  binary — the `_make_zero_payload` helper in `ai.train.dataset`
  injects a fake payload so CI gates don't drag a libvmaf build into
  the Python test surface.
- **`vmaf_tiny_v2` ONNX contract (ADR-0244).** The shipped ONNX
  embeds the StandardScaler `(mean, std)` as Constant `Sub` + `Div`
  nodes that run before the MLP. The runtime feeds raw canonical-6
  feature values; do NOT add an external scaler step. Re-exporting
  via [`ai/scripts/export_vmaf_tiny_v2.py`](scripts/export_vmaf_tiny_v2.py)
  is the only supported path — it pulls `mean` / `std` from the
  trainer checkpoint and bakes them as graph initialisers, so the
  `model/tiny/registry.json` sha256 covers the calibration values
  too. Input name is `features` ([N, 6] float32), output `vmaf`
  ([N] float32); feature column order is fixed at
  `(adm2, vif_scale0, vif_scale1, vif_scale2, vif_scale3, motion2)`
  and must not be reordered without a full Phase-3 re-validation.
- **`vmaf_tiny_v3` ships alongside v2 (ADR-0241).** Same ONNX
  contract as v2 (input `features [N, 6]` float32, output
  `vmaf [N]` float32, opset 17, scaler-baked-into-graph) — only the
  architecture differs (`mlp_medium` 6 → 32 → 16 → 1, 769 params vs
  v2's `mlp_small` 257). **Production default stays v2**;
  [`docs/ai/inference.md`](../docs/ai/inference.md) and the model-card
  cross-references both keep v2 as the recommended `--tiny-model`.
  v3 is the higher-PLCC / lower-variance option (Netflix LOSO mean
  PLCC 0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021). Do NOT replace v2
  with v3 wholesale — both file paths are referenced by name in
  user-facing docs and the registry, and the small mean delta does
  not justify a default flip without multi-seed + KoNViD 5-fold
  parity (documented as Phase-3e follow-up). Same scripts pattern:
  `train_vmaf_tiny_v3.py` / `export_vmaf_tiny_v3.py` /
  `validate_vmaf_tiny_v3.py` / `eval_loso_vmaf_tiny_v3.py` —
  do **not** modify the v2 scripts when iterating on v3.
- **`vmaf_tiny_v3` and `vmaf_tiny_v4` opt-in tiers
  (ADR-0241 / ADR-0242).** v3 (`mlp_medium`, 769 params, ADR-0241)
  and v4 (`mlp_large`, 3 073 params, ADR-0242) ship *alongside* v2,
  not as replacements. Production default stays `vmaf_tiny_v2`. The
  three rungs share the canonical-6 input contract, the bundled
  StandardScaler, and the 90 ep / Adam@1e-3 / MSE / bs=256 recipe;
  only the architecture differs. **Do NOT modify v2 or v3 scripts
  when iterating on later rungs** — each version owns its own
  `train_vmaf_tiny_vN.py` / `export_vmaf_tiny_vN.py` /
  `validate_vmaf_tiny_vN.py` / `eval_loso_vmaf_tiny_vN.py` quartet.
  The arch ladder **stops at v4**: the v3 → v4 LOSO PLCC delta is
  +0.0001 (well below 1 std), demonstrating saturation on the
  canonical-6 + 4-corpus regime. Future quality gains require
  regime change (richer features, larger corpus, ensembles), not
  deeper MLPs. See ADR-0242 § Alternatives considered for the
  mlp_huge rejection rationale.
- **Hardware-capability priors are prior-only (ADR-0335).**
  [`ai/data/hardware_caps.csv`](data/hardware_caps.csv) +
  [`ai/scripts/hardware_caps_loader.py`](scripts/hardware_caps_loader.py)
  ship per-architecture GPU encode-block fingerprints (codecs
  supported, max resolution, encoding-block count, tensor /
  NPU flags, driver floor) sourced exclusively from primary
  vendor docs. The loader's schema rejects benchmark-shaped
  columns (`fps_*`, `throughput`, `mbps`, `latency`, `watts`,
  `tdp`, `score_*`, `vmaf_*`), community-wiki source URLs
  (`wikipedia.org`, `wikichip.org`), empty fields, and zero
  encoding-block rows. Adding throughput / quality numbers to
  this surface is forbidden by ADR-0335 and the companion
  research digest's category-1 NO-GO finding — performance
  signal must come from the corpus's own measured rows, not
  from a static prior table. Schema extensions (new
  capability columns) require a new ADR, not a silent column
  bump.

## `fr_regressor_v1` (C1 baseline — ADR-0249)

The Wave-1 C1 baseline trainer is
[`ai/scripts/train_fr_regressor.py`](scripts/train_fr_regressor.py). It
consumes `runs/full_features_netflix.parquet` (produced by
`ai/scripts/extract_full_features.py` over the local Netflix Public
drop at `.workingdir2/netflix/`), runs 9-fold leave-one-source-out
(LOSO), and exports `model/tiny/fr_regressor_v1.onnx` only when mean
LOSO PLCC ≥ 0.95 against the `vmaf_v0.6.1` per-frame teacher.

**Contract row** (do not regress without an ADR amendment):

- **Input** — `[N, 6]` float32, feature order
  `(adm2, vif_scale0, vif_scale1, vif_scale2, vif_scale3, motion2)`,
  standardised with the per-feature `feature_mean` / `feature_std`
  vectors pinned in the sidecar JSON. Standardisation is **not**
  baked into the ONNX so callers can swap feature pools without
  re-export.
- **Output** — `[N]` float32, VMAF-scale (0–100 typical).
- **Architecture** — stock `vmaf_train.models.FRRegressor` with the
  Wave-1 spec hparams (hidden=64, depth=2, dropout=0.1, GELU). Larger
  / smaller variants must register a new model id, not overwrite this
  one.
- **Ship gate** — mean LOSO PLCC ≥ 0.95 vs `vmaf_v0.6.1`. The trainer
  exits 3 and refuses to overwrite the registry on failure; lowering
  the threshold is a soft-fail of policy, not a code change.

**Rebase-sensitive invariants:**

- The canonical-6 feature order is load-bearing — `vmaf_v0.6.1`
  consumes the same six features in the same order, and the ONNX
  graph weight matrix is column-aligned to it. Reordering the
  sidecar `feature_order` field invalidates the checkpoint.
- Netflix Public Dataset is non-redistributable. CI cannot retrain
  end-to-end; only the smoke path
  (`python ai/scripts/train_fr_regressor.py --epochs 3 --no-export`)
  runs in CI when the parquet is locally available.

## Dynamic-PTQ tiny-MLP family (ADR-0275)

`vmaf_tiny_v3` and `vmaf_tiny_v4` carry dynamic-PTQ int8 sidecars
produced by `ai/scripts/ptq_dynamic.py`. The recipe is identical to
`learned_filter_v1` (ADR-0174) and `nr_metric_v1` (ADR-0248): a
single CLI invocation, no calibration data. The on-disk size win is
proportional to weight mass — `mlp_large` (v4) shrinks 45 %,
`mlp_medium` (v3) shrinks 5 % because Constant scaler nodes and op
metadata dominate that graph. v2 (`mlp_small`) stays fp32: too
little weight mass for an int8 sidecar to be worth the audit cost.

**Invariants:**

- The fp32 `<basename>.onnx` stays on disk as the regression
  baseline; the runtime redirect from ADR-0174 picks the
  `.int8.onnx` sibling only when the registry overlay declares
  `quant_mode != "fp32"`.
- `python ai/scripts/measure_quant_drop.py --all` is the gate. Both
  v3 and v4 sit two orders of magnitude under the 0.01 PLCC
  budget; treat any future drop > 1e-3 as a regression worth
  investigating before merging an int8 refresh.
- Re-running `ptq_dynamic.py` is deterministic on a fixed fp32
  input — but the sha256 of the int8 output can shift across ORT
  versions. When ORT is bumped, regenerate both sidecars and
  refresh `int8_sha256` in `model/tiny/registry.json` +
  `vmaf_tiny_v{3,4}.json` in the same PR.

## Frame loader pixel formats

`ai/src/vmaf_train/data/frame_loader.py` is the direct ffmpeg frame
ingest seam for C2/C3 training. It accepts `gray` as `HxW` arrays and
packed `rgb24` / `bgr24` / `rgba` / `bgra` as `HxWxC` arrays. Do not
silently accept planar or subsampled formats such as `yuv420p` in this
loader; those need explicit plane semantics before they are safe to
feed into training tensors.

## `fr_regressor_v2_ensemble_v1` — probabilistic head (ADR-0279)

The probabilistic successor to the codec-aware
`fr_regressor_v2` is a deep ensemble of N=5 v2 members trained under
distinct seeds, packaged as 5 ONNX files plus a manifest sidecar
`model/tiny/fr_regressor_v2_ensemble_v1.json`. Trainer:
[`ai/scripts/train_fr_regressor_v2_ensemble.py`](scripts/train_fr_regressor_v2_ensemble.py);
evaluator: [`ai/scripts/eval_probabilistic_proxy.py`](scripts/eval_probabilistic_proxy.py);
model card:
[`docs/ai/models/fr_regressor_v2_probabilistic.md`](../docs/ai/models/fr_regressor_v2_probabilistic.md).

**Rebase-sensitive invariants:**

- **Per-member ONNX I/O contract is the v2 two-input shape**: inputs
  `features [N, 6]` (canonical-6, StandardScaler-normalised by the
  manifest's `feature_mean` / `feature_std`) + `codec_onehot
  [N, NUM_CODECS]`; output `score [N]` float32. Each member is a
  stock `FRRegressor(num_codecs=NUM_CODECS)` — flipping that to a
  single-input v1-shaped graph silently invalidates every shipped
  ensemble.
- **Manifest layout is the runtime entry point**, not the registry
  rows. Each member is also added to `model/tiny/registry.json` as
  `kind: "fr"` (with id `<ensemble_id>_seed<N>`) so the existing
  tiny-model verifier can SHA-256-check each member without a
  schema bump, but the manifest sidecar's `members[]` list is the
  canonical ordered set the C-side adapter iterates over. Adding a
  schema-version field to `registry.schema.json` for a `fr_ensemble`
  kind is a future option; until then, ensemble lookups go through
  the manifest, not the registry.
- **Ensemble size is part of the contract.** The manifest's
  `ensemble_size` field pins N; the C-side adapter must open
  `ensemble_size` sessions. Changing N requires a new ensemble id +
  manifest, not an in-place mutation.
- **Confidence rule is a one-of**: `confidence.method` is either
  `"ensemble"` (use `gaussian_z` as the multiplier on `sigma`) or
  `"ensemble+conformal"` (use `conformal_q_residual` instead). The
  trainer emits the conformal scalar only when
  `--conformal-calibration-frac > 0` and the calibration split is
  large enough; otherwise the field stays `null` and the Gaussian
  rule applies.
- **`CODEC_VOCAB` parity with v2 is required.** The manifest pins
  `codec_vocab` + `codec_vocab_version`; the runtime must refuse to
  load when these disagree with the live `ai/src/vmaf_train/codec.py`
  vocabulary. Bumping the vocabulary requires retraining the
  ensemble; the existing closed-vocabulary invariant from ADR-0235
  carries over verbatim.
- **Historical smoke artefacts are retired.** ADR-0303 originally
  shipped synthetic 100-row / 1-epoch ensemble members as load-path
  probes. ADR-0321 replaced the five seed ONNX files with
  full-corpus-trained production weights and added per-seed sidecars.
  Do not reintroduce `smoke: true` for
  `fr_regressor_v2_ensemble_v1_seed{0..4}` unless a future ADR
  explicitly rolls the production flip back.
- **Ensemble registry invariant (ADR-0303)**: each ensemble member's
  `smoke: true` registry row flips to `false` **only after** that
  individual seed clears the `PLCC_i ≥ 0.95` LOSO ship gate
  (ADR-0235 / ADR-0291). The ensemble-mean entry — if/when one is
  added to `model/tiny/registry.json` as a `fr_ensemble`-kind row —
  flips **only after all five seeds clear** the per-seed gate *and*
  the variance bound `max_i(PLCC_i) - min_i(PLCC_i) ≤ 0.005` holds.
  The decision lives in [`scripts/ci/ensemble_prod_gate.py`](../scripts/ci/ensemble_prod_gate.py);
  the trainer that emits the per-seed `loso_seed{N}.json` artefacts
  the gate consumes is
  [`ai/scripts/train_fr_regressor_v2_ensemble_loso.py`](scripts/train_fr_regressor_v2_ensemble_loso.py).
  Do NOT flip individual seed rows by hand without running the gate
  against a real-corpus LOSO output — the variance bound is what
  protects the predictive-distribution semantics; flipping seeds
  ad-hoc would silently bake in an unbounded across-seed spread.
- **Registry-flip happened in ADR-0320**: the five
  `fr_regressor_v2_ensemble_v1_seed{0..4}` rows in
  `model/tiny/registry.json` flipped `smoke: true → false` on
  2026-05-06 against a passing
  `runs/ensemble_v2_real/PROMOTE.json` (mean PLCC = 0.9973,
  spread = 9.5e-4, both gate components green) produced by
  [`ai/scripts/validate_ensemble_seeds.py`](scripts/validate_ensemble_seeds.py).
  The verdict file is committed at
  [`model/tiny/fr_regressor_v2_ensemble_v1_seed_flip_PROMOTE.json`](../model/tiny/fr_regressor_v2_ensemble_v1_seed_flip_PROMOTE.json)
  as the immutable audit trail. **Going-forward invariant**: any
  future registry change for these `ensemble_v1` seed rows (sha256
  bump after retraining, smoke-flag mutation, ONNX path change)
  requires a fresh `PROMOTE.json` verdict with mean PLCC ≥ 0.95 AND
  spread ≤ 0.005 — the same two-part gate ADR-0303 defined and
  ADR-0320 honoured. **Never** flip or mutate these rows during a
  `/sync-upstream` rebase or as a side-effect of any other PR — the
  harness in
  [`ai/scripts/run_ensemble_v2_real_corpus_loso.sh`](scripts/run_ensemble_v2_real_corpus_loso.sh)
  and the validator emit the verdict file but **do not** mutate the
  registry. Auto-flipping on PROMOTE was rejected in ADR-0309's
  alternatives matrix specifically because rebase-time mutation of
  shipped registry rows is the foot-gun this invariant exists to
  prevent.
- **Ensemble production-flip is now done (ADR-0321)**: as of
  2026-05-06 the five `fr_regressor_v2_ensemble_v1_seed{0..4}` rows
  carry `smoke: false` and point at LOSO-gated, full-corpus-trained
  ONNX weights produced by
  [`ai/scripts/export_ensemble_v2_seeds.py`](scripts/export_ensemble_v2_seeds.py).
  Each row has a sidecar
  `model/tiny/fr_regressor_v2_ensemble_v1_seed{N}.json` that mirrors
  the canonical `fr_regressor_v2.json` shape (encoder vocab v2, codec
  block layout, scaler params) plus seed-specific gate evidence from
  `runs/ensemble_v2_real/PROMOTE.json`. **Going-forward rule**: any
  future flip (re-train + re-export) requires a fresh
  `PROMOTE.json` from the LOSO trainer and a re-run of
  `export_ensemble_v2_seeds.py` — both the ONNX bytes and the
  per-seed sidecars must be regenerated together so the
  `test_registry.sh` sha256 + sidecar-presence check stays green.
  Editing one without the other is a foot-gun: the registry test
  catches the sha256 drift, but a stale sidecar's gate-evidence block
  would silently lie about provenance.
- **Canonical-6 JSONL schema is load-bearing (ADR-0319)**: the LOSO
  trainer's `_load_corpus` accepts the schema emitted by
  [`scripts/dev/hw_encoder_corpus.py`](../scripts/dev/hw_encoder_corpus.py)
  bit-for-bit — required keys per row are
  `(src, encoder, cq, frame_index, vmaf, adm2, vif_scale0..3, motion2)`.
  The codec block is materialised as 12-slot `ENCODER_VOCAB` v2
  one-hot (mirrors `train_fr_regressor_v2.py`) + a constant
  `preset_norm = 0.5` (the corpus does not record preset) +
  `crf_norm` = `(cq - cq_min) / (cq_max - cq_min)`. Schema changes
  — column rename, encoder-vocab reorder, new required field —
  require an `ENCODER_VOCAB_VERSION` bump and full ensemble retrain
  per the existing closed-vocabulary invariant. The fold-level
  StandardScaler is fit on the training rows only (mirrors
  `eval_loso_vmaf_tiny_v3.py`); leaking the held-out source's
  distribution into the scaler would silently inflate per-fold
  PLCC. See ADR-0319 §Decision and `_load_corpus`'s docstring.

## Quantization-Aware Training (ADR-0207 / ADR-0208)

The QAT trainer hook lives in [`ai/train/qat.py`](train/qat.py) and
the CLI driver in [`ai/scripts/qat_train.py`](scripts/qat_train.py).
The default config example is
[`ai/configs/learned_filter_v1_qat.yaml`](configs/learned_filter_v1_qat.yaml).

**Pipeline (per ADR-0207 + ADR-0208 implementation bridge):**

1. fp32 warm-start training.
2. FX fake-quant insertion via
   `torch.ao.quantization.quantize_fx.prepare_qat_fx` with the
   default symmetric per-tensor activation + per-channel weight
   qconfig.
3. QAT fine-tune at 10× reduced LR.
4. Copy QAT-conditioned weights into a fresh fp32 module, export
   to ONNX (`dynamo=False`), then ORT static-quantize with a
   calibration set drawn from the QAT distribution. Output is a
   QDQ `.int8.onnx`.

**Rebase-sensitive invariants:**

- The two-step pipeline (PyTorch QAT → fp32 ONNX → ORT
  static-quantize) is load-bearing. Do NOT collapse to
  `convert_fx → torch.onnx.export` — both PyTorch 2.11 ONNX
  exporters refuse the `convert_fx` output (legacy emits
  `quantized::conv2d`; TorchDynamo trips on
  `Conv2dPackedParamsBase.__obj_flatten__`). Re-check on each
  PyTorch upgrade.
- State-dict transfer in `_copy_qat_weights_into_fp32` matches
  by submodule name + tensor shape. Models using top-level
  `nn.Sequential` will break this (FX renames Sequential
  children to numeric indices); the `RuntimeError("0 tensors
  copied")` guard catches it.
- FX preparation runs on CPU (PyTorch 2.11's symbolic tracer is
  flaky on CUDA buffers); the trainer migrates to CPU before
  `prepare_qat_fx` and back to the accelerator afterwards.
- `torch.ao.quantization` is deprecated and will be removed in
  PyTorch 2.10. Migration target is `torchao.quantization.pt2e`
  (`prepare_pt2e` / `convert_pt2e`); only the FX-prep call
  changes — the rest of the pipeline (ORT static-quantize) is
  unaffected.

## Local workflow

```bash
pip install -e ai/
vmaf-train --help
vmaf-train register model/tiny/lpips_sq.onnx   # adds to registry.json
python ai/lpips_export.py                      # re-export LPIPS from the reference repo

# Netflix-corpus training (ADR-0203):
bash ai/scripts/run_training.sh
```

## fr_regressor_v2 — codec block layout (ADR-0272)

`ai/scripts/train_fr_regressor_v2.py` consumes the vmaf-tune Phase A
JSONL corpus and emits `model/tiny/fr_regressor_v2.onnx`. The codec
block layout is **load-bearing** — bumping it requires a re-train.
Pinned invariants:

- `ENCODER_VOCAB = ("libx264", "libx265", "libsvtav1", "libvvenc",
  "libvpx-vp9", "unknown")`. Order matches the encoder-onehot index
  baked into the trained ONNX. Append-only; bump
  `ENCODER_VOCAB_VERSION` and re-train when adding a new entry.
- 8-D codec block layout: `[encoder_onehot[0..5], preset_norm,
  crf_norm]`. Both `preset_norm` and `crf_norm` live in `[0, 1]`.
- `crf_norm = crf / 63.0` — `63` is the union upper bound across
  supported encoders (libsvtav1 / libvpx-vp9 max).
- `preset_norm = preset_ordinal / 9.0` — per-encoder ordinal table
  in `train_fr_regressor_v2.py::PRESET_ORDINAL`. libsvtav1's numeric
  0..13 presets are squashed to 0..9.
- Two-input ONNX: `features` (N, 6) + `codec` (N, 8) -> `score` (N,).
  Mirrors the LPIPS-Sq two-input precedent (ADR-0040 / ADR-0041).
- StandardScaler is applied to `features` only; the codec block
  passes through unscaled. `feature_mean` / `feature_std` ship in
  the sidecar JSON.

The current shipped ONNX is from `--smoke` mode and is registered
`smoke: true` in `model/tiny/registry.json`. Production training run
is gated on a multi-codec Phase A corpus + per-frame feature emission
in the Phase A schema. See ADR-0272 + Research-0054.

## BVI-DVC corpus ingestion (ADR-0310)

The Bristol VI Lab BVI-DVC reference corpus is a second training
shard for `fr_regressor_v2` alongside the Netflix Public drop.
Pipeline: `bvi_dvc_to_full_features.py` (parquet + cached libvmaf
JSON) → `bvi_dvc_to_corpus_jsonl.py` (vmaf-tune `CORPUS_ROW_KEYS`
rows) → `merge_corpora.py` (concatenate with Netflix shard, dedup
by `(src_sha256, encoder, preset, crf)`).

**Rebase-sensitive invariants:**

- BVI-DVC is research-only. The archive
  (`.workingdir2/BVI-DVC Part 1.zip`), the extracted MP4s
  (`.workingdir2/bvi-dvc-extracted/`), the feature parquet
  (`runs/full_features_bvi_dvc_*.parquet`), the JSONL corpus shard
  (`runs/bvi_dvc_corpus.jsonl`), and the cached vmaf JSON
  (`~/.cache/vmaf-tiny-ai-bvi-dvc-full/`) are **never committed**.
  The fork redistributes derived `fr_regressor_v2_*.onnx` weights
  only — corpus-must-be-license-compatible-or-stay-local applies
  uniformly across `ai/` corpora (Netflix Public, BVI-DVC, KoNViD,
  YouTube-UGC).
- The merge contract is `vmaftune.CORPUS_ROW_KEYS` from
  `tools/vmaf-tune/src/vmaftune/__init__.py`. Bumping
  `SCHEMA_VERSION` means re-running the BVI-DVC adapter to backfill
  the new fields. The merge utility refuses any row missing a
  required key — fail-loud is by design.
- The natural-key tuple `(src_sha256, encoder, preset, crf)` is the
  dedup contract. Re-encodes of the same source under a new
  `(preset, crf)` legitimately appear as distinct rows; do not
  fold them by `src_sha256` alone.
- Production-weights flip stays gated on
  [ADR-0303](../docs/adr/0303-fr-regressor-v2-ensemble-flip.md).
  Adding BVI-DVC to the corpus does NOT authorise re-shipping
  `fr_regressor_v2.onnx` without re-running the ensemble gate.

## v5 corpus-expansion probe — research-only (ADR-0287)

The `*_vmaf_tiny_v5.py` scripts
(`fetch_youtube_ugc_subset.py`, `extract_ugc_features.py`,
`train_vmaf_tiny_v5.py`, `eval_loso_vmaf_tiny_v5.py`) are
research infrastructure for the deferred v5 corpus-expansion
probe. **No `vmaf_tiny_v5.onnx` ships** — the 1-σ ship gate did
not clear (Δ PLCC = +0.00005 at seed=0, far below the 1-σ_v2
threshold). When extending these scripts:

- Do not add a `vmaf_tiny_v5` row to
  `model/tiny/registry.json` unless a follow-up run actually
  clears the ship gate documented in
  [ADR-0287](../docs/adr/0287-vmaf-tiny-v5-corpus-expansion.md).
- The fetcher hits a public GCS bucket (`gs://ugc-dataset/`,
  CC-BY); raw videos and the resulting
  `runs/full_features_ugc.parquet` must NEVER be committed
  (the `runs/` and `.workingdir2/` trees are gitignored).
- The dual-arm LOSO trains 18 mlp_small models
  (9 v2-baseline plus 9 v5-candidate); single invocation
  wall-time is ~10–25 min depending on CPU. Do NOT launch it
  concurrently with another training process — the two share
  BLAS threads and serialise badly.

## KonViD-150k MOS-corpus ingestion (ADR-0325)

**Script:** `ai/scripts/konvid_150k_to_corpus_jsonl.py`

### Rebase-sensitive invariants

- The adapter accepts two local layouts under `.workingdir2/konvid-150k/`:
  a URL `manifest.csv` plus `clips/`, or the split score-drop layout
  `k150ka_scores.csv` / `k150kb_scores.csv` plus
  `k150ka_extracted/` / `k150kb_extracted/`. Do not remove the split
  discovery path unless the staged corpus is migrated first.
- An explicit `--manifest-csv` remains strict. If that file is missing,
  the adapter must fail instead of falling back to split discovery; this
  catches typoed operator paths.
- The emitted JSONL schema is still the shared MOS-corpus schema. Split
  score rows do not add a `split` column to output; missing score-drop
  metadata is represented as `mos_std_dev = 0.0` and `n_ratings = 0`.

## CHUG HDR MOS-corpus ingestion (ADR-0426)

**Script:** `ai/scripts/chug_to_corpus_jsonl.py`

### Rebase-sensitive invariants

- CHUG data is local-only under `.workingdir2/chug/`. Do not commit the
  public `chug.csv`, downloaded MP4s, emitted JSONL, trained local
  CHUG heads, or derived features. The README/license mismatch is
  handled by treating the dataset as non-commercial/share-alike until
  clarified.
- CHUG's public `mos_j` column is on a 0-100 axis. The adapter preserves
  it as `mos_raw_0_100` and maps trainer-facing `mos` onto `[1, 5]`
  via `1 + 4 * mos_raw_0_100 / 100` so the existing MOS-head trainer
  does not drop every row as out-of-range. Do not remove the raw field
  or silently change the scale.
- The adapter preserves CHUG HDR / ladder metadata (`chug_bitladder`,
  `chug_resolution`, `chug_bitrate_label`, orientation, manifest
  geometry, and source content name) as optional JSONL fields. Existing
  MOS-head training ignores those columns today; future HDR models may
  consume them explicitly.
- The CHUG feature materialiser is governed by ADR-0427. It pairs each
  distorted row with the matching `chug_content_name` reference row,
  decodes both sides as 10-bit 4:2:0, and scales the distorted side to
  reference geometry before libvmaf extraction. Changing that alignment
  policy changes the training distribution and requires a new ADR.
- CHUG train/validation/test splits are content-level, not row-level.
  `ai/scripts/chug_extract_features.py` hashes `chug_content_name` with
  seed `chug-hdr-v1` into deterministic 80/10/10 partitions and writes
  the chosen `split` plus `chug_split_key` into every feature row. Do
  not split bitrate-ladder rows independently; that leaks the same
  source content across validation.
- The local HDR metadata audit (`--audit-output`) is a pre-training
  gate for CHUG experiments. Preserve its ffprobe transfer / primaries /
  pix-fmt counters and malformed-PQ/HLG-without-BT.2020 row list when
  touching the materialiser.
- `ai/scripts/enrich_k150k_parquet_metadata.py` is the recovery path for
  FULL_FEATURES parquet jobs that were started without `--metadata-jsonl`.
  It must match metadata by `clip_name` / JSONL basename, fill missing
  metadata cells by default, and keep feature/MOS columns unchanged unless
  `--overwrite-metadata` is explicitly passed.

## K150K-A corpus extraction (ADR-0362, ADR-0382, ADR-0431)

**Script:** `ai/scripts/extract_k150k_features.py`
**Branch:** `chore/ensemble-kit-gdrive-quickstart`

### Rebase-sensitive invariants

- **Parquet writes are at-end only — never per-flush (Research-0135 Win 1).**
  Rows are accumulated in memory throughout the run and appended to a JSONL
  staging file (`<out>.rows.jsonl`) for crash durability.  The parquet is
  written exactly once at the end via `_write_parquet_from_rows`.  The old
  `_flush_parquet` helper (which read the growing parquet on every 200-clip
  flush) has been removed.  Restartability is still guaranteed by the
  `.done` checkpoint file; the staging file adds a second durability layer
  so rows are not lost on an unclean exit.  Do not re-introduce per-flush
  parquet writes — they make total parquet I/O O(N²) over the corpus size.
- **Staging file is main-process-only (Research-0135).** `_append_row_to_staging`
  is called only from the main process inside the `as_completed()` loop,
  after `fut.result()` returns.  Worker subprocesses must never write to the
  staging file.  Violating single-writer semantics on the staging file would
  corrupt it without error.
- **ffprobe is skipped when sidecar has geometry (Research-0135 Win 2).**
  `_geometry_from_sidecar(meta)` reads `chug_width_manifest`,
  `chug_height_manifest`, `chug_framerate_manifest`, and optionally
  `chug_bit_depth` from the CHUG JSONL sidecar row.  The sidecar metadata
  is already loaded in `jsonl_meta` for enrichment; no extra I/O is needed.
  If any required field is absent (K150K-A clips, incomplete rows), the
  function returns `None` and `_probe_geometry(mp4)` is called as fallback.
  Do not remove the fallback — K150K-A clips have no sidecar.
- **Binary requirement:** the script requires `libvmaf/build-cpu/tools/vmaf`
  (fork build); the system `/usr/local/bin/vmaf` v3.0.0 lacks `ssimulacra2`
  and `motion_v2`. The `--vmaf-bin` default (in `main()`) now points to
  `libvmaf/build-cpu/tools/vmaf`. Do NOT switch to `build-cuda/tools/vmaf`
  as the default — the CUDA binary has a latent CLI double-write bug when
  `--feature <x>` is combined with the auto-loaded default VMAF model
  (see Research-0096 / ADR-0382 for details).
- **CUDA split invariant:** when operators explicitly pass a CUDA-capable
  `--vmaf-bin`, the script must use explicit CUDA extractor names for the
  GPU-safe pass and `--cpu-vmaf-bin` for the residual CPU pass
  (`float_ssim`, `cambi`). Do not re-collapse this into one generic
  `--backend cuda` invocation; CHUG/K150K 10-bit clips can fail
  `context could not be synchronized` through that path.
- **Parallelism model:** the script uses `concurrent.futures.ProcessPoolExecutor`
  with `--threads-cuda` workers (default 8). Each worker is fully independent.
  The `--threads-cuda` flag is named for historical reasons; it controls outer
  process parallelism for both CPU and split CUDA modes. Do not switch to
  threading — libvmaf subprocess invocations are not thread-safe for concurrent
  parallel pipelines.
- **Checkpoint thread-safety:** `_append_done()` is called only from the main
  process (after `fut.result()` returns in the `as_completed()` loop). Do not
  call it from worker processes — the append-only guarantee relies on single-writer
  semantics.
- **NaN propagation:** `ciede2000` and `psnr_hvs` return `null` from vmaf
  when ref == distorted (identity pair). All-NaN columns are **expected** —
  do not treat them as extraction failures. `np.errstate(all="ignore")`
  in `_aggregate_frames()` suppresses the numpy warning; preserve it.
- **Column-order lock:** `FEATURE_NAMES` (line ~121) defines the 21-feature
  column order (parquet schema v2) that downstream loaders depend on. Appending
  is safe; reordering or removing entries breaks existing parquets and any
  trained model that consumed them. Increment the parquet schema version in a
  separate ADR if reordering becomes necessary. **Schema v2 invariant (ADR-0431):**
  `ssimulacra2` is omitted from K150K/CHUG self-vs-self extraction because in
  identity pairs (ref == distorted) it produces a constant ~100, yielding zero
  training signal while consuming 30–50% of GPU time per clip. When operating in
  FR-from-NR mode (same video on both sides), all difference-based metrics
  (difference-based ssimulacra2, ciede2000, psnr_hvs, ADM, VIF) degenerate; see
  ADR-0362 §Negative consequences. CPU-only ssimulacra2 extraction remains
  available for genuine FR pairs where it is informative.
- **FEATURE_NAMES completeness invariant:** all `FEATURE_NAMES` entries must
  map to JSON keys emitted by the pipeline (CUDA extractors, CPU residual, or
  `--model` dispatch).  The `vmaf` entry is the model composite score emitted
  via the `--model` arg in `_run_feature_passes`; all other entries are raw
  features emitted via `--feature` arguments.
- **vmaf column is computed via vmaf_v0.6.1 (Research-0135):** the `vmaf`
  column in CHUG/K150K output parquets is computed by dispatching the SDR
  `vmaf_v0.6.1` model via `--model version=vmaf_v0.6.1` in the libvmaf CLI
  invocation.  This model is SDR-trained and is mis-calibrated on PQ HDR
  clips; scores are valid for relative bitrate-ladder comparison within a
  content group but are not meaningful as absolute HDR quality targets.
  Replace with the Netflix HDR model when it ships (change the `--model` arg
  in `_run_feature_passes`; no schema change required).  Do NOT remove the
  `--model` arg without an ADR — the vmaf relationship across ladder rungs
  is a required training feature per user direction 2026-05-16.
- **Checkpoint format:** `.done` file is append-only, one clip name per
  line, no header. Changing the format without a migration breaks
  in-progress runs. The `_load_done_set()` / `_append_done()` helpers are
  the single-exit-point for reads and writes; add any format change there.
- **Gitignore:** `runs/full_features_k150k.parquet` and
  `runs/k150k_extract.log` are gitignored (152K-clip output is not tracked).
  Do not commit these files.
- **FR-from-NR adapter:** the script does NOT call `NrToFrAdapter` from
  the Python training harness — it builds the vmaf CLI argv directly with
  ref == distorted, which is the lighter-weight equivalent. Any upstream
  refactor of the Python adapter is irrelevant to this script.

## v3 retrain invariant — `ENCODER_VOCAB` 13 → 16 (ADR-0302)

The `ENCODER_VOCAB_V3` parallel constant in
[`scripts/train_fr_regressor_v2.py`](scripts/train_fr_regressor_v2.py)
documents the target 16-slot vocab (adds `libsvtav1`,
`h264_videotoolbox`, `hevc_videotoolbox` to v2's 13 slots). The
**live `ENCODER_VOCAB` and `ENCODER_VOCAB_VERSION = 2` are the source
of truth** until the follow-up retrain PR clears the LOSO PLCC
ship gate.

**Invariants that the v3 retrain PR must honour** (per ADR-0235 +
ADR-0291 + ADR-0302):

- A schema bump (v2 → v3) requires a fresh LOSO run that clears
  **mean LOSO PLCC ≥ 0.95** across all 9 Netflix sources (matches
  the gate ADR-0291 cleared on v2). The trainer must exit non-zero
  and refuse to overwrite the registry entry on failure — same
  pattern `fr_regressor_v1` already enforces.

  **Status (ADR-0323, 2026-05-06):** The first v3 LOSO run shipped
  under [`ai/scripts/train_fr_regressor_v3.py`](scripts/train_fr_regressor_v3.py)
  on the NVENC-only Phase A corpus (5,640 rows, 9 sources × 4 CQs).
  Mean LOSO PLCC = **0.9975 ± 0.0018** (every source above 0.99) —
  comfortably clears the 0.95 ship gate. The model ships under
  `model/tiny/fr_regressor_v3.onnx` with `smoke: false`. The live
  `ENCODER_VOCAB_VERSION = 2` in [`scripts/train_fr_regressor_v2.py`](scripts/train_fr_regressor_v2.py)
  **stays authoritative for `fr_regressor_v2.onnx`** until a separate
  "promote v3 to authoritative" PR — this PR ships v3 as a parallel
  checkpoint, not a v2 replacement. Future v3 retrains (on a
  multi-codec corpus drop) must continue to clear the 0.95 floor and
  must additionally measure the ADR-0235 multi-codec lift floor
  (≥+0.005 PLCC over `fr_regressor_v1`); the lift floor is not yet
  measurable on the NVENC-only corpus, so this PR's gate is the 0.95
  floor only.
- Multi-codec lift over the v1 single-input regressor must remain
  **≥ +0.005 PLCC**. ADR-0235 set this as the codec-block invariant;
  the v2 production checkpoint cleared it comfortably and v3 must
  not regress.
- The in-tree v2 ONNX (`model/tiny/fr_regressor_v2.onnx`) **must
  not be replaced** until the new v3 ONNX clears the gate. The
  load-fallback shim collapses unknown v3 strings into the v2
  `unknown` column and lets v2 keep serving every consumer in the
  meantime.
- Append-only ordering is load-bearing — the 13 v2 slot indices
  (0..12) keep their column positions verbatim under v3; the three
  new slots append at indices 13/14/15. Reordering silently
  invalidates every shipped `fr_regressor_v2_*.onnx`. ADR-0235
  documents this rule for `CODEC_VOCAB`; ADR-0302 §Decision
  re-asserts it for `ENCODER_VOCAB`.
- Slot strings must match the vmaf-tune codec-adapter registry keys
  exactly (`libsvtav1`, `h264_videotoolbox`, `hevc_videotoolbox`).
  ADR-0235 §References pins this rule globally for all corpus
  emitters.

## `fr_regressor_*` namespace map (ADR-0349)

The `fr_regressor` lineage carries two orthogonal axes. Encoder-vocab
versioning runs on `_v{N}` (v1 = no codec block, v2 = 13-slot, v3 = 16-slot).
Feature-set versioning runs as a `_v{N}plus_features` suffix on the matching
encoder-vocab base. Names below are claimed; do **not** reuse them for
unrelated workstreams.

| Name | Encoder vocab | Feature axis | Status |
|---|---|---|---|
| `fr_regressor_v1` | none (single-input) | canonical-6 | shipped (ADR-0249) |
| `fr_regressor_v2` | v2 (13-slot) | canonical-6 + 8-D codec block | shipped (ADR-0272 / ADR-0291) |
| `fr_regressor_v2_ensemble_v1_seed{0..4}` | v2 (13-slot) | canonical-6 + 8-D codec block | shipped (ADR-0279) |
| `fr_regressor_v3` | v3 (16-slot) | canonical-6 + 18-D codec block | shipped (ADR-0302 / ADR-0323) |
| `fr_regressor_v3plus_features` | v3 (16-slot) | canonical-6 + `encoder_internal` + shot-boundary + `hwcap` | **reserved** (ADR-0349) — registry row lands with the future PR that ships the `.onnx` |

The reservation is documentation-only because
[`libvmaf/test/dnn/test_registry.sh`](../libvmaf/test/dnn/test_registry.sh)
treats every registry row as a hard contract (file must exist, sha256 must
match, sidecar must accompany every `smoke: false` entry); a stub row would
fail CI on day one. The future `_v3plus_features` PR populates the row in
the same commit that ships the `.onnx`. See
[ADR-0349](../docs/adr/0349-fr-regressor-v3-namespace.md) for the namespace
decision and the rejected alternatives.

## MOS-head v1 invariants — `konvid_mos_head_v1` (ADR-0336, Phase 3 of ADR-0325)

The fork's first head trained against subjective MOS (not VMAF)
ships under
[`ai/scripts/train_konvid_mos_head.py`](scripts/train_konvid_mos_head.py),
with the trained ONNX and human-readable model card at
[`model/konvid_mos_head_v1.onnx`](../model/konvid_mos_head_v1.onnx) and
[`model/konvid_mos_head_v1_card.md`](../model/konvid_mos_head_v1_card.md).
Invariants that any follow-up retrain or corpus-shape PR must honour:

- **Feature-column order is load-bearing.** `FEATURE_COLUMNS =
  CANONICAL_6 + EXTRA_FEATURES` is the exact 11-D layout baked into
  the trained ONNX and consumed by
  `tools/vmaf-tune/src/vmaftune/predictor.py::_predict_mos_via_head`.
  The 6 canonical columns occupy indices 0..5; the 5 extras
  (`saliency_mean`, `saliency_var`, `shot_count_norm`,
  `shot_mean_len_norm`, `shot_cut_density`) occupy 6..10. Reordering
  silently invalidates every shipped `konvid_mos_head_v1.onnx`.
- **ENCODER_VOCAB v4 expansion is append-only.** The v4 vocab ships
  with a single `"ugc-mixed"` slot per ADR-0325 §Decision. When
  LSVQ + YouTube-UGC ingestion lands, the new slots append at the
  end; existing trained ONNX stays loadable and the predictor's
  per-shot one-hot widens transparently.
- **MOS range is `[1.0, 5.0]` and is baked into the graph.** The
  trainer wraps the MLP output in
  `MOS_MIN + (MOS_MAX - MOS_MIN) * sigmoid(raw)`; adversarial input
  cannot drive the prediction outside `[1, 5]`. Predictor surfaces
  (`Predictor.predict_mos` + `_predict_mos_via_head`) carry an
  additional clamp as belt-and-braces. Do not change the range
  without a schema bump + retrain.
- **Production-flip gate is not lowered on real-corpus failures.**
  Per memory `feedback_no_test_weakening` and ADR-0325 §Production-
  flip gate, a failing real-corpus retrain ships the head with
  `Status: Proposed`, *not* a relaxed gate. Threshold values
  (`PLCC ≥ 0.85`, `SROCC ≥ 0.82`, `RMSE ≤ 0.45`, `spread ≤ 0.005`)
  are constants in the trainer (`GATE_*`); changing them requires
  a new ADR.
- **Predictor fallback path is documented behaviour, not a bug.**
  When the ONNX is missing, `Predictor.predict_mos` returns
  `(predicted_vmaf - 30) / 14` clamped to `[1, 5]`. That is the
  documented contract; tests
  (`tools/vmaf-tune/tests/test_predict_mos.py::test_predict_mos_falls_back_when_onnx_missing`)
  pin it. Removing the fallback breaks every dev host that hasn't
  pulled the ONNX.

## Knob-sweep recipe-regression policy (ADR-0308)

Cited from the regression-detection invariant in
[ADR-0305](../docs/adr/0305-encoder-knob-space-pareto-analysis.md)
and the policy decision in
[ADR-0308](../docs/adr/0308-encoder-knob-sweep-recipe-regression-policy.md);
populated findings are in
[Research-0080](../docs/research/0080-encoder-knob-sweep-findings.md).
When extending `ai/scripts/analyze_knob_sweep.py` or anything that
consumes its output:

- A recipe regression is *structural* iff it reproduces on **≥7 of
  the 9** corpus sources within a single
  `(codec, rc_mode, recipe, preset, q)` cell. Structural regressions
  are forbidden as `tools/vmaf-tune/codec_adapters/*` defaults and
  forbidden as `vmaf-tune recommend` outputs without explicit
  override. The known-structural cells are listed in
  Research-0080 §Aggregated-bad-recipe-patterns; do not promote any
  of them to an adapter-level default in a follow-up PR.
- A recipe regression that hits 1-6 sources is *content-dependent*
  and is filtered at recommend-time via the per-slice hull lookup,
  not at adapter-default time.
- Do NOT modify `ai/scripts/analyze_knob_sweep.py` to relax
  `bitrate_tol_pct` (default 5.0) or `vmaf_tol` (default 0.1)
  without an ADR — those tolerances are calibrated against the
  per-frame VMAF noise floor and bitrate quantisation in
  libavformat muxers, and loosening them silently masks the
  structural cluster (see ADR-0305 §Consequences).
- The detector is an **offline** gate (3-hour sweep, ~2 GiB JSONL,
  single-host variance); do not wire it into CI without first
  designing a smaller stratified sample that reproduces the
  structural patterns. Tracked as a follow-up in ADR-0308 §Decision
  point 4.
- The corpus producer (`hw_encoder_corpus.py`) currently emits
  `(src, actual_kbps, vmaf, enc_ms, recipe)` while
  `analyze_knob_sweep.SweepRow` consumes
  `(source, bitrate_kbps, vmaf_score, encode_time_ms,
  is_bare_default)`. Until the producer-side rename lands
  (SCHEMA_VERSION=3 follow-up per ADR-0308 §Decision point 5),
  any analysis run goes through a throw-away wrapper that performs
  the rename in-process — do NOT modify
  `analyze_knob_sweep.py` to accept both spellings.


## `u2netp` fork-local mirror invariants (ADR-0325)

The fork ships a release-artefact mirror for the upstream U-2-Net
`u2netp` checkpoint via GitHub Release attachments. The scaffold
(license, model card, operator doc, supply-chain staging step)
landed in PR scope ADR-0325; the binary upload is a separate PR.

- **Never commit `model/u2netp_mirror.onnx` or
  `model/u2netp_mirror.pth` to git.** Both paths are gitignored
  (see `.gitignore`). The binary lives in GitHub Release assets
  only — signed via Sigstore, hashed for SLSA, paired with
  `LICENSES/Apache-2.0-u2netp.txt` at upload time. If a binary
  upload PR ever attempts to commit either file, the ADR-0325
  contract is broken; reject the PR.
- **The recommended saliency weights remain
  `saliency_student_v1`** (ADR-0286, fork-trained DUTS student
  under BSD-3-Clause-Plus-Patent). `u2netp_mirror` is the named
  *fallback* for upstream-lineage citation, comparative
  evaluation, or downstream pipelines pinned to upstream
  behaviour. Do NOT flip `model/tiny/registry.json`'s default
  `mobilesal` resolution to `u2netp_mirror_v1` without an ADR
  superseding ADR-0286.
- **Apache-2.0 §4 (a) + (c) compliance is non-negotiable.**
  Every release that carries `u2netp_mirror_v*` must also carry
  `LICENSES/Apache-2.0-u2netp.txt` with its attribution block
  intact. The supply-chain.yml staging step pairs them
  automatically; if a future refactor decouples them, downstream
  operators inherit a license-non-compliant artefact. §4 (b)
  applies only to ONNX rewraps (the export script writes a
  `metadata_props` block recording the conversion provenance);
  verbatim `.pth` redistribution does not trigger (b). §4 (d) is
  moot — upstream ships no NOTICE file.
- **The binary upload PR re-pins the upstream commit.** The
  scaffold-time pin is HEAD `ac7e1c81`; the binary upload PR
  must verify the upstream `LICENSE` SPDX is still Apache-2.0
  and the tree still carries no NOTICE file at the
  upload-time HEAD, then bump the model card's commit pin
  accordingly.

- The MOS-corpus row schema emitted by

  `ai/scripts/lsvq_to_corpus_jsonl.py` (ADR-0367) is byte-identical
  to the KonViD-150k Phase 2 adapter
  (`ai/scripts/konvid_150k_to_corpus_jsonl.py`) modulo the
  `corpus` and `corpus_version` literals. Both are consumed
  through one trainer-side data loader. Do NOT widen the schema
  in only one adapter — adding or removing a column means a
  lockstep edit across both, plus a `corpus_version` bump.


  `ai/scripts/youtube_ugc_to_corpus_jsonl.py` (ADR-0368) is
  byte-identical to the LSVQ adapter
  (`ai/scripts/lsvq_to_corpus_jsonl.py`, ADR-0333) and the
  KonViD-150k Phase 2 adapter modulo the `corpus` and
  `corpus_version` literals. All three are consumed through one
  trainer-side data loader. Do NOT widen the schema in only one
  adapter — adding or removing a column means a lockstep edit
  across all three, plus a `corpus_version` bump. The synthesised
  bucket-URL path (`--bucket-prefix` flag) is YouTube-UGC-specific
  because the canonical `original_videos.csv` ships without a
  `url` column; do not back-port that synthesis seam to the LSVQ
  / KonViD-150k adapters where it would mask manifest-CSV bugs.

  `ai/scripts/waterloo_ivc_to_corpus_jsonl.py` (ADR-0369) is
  byte-identical to the LSVQ adapter
  (`ai/scripts/lsvq_to_corpus_jsonl.py`, ADR-0333) and the
  KonViD-150k Phase 2 adapter
  (`ai/scripts/konvid_150k_to_corpus_jsonl.py`, ADR-0325 Phase 2)
  modulo the `corpus` and `corpus_version` literals. All three
  adapters are consumed through one trainer-side data loader. Do
  NOT widen the schema in only one adapter — adding or removing
  a column means a lockstep edit across all three (plus a
  `corpus_version` bump on each). The Waterloo IVC adapter
  records MOS verbatim on the dataset's native **0–100** scale,
  diverging from KonViD / LSVQ's 1–5 Likert scale; cross-corpus
  rescaling is a trainer-side concern and is NOT applied at
  ingest time on either adapter. The trainer-side normaliser
  must read each row's `corpus` literal to pick the correct
  per-shard rescale factor.
