# `tools/vmaf-tune/` — agent notes

Quality-aware encode automation harness. See
[`docs/adr/0237-quality-aware-encode-automation.md`](../../docs/adr/0237-quality-aware-encode-automation.md)
for the umbrella spec and
[`docs/research/0044-quality-aware-encode-automation.md`](../../docs/research/0044-quality-aware-encode-automation.md)
for the option-space digest.

## Rebase-sensitive invariants

- **The Phase A JSONL corpus row schema is the API contract for Phase
  B / C.** Phase B (target-VMAF bisect) and Phase C (per-title CRF
  predictor) read corpora produced by this tool. Adding optional keys
  with a default is fine; renaming or removing keys, or changing their
  type/semantics, requires bumping `vmaftune.SCHEMA_VERSION` and
  updating every downstream consumer in the same PR. The canonical
  key list lives in `src/vmaftune/__init__.py` (`CORPUS_ROW_KEYS`)
  and is asserted on every emitted row by `corpus._row_for`. Schema
  v3 ([ADR-0331](../../docs/adr/0331-corpus-schema-v3.md)) added 12
  canonical-6 per-feature aggregate columns (`adm2_mean`,
  `vif_scale[0..3]_mean`, `motion2_mean` plus matching `_std`);
  they are sourced from libvmaf's `pooled_metrics.<feature>` block
  and **must surface as `NaN` — never `0.0` — when libvmaf does not
  expose the feature** so trainers can drop the row instead of
  fitting on synthetic zeros. The reader (`corpus.read_jsonl`)
  back-fills missing v3 columns on legacy v2 rows with `NaN`; the
  on-disk `schema_version` is preserved so consumers can filter on
  `>= 3` when they need real per-feature data.
- **The `vmaf_model` JSONL field is now per-row, not per-job.** Since
  ADR-0289 (resolution-aware model selection), `corpus._row_for`
  populates `vmaf_model` from `score_res.request.model`, which in
  turn comes from `resolution.select_vmaf_model_version(width, height)`
  when `CorpusOptions.resolution_aware` is True. Mixed-ladder corpora
  legitimately contain multiple distinct `vmaf_model` values across
  rows. Downstream consumers (Phase B/C/D) must group/filter by
  `vmaf_model` rather than assuming a constant.
- **`resolution.py` decision rule is height-only.** `height >= 2160`
  picks `vmaf_4k_v0.6.1`; everything else picks `vmaf_v0.6.1`. Width
  is accepted in the API for symmetry but ignored in the body. Do not
  add per-codec / per-pixel-count branches without an ADR-0289
  follow-up — the rule mirrors Netflix's published guidance and is
  the only defensible default until the fork ships its own
  intermediate models.
- **The codec-adapter contract is multi-codec from day one.** Phase A
  wires `libx264` end-to-end; `libaom-av1`
  ([ADR-0279](../../docs/adr/0279-vmaf-tune-codec-adapter-libaom.md))
  joins as a metadata-and-argv-helper adapter (its argv shape uses
  `-cpu-used`, not `-preset`, so the encode driver gains a second
  argv path when the codec-pluggable encode wiring lands).
  `codec_adapters/__init__.py` exposes a registry the search loop
  must use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`.
- **Adapter preset vocabulary is the cross-codec sweep axis.** The
  ten-name preset tuple (`placebo, slowest, slower, slow, medium,
  fast, faster, veryfast, superfast, ultrafast`) is shared across
  AV1-family adapters so a single `--preset` axis covers x264 / x265
  / svtav1 / libaom-av1 / libvpx-vp9 in one sweep. Each adapter maps the name
  onto its codec-specific knob (cpu-used, preset enum, ...). Do not
  introduce per-adapter preset names; if the codec needs a knob the
  shared vocabulary cannot express, route it through `extra_params`
  rather than splitting the preset axis.
- **`libvpx-vp9` two-pass is FFmpeg-generic, encoder-stats is not.**
  The adapter may set `supports_two_pass = True` because FFmpeg's
  libvpx wrapper honours `-pass` / `-passlogfile`, but
  `supports_encoder_stats` stays `False`: VP9 first-pass stats are a
  binary libvpx packet stream, not the x264/x265 text stats schema
  consumed by `encoder_stats.py`.
- **The codec-adapter contract is multi-codec from day one.**
  `codec_adapters/__init__.py` exposes a registry the search loop must
  use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`. Wired today: `libx264`
  (Phase A scaffold) and `libx265` (ADR-0288). One narrow exception
  lives in `encode.parse_versions(stderr, encoder=…)` — the per-codec
  banner regex (x264's `x264 - core <N>` vs x265's
  `x265 [info]: HEVC encoder version <V>`) cannot be expressed as a
  single pattern, so the function dispatches on the encoder name. This
  branch is allowed; the corpus emitter and the search loop must still
  go through the registry.
  wires `libx264` plus the NVENC family (`h264_nvenc`,
  `hevc_nvenc`, `av1_nvenc` — see
  [ADR-0290](../../docs/adr/0290-vmaf-tune-nvenc-adapters.md)).
  `codec_adapters/__init__.py` exposes a registry the search loop
  must use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`. Hardware-encoder
  families share private helpers (e.g. `_nvenc_common.py`) — keep
  the mnemonic preset map and CQ window in one place per family so
  the per-codec files stay thin.
  wires `libx264` and `libsvtav1` (ADR-0294); `codec_adapters/__init__.py`
  exposes a registry the search loop must use uniformly. Do not branch
  on codec name in `corpus.py` / `encode.py` / `score.py`; route via
  the adapter. New codecs are one-file additions under
  `codec_adapters/`.
- **`PRESET_NAME_TO_INT` in `codec_adapters/svtav1.py` is closed and
  order-stable** (ADR-0278). The mapping (`placebo`→`0`, `slowest`→`1`,
  `slower`→`3`, `slow`→`5`, `medium`→`7`, `fast`→`9`, `faster`→`11`,
  `veryfast`→`13`) is exercised by every corpus row that records
  `encoder == "libsvtav1"`. Adding a name is a schema bump for any
  fr_regressor_v2 corpus that pinned the previous mapping; reordering
  silently changes the integer SVT-AV1 receives. Editing this table
  requires a same-PR doc + ADR update.
- **The `ffmpeg_preset_token()` adapter hook is optional** —
  `corpus.iter_rows` falls back to forwarding the preset name verbatim
  when an adapter does not implement it (the libx264 path). Adapters
  that need a non-string preset translation (libsvtav1 today,
  libsvthevc / future codecs tomorrow) implement the hook and return
  a string for argv. Do not promote it to a required protocol method
  without a same-PR pass over every existing adapter.
- **Subprocess boundary is the test seam.** `encode.run_encode` and
  `score.run_score` accept a `runner` argument that defaults to
  `subprocess.run`. Tests inject a fake; production callers leave it
  default. Do not reach for `os.system` / `popen` shortcuts —
  `tests/test_corpus.py` will silently stop covering the path.
- **Fast-path is opt-in; the grid stays canonical
  ([ADR-0276](../../docs/adr/0276-vmaf-tune-fast-path.md)).** The
  `fast` subcommand under `src/vmaftune/fast.py` accelerates the
  *recommendation* use case via proxy + Bayesian + GPU-verify, but
  must never automatically replace the Phase A grid path. The grid
  is the ground-truth corpus generator that Phase B/C/D consume;
  removing or re-routing it breaks the Phase A.5 → Phase A
  fallback contract for proxy-OOD sources. The `fast` subcommand
  surfaces its smoke vs production mode in the CLI output's
  `notes` field — keep that visibility when extending the loop.
- **Fast-path time budgets are enforced by Optuna, not just reported.**
  `fast.fast_recommend(time_budget_s=...)` passes the value to
  `study.optimize(timeout=...)`, and the emitted `n_trials` field is
  the number of completed trials, not the requested cap. Preserve that
  distinction so wrappers can tell when the budget cut a search short.
- **`vmaf-tune fast` CLI exit-code contract is the fall-back
  signal** (HP-3, ADR-0276 § Status update 2026-05-08). `_run_fast`
  in `cli.py` exits `0` for an in-tolerance recommendation, `2`
  for argument errors, and **`3`** for the OOD case where the
  proxy/verify gap exceeds `--proxy-tolerance`. The `||
  vmaf-tune recommend ...` fall-back idiom in
  `docs/usage/vmaf-tune.md` depends on the non-zero exit when
  the gap exceeds tolerance — do not silently downgrade to `0`
  or print a warning instead. The CLI is the **only** seam that
  injects `sample_extractor` (canonical-6 from probe encode +
  libvmaf JSON parse) and `encode_runner` (verify pass) into
  `fast.fast_recommend`; downstream callers that need to re-use
  the wiring import `_build_fast_sample_extractor` /
  `_build_fast_encode_runner` rather than re-implementing them.
  Output schema is the same JSON shape `recommend` and `predict`
  emit (single source of truth) plus the fast-path-specific
  `verify_vmaf` / `proxy_verify_gap` / `score_backend` fields.
- **Optuna is an optional runtime dep.** Importing it at module
  scope outside `src/vmaftune/fast.py` (or its tests) is a bug —
  the core install path stays zero-dep so corpus generation works
  on hosts that never run the fast path. The lazy-import guard in
  `fast.py` is the only correct entry point; tests that exercise
  `fast.py` use `pytest.importorskip("optuna")`.
- **Usage docs describe shipped implementation status.** The
  dedicated `docs/usage/vmaf-tune-*.md` pages and the umbrella
  `docs/usage/vmaf-tune.md` page are user-discoverable contracts,
  not backlog scratch space. When a tune surface leaves scaffold
  state, update both the standalone page and the umbrella page in
  the same PR; do not leave `(stub)`, `scaffold-only`, or stale CLI
  names on paths backed by implementation and tests.
- **Local sidecar CLI mirrors the programmatic sidecar contract
  (ADR-0394).** `vmaf-tune sidecar` is the operator surface for
  `vmaftune.sidecar.SidecarPredictor`: it must keep the same
  cache layout (`<cache>/<predictor-version>/<codec>/state.json`),
  same random host UUID posture, and same `ShotFeatures` column
  semantics as the Python API. Do not add upload, hostname-derived
  identifiers, or predictor mutation to this CLI; community pooling
  and non-linear sidecars require a separate ADR / PR.
- **Ladder uncertainty is post-hull / pre-knee.** `vmaf-tune ladder
  --with-uncertainty` must run the ADR-0279 prune/insert recipe only
  after `convex_hull()` and before `select_knees()`. Preserve corpus
  row `vmaf_interval` payloads when present; when rows are point-only,
  use the active `wide_interval_min_width` as the conservative centred
  fallback interval so point-only corpora still participate in midpoint
  insertion.
- **Saliency inference consumes RGB, not luma-replicated input
  (ADR-0430).** `saliency.compute_saliency_map()` reads yuv420p Y/U/V,
  nearest-neighbour upsamples chroma, converts BT.709 limited-range
  YUV to RGB, and only then applies ImageNet normalisation for
  `saliency_student_v1`. Do not reintroduce the old luma-only tensor
  path unless the model card and operator docs explicitly change.
- **Saliency temporal aggregation is a CLI-visible contract
  (ADR-0396 Phase 1).** `recommend-saliency --saliency-aggregator`
  exposes `mean`, `ema`, `max`, and `motion-weighted`. `mean` is the
  compatibility default; changing that default or removing a reducer
  changes user-visible encode behaviour and needs a same-PR usage-doc
  update plus an ADR-0396 follow-up.
- **`auto` non-smoke source probing is a real planning path.**
  `run_auto(smoke=False, meta_override=None)` must route source
  metadata through `_probe_source_meta`: ffprobe geometry, ffprobe
  duration, and `hdr.detect_hdr` share the same subprocess runner seam.
  Keep failures conservative (1920x1080 SDR, `duration_s=0.0`) so the
  planner can still emit an auditable JSON plan instead of depending on
  host ffprobe quirks or reintroducing `NotImplementedError`.
- **`auto` emits one selected winner.** `run_auto` must keep
  `metadata.winner` aligned with the single `cells[].selected == true`
  row whenever the winner status has a `cell_index`; evidence-failure
  plans may report `no_eligible_cells` with no selected row. The
  selector is quality/budget ordered per ADR-0428: first
  in-budget target passes, then target passes with the smallest budget
  overage, then the closest quality miss. Do not make callers infer the
  winner from cell order.
- **Fast-path proxy invariant
  ([ADR-0304](../../docs/adr/0304-vmaf-tune-fast-path-prod-wiring.md)).**
  The production proxy is **always** `fr_regressor_v2` (no smoke
  models in the production path; ADR-0291 flipped v2 to
  production). Every consumer goes through
  `vmaftune.proxy.run_proxy(...)` — a single seam over
  onnxruntime + the 14-D codec block (12-way ENCODER_VOCAB v2
  one-hot + preset_norm + crf_norm). Do not call onnxruntime
  directly from `fast.py` / `recommend.py` / `per_shot.py`; future
  probabilistic-head / ensemble migrations (ADR-0279 follow-up)
  must land in `proxy.py` so callers see no diff. Onnxruntime and
  numpy stay lazy-imported inside `proxy.py` so the corpus path
  on hosts without those deps stays zero-dep. A **single** GPU
  verify pass at `fast_recommend` end is mandatory — proxy alone
  never wins, regardless of how confident the proxy looks.
  Verification uses the existing `score_backend.select_backend`
  selector (ADR-0299); `verify_vmaf` and `proxy_verify_gap` ride
  on the result dict. When the gap exceeds the configured
  tolerance the result is flagged OOD; the operator falls back to
  the slow Phase A grid (ADR-0276 fallback contract).
  `ENCODER_VOCAB_V2` ordering is frozen by ADR-0291; reordering
  silently invalidates every shipped v2 inference.
- **`recommend` is a pure consumer of the corpus schema.** The
  `recommend` subcommand reads `vmaf_score`, `bitrate_kbps`, `crf`,
  `preset`, `encoder`, `exit_status` directly from rows produced by
  `corpus.py` (or loaded via `--from-corpus` from a previous run).
  No new schema, no parallel data path. If `SCHEMA_VERSION` bumps,
  `recommend.py`'s row-reader is one of the downstream consumers
  that must be updated in the same PR — the contract is checked by
  `test_recommend.py` against `CORPUS_ROW_KEYS`.
- **Predicate semantics are part of the user-visible contract.**
  `--target-vmaf T` returns the *smallest CRF* whose `vmaf_score >=
  T` (falling back to closest-miss when nothing clears, marked
  `(UNMET)`). `--target-bitrate KBPS` returns the row with minimum
  `|bitrate_kbps - KBPS|`, ties broken by smaller CRF. The two
  flags are mutually exclusive at the argparse layer (exit code 2
  when both are passed). Changing any of these defaults is a
  user-visible behaviour change requiring an ADR.
- **Phase F 2-pass goes through the adapter, not the driver (ADR-0333).**
  Codecs opting into 2-pass encoding declare `supports_two_pass = True`
  and override `two_pass_args(pass_number, stats_path) -> tuple[str, ...]`
  on their adapter (today: `X264Adapter`, returning
  `('-pass', str(N), '-passlogfile', str(path))`, and `X265Adapter`,
  returning `('-x265-params', f'pass={N}:stats={path}')`). The encode driver
  (`encode.py`) calls the adapter via `getattr(adapter, "supports_two_pass", False)`
  + `adapter.two_pass_args(...)` — it never branches on codec name.
  `EncodeRequest` carries `pass_number: int = 0` (0 = single-pass /
  default; 1 / 2 = pass index) and `stats_path: Path | None = None`.
  `build_ffmpeg_command` redirects pass-1 output to `-f null -` so
  the throwaway encoded bitstream isn't written. The 2-pass loop
  itself lives in `run_two_pass_encode` in `encode.py`; it
  materialises the stats file in a `tempfile.mkdtemp` (or a
  caller-supplied `scratch_dir`) and removes it (plus known encoder
  sidecars such as libx265's `.cutree`) on exit. When
  `supports_two_pass = False`, the
  driver falls back to single-pass with a stderr warning by default
  (`on_unsupported="fallback"`), or raises with
  `on_unsupported="raise"` — matches the saliency.py
  "unsupported ROI encoder, fallback to plain encode" precedent.
  Sibling codec adapters (libsvtav1, libvvenc, libaom-av1) inherit
  this seam without touching the driver — their PRs only need to override
  `supports_two_pass` + `two_pass_args` on the adapter file. NVENC's
  `-multipass` is **not** this seam (single-invocation lookahead, not
  a stats-file two-call sequence); a separate adapter contract is the
  follow-up if demand surfaces.
- **AMF preset compression is fixed (ADR-0282).** The 7-into-3 preset
  table in `codec_adapters/_amf_common.py` (`_PRESET_TO_AMF`) is the
  cross-codec axis Phase B / C consumers depend on. Do not extend
  `presets` beyond the canonical 7 names without amending ADR-0282 —
  the registry uniformity that lets the search loop ignore codec
  identity rests on every codec accepting the same preset vocabulary.
  AV1 (`av1_amf`) is RDNA3+ only; `ensure_amf_available` is the
  runtime gate.

- **Phase E ladder math is two-pass and order-sensitive.** `convex_hull`
  in `ladder.py` runs (1) Pareto filter sorted by bitrate ascending,
  vmaf descending tie-break; (2) upper-convex envelope with `cross >= 0`
  pop predicate (drops accelerating-returns interior points so the
  hull is concave / diminishing-returns end-to-end). Re-deriving the
  hull from a different starting condition is easy to get subtly
  wrong — the algorithm is pinned by `test_ladder.py` invariants
  (monotonic both axes, no domination). Don't refactor without
  re-running that suite.
- **Phase E spacing names are part of the CLI contract.** `--spacing
  log_bitrate` is the default, `--spacing vmaf` is the documented
  perceptual-spacing mode, and `uniform` is a legacy alias for `vmaf`.
  Keep the CLI choices and `ladder.select_knees()` aliases in lockstep
  so argparse cannot accept a value the library rejects.
- **Phase E sampler is pluggable; default is a 5-point CRF sweep
  (ADR-0307).** `ladder.build_ladder` accepts an explicit `sampler=`
  callback; when omitted, `_default_sampler` composes
  `corpus.iter_rows` (Phase A encode+score) with
  `recommend.pick_target_vmaf` (smallest CRF clearing the target VMAF)
  over the canonical sweep
  `DEFAULT_SAMPLER_CRF_SWEEP = (18, 23, 28, 33, 38)` at the codec
  adapter's mid-range preset (`"medium"` for libx264 / libx265 /
  libsvtav1). The 5-point sweep is the load-bearing default; do not
  widen it without an ADR-0307 follow-up — Phase E callers downstream
  size their wall-time budget against five encodes per
  (resolution, target_vmaf) cell. Callers needing a finer grid, a
  Bayesian bisect, or a precomputed corpus stream pass an explicit
  `sampler=` — that seam stays open. Tests stub `iter_rows` via
  `monkeypatch.setattr(corpus_module, "iter_rows", ...)`; the lazy
  `from .corpus import iter_rows` inside `_default_sampler` resolves
  through the patched module attribute on every call.
- **Saliency signal blend matches `vmaf-roi` (ADR-0293).**
  `saliency.py` deliberately mirrors `vmaf-roi`'s ADR-0247 signal
  blend (`offset = (2*sal − 1) * foreground_offset`, clamped to
  ±12). If `vmaf-roi`'s C-side blend changes, `saliency.py` follows
  in the same PR — the bit-for-bit equivalence is pinned by
  `tests/test_saliency.py` and is the contract that lets us swap
  the Python implementation for a `vmaf-roi` shell-out later
  without behaviour drift. The ONNX session is the second test
  seam (`session_factory` parameter) — production callers leave it
  default; tests inject a fake. Do not import `onnxruntime` at
  module top-level; lazy-load via `_import_onnxruntime` so the
  corpus subcommand and unit tests work without it installed.
- **Compare predicate is the recommend seam.** `compare.compare_codecs`
  takes a `predicate(codec, src, target_vmaf) -> RecommendResult`
  callable. The programmatic default predicate returns `ok=False`
  pointing callers at `bisect.make_bisect_predicate(target_vmaf, *,
  width=..., height=..., framerate=..., duration_s=...)` because the
  bare predicate signature does not carry source geometry. The
  `vmaf-tune compare` CLI binds that Phase B
  ([ADR-0326](../../docs/adr/0326-vmaf-tune-phase-b-bisect.md))
  predicate from its explicit geometry flags by default; the
  `--predicate-module MODULE:CALLABLE` hook is the only supported
  way to bypass real bisect. `tests/test_compare.py` injects fake
  predicates so ranking is exercised without `ffmpeg` / `vmaf`
  binaries. Do not branch on codec name inside `compare.py` — route
  every per-codec call through the predicate / adapter registry.
- **Phase G benchmark is read-only corpus analysis (ADR-0424).**
  `vmaf-tune benchmark` consumes existing Phase-A JSONL rows and must
  not call `ffmpeg`, `vmaf`, `compare.compare_codecs`, or Phase-B
  bisect. Its contract is one summary row per encoder: lowest-bitrate
  corpus point clearing `--target-vmaf`, with closest misses preserved
  as `status="unmet"`. Live encode comparisons stay in `compare`;
  offline corpus reports stay in `benchmark`.
- **Phase B bisect assumes monotone-decreasing VMAF in CRF
  ([ADR-0326](../../docs/adr/0326-vmaf-tune-phase-b-bisect.md)).**
  `vmaftune.bisect.bisect_target_vmaf` aborts with a clear error when
  two non-adjacent samples violate this contract by more than 0.5
  VMAF (looser than measurement noise). Never weaken to a fall-back
  search strategy on monotonicity violation — the contract is part
  of the public surface, and surfacing the violation is more useful
  than papering over it. Real-world content + modern codecs satisfy
  the contract; pathological exceptions are encoder bugs we want to
  see, not absorb. Subprocess seam mirrors `encode.run_encode` /
  `score.run_score`: tests inject `encode_runner` / `score_runner`
  stubs; production callers leave them `None`.
- **`COMPARE_ROW_KEYS` is the JSON / CSV output contract** for
  `vmaf-tune compare`. Same maintenance discipline as
  `CORPUS_ROW_KEYS`: adding optional keys with a default is fine,
  renaming or removing keys requires bumping the schema and updating
  every downstream consumer in the same PR.
- **Score backend selection is strict-by-default
  ([ADR-0299](../../docs/adr/0299-vmaf-tune-gpu-score.md)).**
  `score_backend.select_backend(prefer)` honours `cuda` / `vulkan` /
  `sycl` / `cpu` exactly — if the requested backend is not available,
  it raises `BackendUnavailableError` rather than silently falling back
  to CPU. Only `prefer="auto"` walks the fallback chain. Do not "fix"
  a strict-mode test that fails on a CI runner without GPU by adding
  silent fallback to `select_backend`; the strict guarantee is
  load-bearing for operator wall-clock expectations. Mock the
  `available` argument or `runner` instead.
- **`--score-backend` argparse choices are kept in sync with
  `score_backend.ALL_BACKENDS` and libvmaf's `--backend NAME`
  vocabulary ([ADR-0314](../../docs/adr/0314-vmaf-tune-score-backend-vulkan.md)).**
  Do NOT add a new value (e.g. `hip`, `metal`) to the argparse
  `choices` tuple in `cli.py` without the corresponding libvmaf-side
  wiring landing in the same release. The four current values
  (`cpu`, `cuda`, `sycl`, `vulkan`) are the exact set the libvmaf CLI
  accepts; widening the harness without widening the binary produces
  silent strict-mode failures on hosts that probe positively for the
  new value. Cross-reference: `libvmaf/tools/cli_parse.c` `--backend`
  alternation.
- **HDR detection is fail-safe to SDR (ADR-0295).** `hdr.detect_hdr`
  returns `None` on any classification ambiguity (missing file,
  ffprobe failure, malformed JSON, mismatched primaries vs.
  PQ/HLG transfer). Misclassifying SDR as HDR is the dangerous
  failure mode (would inject mismatched signaling into a Rec.709
  encode); misclassifying HDR as SDR is recoverable. Do not relax
  the BT.2020 primaries gate in `_classify_payload` without an ADR
  superseding 0261.
- **The HDR codec dispatch table is the contract for codec adapters.**
  `hdr.hdr_codec_args` dispatches per `encoder` name. When a new
  codec adapter (libx265, libsvtav1, ...) lands under
  `codec_adapters/`, it inherits the dispatch row that already
  exists; adapters do not roll their own HDR flag set.
- **`auto` records HDR args through the same dispatch table.**
  `run_auto` must call `hdr_codec_args(codec, info)` per cell when
  `meta.is_hdr` is true. A generic tuple such as
  `("-color_primaries", "bt2020", "-color_trc", "smpte2084")`
  is insufficient because x265, SVT-AV1, HEVC hardware encoders,
  AV1 hardware encoders, and VVenC use different ffmpeg flag
  families. Hardware HEVC rows force `p010le` + `main10`; hardware
  AV1 rows force `p010le`; codec-private SEI flags stay limited to
  families with stable FFmpeg knobs. Tests in
  `tests/test_auto_short_circuits.py` lock this per-codec shape.
- **`select_hdr_vmaf_model` falls back silently.** When
  `model/vmaf_hdr_*.json` is absent (current state — fork hasn't
  ported Netflix's HDR model yet), `_resolve_vmaf_model` logs a
  warning and returns the SDR model. Do not change this to raise —
  HDR encode-side correctness ships independently of HDR scoring.
- **`model/vmaf_hdr_model_card.md` is documentation, not weights**
  ([research-0089](../../docs/research/0089-hdr-vmaf-model-search.md);
  ADR-0300 status update 2026-05-09). The file is a `.md`, not a
  `.json`, so `select_hdr_vmaf_model`'s `vmaf_hdr_*.json` glob does
  **not** match it and continues to return `None`. Do not rename
  the card to `.json`, do not relax the resolver glob to also match
  `.md`, and do not synthesise placeholder weights — the SDR-fallback
  path with a one-shot warning is the deliberate Path C outcome
  until either Netflix open-sources `vmaf_hdr_v0.6.1.json` upstream
  or the fork acquires a permissively-licensed HDR-MOS-labelled
  training corpus.
- **HDR is resolved once per source in `corpus.iter_rows`** (HP-2,
  ADR-0300 status update 2026-05-08). `_resolve_hdr` returns
  `(HdrInfo | None, forced: bool)`; `hdr_codec_args` runs once and
  the resulting argv tail rides on every cell's
  `EncodeRequest.extra_params`. Do **not** re-probe ffprobe per
  cell (would burn an ffprobe per encode for a constant signal),
  and do not move the HDR-mode resolution into `_row_for` (the
  decision drives the encode argv, so it must precede the encode).
  The one-shot HDR-VMAF-model warning fires once per `iter_rows`
  invocation via the `score_model_warned` mutable flag — keep
  that semantics or operators get N spurious warnings on a
  single corpus run.
- **Cache key fields are load-bearing
  ([ADR-0298](../../docs/adr/0298-vmaf-tune-cache.md)).** The
  `cache_key()` function in `cache.py` digests six fields:
  `src_sha256`, `encoder`, `preset`, `crf`, `adapter_version`,
  `ffmpeg_version`. Dropping any one of them is a silent
  correctness bug — stale entries shadow real results when the
  adapter or ffmpeg is upgraded. The contract is asserted by
  `test_cache_key_diffs_on_each_field`. When adding a new codec
  adapter, set `adapter_version: str` on the dataclass; the
  registry `Protocol` already requires it. Bump the string when
  the adapter's argv shape, preset list, or quality range changes.
- **Cache content stays opaque.** The cache value is the parsed
  `(bitrate, vmaf, encode_time, score_time)` tuple plus an opaque
  `<key>.bin` blob. Do not bake cache contents into the JSONL row —
  the row is the canonical record, the cache is a sidecar. A cache
  hit must produce a row that is bit-identical to a cache miss
  (modulo `encode_path`, which stays empty unless `--keep-encodes`).
- **Sample-clip windows are mirrored on both sides** ([ADR-0301](../../docs/adr/0301-vmaf-tune-sample-clip.md)).
  The encode side uses FFmpeg input-side `-ss <start> -t <N>`
  (rawvideo demuxer fast-seek); the score side uses libvmaf's
  `--frame_skip_ref` / `--frame_cnt`. They MUST stay in sync — the
  centre-anchored window is computed once in `_resolve_sample_clip`
  for corpus rows or `_sample_clip_window` for Phase-B bisect and
  threaded through both `EncodeRequest` and `ScoreRequest`. Do not
  slice the reference YUV on disk into a temp file (the zero-I/O
  frame-skip path is the design); do not use output-side `-ss` (it
  decodes the full source first, defeating the speedup).
- **Coarse-to-fine search is layered on `iter_rows`, not duplicated
  (ADR-0296).** `corpus.coarse_to_fine_search()` builds two
  `dataclasses.replace(job, cells=...)` jobs (coarse + fine) and
  delegates to `iter_rows` for each. Do **not** factor out a parallel
  encoder dispatch path inside the search loop — the JSONL row schema,
  encode-failure handling, and `keep_encodes` cleanup all live in
  `iter_rows`, and forking the search loop loses them. New search
  strategies (binary, Bayesian) should follow the same pattern: build
  a list of `(preset, crf)` cells, call `iter_rows`, post-process the
  emitted rows.
- **Adapter `quality_range` is the search-space boundary, not a
  user-input gate (ADR-0296).** Widening libx264's range from `(15,
  40)` to `(0, 51)` was deliberate: the recommend / coarse-to-fine
  flow must be allowed to probe boundary CRFs to bracket the answer.
  If a future codec adapter wants to restrict the *user-visible* range
  on `--crf NNN`, do that at the CLI layer, not in `adapter.validate`.

## Phase scope

Phase A (this scaffold): grid sweep + JSONL emit, x264 only.
Phase A.5 (this PR): opt-in `fast` subcommand scaffold (proxy +
Bayesian + GPU-verify, smoke-mode validated; production loop
deferred to follow-up). Phases B–F per ADR-0237 are explicitly out
of scope here; do not add bisect / predictor / ladder / MCP code
into this tree without an ADR-0237 follow-up promoting the
corresponding phase.
Phase A (this scaffold): grid sweep + JSONL emit. Wired codecs:
`libx264` (initial scaffold) and `libx265` (ADR-0288). Further codecs
(`libsvtav1`, `libvpx-vp9`, `libvvenc`, `libaom`, neural-codec extras)
are one-file adapter additions under `codec_adapters/` per ADR-0237.
Phases B–F per ADR-0237 are explicitly out of scope here; do not add
bisect / predictor / ladder / MCP code into this tree without an
ADR-0237 follow-up promoting the corresponding phase.
  wired `libx264`; ADR-0281 widened the registry with the three
  Intel QSV adapters (`h264_qsv`, `hevc_qsv`, `av1_qsv`). The
  search loop must use the registry uniformly. Do not branch on
  codec name in `corpus.py` / `encode.py` / `score.py`; route via
  the adapter. New codecs are one-file additions under
  `codec_adapters/`.
- **The QSV adapters share `_qsv_common.py`.** Three encoders with
  identical parameter shape (preset vocabulary, ICQ
  `global_quality` window) is a deliberate exception to the
  "one file per codec, nothing shared" Phase A convention. Per
  ADR-0281, future codec families that share parameter shape
  (NVENC's three encoders, AMF's three encoders, VideoToolbox's
  two H.264 + HEVC encoders) follow the same pattern: one
  `_<family>_common.py` private module, thin dataclass adapters.
  Single-codec families stay flat.
- **Apple VideoToolbox adapters share `_videotoolbox_common.py`
  (ADR-0283 + ADR-0283 *Status update 2026-05-09*).** Three
  encoders (`h264_videotoolbox`, `hevc_videotoolbox`,
  `prores_videotoolbox`) reuse the nine-name preset → `-realtime`
  boolean mapping. H.264 and HEVC share a single `-q:v` 0..100
  quality knob (higher = better; `invert_quality=False`). ProRes
  uses `-profile:v` instead — it is a fixed-rate intermediate
  codec, so the harness's `crf` slot carries the integer tier id
  (0=`proxy` → 5=`xq`); the adapter has its own validator
  `validate_prores_videotoolbox()` and an integer-id-to-FFmpeg-alias
  helper `prores_profile_name()`. Per the codec-adapter contract,
  the search loop never branches on adapter identity — it consumes
  `quality_range` + `ffmpeg_codec_args(...)` uniformly. AV1
  hardware encoding is intentionally absent — Apple Silicon has
  no AV1 hardware encoder block as of 2026 and FFmpeg exposes no
  `av1_videotoolbox`. Tests mock `subprocess.run`; the suite runs
  on Linux CI without macOS. End-to-end VT exercise is left to
  contributors with macOS + VideoToolbox available locally
  (ProRes additionally requires M1 Pro / Max / Ultra or later —
  Intel Macs with T2 do not have the ProRes hardware block).
- **The encode pipeline (`encode.py`) is still x264-CRF-tied.**
  ADR-0281 added the QSV adapter classes but did not widen
  `build_ffmpeg_command` to dispatch on `adapter.quality_knob`.
  Until that follow-up lands, the QSV adapters validate
  `(preset, global_quality)` correctly but the harness will not
  yet successfully drive a QSV encode end-to-end.
- **Subprocess boundary is the test seam.** `encode.run_encode`,
  `score.run_score`, and the QSV `ffmpeg_supports_encoder` probe
  accept a `runner` argument that defaults to `subprocess.run`.
  Tests inject a fake; production callers leave it default. Do
  not reach for `os.system` / `popen` shortcuts —
  `tests/test_corpus.py` and `tests/test_codec_adapter_qsv.py`
  will silently stop covering the path.

## Phase scope

Phase A (the original scaffold): grid sweep + JSONL emit, x264
only. ADR-0281 added the three QSV codec adapters as a one-file
extension off the registry; the encode-pipeline widening that
makes them functional is itself a separate Phase A follow-up.
Phases B–F per ADR-0237 (bisect / predictor / ladder / MCP) remain
explicitly out of scope here; do not add that code into this tree
without an ADR-0237 follow-up promoting the corresponding phase.
Phase A (corpus generation): grid sweep + JSONL emit, x264 only.
Phase D (per-shot CRF tuning, ADR-0276): orchestrates shot detection
(via the C-side `vmaf-perShot` binary, ADR-0222), extracts each shot
to raw YUV, and binds the pluggable per-shot CRF predicate to Phase B's
real bisect backend by default. The CLI deliberately stops before
running the final segment encodes — it emits an FFmpeg encoding plan as
JSON plus an optional shell script. `--predicate-module` remains the
advanced custom/test escape hatch; it is no longer the production path.

Phases B (target-VMAF bisect), C (per-title CRF predictor), E
(Pareto ABR ladder) and F (MCP tools) per ADR-0237 are explicitly
out of scope here; do not add bisect / predictor / ladder / MCP code
into this tree without an ADR-0237 follow-up promoting the
corresponding phase.

## Phase D rebase-sensitive invariants

- **Predicate signature is the Phase B contract.** The
  ``PredicateFn`` type alias in ``per_shot.py`` is
  ``(Shot, target_vmaf: float, encoder: str) -> (crf: int,
  measured_or_predicted_vmaf: float)``. The CLI adapter around
  Phase-B bisect must conform to this signature; widening the return
  tuple is a coordinated change that bumps the public-API surface
  across both modules in the same PR.
- **CLI default is real per-shot bisect.** `vmaf-tune tune-per-shot`
  must call the Phase-B bisect backend unless
  `--predicate-module MODULE:CALLABLE` is explicitly supplied. Do not
  reintroduce the adapter-default CRF as CLI behaviour; that fallback
  exists only for library dry runs that call `tune_per_shot()` without
  a predicate.
- **Bisect inputs are temporary raw YUV shots.** `bisect_target_vmaf`
  expects raw YUV geometry, so the CLI extracts each detected
  half-open shot range to a temporary raw-YUV file before calling it.
  Raw `.yuv` / `.raw` sources are opened with explicit rawvideo
  demuxer flags (`--width`, `--height`, `--pix-fmt`, `--framerate`);
  container and Y4M sources are left to FFmpeg's demuxer.
- **Shot ranges are half-open inside Python.** The C-side
  ``vmaf-perShot`` JSON/CSV sidecar uses inclusive ``end_frame``;
  ``per_shot.py`` normalises into ``[start_frame, end_frame)`` at
  the parse boundary. ``Shot.length`` and the
  ``-frames:v`` arg in ``_segment_command`` both depend on the
  half-open form. Do not "round-trip back to inclusive" — every
  downstream consumer assumes the half-open form.
- **The ``vmaf-perShot`` binary surface is the canonical detector.**
  Do not add a parallel ONNX-Runtime-from-Python detector path.
  When TransNet V2 is hot-pathed (e.g. Phase E ladder generation
  re-running detection), extend ``detect_shots`` to call
  ``vmaf-perShot`` once and cache, not to bypass the binary.
- **Shot detection runs once per source, never per cell.** The
  corpus driver (``corpus._resolve_shot_metadata``) calls
  ``_detect_shots_with_status`` at the top of ``iter_rows`` and
  passes the resulting ``ShotMetadata`` down to every
  ``(preset, crf)`` row via ``_row_for``. Moving the call inside
  the cell loop roughly doubles corpus wall time on TransNet-V2.
  ``_detect_shots_with_status`` is the only API that returns the
  ``(shots, ok)`` tuple needed to distinguish a real single-shot
  source from a "binary failed" fallback — the public
  ``detect_shots`` shape cannot carry that flag.
- **HDR VMAF model resolution goes through
  ``hdr.select_hdr_vmaf_model``.** The canonical filename is
  ``vmaf_hdr_v0.6.1.json`` (Netflix's research-artefact name).
  Route lookups through ``hdr_model_name_for(transfer)`` so a
  future Dolby-Vision-specific model entry is one
  dispatch-table row away. The "HDR model not shipped" warning
  is single-shot per process; clear it from tests via
  ``hdr.reset_hdr_model_warning()``.
Phase A (this scaffold): grid sweep + JSONL emit. Codecs wired so
far: `libx264` (ADR-0237) and `libsvtav1` (ADR-0294). Phases B–F per
ADR-0237 are explicitly out of scope here; do not add bisect /
predictor / ladder / MCP code into this tree without an ADR-0237
follow-up promoting the corresponding phase.
Phase A (the corpus scaffold): grid sweep + JSONL emit, x264 only.
Phase E (this scaffold): per-title bitrate-ladder generator (Pareto
hull + manifest emit), sampler-pluggable, smoke-only until Phase B
merges. Phases B / C / D / F per ADR-0237 are explicitly out of scope
here; do not add bisect / predictor / per-shot / MCP code into this
tree without an ADR-0237 follow-up promoting the corresponding phase.

- **The seven F.2 short-circuit predicates in ``auto.py`` are an
  ordered tuple, not a set.** ``SHORT_CIRCUIT_PREDICATES`` declares
  ``ShortCircuit.LADDER_SINGLE_RUNG`` first and
  ``ShortCircuit.SKIP_PER_SHOT`` last; the order is part of the
  public contract because tests assert determinism across
  `evaluate_short_circuits` invocations and the JSON schema records
  the canonical-order list under ``plan.metadata.short_circuits``.
  Adding an eighth short-circuit (F.3+ follow-ups) appends to the
  tuple; never insert in the middle. The Phase D thresholds
  (`PHASE_D_DURATION_GATE_S = 300.0` and
  `PHASE_D_SHOT_VARIANCE_GATE = 0.15`) are placeholders pending F.3
  empirical fit — change them via an ADR-0325 follow-up, not a
  drive-by tweak. See [ADR-0325](../../docs/adr/0325-vmaf-tune-phase-f-auto.md).

- **F.3 confidence-aware thresholds are corpus-derived; do not
  hand-pick.** `DEFAULT_TIGHT_INTERVAL_MAX_WIDTH = 2.0` and
  `DEFAULT_WIDE_INTERVAL_MIN_WIDTH = 5.0` in `auto.py` are an
  emergency floor (Research-0067), not a target. The production
  thresholds load from a calibration JSON sidecar emitted by the
  conformal-VQA pipeline (ADR-0279) — keys
  `tight_interval_max_width` and `wide_interval_min_width`.
  `load_confidence_thresholds` falls back to the defaults with a
  one-line WARNING when no sidecar is found; do not silence that
  warning, and do not "tune" the defaults to make a failing
  integration test pass. The fix for surprising cell escalations on
  real data is a recalibration PR, not a threshold loosening here
  (CLAUDE.md `feedback_no_test_weakening`). The decision helper
  `_confidence_aware_escalation` is a pure function of
  `(verdict, interval_width, thresholds)` so it stays trivially
  unit-testable; keep it pure when extending the decision table.
  `run_auto` must pass the recipe-adjusted `effective_thresholds`
  from `_apply_recipe_override` into every F.3 decision and into
  `plan.metadata.confidence_thresholds`; computing the adjusted
  value and then falling back to `ConfidenceThresholds()` is a
  user-visible planning bug.

- **F.4 recipe overrides are read-only factories, not literal
  dicts.** `_CONTENT_RECIPE_TABLE` in `auto.py` stores **callables**
  (`_animation_recipe`, `_screen_content_recipe`,
  `_live_action_hdr_recipe`, `_ugc_recipe`, `_empty_recipe`); every
  call returns a fresh dict so a caller mutating the return value
  cannot leak the mutation into the next `run_auto` invocation. Tests
  in `tests/test_auto_recipe_overrides.py` assert this invariant
  explicitly. Adding a new content class means adding a factory
  function and a `RECIPE_CLASS_<NAME>` constant; never inline a
  literal dict into the table or mutate one in place. The four
  override keys (`tight_interval_max_width`, `force_single_rung`,
  `saliency_intensity`, `target_vmaf_offset`) are the only keys the
  driver honours — `get_recipe_for_class` filters by the
  `_RECIPE_KEYS` allowlist as defence-in-depth. Every threshold value
  shipped at F.4 is `[provisional, calibrate against real corpus in
  F.5]`; do not promote a placeholder to "calibrated" in a drive-by
  edit. Per memory `feedback_no_test_weakening`,
  `target_vmaf_offset` shifts only the predictor's effective target;
  the input `--target-vmaf` (the gate that ships models) is
  preserved verbatim in `plan.metadata.target_vmaf`. See
  [ADR-0325](../../docs/adr/0325-vmaf-tune-phase-f-auto.md) §F.4.

## ADR-0332 invariants (encoder-internal stats capture)

- The corpus row schema is at v3; new columns added to
  ``CORPUS_ROW_KEYS`` and ``SCHEMA_VERSION`` must keep the v3 ten
  ``enc_internal_*`` columns positionally stable so v2 readers see a
  zero rather than a missing key. Coordinates with ADR-0302.
- Every adapter in ``codec_adapters/`` must declare
  ``supports_encoder_stats: bool`` (no Protocol default). x264 / x265
  set True; everything else False until a codec-specific parser
  lands. x265's ``q-aq`` and ``icu`` / ``pcu`` / ``scu`` pass-1 aliases
  are intentionally normalised in ``encoder_stats.py`` so corpus rows
  keep the same ten ``enc_internal_*`` columns as x264.
- ``run_encode_with_stats`` doubles per-encode wall-clock on opt-in
  adapters by design. Do not collapse the pass-1 + pass-2 calls into
  one — the encoder won't emit a parseable stats file outside
  ``-pass 1`` mode.


## Sidecar (ADR-0325) rebase-sensitive invariants

- **`FEATURE_DIM = 14` and the column order in
  `sidecar._feature_vector` are the load-bearing pin** for the
  online-ridge state. Adding or reordering features without
  bumping `SIDECAR_SCHEMA_VERSION` will silently align saved
  weights to the wrong column on load. The leading `1.0` bias /
  intercept term must stay at column 0; it is what lets the
  ridge fit absorb a constant offset between predicted and
  observed VMAF.
- **`SidecarConfig.predictor_version` is the contract that
  invalidates stale corrections** when the shipped predictor
  upgrades. Tag mismatch on `SidecarModel.from_dict` raises and
  the caller (`SidecarModel.load`) falls back to a cold-start
  model. Do not catch the mismatch and "rescale" — a stale
  correction trained against the previous predictor's residuals
  is worse than no correction.
- **The host UUID is anonymous by construction.** It is generated
  by `secrets.token_hex(16)` on first install and persisted at
  `<cache_dir>/host-uuid`. **Never** swap it for `uuid.getnode()`
  / `socket.gethostname()` / `/etc/machine-id` / CPUID — that
  would re-identify the operator and break the privacy
  precondition for the future opt-in upload PR (ADR-0325
  §Future work).
- **Sidecar state is local-only by default.** The harness has no
  upload code path. Adding one requires the dedicated opt-in
  upload ADR + signing chain spelled out in ADR-0325 §Future
  work — do not slip a network call into `SidecarPredictor` or
  any of its callers without that ADR landing first.

## Predictor stub-models policy (ADR-0325)

The fork ships one `model/predictor_<codec>.onnx` per codec adapter.
As of 2026-05-14 the NVENC / QSV predictors (`h264_nvenc`,
`hevc_nvenc`, `av1_nvenc`, `h264_qsv`, `hevc_qsv`, `av1_qsv`) are
real-corpus retrains from `runs/phase_a/full_grid/comprehensive.jsonl`
and their cards carry `corpus.kind: real-N=<rows>`. Software and AMF
predictors remain synthetic stubs until matching real corpora exist.
The trainer
(`tools/vmaf-tune/src/vmaftune/predictor_train.py`) sources its
`CODECS` tuple from `predictor._DEFAULT_COEFFS` so the two stay
single-source. When a new codec adapter is added (e.g. a future
`vp9_qsv` row in `_DEFAULT_COEFFS`), the same PR must:

1. Re-run `python3 -m vmaftune.predictor_train --output-dir model`
   to produce the matching `predictor_<codec>.onnx` + card.
2. Commit the new ONNX bytes — the shipped-model smoke test
   parameterises over `CODECS` and fails if a coefficient row has
   no shipped artefact.
3. Refresh the model card's `corpus.kind` line on every retrain
   (the trainer does this automatically; review the diff).

Stub models are explicitly **not** for production CRF picks. The
synthetic target *is* the analytical fallback, so PLCC / SROCC
numbers in stub cards are artificially high. Real-corpus retrains
follow the same trainer entry point with `--corpus path/to/file.jsonl`
or `--corpus path/to/corpus-dir/` and produce honest metrics. Directory
corpus inputs are recursive and sorted so `.workingdir2/corpus_run/`
trains deterministically without a manual concatenation step. Keep that
directory handling reachable from both `train_all_codecs()` and the CLI;
file-only `is_file()` guards above `load_corpus()` silently turn real
corpus directories back into synthetic stubs. The loader accepts both
canonical `encoder` / `crf` / `vmaf_score` /
`bitrate_kbps` rows and historical hardware-sweep `codec` / `q` /
`vmaf` / `actual_kbps` aliases; do not reintroduce external conversion
scripts for those local corpora.
