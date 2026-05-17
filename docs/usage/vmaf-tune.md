# `vmaf-tune` — quality-aware encode automation harness

`vmaf-tune` is a fork-added Python tool
([ADR-0237](../adr/0237-quality-aware-encode-automation.md),
[Research-0044](../research/0044-quality-aware-encode-automation.md))
that drives FFmpeg over an encoder parameter grid, scores each encode
with [`vmaf`](cli.md), and emits a JSONL corpus of
`(source, encoder, params, bitrate, vmaf)` rows.

## Subcommands at a glance

| Subcommand            | Purpose                                                | ADR / research                                                                                       |
|-----------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `corpus`              | Multi-codec encoder grid sweep + scoring               | [ADR-0237](../adr/0237-quality-aware-encode-automation.md)                                           |
| `recommend`           | Target-VMAF / target-bitrate predicate (Phase B)       | [Research-0061](../research/0061-vmaf-tune-capability-audit.md), buckets 4+5                         |
| `tune-per-shot`       | Per-shot CRF zones                                     | [ADR-0392](../adr/0392-vmaf-tune-phase-d-per-shot.md)                                                |
| `recommend-saliency`  | Saliency-aware ROI tuning                              | [ADR-0287](../adr/0287-vmaf-tiny-v5-corpus-expansion.md) ecosystem; consumes `vmaf-roi` sidecars     |
| `ladder`              | Per-title bitrate ladder (Pareto ABR)                  | [ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md)                                          |
| `fast`                | Predicted-CRF fast path                                | [ADR-0276](../adr/0276-vmaf-tune-fast-path.md)                                                       |
| `hdr`                 | HDR-aware tuning + HDR-VMAF scoring                    | (PR #434, bucket #9)                                                                                 |
| `compare`             | Apples-to-apples codec comparison at matched VMAF      | (PR #435)                                                                                            |
| `benchmark`           | Offline cross-codec report from an existing JSONL      | [ADR-0424](../adr/0424-vmaf-tune-corpus-benchmark.md)                                                |
| `sidecar`             | Local predictor bias-correction training / inspection  | [ADR-0394](../adr/0394-local-sidecar-training.md)                                                    |

## Codec adapters

18 adapters under `tools/vmaf-tune/src/vmaftune/codec_adapters/`:

| Family | Software           | NVIDIA NVENC                                                                                               | Intel QSV     | AMD AMF       | Apple VideoToolbox    |
|--------|--------------------|------------------------------------------------------------------------------------------------------------|---------------|---------------|-----------------------|
| AV1    | `libaom`, `svtav1` | `av1_nvenc` ([ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md))                                          | `av1_qsv`     | `av1_amf`     | —                     |
| H.264  | `x264` (Phase A)   | `h264_nvenc`                                                                                               | `h264_qsv`    | `h264_amf`    | `h264_videotoolbox`   |
| HEVC   | `x265` ([ADR-0288](../adr/0288-vmaf-tune-codec-adapter-x265.md)) | `hevc_nvenc`                                                              | `hevc_qsv`    | `hevc_amf`    | `hevc_videotoolbox`   |
| VP9    | `libvpx-vp9`       | —                                                                                                          | —             | —             | —                     |
| VVC    | `vvenc`            | —                                                                                                          | —             | —             | —                     |

## Pipeline

```text
ref.yuv ──► vmaf-tune corpus ──► encode (libx264) ──► vmaf score ──► corpus.jsonl
ref.yuv ──► vmaf-tune corpus ──► encode (libx264|libx265) ──► vmaf score ──► corpus.jsonl
              │
              └─► encodes are written to --encode-dir, deleted post-score
                  unless --keep-encodes

corpus.jsonl ──► vmaf-tune recommend --target-vmaf T   ──► smallest CRF >= T
                                    --target-bitrate B ──► CRF closest to B
corpus.jsonl ──► vmaf-tune benchmark --target-vmaf T   ──► encoder ranking
```

## Install

The tool ships under `tools/vmaf-tune/` as a standalone Python package.
Phase A has zero runtime dependencies beyond the standard library.

```shell
pip install -e tools/vmaf-tune
# or run directly from the checkout:
python tools/vmaf-tune/vmaf-tune --help
```

External binaries required at runtime:

- `ffmpeg` with `--enable-libx264` (and `--enable-libsvtav1` if you
  pass `--encoder libsvtav1`) on `PATH` (or `--ffmpeg-bin`).
- `vmaf` (this fork's CLI, built via meson) on `PATH` (or `--vmaf-bin`).

## Predictor Training Corpora

`vmaftune.predictor_train` trains the per-codec ONNX predictors consumed
by the fast, per-shot, ladder, and auto paths. `--corpus` accepts either
a single Phase-A JSONL file or a directory of JSONL shards; directory
inputs are scanned recursively in sorted order so the trainer can consume
`.workingdir2/corpus_run/` directly:

```shell
python -m vmaftune.predictor_train \
    --corpus .workingdir2/corpus_run \
    --codec libx264 \
    --output-dir .workingdir2/predictor-real
```

Rows are filtered per codec after schema aliases are normalised. The
trainer accepts both current `corpus.py` rows (`encoder`, `crf`,
`vmaf_score`, `bitrate_kbps`) and older hardware-sweep rows (`codec`,
`q`/`cq`, `vmaf`, `actual_kbps`). If a codec has no usable rows, that
codec still falls back to the documented synthetic-stub corpus and the
model card records `corpus.kind: synthetic-stub-*`. Directory inputs
are passed through the same loader as single files; a codec with rows in
any shard records `corpus.kind: real-N=<rows>`.

## Local Sidecar Bias Correction

`vmaf-tune sidecar` exposes the local sidecar model from
[`docs/ai/local-sidecar-training.md`](../ai/local-sidecar-training.md)
as an operator CLI. It never uploads captures and never mutates the
shipped predictor; it stores only the online-ridge correction under
`${XDG_CACHE_HOME:-~/.cache}/vmaf-tune/sidecar/`.

Inspect the current sidecar state:

```shell
vmaf-tune sidecar status --codec libx264 --json
```

Record one observed encode. `features.json` may be either a flat
`ShotFeatures` object or `{ "features": { ... } }`; the required
fields are `probe_bitrate_kbps`, `probe_i_frame_avg_bytes`,
`probe_p_frame_avg_bytes`, and `probe_b_frame_avg_bytes`.

```shell
vmaf-tune sidecar record \
    --codec libx264 \
    --features-json features.json \
    --crf 28 \
    --observed-vmaf 94.2
```

Batch training accepts JSONL with one row per observed encode:

```json
{"features":{"probe_bitrate_kbps":3000,"probe_i_frame_avg_bytes":10000,"probe_p_frame_avg_bytes":2000,"probe_b_frame_avg_bytes":1000,"width":1920,"height":1080,"fps":24},"crf":28,"observed_vmaf":94.2}
```

```shell
vmaf-tune sidecar batch-record --codec libx264 --captures-jsonl captures.jsonl
```

Prediction reports the bare predictor score, sidecar correction, and
clamped final score:

```shell
vmaf-tune sidecar predict --codec libx264 --features-json features.json --crf 28 --json
```

## Quick start

Generate a 6-cell corpus row over `(medium, slow) × (22, 28, 34)` for one
1080p source clip:

```shell
vmaf-tune corpus \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --preset medium --preset slow \
    --crf 22 --crf 28 --crf 34 \
    --output corpus.jsonl
```

`--source` is repeatable — pass one flag per source clip. The grid is the
Cartesian product of `--preset × --crf`.

### SVT-AV1 example (ADR-0278)

The `libsvtav1` adapter accepts the same x264-style preset names — they
are translated to SVT-AV1's integer presets internally. AV1 CRF values
live in `0..63`; the Phase A informative window is `(20, 50)`:

```shell
vmaf-tune corpus \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --encoder libsvtav1 \
    --preset medium --preset slow \
    --crf 28 --crf 35 --crf 42 \
    --output corpus_av1.jsonl
```

The corpus row records the human-readable preset name (`"medium"`),
while the FFmpeg argv carries the integer SVT-AV1 expects
(`-preset 7`).

## Codec adapter parameter ranges

Each adapter declares its own quality knob, range, and preset
vocabulary. The harness validates `(preset, crf)` against the adapter
before invoking FFmpeg.

| Encoder | CRF (absolute) | Phase A CRF window | Default CRF | Presets |
| --- | --- | --- | --- | --- |
| `libx264` | `0..51` | `(15, 40)` | `23` | `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow` |
| `libsvtav1` | `0..63` | `(20, 50)` | `35` | `placebo`, `slowest`, `slower`, `slow`, `medium`, `fast`, `faster`, `veryfast` |
| `libvpx-vp9` | `0..63` | `(0, 63)` | `32` | `placebo`, `slowest`, `slower`, `slow`, `medium`, `fast`, `faster`, `veryfast`, `superfast`, `ultrafast` |

### SVT-AV1 preset name -> integer mapping

SVT-AV1 uses integer presets `0..13` (`0` = slowest / best,
`13` = fastest). The harness maps the x264-style names below to AV1
integers so the corpus row schema is codec-independent:

| Name | SVT-AV1 integer | Notes |
| --- | --- | --- |
| `placebo` | `0` | Slowest; research-grade only. |
| `slowest` | `1` | |
| `slower` | `3` | |
| `slow` | `5` | |
| `medium` | `7` | SVT-AV1 default. |
| `fast` | `9` | |
| `faster` | `11` | |
| `veryfast` | `13` | Fastest. |

The mapping is closed and order-stable; see
[ADR-0294](../adr/0294-vmaf-tune-codec-adapter-svtav1.md).

## CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--source PATH` | — | Required. Repeatable for multi-source sweeps. |
| `--width / --height` | — | Required. Source resolution. |
| `--pix-fmt PFMT` | `yuv420p` | Forwarded to ffmpeg `-pix_fmt`. |
| `--framerate F` | `24.0` | Source framerate. |
| `--duration S` | `0` | Source duration in seconds (used for bitrate calc). |
| `--encoder NAME` | `libx264` | One of `libx264`, `libx265`. |
| `--encoder NAME` | `libx264` | One of `libx264`, `h264_amf`, `hevc_amf`, `av1_amf`. |
| `--preset P` | — | Required. Repeatable. x264 preset name. |
| `--crf N` | — | Required. Repeatable. x264 CRF integer. |
| `--encoder NAME` | `libx264` | Currently wired: `libx264`, `libsvtav1` (ADR-0294). |
| `--preset P` | — | Required. Repeatable. Preset name (see codec table below). |
| `--crf N` | — | Required. Repeatable. CRF integer (range varies by codec). |
| `--output PATH` | `corpus.jsonl` | JSONL destination. |
| `--encode-dir PATH` | `.workingdir2/encodes` | Scratch dir; gitignored by convention. |
| `--keep-encodes` | off | Retain encoded files after scoring. |
| `--vmaf-model NAME` | `vmaf_v0.6.1` | Forwarded to `vmaf --model`. Only used when `--no-resolution-aware` is set; otherwise auto-picked per encode resolution (see "Resolution-aware mode" below). |
| `--resolution-aware` / `--no-resolution-aware` | on | Auto-pick the VMAF model per encode resolution. Default on. |
| `--ffmpeg-bin PATH` | `ffmpeg` | Override the ffmpeg binary. |
| `--ffprobe-bin PATH` | `ffprobe` | Override the ffprobe binary (used for HDR detection). |
| `--vmaf-bin PATH` | `vmaf` | Override the vmaf binary. |
| `--score-backend NAME` | `auto` | libvmaf scoring backend — `auto\|cpu\|cuda\|sycl\|vulkan`. See below. |
| `--no-source-hash` | off | Skip `src_sha256` (faster on large YUVs; loses provenance). |
| `--auto-hdr` | (default) | Probe each source via ffprobe; inject HDR flags + HDR model when PQ / HLG signaling is detected. |
| `--force-sdr` | off | Treat all sources as SDR; skip HDR detection. |
| `--force-hdr-pq` | off | Treat all sources as HDR PQ (SMPTE-2084) without probing. Useful for raw YUV refs that ffprobe cannot read color metadata from. |
| `--force-hdr-hlg` | off | Treat all sources as HDR HLG (ARIB STD-B67) without probing. |
| `--two-pass` | off | Phase F (ADR-0333). Run a 2-pass encode for codecs whose adapter sets `supports_two_pass = True` (today: `libx264`, `libx265`, `libvpx-vp9`). Codecs without 2-pass support fall back to single-pass with a stderr warning. Doubles encode wall time. |

## Resolution-aware mode

VMAF is a resolution-aware metric: the fork ships two production-grade
pooled-mean models — `vmaf_v0.6.1` (trained on a 1080p viewing setup)
and `vmaf_4k_v0.6.1` (re-fit for a 4K display). Scoring 4K content
against the 1080p model under-counts spatial detail; scoring 1080p
content against the 4K model over-counts coding artefacts. The bias is
several VMAF points either way — large enough to poison a
mixed-resolution ABR-ladder corpus.

When `--resolution-aware` is on (the default), `vmaf-tune` picks the
model per encode according to a height-only rule that mirrors Netflix's
published guidance:

| Encode height | Selected model |
| --- | --- |
| `≥ 2160` (UHD-1 and up) | `vmaf_4k_v0.6.1` |
| `< 2160` (everything else, including 1440p / 720p / SD) | `vmaf_v0.6.1` |

The fork has no 720p / 1440p / SD model — `vmaf_v0.6.1` is the
canonical fallback for all sub-2160p content (matches Netflix's
recommendation).

The emitted JSONL row's `vmaf_model` field now records the **effective**
model used for *that row*, not the global `--vmaf-model` opt. Mixed-ladder
corpora legitimately contain multiple distinct `vmaf_model` values across
rows; downstream consumers should group / filter by `vmaf_model` rather
than assume a constant.

To force a single model regardless of resolution (e.g. to reproduce a
legacy single-model corpus), pass `--no-resolution-aware`:

```shell
vmaf-tune corpus \
    --source ref_4k.yuv --width 3840 --height 2160 \
    --preset medium --crf 23 \
    --no-resolution-aware --vmaf-model vmaf_v0.6.1 \
    --output corpus.jsonl
```

The Python API exposes the decision rule directly for callers that
need to consult it outside the corpus loop:

```python
from vmaftune.resolution import (
    select_vmaf_model_version,    # str: "vmaf_v0.6.1" or "vmaf_4k_v0.6.1"
    select_vmaf_model,            # Path: in-tree model JSON file
    crf_offset_for_resolution,    # int: -2 / 0 / +2 / +4 by resolution band
)

assert select_vmaf_model_version(3840, 2160) == "vmaf_4k_v0.6.1"
assert select_vmaf_model_version(1920, 1080) == "vmaf_v0.6.1"
```

`crf_offset_for_resolution` returns a small integer offset that the
future search layer (Phase B target-VMAF bisect) can apply when seeding
bisect bounds across an ABR ladder. The shipped defaults are
codec-agnostic and conservative; Phase B/C/D will learn per-codec
offsets from real corpora and override them via the same function
signature. See
[ADR-0289](../adr/0289-vmaf-tune-resolution-aware.md) and
[Research-0054](../research/0064-vmaf-tune-resolution-aware.md) for the
full rationale.

## GPU scoring backend

Per [ADR-0299](../adr/0299-vmaf-tune-gpu-score.md), `vmaf-tune corpus`
forwards a `--backend NAME` argument to the libvmaf CLI so scoring runs
on a GPU when one is present. CPU VMAF runs at ~1–2 fps on 1080p; the
CUDA / Vulkan / SYCL backends shipped with this fork
([ADR-0127](../adr/0127-vulkan-compute-backend.md),
[ADR-0175](../adr/0175-vulkan-backend-scaffold.md),
[ADR-0186](../adr/0186-vulkan-image-import-impl.md)) deliver
**10–30× speedup** on the score axis.

### Modes

| Value | Behaviour |
| --- | --- |
| `auto` (default) | Detect what the local `vmaf` binary supports + what the host hardware advertises, then pick the fastest available backend in the order `cuda → vulkan → sycl → cpu`. Falls back to CPU silently only because no GPU was found. |
| `cuda` / `vulkan` / `sycl` | Strict mode. Errors out with `BackendUnavailableError` if the local `vmaf` binary does not support the requested backend or the host hardware is missing. **No silent downgrade to CPU** — that would mask hardware/build mismatches and lie about wall-clock expectations. |
| `cpu` | Force the CPU path. Useful for reproducibility against the Netflix golden-data gate or to bypass a known-bad GPU driver day. |

### Detection heuristics

`vmaf-tune` inspects the `vmaf --help` output to learn which backends
the binary advertises (the CLI prints a line of the form
`--backend $name: ...auto|cpu|cuda|sycl|vulkan`), then runs cheap
hardware probes:

- **CUDA**: `nvidia-smi -L` returns at least one `GPU` line.
- **Vulkan**: `vulkaninfo --summary` reports a `deviceName`.
- **SYCL**: `sycl-ls` lists at least one `:gpu` device.

Missing tools degrade to "backend not available" — they never raise
hard errors. CPU is always considered available even if the help line
is missing.

### Wall-clock expectation (60 s 1080p source, indicative)

| Score backend | Hardware | Wall-clock | Throughput |
| --- | --- | --- | --- |
| `cpu` | AVX2 desktop CPU | ~600–1200 s | ~1.2–2.5 fps |
| `cuda` | RTX 30/40-class GPU | ~50–120 s | ~12–30 fps |
| `vulkan` | RTX 30/40 / RDNA3 | ~60–140 s | ~10–25 fps |
| `sycl` | Intel Arc / Iris Xe | ~80–180 s | ~8–18 fps |

Numbers are order-of-magnitude only; exact figures depend on the
specific feature extractors enabled by the model
(`vmaf_v0.6.1` versus tiny-AI variants), whether `--keep-encodes` is
on, and host I/O bandwidth. Cross-backend numerical parity is
guaranteed to `places=4` by the
[ADR-0214](../adr/0214-gpu-parity-ci-gate.md) CI gate.

### Examples

```shell
# Default — auto-pick the fastest backend.
vmaf-tune corpus --source ref.yuv --width 1920 --height 1080 \
    --preset medium --crf 22 --crf 28

# Force CUDA. Errors out clearly if /opt/cuda is missing or the
# vmaf binary was built without CUDA support.
vmaf-tune corpus --source ref.yuv --width 1920 --height 1080 \
    --preset medium --crf 22 --score-backend cuda

# Pin CPU for reproducibility against the Netflix golden gate.
vmaf-tune corpus --source ref.yuv --width 1920 --height 1080 \
    --preset medium --crf 22 --score-backend cpu
```

### Vulkan score backend (`--score-backend=vulkan`)

Per [ADR-0314](../adr/0314-vmaf-tune-score-backend-vulkan.md), the
`vulkan` value of `--score-backend` is the **vendor-neutral** GPU
score path. Use it whenever the host is not an NVIDIA box (or when
the NVIDIA box has no CUDA toolkit installed).

#### Supported platforms

| Host | Driver | Status |
| --- | --- | --- |
| Linux + AMD RDNA2/RDNA3 | Mesa RADV | Production. |
| Linux + Intel Arc / Iris Xe | Mesa anv | Production. |
| Linux + NVIDIA | Mesa NVK or proprietary `nvidia` driver | Production; coexists with `--score-backend=cuda`. |
| Linux CI / no GPU | Mesa lavapipe (software rasteriser) | Slow but functional — the cross-backend parity gate ([ADR-0214](../adr/0214-gpu-parity-ci-gate.md)) runs on lavapipe. |
| macOS (Apple Silicon + Intel) | MoltenVK 1.2 layered over Metal | Functional. |
| Windows | Vendor-supplied Vulkan ICD | Functional; not gated in CI yet. |

The libvmaf binary needs to be built with `-Denable_vulkan=true`
(default in fork release artefacts). The `vulkan` value will fail
strict-mode validation otherwise — `vmaf --help` will not advertise
`vulkan` in its `--backend` alternation.

#### Verifying Vulkan availability

`vmaf-tune` runs the same probe libvmaf does — `vulkaninfo --summary`
must succeed and report at least one `deviceName`:

```shell
$ vulkaninfo --summary | grep deviceName
        deviceName        = AMD Radeon RX 7900 XTX (RADV NAVI31)
```

If that command is missing, install the Vulkan SDK loader (Linux:
`vulkan-tools` package; macOS: `brew install vulkan-tools`).

#### Example

```shell
# Vendor-neutral GPU scoring on AMD / Intel Arc / MoltenVK hosts.
vmaf-tune corpus --source ref.yuv --width 1920 --height 1080 \
    --preset medium --crf 22 --crf 28 --score-backend vulkan
```

Failure mode (no Vulkan loader installed):

```text
vmaf-tune: backend 'vulkan' requested but not available on this host
(available: cpu). Check that the local vmaf binary was built with the
matching backend support and the corresponding runtime/driver is
installed.
```

The exit code is `2` and no encodes are dispatched — the strict-mode
guarantee from [ADR-0299](../adr/0299-vmaf-tune-gpu-score.md) (no
silent CPU downgrade) is preserved across all four backend values.

| `--no-cache` | off | Disable the content-addressed encode/score cache (default: ON). |
| `--cache-dir PATH` | `$XDG_CACHE_HOME/vmaf-tune` | Override cache location (falls back to `~/.cache/vmaf-tune`). |
| `--cache-size-gb N` | `10` | LRU eviction cap in GiB. |
| `--sample-clip-seconds N` | `0` | Encode/score only the centre `N`-second slice of each source. `0` (default) keeps the legacy full-source behaviour. See [Sample-clip mode](#sample-clip-mode). |

## Sample-clip mode

Set `--sample-clip-seconds N` to evaluate each grid cell on the centre
`N`-second slice of the source instead of the full reference. This is a
runtime/accuracy trade-off, formalised in
[ADR-0301](../adr/0301-vmaf-tune-sample-clip.md).

- **Speedup.** Encode wall-time scales roughly linearly with slice
  length, so e.g. a 10-second slice of a 60-second source is a ~6x
  speedup per grid point. The libvmaf scoring pass shrinks by the same
  factor (it reads only the matching reference window via
  `--frame_skip_ref` / `--frame_cnt`).
- **Accuracy delta.** Expect ~1-2 VMAF points of drift versus
  full-clip on diverse content (mixed-shot trailers, sports, action),
  tighter (~0.3-0.5 VMAF points) on uniform content (single-shot
  interviews, animation, static stills). The delta is per-cell
  consistent — relative ordering between (preset, crf) cells survives,
  which is what Phase B (target-VMAF bisect) and Phase C (per-title
  CRF predictor) actually consume. Full-clip rescoring of the
  predictor's pick is the recommended Phase C epilogue.
- **Window placement.** Naive centre-anchored: the slice is
  `(duration_s − N) / 2 .. (duration_s + N) / 2`. Smarter shot-aware
  placement (e.g. via `transnet_v2`) is on the follow-up backlog.
- **Fallback.** If `N >= duration_s` the harness silently falls back
  to full-clip mode and tags the row `clip_mode="full"` — the request
  is treated as "use the whole source", not as an error.
- **Bitrate semantics.** `bitrate_kbps` is computed against the
  encoded duration, so sample-clip rows aren't biased low by dividing
  slice-bytes by full-source seconds. `duration_s` keeps the original
  source provenance.

```shell
# 6x faster grid sweep — 10s of a 60s source per cell.
vmaf-tune corpus \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 60 \
    --preset medium --preset slow \
    --crf 22 --crf 28 --crf 34 \
    --sample-clip-seconds 10 \
    --output corpus.jsonl
```

Each emitted row carries `clip_mode="sample_10s"` (or `"full"`),
letting Phase B/C either filter sample rows out, weight them
differently, or rescore the chosen cell on the full source.

## `benchmark` subcommand — corpus-level codec ranking

`vmaf-tune benchmark` is Phase G of the tune workflow. It does not run
FFmpeg or libvmaf; it reads an existing Phase-A JSONL corpus and ranks
encoders by their best matched-quality point.

For each encoder in the corpus, the command filters to successful rows
with finite `vmaf_score` and `bitrate_kbps`, then chooses the lowest
bitrate row whose VMAF clears `--target-vmaf`. Encoders that never clear
the target stay in the report as `status=unmet` using their closest VMAF
miss, so a too-narrow CRF sweep is visible instead of silently dropped.

```shell
vmaf-tune benchmark \
    --from-corpus corpus.jsonl \
    --target-vmaf 92 \
    --baseline-encoder libx264 \
    --format markdown
```

Output formats:

| Format | Use |
| --- | --- |
| `markdown` | PR comments and human review. |
| `json` | Notebooks, dashboards, and follow-up automation. |
| `csv` | Spreadsheets and quick plots. |

`--baseline-encoder` controls the bitrate-delta column. When omitted,
the baseline is the lowest-bitrate encoder that clears the target. The
report inherits the corpus coverage: if `libx264` was swept over 20 CRFs
and `libx265` over 3 CRFs, the ranking reflects those sampled points
only.

## Corpus JSONL schema

Each row is one JSON object on its own line. The full key list is
exported as `vmaftune.CORPUS_ROW_KEYS` for programmatic consumers and
versioned via `vmaftune.SCHEMA_VERSION` (currently `3` — v2 added
`clip_mode` for sample-clip mode under ADR-0301; v3 added the HDR
provenance triple `hdr_transfer` / `hdr_primaries` / `hdr_forced`
when `corpus.iter_rows` was wired to `hdr.detect_hdr` +
`hdr.hdr_codec_args` per the ADR-0300 status update of 2026-05-08).
Bumping the schema is a coordinated change with Phase B/C; do not
edit row shape without bumping the version.

| Key | Type | Description |
| --- | --- | --- |
| `schema_version` | int | Currently `3`. v3 adds the `enc_internal_*` aggregates (ADR-0332). |
| `run_id` | str | Per-row UUID4 hex. |
| `timestamp` | str | UTC ISO-8601 (seconds precision). |
| `src` | str | Path to the reference YUV. |
| `src_sha256` | str | SHA-256 of the reference (empty if `--no-source-hash`). |
| `width` / `height` | int | Source dimensions. |
| `pix_fmt` | str | Source pixel format. |
| `framerate` | float | Source framerate. |
| `duration_s` | float | Source duration in seconds. |
| `encoder` | str | Codec adapter name (e.g. `libx264`). |
| `encoder_version` | str | Detected encoder version (e.g. `libx264-164`). |
| `preset` | str | Encoder preset. |
| `crf` | int | Quality knob value. |
| `extra_params` | list[str] | Additional encoder argv (Phase A: `[]`). |
| `encode_path` | str | Path to encoded file (empty if not retained). |
| `encode_size_bytes` | int | Encoded file size. |
| `bitrate_kbps` | float | `(encode_size_bytes × 8 / 1000) / duration_s`. |
| `encode_time_ms` | float | Wall-clock encode time. |
| `vmaf_score` | float | Pooled-mean VMAF (`NaN` if scoring skipped/failed). |
| `vmaf_model` | str | Model version string (e.g. `vmaf_v0.6.1`). |
| `score_time_ms` | float | Wall-clock scoring time. |
| `ffmpeg_version` | str | Detected ffmpeg version. |
| `vmaf_binary_version` | str | Detected vmaf binary version. |
| `exit_status` | int | First non-zero of (encode, score) exit codes. |
| `hdr_transfer` | str | `""` (SDR), `"pq"` (SMPTE-2084) or `"hlg"` (ARIB STD-B67). Schema v3+. |
| `hdr_primaries` | str | Raw ffprobe `color_primaries` (e.g. `bt2020`); empty for SDR. Schema v3+. |
| `hdr_forced` | bool | `true` iff the user overrode detection via `--force-hdr-*` / `--force-sdr`. Schema v3+. |
| `clip_mode` | str | `"full"` (default) or `"sample_<N>s"` per `--sample-clip-seconds`. Schema v2+. |
| `shot_count` | int | Number of TransNet-V2 shots in the source (`0` when shot detection unavailable). v3+. |
| `shot_avg_duration_sec` | float | Mean shot length in seconds (`0.0` when unavailable). v3+. |
| `shot_duration_std_sec` | float | Population std of shot lengths in seconds — content-class proxy (animation: low; live action: high). v3+. |
| `adm2_mean` | float | Per-frame ADM2 mean (canonical-6). `NaN` when scoring skipped. v3+ (ADR-0366). |
| `vif_scale0_mean` … `motion2_std` | float | Remaining canonical-6 mean/std aggregates (12 columns total). v3+ (ADR-0366). |
| `enc_internal_qp_mean` | float | Per-frame QP mean from x264 pass-1 stats. `0.0` for opt-out adapters. v3+ (ADR-0332). |
| `enc_internal_qp_std` | float | Per-frame QP standard deviation. v3+ (ADR-0332). |
| `enc_internal_bits_mean` | float | Per-frame bit-cost mean (`tex+mv+misc`). v3+ (ADR-0332). |
| `enc_internal_bits_std` | float | Per-frame bit-cost standard deviation. v3+ (ADR-0332). |
| `enc_internal_mv_mean` | float | Per-frame motion-vector bit-cost mean. v3+ (ADR-0332). |
| `enc_internal_mv_std` | float | Per-frame motion-vector bit-cost standard deviation. v3+ (ADR-0332). |
| `enc_internal_itex_mean` | float | Mean intra-texture cost across I/i frames. v3+ (ADR-0332). |
| `enc_internal_ptex_mean` | float | Mean predicted-texture cost across P/B/b frames. v3+ (ADR-0332). |
| `enc_internal_intra_ratio` | float | Fraction of macroblocks coded as intra. v3+ (ADR-0332). |
| `enc_internal_skip_ratio` | float | Fraction of macroblocks coded as skip. v3+ (ADR-0332). |

The ten `enc_internal_*` columns are populated for adapters that
declare `supports_encoder_stats = True` (currently libx264 and
libx265). The parser normalizes x264 macroblock counters and x265
`icu` / `pcu` / `scu` CTU counters into the same intra / predicted /
skip ratio columns.
Hardware encoders (NVENC / AMF / QSV / VideoToolbox) and AV1 software
encoders (libaom-av1 / libsvtav1 / libvvenc) opt out and emit `0.0`
for every column so the schema is uniform across the corpus. The
trade-off: per-encode wall-clock cost roughly doubles for opt-in
adapters because the harness runs a stats-only `-pass 1` invocation
before the production CRF encode.

### Example row

```json
{
  "schema_version": 3,
  "run_id": "0a3b1c8b...",
  "timestamp": "2026-05-03T16:00:00+00:00",
  "src": "ref.yuv",
  "src_sha256": "",
  "width": 1920, "height": 1080, "pix_fmt": "yuv420p",
  "framerate": 24.0, "duration_s": 10.0,
  "encoder": "libx264", "encoder_version": "libx264-164",
  "preset": "medium", "crf": 28,
  "extra_params": [],
  "encode_path": "",
  "encode_size_bytes": 845210,
  "bitrate_kbps": 676.168,
  "encode_time_ms": 4321.0,
  "vmaf_score": 92.41,
  "vmaf_model": "vmaf_v0.6.1",
  "score_time_ms": 1820.5,
  "ffmpeg_version": "6.1.1",
  "vmaf_binary_version": "3.0.0-lusoris.0",
  "exit_status": 0,
  "clip_mode": "full",
  "shot_count": 12,
  "shot_avg_duration_sec": 0.83,
  "shot_duration_std_sec": 0.41,
  "adm2_mean": 9.73, "adm2_std": 0.12,
  "enc_internal_qp_mean": 25.23,
  "enc_internal_qp_std": 0.12,
  "enc_internal_bits_mean": 4975.0,
  "enc_internal_bits_std": 1820.5,
  "enc_internal_mv_mean": 60.3,
  "enc_internal_mv_std": 32.1,
  "enc_internal_itex_mean": 8000.0,
  "enc_internal_ptex_mean": 1500.0,
  "enc_internal_intra_ratio": 0.07,
  "enc_internal_skip_ratio": 0.16
}
```

## `recommend` subcommand — target-VMAF / target-bitrate

`vmaf-tune recommend` consumes the corpus (either pre-built via
`--from-corpus PATH.jsonl` or generated on the fly from `--source` +
grid flags) and applies one of two predicates:

- `--target-vmaf T` — return the row with the **smallest CRF** whose
  `vmaf_score >= T`. If no row clears the bar, the row with the
  highest VMAF is returned and the predicate is annotated `(UNMET)`
  in the output. Exit code is still `0` for an honest closest-miss.
- `--target-bitrate KBPS` — return the row whose `bitrate_kbps` is
  **closest** (absolute distance) to `KBPS`. Ties on distance go to
  the smaller CRF (higher quality).

The two flags are mutually exclusive — argparse rejects passing both
with exit code `2`.

When reading a pre-built corpus, `recommend` ignores rows whose
`exit_status` is non-zero and rows with missing or non-finite
`vmaf_score`. The `--encoder` and `--preset` flags act as filters in
that mode, so a mixed-codec corpus can be reused without first splitting
it into per-codec files.

### Use a pre-built corpus

```shell
# Phase A — build once.
vmaf-tune corpus --source ref.yuv --width 1920 --height 1080 \
    --framerate 24 --duration 10 \
    --preset medium --crf 18 --crf 22 --crf 26 --crf 30 --crf 34 \
    --output corpus.jsonl

# Smallest CRF whose VMAF >= 92.
vmaf-tune recommend --from-corpus corpus.jsonl --target-vmaf 92.0

# CRF whose bitrate is closest to 5 Mbps.
vmaf-tune recommend --from-corpus corpus.jsonl --target-bitrate 5000
```

### Build the corpus on the fly

```shell
vmaf-tune recommend \
    --source ref.yuv --width 1920 --height 1080 \
    --framerate 24 --duration 10 \
    --preset medium --crf 18 --crf 22 --crf 26 --crf 30 \
    --target-vmaf 92.0
```

If `--preset` and `--crf` are omitted, `recommend` sweeps `medium` ×
`range(18, 36, 2)` as a sensible default for ad-hoc runs.

### Output

Default output is a single human-readable line on stdout, e.g.

```text
encoder=libx264 preset=medium crf=22 vmaf=95.000 bitrate_kbps=5000.00 \
  predicate=target_vmaf>=92.0 margin=+3.000
```

Pass `--json` to get the full corpus row as a JSON object on stdout
instead — convenient for piping into other tooling.

### `recommend` flags

| Flag | Default | Notes |
|---|---|---|
| `--target-vmaf T` | — | Smallest CRF whose `vmaf_score >= T`. |
| `--target-bitrate KBPS` | — | CRF whose `bitrate_kbps` is closest to `KBPS`. |
| `--from-corpus PATH` | — | Read rows from a pre-built JSONL. Skips encode + score. |
| `--source / --width / --height / --framerate / --duration` | — | Build a corpus on the fly. Required when `--from-corpus` is omitted. |
| `--encoder / --preset / --crf` | `libx264` / `medium` / `[18,20,...,34]` | Sweep grid (when building). Filter (when loading). |
| `--json` | off | Emit the winning row as JSON instead of the prose summary. |

## `fast` subcommand — proxy + Bayesian + GPU-verify (Phase A.5)

`vmaf-tune fast` is the seconds-to-minutes alternative to the Phase A
grid for the recommendation use case. It runs an Optuna TPE search
over the integer CRF axis, scores each trial with the
`fr_regressor_v2` proxy ([ADR-0291](../adr/0291-fr-regressor-v2-prod-flip.md))
on the canonical-6 libvmaf features extracted from a short probe
encode, then runs **one** real-encode + libvmaf verify pass at the
recommended CRF before reporting. The slow grid stays canonical
([ADR-0276](../adr/0276-vmaf-tune-fast-path.md)) — `fast` is opt-in,
and falls back to the grid when the proxy/verify gap exceeds the
configured tolerance.

### Install

The fast-path needs Optuna in addition to the core install:

```shell
pip install 'vmaf-tune[fast]'
```

The shipped `[fast]` extra is the only correct install path; the core
package stays zero-extra-dep so corpus generation works on hosts that
never run the fast path.

### Smoke run (no ffmpeg / no ONNX / no GPU)

The `--smoke` flag swaps the proxy + verify pipeline for a
deterministic synthetic CRF→VMAF curve so CI on bare hosts still
exercises the search loop:

```shell
vmaf-tune fast --target-vmaf 92.0 --smoke --n-trials 12
```

```json
{
  "encoder": "libx264",
  "n_trials": 12,
  "notes": "smoke mode — synthetic predictor; no ffmpeg / ONNX / GPU. See ADR-0276 + ADR-0304 + Research-0076 for the production path.",
  "predicted_kbps": 1954.27,
  "predicted_vmaf": 82.65,
  "proxy_verify_gap": null,
  "recommended_crf": 27,
  "smoke": true,
  "target_vmaf": 92.0,
  "verify_vmaf": null
}
```

### Production run

```shell
vmaf-tune fast \
    --src ref.yuv --width 1920 --height 1080 \
    --framerate 24 --pix-fmt yuv420p \
    --encoder libx264 --preset medium \
    --target-vmaf 92.0 \
    --crf-min 18 --crf-max 40 \
    --n-trials 30 \
    --score-backend auto \
    --output recommendation.json
```

The recommendation lands as a single JSON object — same schema
`recommend` and `predict` already emit, plus the fast-path-specific
`verify_vmaf` and `proxy_verify_gap` diagnostics:

```json
{
  "encoder": "libx264",
  "target_vmaf": 92.0,
  "recommended_crf": 22,
  "predicted_vmaf": 92.41,
  "predicted_kbps": 4820.0,
  "n_trials": 30,
  "smoke": false,
  "notes": "production: TPE over 30 trials with v2 proxy; GPU verify gap = 0.612 VMAF (tolerance 1.50).",
  "verify_vmaf": 91.80,
  "proxy_verify_gap": 0.612,
  "score_backend": "cuda"
}
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Recommendation produced; proxy/verify gap within tolerance. |
| `2` | Argument validation error (missing `--src`, bad CRF range, ...). |
| `3` | Out-of-distribution: proxy/verify gap exceeded `--proxy-tolerance`. The recommendation is still emitted; callers should fall back to the slow Phase A grid (`vmaf-tune corpus` + `vmaf-tune recommend`). |

### Fall-back idiom

```shell
vmaf-tune fast --src ref.yuv --width 1920 --height 1080 \
    --target-vmaf 92.0 --output rec.json \
  || vmaf-tune recommend --source ref.yuv --width 1920 --height 1080 \
        --preset medium --target-vmaf 92.0 --output rec.json
```

The `||` chain captures both the production-error case (`rc=2`) and
the OOD case (`rc=3`), so the slow grid is the safety net whenever
the fast-path is not confident.

### `fast` flags

| Flag | Default | Notes |
|---|---|---|
| `--src PATH` | — | Source video. Required outside `--smoke`. |
| `--width / --height` | `0` | Raw-YUV geometry. Required outside `--smoke`. |
| `--pix-fmt` | `yuv420p` | ffmpeg pix_fmt for the probe + verify encodes. |
| `--framerate` | `24.0` | Reference framerate. |
| `--target-vmaf T` | — | Quality target on the standard `[0, 100]` scale. **Required.** |
| `--encoder` | `libx264` | Codec adapter; must be in `ENCODER_VOCAB_V2` for production mode. |
| `--preset` | `medium` | Encoder preset for the probe + verify encodes. |
| `--crf-min / --crf-max` | `10` / `51` | TPE search range over the integer CRF axis. |
| `--n-trials` | `30` (prod), `50` (smoke) | TPE trial budget. |
| `--time-budget-s` | `300` | Soft wall-clock cap for Optuna. TPE stops scheduling new trials after the timeout; an in-flight probe finishes. |
| `--proxy-tolerance` | `1.5` | Max abs proxy/verify gap before exit code `3`. |
| `--sample-chunk-seconds` | `5.0` | Probe-slice duration per TPE trial. |
| `--smoke` | off | Synthetic curve; no ffmpeg / ONNX / GPU. |
| `--score-backend` | `auto` | Verify-pass backend (`auto`/`cpu`/`cuda`/`vulkan`/`sycl`). |
| `--ffmpeg-bin / --vmaf-bin` | `ffmpeg` / `vmaf` | Tool paths. |
| `--vmaf-model` | `vmaf_v0.6.1` | libvmaf model for the verify pass. |
| `--encode-dir` | `.workingdir2/fast` | Scratch dir for probe + verify encodes. |
| `--output` | stdout | JSON destination for the recommendation payload. |

## Codec adapters

Phase A wires `libx264` end-to-end through the search loop. Additional
codec adapters land as one-file additions under
`tools/vmaf-tune/src/vmaftune/codec_adapters/` and join the registry
without touching the search loop. The currently-registered adapters are
discoverable via `vmaftune.codec_adapters.known_codecs()`.

### `libaom-av1`

Google's reference AV1 encoder. Adapter shipped via
[ADR-0279](../adr/0279-vmaf-tune-codec-adapter-libaom.md).

- Encoder name (FFmpeg): `libaom-av1`.
- Quality knob: `-crf` integer in `[0, 63]` (default `35`); higher CRF
  is lower quality.
- Speed knob: `-cpu-used` integer in `[0, 9]` (0 = slowest/best,
  9 = fastest). The adapter exposes a human-readable preset vocabulary
  that matches x264/x265 so a single sweep axis covers all four
  encoders.

| `--preset` name | libaom `-cpu-used` |
|---|---|
| `placebo`   | 0 (slowest, highest quality) |
| `slowest`   | 1 |
| `slower`    | 2 |
| `slow`      | 3 |
| `medium`    | 4 (default) |
| `fast`      | 5 |
| `faster`    | 6 |
| `veryfast`  | 7 |
| `superfast` | 8 |
| `ultrafast` | 9 (fastest) |

Sample FFmpeg invocation produced by the adapter (Phase B+ wires the
codec args into `vmaftune.encode`):

```shell
ffmpeg -i ref.y4m -c:v libaom-av1 -crf 35 -cpu-used 4 -an -y out.mkv
```

### libaom vs SVT-AV1 trade-offs

`libaom-av1` and `libsvtav1` both target the AV1 bitstream but sit at
different points on the speed/quality curve. Use the table below as a
rough decision aid; the per-corpus numbers belong in your local sweep
output, not here.

| Concern | libaom-av1 | libsvtav1 |
|---|---|---|
| Encode wall time at matched preset | meaningfully slower | meaningfully faster |
| Quality at slow presets (matched bitrate) | slightly higher per AOM benchmarks | slightly lower |
| Quality at fast presets | comparable | comparable, sometimes ahead |
| CRF range | `0..63` | `0..63` |
| Best fit | offline / high-quality archive encodes | live, batch, large catalog |

The fork's `vmaf-tune` corpus rows record the exact `(encoder, preset,
crf, vmaf_score, encode_time_ms, bitrate_kbps)` tuple, so Phase C/D
predictors can pick whichever encoder dominates the relevant region of
the rate-distortion plane on a given source.

## Hardware encoders (NVENC)

Phase A also wires the NVIDIA NVENC family for hardware-accelerated
sweeps:

| Adapter `--encoder` | FFmpeg encoder | Hardware required |
|---|---|---|
| `h264_nvenc` | `h264_nvenc` | NVIDIA Kepler+ (most modern GPUs) |
| `hevc_nvenc` | `hevc_nvenc` | NVIDIA Maxwell 2nd-gen+ (GTX 960+) |
| `av1_nvenc` | `av1_nvenc` | NVIDIA Ada Lovelace+ (RTX 40-series, L40, L4) |

NVENC's quality knob is `-cq` (constant quantizer), the closest
analogue to libx264 CRF. The fork's CQ window is the same `[15, 40]`
perceptually informative range used for `libx264`; the hardware
accepts `[0, 51]`.

NVENC has seven preset levels (`p1` fastest → `p7` slowest). The CLI
takes the same mnemonic preset names as `libx264` and maps them:

| Mnemonic | NVENC preset |
|---|---|
| `ultrafast`, `superfast`, `veryfast` | `p1` |
| `faster` | `p2` |
| `fast` | `p3` |
| `medium` (default) | `p4` |
| `slow` | `p5` |
| `slower` | `p6` |
| `slowest`, `placebo` | `p7` |

## Hardware encoders (AMF)

`vmaf-tune` ships software (`libx264`) and hardware adapters under one
codec-adapter contract — the harness search loop is identical for all of
them. AMD AMF (Advanced Media Framework) covers H.264, HEVC, and AV1 on
AMD GPUs (see [ADR-0282](../adr/0282-vmaf-tune-amf-adapters.md)):

| Encoder | Codec | Hardware | FFmpeg flag | Quality knob |
| --- | --- | --- | --- | --- |
| `h264_amf` | H.264 / AVC | Any AMD GPU with AMF | `-c:v h264_amf` | `-qp_i / -qp_p` (cqp) |
| `hevc_amf` | HEVC / H.265 | Any AMD GPU with AMF | `-c:v hevc_amf` | `-qp_i / -qp_p` (cqp) |
| `av1_amf` | AV1 | RDNA3+ only (RX 7000 series and newer) | `-c:v av1_amf` | `-qp_i / -qp_p` (cqp) |

Requirements:

- AMD GPU with AMF support (AV1 needs RDNA3 silicon or newer).
- FFmpeg built with `--enable-amf`.
- The AMF runtime / driver installed (Adrenalin on Windows; the
  open-source Mesa AMF stack or AMD Pro driver on Linux).

Behavioural notes:

- **Preset compression — 7 levels collapse to 3.** AMF exposes only
  three `-quality` rungs (`quality`, `balanced`, `speed`) where
  `libx264` / NVENC / QSV expose seven preset names. The adapter maps
  preset names onto AMF rungs as follows:

  | Preset names | AMF `-quality` |
  | --- | --- |
  | `placebo`, `slowest`, `slower`, `slow` | `quality` |
  | `medium` (default) | `balanced` |
  | `fast`, `faster`, `veryfast`, `superfast`, `ultrafast` | `speed` |

  This is opinionated: AMF's hardware pipeline does not expose finer
  steps. Callers that need finer granularity should pin `-qp_i` / `-qp_p`
  (the `qp` quality knob) instead of the preset.

- **Rate control is constant-QP.** AMF `-rc cqp` plus matched
  `-qp_i` / `-qp_p` is the closest analogue to x264 CRF. Range is 0..51;
  the harness exposes the (15, 40) Phase A informative window for
  cross-codec comparability.

Example:

## Coarse-to-fine CRF search (ADR-0306)

Sweeping every CRF from 0..51 (52 encodes per source × preset) is
wasteful when the only question is *"what's the smallest CRF whose
VMAF still meets my target?"*. The fork ships a 2-pass coarse-to-fine
search that visits ~15 points instead of 52, a ~3.5× wall-time
speedup with no measurable quality regression.

**How it works:**

1. **Coarse pass** at `--coarse-step` over `[10, 50]` — by default
   that's `[10, 20, 30, 40, 50]` (5 encodes).
2. **Fine pass** at `--fine-step` within `±--fine-radius` of the
   best-coarse CRF. With defaults that's the 10 unique CRFs around the
   best (e.g. `[25..29, 31..35]` if best-coarse is `30`).
3. **1-pass shortcut**: when the highest-CRF coarse point already
   meets the VMAF target, no refinement is needed (lower bitrate would
   have to come from CRFs above the coarse grid, which the fine pass
   wouldn't probe anyway). The search stops after the coarse pass.

### `corpus --coarse-to-fine`

```shell
vmaf-tune corpus \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --encoder h264_nvenc \
    --preset medium --preset slow \
    --crf 23 --crf 28 --crf 34 \
    --output corpus_nvenc.jsonl
```

(The `--crf` flag carries the quality value regardless of whether the
encoder names it CRF or CQ; the adapter forwards it as `-cq` for
NVENC.)

### Hardware vs software trade-off

NVENC is **10–100× faster** than the software encoders at the cost of
quality. Empirically, `h264_nvenc` at `medium` typically loses
**3–5 VMAF points** versus `libx264 medium` at the same bitrate,
depending on content complexity. The Pareto frontier is genuinely
different — that is precisely why the harness treats NVENC as
separate codec entries rather than a flag on `libx264`. Use NVENC
when you need a large corpus quickly or when the production pipeline
is GPU-encoded; use software when you need the perceptually best
encode at a given bitrate.

If FFmpeg reports `Encoder h264_nvenc not found` (or one of the
sibling encoders), the FFmpeg build wasn't compiled with
`--enable-nvenc` or the GPU lacks the relevant generation. The
harness records the failure as `exit_status != 0` and skips scoring,
so a partial corpus over a heterogeneous fleet is still well-formed.

```shell
vmaf-tune corpus \
    --encoder h264_amf \
    --preset slow --preset medium --preset fast \
    --crf 23 --crf 28 --crf 34 \
    --output corpus_amf.jsonl
```

The raw FFmpeg invocation the adapter emits looks like:

```shell
ffmpeg -i ref.yuv -c:v h264_amf \
       -quality balanced -rc cqp -qp_i 23 -qp_p 23 \
       -an out.mkv
```

## Hardware encoders (QSV)

Beyond `libx264`, the registry exposes the three Intel QSV (Quick Sync
Video) hardware adapters added in
[ADR-0281](../adr/0281-vmaf-tune-qsv-adapters.md). They share the QSV
preset vocabulary (`veryslow, slower, slow, medium, fast, faster,
veryfast` — same names as x264's medium-and-down subset) and use
`-global_quality N` (ICQ rate control, range `1..51`, semantically
similar to CRF).

| Adapter name | FFmpeg encoder | Quality knob | Hardware required |
| --- | --- | --- | --- |
| `h264_qsv` | `h264_qsv` | `global_quality` (1–51) | Intel iGPU 7th-gen+ (Kaby Lake or newer) or Arc / Battlemage |
| `hevc_qsv` | `hevc_qsv` | `global_quality` (1–51) | Intel iGPU 7th-gen+ (10-bit needs 11th-gen+) or Arc / Battlemage |
| `av1_qsv` | `av1_qsv` | `global_quality` (1–51) | Intel iGPU 12th-gen+ only or Arc / Battlemage |

The underlying FFmpeg invocations look like:

```shell
ffmpeg -i src.mkv -c:v h264_qsv -preset medium -global_quality 23 -an out.mkv
ffmpeg -i src.mkv -c:v hevc_qsv -preset medium -global_quality 23 -an out.mkv
ffmpeg -i src.mkv -c:v av1_qsv  -preset medium -global_quality 23 -an out.mkv
```

`vmaf-tune` validates the `(preset, global_quality)` pair before
spawning ffmpeg and probes `ffmpeg -encoders` for the requested
encoder; if libmfx / VPL is not compiled in, the harness raises
`RuntimeError` with a build-time hint rather than letting ffmpeg
emit an `Encoder not found` line buried in stderr.

## Apple VideoToolbox adapters

The `h264_videotoolbox`, `hevc_videotoolbox`, and `prores_videotoolbox`
registry entries cover Apple Silicon (M-series) and T2-equipped Intel
Macs via the `VideoToolbox.framework` hardware encoder. The H.264 and
HEVC adapters were added in
[ADR-0283](../adr/0283-vmaf-tune-videotoolbox-adapters.md); the ProRes
adapter follows the same registry pattern (see ADR-0283 *Status update
2026-05-09*). All three share `_videotoolbox_common.py` for the
preset → `-realtime` mapping; H.264 / HEVC also share the `-q:v`
quality knob, while ProRes uses the integer **profile tier** instead.

| Adapter name | FFmpeg encoder | Quality knob | Hardware required |
| --- | --- | --- | --- |
| `h264_videotoolbox` | `h264_videotoolbox` | `q:v` (0..100, higher = better) | Apple Silicon or Intel Mac with T2 |
| `hevc_videotoolbox` | `hevc_videotoolbox` | `q:v` (0..100, higher = better) | Apple Silicon or Intel Mac with T2 |
| `prores_videotoolbox` | `prores_videotoolbox` | `profile:v` (0..5, higher tier = better) | Apple Silicon **M1 Pro / Max / Ultra or later** |

VideoToolbox exposes only a binary `-realtime {0,1}` flag instead of
a multi-valued preset, so the harness's nine-name preset vocabulary
collapses onto that boolean per the table in
`_videotoolbox_common.py`:
`ultrafast`/`superfast`/`veryfast`/`faster`/`fast` → `realtime=1`
(low-latency fast path); `medium`/`slow`/`slower`/`veryslow` →
`realtime=0` (offline / quality-priority). The mapping is
intentionally lossy — VT cannot expose a finer dial.

The underlying FFmpeg invocations look like:

```shell
ffmpeg -i src.mkv -c:v h264_videotoolbox -realtime 0 -q:v 60 -an out.mkv
ffmpeg -i src.mkv -c:v hevc_videotoolbox -realtime 0 -q:v 60 -an out.mkv
ffmpeg -i src.mkv -c:v prores_videotoolbox -realtime 0 -profile:v hq -an out.mov
```

AV1 hardware encoding is intentionally **not wired** — Apple Silicon
has no AV1 hardware encoder block as of 2026 and FFmpeg exposes no
`av1_videotoolbox`. Use `libaom-av1` or `libsvtav1` for AV1 on
macOS.

`vmaf-tune` validates the `(preset, quality)` pair via the adapter
(`-q:v` for H.264 / HEVC; integer tier id for ProRes) and probes
`ffmpeg -encoders` for the requested encoder; if VideoToolbox is
unavailable (e.g. the host is Linux), the harness raises
`RuntimeError` rather than letting ffmpeg emit `Encoder not found`.

### ProRes tier reference

ProRes is a fixed-rate intermediate codec — there is no CRF / QP
scalar. Quality is selected entirely by the **tier**, and bitrate is
implicit in the tier × resolution × frame-rate combination. The
harness's `--crf` flag carries the integer tier id (the FFmpeg
`profile:v` AVOption value); the adapter emits the canonical FFmpeg
alias on the argv for diagnosability.

| `crf` value | FFmpeg alias | Marketing name | Typical use |
| --- | --- | --- | --- |
| 0 | `proxy` | ProRes 422 Proxy | Offline editing, dailies |
| 1 | `lt` | ProRes 422 LT | Broadcast acquisition |
| 2 | `standard` | ProRes 422 | Mainline broadcast master |
| 3 | `hq` | ProRes 422 HQ | High-end broadcast / film master (default) |
| 4 | `4444` | ProRes 4444 | Graphics, alpha, colour grading |
| 5 | `xq` | ProRes 4444 XQ | High-dynamic-range / wide-gamut master |

Source: FFmpeg `libavcodec/videotoolboxenc.c` `prores_options`
AVOption table (verified against an FFmpeg n8.1.1 checkout).

ProRes is intra-only — every frame is a keyframe — so `--keyint` /
`--force-keyframes` flags are accepted but have no rate-distortion
effect. The harness still emits them so the muxer's seek-table
density is predictable across codecs.

## Saliency-aware encoding (`recommend-saliency --saliency-aware`)

Bucket #2 of the [PR #354](https://github.com/lusoris/vmaf/pull/354)
audit (see [ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md)) wires
the fork-trained `saliency_student_v1` ONNX model
([ADR-0286](../adr/0286-saliency-student-fork-trained-on-duts.md)) into
`vmaf-tune` so a single command can produce an encode that biases bits
toward salient regions (faces, focal subjects, action) and saves bits
on background.

### Synopsis

```shell
vmaf-tune recommend-saliency \
    --src ref.yuv --width 1920 --height 1080 --framerate 24 \
    --duration-frames 240 \
    --preset medium --crf 23 \
    --saliency-aware \
    [--saliency-offset -4] \
    [--saliency-model model/tiny/saliency_student_v1.onnx] \
    [--saliency-aggregator mean|ema|max|motion-weighted] \
    --output out.mp4
```

### How it works

1. `compute_saliency_map()` samples the requested frame window from
   the source YUV, runs those frames through
   `saliency_student_v1.onnx` (ImageNet-normalised RGB derived from
   YUV, NCHW `[1, 3, H, W]`), and reduces the per-frame saliency
   outputs into one mask in `[0, 1]`.
2. `saliency_to_qp_map()` linearly maps the mask to per-pixel QP
   deltas — `--saliency-offset` is the QP delta at peak saliency
   (negative means **better** quality on salient regions). Background
   gets the symmetric positive delta. The output is clamped to
   `[-12, +12]` (matching the [`vmaf-roi`](vmaf-roi.md) sidecar
   convention from ADR-0247).
3. The pixel-level QP-offset map is dispatched to the encoder-specific
   ROI channel (see the encoder-targets table below); each channel
   reduces the map to its native granularity before writing any sidecar
   file or composing the argv slice.
4. The augmented `extra_params` (qpfile path, zones string, or ROI-map
   path) are appended to the FFmpeg command and the normal encode path
   runs.

### Trade-off

| Axis | Direction |
| --- | --- |
| Bitrate (same VMAF) | **−10 % to −20 %** for content with strong attention focus (faces, action, sport). Background-uniform content sees little change. |
| Encode time | **+5 %** typical (saliency inference + per-MB reduce; per-frame model time is sub-millisecond on CPU at SD/HD). |
| Decode time | unchanged (the bitstream is standard-compliant for all five supported encoders). |
| Quality (VMAF) | unchanged at the **clip-mean** level; concentrated where the eye looks. |

Numbers are indicative. Today's `recommend-saliency` subcommand is a
one-shot encode at `--crf` (or the adapter default); target-VMAF
selection remains the job of `recommend`, `compare`, or a
caller-provided bisect loop.

### Temporal aggregation

`--saliency-aggregator` controls how sampled frame masks become the
single ROI pattern applied to the encode:

| Aggregator | Behaviour | Use when |
| --- | --- | --- |
| `mean` | Per-pixel arithmetic mean; preserves the original implementation. | Default, stable clips, and baseline comparisons. |
| `ema` | Exponential moving average with `--saliency-ema-alpha` as the current-frame weight. | Motion or cuts make the latest sampled frames more representative. |
| `max` | Per-pixel maximum over sampled masks. | Missing a brief salient object is worse than over-protecting background. |
| `motion-weighted` | Weighted mean where each sampled frame is weighted by luma delta from the previous sampled frame. | Motion-heavy clips where changing frames should dominate the aggregate. |

All four reducers use the same `saliency_student_v1` weights and the
same downstream QP mapping, so changing the reducer does not change the
model contract or encoder sidecar format.

### Graceful fallback

If `onnxruntime` is not installed or
`model/tiny/saliency_student_v1.onnx` cannot be loaded,
`recommend-saliency --saliency-aware` logs a warning and falls back to a
plain encode. Callers always get a result; the saliency bias is
opportunistic. This matches the
[`vmaf-roi`](vmaf-roi.md) C sidecar's posture.

### Encoder targets

The saliency pipeline supports five encoder ROI mechanisms:

| Encoder | ROI channel | Granularity | argv slot |
| --- | --- | --- | --- |
| `libx264` | ASCII `--qpfile` (x264 r2390+) | 16×16 luma MB | `-x264-params qpfile=…` |
| `libaom-av1` | patched FFmpeg `-qpfile` ROI bridge | 16×16 luma MB mapped onto libaom MI cells | `-qpfile …` |
| `libx265` | `--zones` QP delta (per-clip mean) | full-clip spatial mean | `-x265-params zones=0,N,q=<delta>` |
| `libsvtav1` | `--qp-file` offset map (SVT-AV1 v1.7+) | 64×64 super-block | `-svtav1-params qp-file=…` |
| `libvvenc` | `ROIFile` CSV (VVenC v1.14.0+) | 64×64 CTU | `-vvenc-params ROIFile=…` |

See [ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md) (x264 baseline) and
[ADR-0370](../adr/0370-saliency-roi-x265-svtav1-vvenc.md) (x265 / SVT-AV1 / VVenC).
The libaom-av1 row uses the shared `-qpfile` bridge documented in
[`vmaf-tune-ffmpeg.md`](vmaf-tune-ffmpeg.md#libaom-av1-full-roi-bridge).

Per-adapter helpers in `vmaftune.saliency`:

- `write_x265_zones_arg(block_offsets, duration_frames)` → zones string
- `write_x264_qpfile(block_offsets, out_path, duration_frames)` → x264/libaom qpfile
- `write_svtav1_qpoffset_map(block_offsets, out_path, duration_frames)` → Path
- `write_vvenc_roi_csv(block_offsets, out_path, duration_frames)` → Path

### Caveats

- **Aggregate mask, not per-frame ROI.** The current implementation
  reduces saliency across the sampled frames and applies one delta
  pattern across the whole clip. Per-frame ROI is on the roadmap.
- **x265 zones: spatial mean only.** x265's `--zones` has temporal
  granularity but not per-block spatial granularity. The zone carries
  the mean QP delta across all blocks. Per-block x265 ROI requires a
  future x265 qpfile port (different format from x264).
- **libaom segment quantisation.** The FFmpeg bridge maps 16×16 qpfile
  deltas onto libaom's MI grid and at most eight segment QPs. Very
  fine-grained deltas are therefore quantised to the nearest segment.
- **SVT-AV1 / VVenC: 64×64 granularity.** Both encoders document
  64×64 as their ROI-map unit. The saliency mask is reduced to this
  grid via `reduce_qp_map_to_blocks(qp_map, block=64)` before writing.
- **RGB saliency input.** The student model receives BT.709-limited
  yuv420p converted to ImageNet-normalised RGB. Chroma is
  nearest-neighbour upsampled to luma resolution before inference.
- **Don't use the placeholder.** `mobilesal_placeholder_v0` and the
  radial fallback inside `vmaf-roi` are smoke-test stubs. Pass an
  explicit `--saliency-model` pointing at the real fork-trained
  weights when you want a perceptual benefit.

### Reproducer

The test suite mocks the ONNX session and the encode runner so it
runs without ffmpeg or onnxruntime installed:

```shell
pytest tools/vmaf-tune/tests/test_saliency.py \
       tools/vmaf-tune/tests/test_saliency_roi_adapters.py \
       tools/vmaf-tune/tests/test_saliency_roi_codec.py -v
```

## Codec adapter contract

The encode driver
([`tools/vmaf-tune/src/vmaftune/encode.py`](../../tools/vmaf-tune/src/vmaftune/encode.py))
is **codec-agnostic** as of [ADR-0297](../adr/0297-vmaf-tune-encode-multi-codec.md).
It looks up the codec adapter via
`vmaftune.codec_adapters.get_adapter(req.encoder)` and asks the
adapter for its FFmpeg argv slice — the harness itself never branches
on codec identity. Adding a new codec is one file under
`tools/vmaf-tune/src/vmaftune/codec_adapters/` plus a registry entry;
the search loop, corpus row schema, and FFmpeg invocation stay
untouched.

A codec adapter is a frozen dataclass exposing:

| Member | Type | Purpose |
| --- | --- | --- |
| `name` | `str` | Human-readable codec id (`"libx264"`). |
| `encoder` | `str` | FFmpeg `-c:v` value (`"libx264"`, `"h264_nvenc"`, ...). |
| `quality_knob` | `str` | Name of the quality knob (`"crf"`, `"cq"`, `"qp"`, ...). |
| `quality_range` | `tuple[int, int]` | Inclusive `(min, max)` for the knob. |
| `quality_default` | `int` | Default quality value. |
| `invert_quality` | `bool` | True when a higher value means lower quality (CRF / QP). |
| `presets` | `tuple[str, ...]` | Allowed preset names. |
| `validate(preset, quality) -> None` | method | Raises `ValueError` on out-of-range input. |
| `ffmpeg_codec_args(preset, quality) -> list[str]` | method | Codec-specific argv slice (e.g. `["-c:v", "libx264", "-preset", "medium", "-crf", "23"]`). |
| `extra_params() -> tuple[str, ...]` | method (optional) | Additional non-codec argv (e.g. `("-svtav1-params", "tune=0")`). |

The dispatcher composes the final ffmpeg command as:

```text
[ffmpeg, -y, -hide_banner, -loglevel info,
 -f rawvideo -pix_fmt <pf> -s WxH -r FR -i <src>,
 *adapter.ffmpeg_codec_args(preset, quality),
 *adapter.extra_params(),
 *req.extra_params,
 <output>]
```

Adapters that do not yet implement `ffmpeg_codec_args` (or for which
`get_adapter` raises `KeyError`) fall back to the legacy x264-CRF
shape (`-c:v <encoder> -preset <p> -crf <q>`) so partial adapters
stay drivable end-to-end while their per-codec PRs are in flight.

`parse_versions(stderr, encoder=...)` selects a per-codec version
probe; missing matches degrade to `"unknown"` rather than raising.

### Adapters in flight

| Codec | PR | Status |
| --- | --- | --- |
| `libx264` | shipped (Phase A) | green |
| `libx265` | #362 | adapter ships; dispatcher unblocks end-to-end |
| `libsvtav1` | #370 | adapter ships; dispatcher unblocks end-to-end |
| `libaom-av1` | #360 | adapter ships; dispatcher unblocks end-to-end |
| `libvvenc` | #368 | adapter ships; dispatcher unblocks end-to-end |
| `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` | #364 | adapter ships; dispatcher unblocks end-to-end |
| `h264_qsv` / `hevc_qsv` | #367 | adapter ships; dispatcher unblocks end-to-end |
| `h264_amf` / `hevc_amf` | #366 | adapter ships; dispatcher unblocks end-to-end |
| `h264_videotoolbox` / `hevc_videotoolbox` | #373 | adapter ships; dispatcher unblocks end-to-end |

## Codec comparison

`vmaf-tune compare` answers the perennial *"should I migrate from x264
to SVT-AV1 yet?"* question per-source: given one reference and a target
VMAF, run each codec's recommend predicate in parallel and rank the
results by smallest file. This is Bucket #7 of the
[`vmaf-tune` capability audit](../research/0061-vmaf-tune-capability-audit.md);
the default CLI backend is Phase B target-VMAF bisect per
[ADR-0326](../adr/0326-vmaf-tune-phase-b-bisect.md).

```shell
vmaf-tune compare \
    --src ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --sample-clip-seconds 4 \
    --target-vmaf 92 \
    --encoders libx264,libx265,libsvtav1,libaom,libvvenc \
    --crf-min 15 --crf-max 40 \
    --format markdown
```

The real bisect backend needs source geometry because raw YUV does not
self-describe. Pass `--width` and `--height` explicitly; `--pix-fmt`,
`--framerate`, and `--duration` default to the common SDR 24 fps shape
but should be set for accurate scoring and bitrate math. Pass
`--sample-clip-seconds N` to evaluate the centre `N` seconds of the
source per bisect iteration; compare forwards matching
`--frame_skip_ref` / `--frame_cnt` scorer bounds and normalises
bitrate against the sample duration. For custom rankers or tests,
`--predicate-module MODULE:CALLABLE` still accepts any importable
`(codec, src, target_vmaf) -> RecommendResult` callable and bypasses
the bisect backend.

Sample output (`--format markdown`, abridged):

```markdown
# Codec comparison — target VMAF 92

- Source: `ref.yuv`
- Tool: `vmaf-tune 0.0.1`
- Wall time: 6421.3 ms

| Rank | Codec    | Encoder         | Best CRF | Bitrate (kbps) | Encode time (ms) | VMAF  | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1  | libaom    | libaom-3.8.0    | 30       |         1500.0 |          18000.0 | 92.40 | ok     |
| 2  | libx265   | libx265-3.5     | 26       |         1700.0 |           4200.0 | 92.00 | ok     |
| 3  | libsvtav1 | libsvtav1-1.7.0 | 32       |         1900.0 |           2800.0 | 92.30 | ok     |
| 4  | libx264   | libx264-164     | 23       |         2400.0 |           1500.0 | 92.10 | ok     |

**Smallest file**: `libaom` at CRF 30 → 1500.0 kbps (VMAF 92.40).
```

### `compare` CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--src PATH` | — | Required. Single reference clip. |
| `--target-vmaf F` | `92.0` | VMAF the bisect backend targets. |
| `--encoders LIST` | every registered adapter | Comma-separated codec names; e.g. `libx264,libx265,libsvtav1,libaom`. |
| `--width / --height` | — | Required for the default real-bisect backend. |
| `--pix-fmt` | `yuv420p` | Source pixel format forwarded to the scorer. |
| `--framerate` | `24.0` | Source framerate. |
| `--duration` | `0.0` | Source duration in seconds, used for bitrate math. |
| `--sample-clip-seconds` | `0.0` | `0.0` scores the full source. Positive values shorter than `--duration` use a centre sample window for encode, score, and bitrate math (ADR-0301). |
| `--preset` | adapter default | Preset forwarded to the codec adapter. |
| `--crf-min / --crf-max` | adapter range | Inclusive CRF search window. Pass both or neither. |
| `--max-iterations` | `8` | Encode+score round-trip cap per codec. |
| `--vmaf-model` | `vmaf_v0.6.1` | VMAF model forwarded to the scorer. |
| `--score-backend` | scorer default | `cpu`, `cuda`, `sycl`, `vulkan`, or `auto`. |
| `--ffmpeg-bin / --vmaf-bin` | `ffmpeg` / `vmaf` | Binary overrides. |
| `--format` | `markdown` | One of `markdown`, `json`, `csv`. |
| `--no-parallel` | off | Run codecs sequentially (default: thread pool, one per codec). |
| `--max-workers N` | `len(encoders)` | Cap on the parallel thread pool. |
| `--predicate-module MOD:FN` | off | Advanced hook that bypasses the bisect backend. |
| `--output PATH` | stdout | Write the rendered report to PATH instead of stdout. |

### `compare` output schema

The JSON / CSV columns are exported as `vmaftune.compare.COMPARE_ROW_KEYS`:
`codec`, `encoder_version`, `best_crf`, `bitrate_kbps`, `encode_time_ms`,
`vmaf_score`, `target_vmaf`, `ok`, `error`. Failed rows trail successful
ones in the ranking; `ok=False` rows carry a human-readable `error` and
sentinel numerics (`-1` for `best_crf`, `NaN` for the floats).

> **Encode-time normalisation**: the `encode_time_ms` column is
> wall-clock on whatever machine ran the predicate. Cross-codec time
> comparisons only make sense when every predicate was run on the same
> hardware in the same configuration — see
> [Research-0061 Bucket #7](../research/0061-vmaf-tune-capability-audit.md).

## HDR-aware tuning (Bucket #9, ADR-0300)

Phase A auto-detects HDR sources and injects codec-appropriate HDR
encode flags + HDR VMAF scoring. Detection runs `ffprobe` against
each `--source` once at corpus start; the per-source encode argv
gets the resulting HDR flag set appended.

### What gets detected

A source is classified as HDR iff its first video stream carries
**both** of:

- `color_transfer` ∈ {`smpte2084` (PQ), `arib-std-b67` / `hlg`} **and**
- `color_primaries` ∈ {`bt2020`, `bt2020nc`, `bt2020-ncl`, `bt2020c`, `bt2020-cl`}.

Mismatched signaling (e.g. PQ transfer with BT.709 primaries) is
treated as SDR — misclassifying SDR as HDR is the dangerous failure
mode. Mastering-display + max-CLL SEI side data is read when present
and propagated to encoders that expose stable FFmpeg SEI flags (x265,
SVT-AV1, HEVC NVENC).

### Detection modes

| Mode | When to use |
| --- | --- |
| `--auto-hdr` (default) | Mixed corpora; let ffprobe classify each source. |
| `--force-sdr` | Disable HDR injection entirely (override probe). |
| `--force-hdr-pq` | Raw YUV refs with no container metadata; you know the source is PQ. |
| `--force-hdr-hlg` | Same, for HLG. |

The four flags are mutually exclusive.

### Codec dispatch

| Encoder | HDR signaling carrier |
| --- | --- |
| `libx264` | Container-level `-color_*` flags only (x264 has no in-stream HDR SEI). |
| `libaom-av1` | Global `-color_*` tags only; no fork-owned private SEI mapping yet. |
| `libx265` | Global `-color_*` + `-x265-params colorprim=bt2020:transfer=...:colormatrix=bt2020nc[:master-display=...:max-cll=...:hdr10-opt=1]`. |
| `libsvtav1` | Global `-color_*` + `-svtav1-params color-primaries=9:transfer-characteristics=16` (PQ) or `=18` (HLG) `:matrix-coefficients=9`. |
| `hevc_nvenc` | `-pix_fmt p010le -profile:v main10` + global `-color_*` + `-master_display` / `-max_cll` (when ffmpeg supports them). |
| `av1_nvenc` | `-pix_fmt p010le` + global `-color_*` tags. |
| `hevc_qsv` / `hevc_amf` / `hevc_videotoolbox` | `-pix_fmt p010le -profile:v main10` + global `-color_*` tags. |
| `av1_qsv` / `av1_amf` | `-pix_fmt p010le` + global `-color_*` tags. |
| `libvvenc` | Global `-color_*` only (SEI options live behind `--vvenc-params` in newer ffmpeg builds). |

Encoders not in the dispatch table emit no HDR flags and the corpus
row's `hdr_*` fields still record the detection result.

### HDR VMAF scoring (model-port slot)

`vmaftune.hdr.select_hdr_vmaf_model(model_dir, transfer="pq"|"hlg")`
resolves an HDR-trained model JSON via a two-stage lookup:

1. canonical filename — `model/vmaf_hdr_v0.6.1.json` (the Netflix
   research artefact name); preferred when `transfer` is `"pq"` or
   `"hlg"`.
2. glob fallback — `model/vmaf_hdr_*.json` (so future revisions can
   land without code changes).

The fork **does not ship the JSON** in this PR. Verified
2026-05-08 against `Netflix/vmaf` master `model/`: no
`vmaf_hdr_*.json` is present in the upstream public tree; Netflix
publishes the artefact in a separate research bundle outside the
repo. A fork-local license review is the gating follow-up
([ADR-0300 § Status update 2026-05-08](../adr/0300-vmaf-tune-hdr-aware.md#status-update-2026-05-08-iter_rows-integration-transfer-aware-model-slot)).
Until then, HDR sources are scored against the SDR model with a
**one-shot warning** logged on the first miss (subsequent misses
stay quiet). Resulting `vmaf_score` values trend low for
high-luminance regions and are not directly comparable to SDR
scores. Drop a licensed copy at `model/vmaf_hdr_v0.6.1.json` and
the harness picks it up automatically — no code change required.

## Content-addressed cache

Re-running a corpus sweep after adjusting an unrelated flag should
not re-encode and re-score tuples that have not changed. The
content-addressed cache turns repeated `(src, encoder, preset, crf)`
combinations into a free hit on the second run, restoring the parsed
`(bitrate, vmaf, encode_time, score_time)` tuple from disk and
skipping both subprocess calls. See
[ADR-0298](../adr/0298-vmaf-tune-cache.md) for the design.

### Key composition

The cache key is `sha256` of the canonical-JSON-encoded six-tuple:

1. `src_sha256` — content hash of the reference YUV
2. `encoder` — adapter name (`libx264`, …)
3. `preset` — encoder preset string
4. `crf` — quality knob value (int)
5. `adapter_version` — bumps when the codec adapter's argv shape changes
6. `ffmpeg_version` — host ffmpeg version string

Dropping any one of these would let stale entries shadow real
results when the adapter or ffmpeg is upgraded — the test suite
asserts each field flips the key.

### Layout

The cache lives at `$XDG_CACHE_HOME/vmaf-tune/` (or
`~/.cache/vmaf-tune/` if the env var is unset). Override with
`--cache-dir`. Layout:

```text
<cache-dir>/
  meta/<key>.json     — parsed (bitrate, vmaf, encode_time, ...) tuple
  blobs/<key>.bin     — opaque encoded artifact
  __index__.json      — last-access timestamps for LRU eviction
```

### Eviction

LRU with a default 10 GiB cap (configurable via `--cache-size-gb`).
On every `put`, the oldest entries are dropped until the total
on-disk size sits at or below the cap.

### Disabling

Pass `--no-cache` to force a re-encode/re-score on every cell. The
cache is also automatically skipped when `--no-source-hash` is
active (no stable content key) or when `ffmpeg -version` cannot be
probed before the run.

### Caveats

- The cache is **not** baked into the JSONL row; the row stays the
  canonical record, the cache is an opaque sidecar.
- Cache hits do not write a synthetic `encode_path` — that field
  remains empty unless `--keep-encodes` is set.
- Concurrent runs against a shared cache dir (e.g. NFS) work for
  reads; writes are last-writer-wins and both writers' bytes are
  valid by content addressing.

```shell
vmaf-tune corpus --source ref.yuv \
    --width 1920 --height 1080 --framerate 24 --duration 10 \
    --preset medium \
    --coarse-to-fine --target-vmaf 92 \
    --output corpus.jsonl
```

`--crf` is no longer required — the CRF axis is generated by the
search. `--target-vmaf` is optional here; without it the search still
runs both passes and refines around the highest-VMAF coarse point.

### `recommend` — pick a CRF for a quality target

The `recommend` subcommand always runs coarse-to-fine and prints the
single recommended (preset, crf) pair plus its measured VMAF. It also
writes the visited points to `--output` so callers have the corpus row
for downstream analysis.

```shell
vmaf-tune recommend \
    --source ref.yuv \
    --width 1920 --height 1080 --framerate 24 --duration 10 \
    --preset medium \
    --target-vmaf 92
# stdout: src=ref.yuv preset=medium crf=27 vmaf=92.341 (visited 15 encodes)
```

### Tunables

| Flag | Default | Notes |
|---|---|---|
| `--coarse-to-fine` | off (corpus); on (recommend) | Activate the 2-pass search. |
| `--coarse-step N` | `10` | Step for the coarse pass. With defaults gives `[10, 20, 30, 40, 50]`. |
| `--fine-radius R` | `5` | ±R around best-coarse for the fine pass. |
| `--fine-step S` | `1` | Step for the fine pass. |
| `--target-vmaf V` | unset | Required for `recommend`; optional for `corpus`. |

### Timing comparison

Numbers below are illustrative — actual encode + score wall time per
point varies with source resolution, preset, and the libvmaf backend
(`cpu`/`cuda`/`sycl`/`vulkan`). The relevant ratio is **points
visited**, not seconds:

| Mode | Points visited | Relative wall time |
|---|---:|---:|
| Full grid `--crf 0 ... 51` | 52 | 1.00× (baseline) |
| Coarse-to-fine, defaults, target met mid-range | 15 | **~0.29×** (3.46× faster) |
| Coarse-to-fine, 1-pass shortcut (target met at coarse max) | 5 | **~0.10×** (10.4× faster) |
| Coarse-to-fine, target unmet (full fine pass anyway) | 15 | ~0.29× |

For a 1080p `--preset medium` clip where one (encode + score) pass
takes ~5 s, the coarse-to-fine path drops a single recommend run from
~260 s to ~75 s.

## What Phase A does **not** do

- No target-VMAF bisect (Phase B).
- No per-title CRF prediction (Phase C).
- No Pareto ABR ladder generation (Phase E).

## Per-title ladder (Phase E)

Phase E ships the `vmaf-tune ladder` subcommand — given one source,
sample (resolution × target-VMAF) points, take the Pareto upper-convex
hull on (bitrate, vmaf), pick `n` evenly-spaced rungs along the hull,
and emit the result as an HLS master playlist, DASH MPD, or JSON
descriptor. This is the
"[per-title encoding](https://netflixtechblog.com/per-title-encode-optimization-7e99442b62a2)"
loop in one command — a fixed authoring-spec ladder is replaced by the
ladder that's actually optimal for *this* title.

See [ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md) for
the design and the alternatives considered (geometric ladder, JND-
spaced, fixed Apple HLS).

The default sampler is wired: for each `(resolution, target_vmaf)` cell
it runs the canonical 5-point CRF sweep `18,23,28,33,38` through the
normal Phase A encode-and-score path, picks the closest row to the
target, and feeds those points into hull and knee selection. A custom
`sampler=` callback remains supported for callers that want a finer
grid, a bisect loop, or a precomputed corpus stream.

### Canonical 5-rung invocation

The default rendition set is the canonical 5-rung
1080p/720p/480p/360p/240p ladder against VMAF targets
{95, 90, 85, 75, 65}:

```shell
vmaf-tune ladder \
    --src episode01.yuv \
    --encoder libx264 \
    --resolutions 1920x1080,1280x720,854x480,640x360,426x240 \
    --target-vmafs 95,90,85,75,65 \
    --quality-tiers 5 \
    --format hls \
    --output episode01_ladder.m3u8
```

The output is an HLS master playlist with one `#EXT-X-STREAM-INF` per
rung; bandwidth (in bps) is monotonically increasing. Variant URIs are
placeholders — re-point them at your per-rendition playlists when
packaging the encoded segments.

### Other manifest formats

```shell
# DASH MPD
vmaf-tune ladder --src ep01.yuv --format dash --output ladder.mpd

# JSON descriptor (machine-readable, vmaf-tune-ladder/v1 schema)
vmaf-tune ladder --src ep01.yuv --format json --output ladder.json
```

### Rung spacing

`--spacing log_bitrate` (default) doubles bandwidth per rung — Apple
HLS authoring-spec convention. `--spacing vmaf` spaces rungs by equal
VMAF gaps, matching how viewers perceive quality steps. `uniform` is
accepted as a legacy alias for `vmaf`.

### Phase E ladder CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--src PATH` | — | Required. Source label (sampling currently mocked). |
| `--encoder NAME` | `libx264` | Codec adapter (Phase A wires `libx264` only). |
| `--resolutions WxH,...` | `1920x1080,1280x720,854x480,640x360,426x240` | Canonical 5-rung. |
| `--target-vmafs F,...` | `95,90,85,75,65` | VMAF targets per resolution. |
| `--quality-tiers N` | `5` | Rungs to pick from the Pareto hull. |
| `--spacing` | `log_bitrate` | `log_bitrate` (HLS spec) or `vmaf` (perceptual); `uniform` is a legacy alias for `vmaf`. |
| `--format` | `hls` | `hls`, `dash`, or `json`. |
| `--with-uncertainty` | off | Apply the ADR-0279 prune/insert recipe. Sampled `vmaf_interval` payloads win; point-only rows use the active wide-interval threshold as a conservative fallback. |
| `--uncertainty-sidecar PATH` | default thresholds | Calibration sidecar for the uncertainty recipe. |
| `--rung-overlap-threshold F` | `0.5` | Adjacent-rung interval overlap threshold for pruning. |
| `--output PATH` | stdout | Manifest destination. |

## Phase F — multi-pass encoding (ADR-0333)

Phase F lights up 2-pass encoding for codecs that benefit. Default
behaviour stays single-pass; opting in via `--two-pass` runs the
encoder twice — pass 1 analyses the source and writes a stats file
to a temp directory, pass 2 reads those stats to make better
rate-allocation decisions.

### When to use it

2-pass encoding pays off most clearly in **target-bitrate** workflows
(VOD ladder generation, codec comparisons at fixed bitrate). Constant-
quality (CRF) encodes already adapt QPs frame-by-frame from the
encoder's lookahead, so the win at fixed CRF is more modest. Expect:

- **+1 to +3 VMAF points** at a fixed bitrate target on libx265 vs
  1-pass ABR (typical-content range; see x265 rate-control docs).
- **~2× encode wall time** — the second pass roughly doubles the cost.

### Quick start

```shell
vmaf-tune corpus \
  --source ref.yuv --width 1920 --height 1080 \
  --pix-fmt yuv420p --framerate 24 --duration 5 \
  --encoder libx265 --preset medium --crf 23 \
  --two-pass \
  --output corpus_2pass.jsonl
```

The driver materialises a per-encode stats file under
`tempfile.gettempdir()` (e.g. `/tmp/vmaftune-2pass-XXXXXX/`), runs
both passes back-to-back, and removes the stats file plus known
encoder sidecars when the run completes — successful or not.

### Codec support matrix

| Codec | `supports_two_pass` | Notes |
| --- | --- | --- |
| `libx265` | yes | First Phase F implementation. ADR-0333. |
| `libx264` | yes | FFmpeg-native `-pass` / `-passlogfile` wiring. |
| `libvpx-vp9` | yes | FFmpeg-native `-pass` / `-passlogfile`; adapter emits VP9 CRF mode via `-b:v 0`. |
| `libsvtav1` | not yet | Sibling PR planned. Uses `-svtav1-params passes=2`. |
| `libvvenc` | not yet | Sibling PR planned. Uses `-vvenc-params`. |
| `libaom-av1` | not yet | Possible sibling PR; encode time prohibitive on long sources. |
| `*_nvenc` (NVIDIA) | no | NVENC's `-multipass` is a single-invocation lookahead, not the stats-file two-call sequence. Separate adapter contract. |
| `*_amf` / `*_qsv` / `*_videotoolbox` | no | Hardware encoders use internal lookahead; no stats-file 2-pass exposed. |

When `--two-pass` is set against a codec where `supports_two_pass = False`,
vmaf-tune writes a one-line warning to stderr and runs single-pass.
(Mirrors the saliency.py "unsupported ROI encoder, fallback to plain encode"
precedent.) To fail loud instead, callers using the Python API can
pass `on_unsupported="raise"` to `run_two_pass_encode`.

### Cache interaction

The content-addressed encode cache (ADR-0298) keys on pass count, so
a 1-pass encode and a 2-pass encode of the same `(src, codec, preset,
crf)` are distinct cache entries — the cache will never serve a
1-pass encode for a 2-pass request.

### Sample-clip composition

Sample-clip mode (ADR-0297) composes with 2-pass: both passes apply
the same `-ss <start> -t <N>` input slice, and the per-encode stats
file is unique per slice. No special handling required.

## What Phase A / E / F do **not** do

- No target-VMAF bisect (Phase B). Phase E uses the canonical
  5-point CRF sweep as its production default sampler.
- No per-title or per-shot CRF prediction (Phase C / D).
- No real-corpus end-to-end ladder validation against a Netflix per-
  title baseline yet.
- 2-pass on codecs other than `libx264` / `libx265` / `libvpx-vp9`
  (Phase F sibling PRs land one-file-at-a-time per the ADR-0288 /
  ADR-0333 pattern).
- Encoder-stats parsing for `libvpx-vp9`: FFmpeg's libvpx passlog is
  binary first-pass data, not the x264/x265 text schema consumed by
  `encoder_stats.py`.

## Phase A.5 — opt-in `fast` recommend

The `fast` subcommand
([ADR-0276](../adr/0276-vmaf-tune-fast-path.md),
[Research-0060](../research/0060-vmaf-tune-fast-path.md)) is an
opt-in recommendation surface that combines three acceleration
levers — VMAF proxy via `fr_regressor_v2`, Bayesian search via
Optuna's TPE sampler, and GPU-accelerated VMAF for the verify step —
to replace the exhaustive grid for the *recommendation* use case.
The slow `corpus` path stays the canonical ground truth. Production
mode runs the sample encode, extracts canonical-6 features, calls the
`fr_regressor_v2` ONNX proxy, and performs a single verify pass through
the selected VMAF backend. `--smoke` is still available for dependency-
free CI and local plumbing checks.

### Install

```shell
pip install -e 'tools/vmaf-tune[fast]'   # adds Optuna
```

The core install path stays zero-dependency; the `[fast]` extra is
strictly opt-in.

### Smoke-mode quick start

```shell
vmaf-tune fast --smoke --target-vmaf 92
```

Smoke mode runs Optuna over a synthetic x264-shaped CRF→VMAF curve.
No ffmpeg, no ONNX Runtime, no GPU is touched. Output is a single
JSON object:

```json
{
  "encoder": "libx264",
  "target_vmaf": 92.0,
  "recommended_crf": 18,
  "predicted_vmaf": 92.39,
  "predicted_kbps": 4121.09,
  "n_trials": 50,
  "smoke": true,
  "notes": "smoke mode — synthetic predictor; ..."
}
```

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--src PATH` | — | Source video. Required outside `--smoke`. |
| `--target-vmaf F` | `92.0` | Quality target on VMAF [0, 100] scale. |
| `--encoder NAME` | `libx264` | Codec adapter (Phase A.5: x264 only). |
| `--crf-lo N` | `10` | Lower bound of the CRF search. |
| `--crf-hi N` | `51` | Upper bound of the CRF search. |
| `--n-trials N` | `50` | Optuna TPE trial count. |
| `--time-budget-s N` | `300` | Soft wall-clock budget for Optuna. Completed trial count may be lower than `--n-trials` when the timeout is hit. |
| `--smoke` | off | Synthetic predictor — exercises the pipeline without ffmpeg / ONNX. |

### Speedup model

Per Research-0060 §Speedup model:

| Combination | Speedup vs Phase A grid |
| --- | --- |
| Phase A grid (baseline) | 1× |
| `fast` (proxy + Bayesian + GPU verify) | ≈20–50× |
| `fast` + NVENC (lever C, follow-up) | ≈100–500× |

These are upper bounds. The production claim is gated on a
recommendation-quality benchmark against the slow grid.

### Production limits

`fast` is a recommendation shortcut, not a corpus generator. It still
needs a representative source clip, a usable encoder, and a VMAF binary
for verification. When the proxy / verify gap exceeds
`--proxy-tolerance`, the CLI exits with code `3` so operators can fall
back to the slow grid:

```shell
vmaf-tune fast --src ref.yuv --target-vmaf 92 || \
    vmaf-tune recommend --from-corpus corpus.jsonl --target-vmaf 92
```

Per-shot parallelisation remains a separate integration with TransNet
V2 and `vmaf-perShot`.

`fast` accepts any registered codec adapter in production mode. Smoke
mode remains synthetic and x264-shaped by design.

## libvpx-vp9 example

The VP9 adapter routes `--encoder libvpx-vp9` through FFmpeg's
`libvpx-vp9` wrapper. It maps the shared preset vocabulary to
`-deadline good -cpu-used N`, emits `-crf`, and forces VP9
constant-quality mode with `-b:v 0`.

```shell
vmaf-tune corpus \
    --encoder libvpx-vp9 \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --preset medium --preset fast \
    --crf 28 --crf 32 --crf 36 \
    --output corpus_vp9.jsonl
```

`--two-pass` is supported for VP9 via FFmpeg's generic `-pass` /
`-passlogfile` switches. Per-frame encoder-stats columns remain zero
for VP9 until a binary libvpx first-pass parser lands.

## x265 example

x265 ships ten presets (`ultrafast` … `placebo`) on the same 0..51 CRF
scale as x264; the harness routes `--encoder libx265` through ffmpeg's
`-c:v libx265` path. External `ffmpeg` must be built with
`--enable-libx265`.

```shell
vmaf-tune corpus \
    --encoder libx265 \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --preset medium --preset slow --preset placebo \
    --crf 23 --crf 28 --crf 34 \
    --output corpus_x265.jsonl
```

The 10-bit pipeline is enabled by setting `--pix-fmt yuv420p10le`; the
adapter reports the corresponding HEVC profile (`main10`) via
`X265Adapter.profile_for(pix_fmt)` for downstream consumers that need
it.

The software adapters (`libx264`, `libx265`, `libsvtav1`,
`libaom-av1`, `libvpx-vp9`, `libvvenc`) all route through the same
codec-adapter registry; the search loop does not branch on codec name.

## VVenC (H.266 / VVC)

`vmaf-tune` ships a `libvvenc` codec adapter
([ADR-0285](../adr/0285-vmaf-tune-vvenc-nnvc.md)) that drives
Fraunhofer HHI's open-source VVC encoder via FFmpeg's `-c:v libvvenc`
wrapper. VVC is the ITU-T / ISO standard that succeeds HEVC and
delivers roughly 30-50% better compression at equal quality. As a
rough rule of thumb, VVenC `slow` is ~5-10% better quality than HEVC
`slower` at the same bitrate but ~3-5× slower wall-clock — VVC is the
"opt in to longer encodes for tighter bitrates" branch of the adapter
set.

### Quality knob and presets

| Property | Value |
| --- | --- |
| Quality knob | `qp` (forwarded as the integer that the harness's `--crf` flag carries; VVenC's wrapper accepts the value regardless of label) |
| Quality range | `[17, 50]` (perceptually informative window; full VVenC scale is 0..63) |
| Default | `32` |
| Native presets | `faster, fast, medium, slow, slower` (5 levels) |

The harness's canonical 7-name preset vocabulary
(`placebo / slowest / slower / slow / medium / fast /
faster / veryfast / superfast / ultrafast`) compresses onto VVenC's
native 5 levels via a static map: anything strictly slower than `slow`
pins to `slower`; anything strictly faster than `fast` pins to
`faster`; the central three names map identically. This matches the
projection rule used by the parallel HEVC and AV1 adapters so that
predictor inputs stay codec-uniform.

### Tuning surface (real VVenC 1.14.0 knobs)

The adapter exposes a curated subset of VVenC's config keys via
FFmpeg's `-vvenc-params key=value:key=value` channel. Keys are sourced
verbatim from
[`source/Lib/apputils/VVEncAppCfg.h`](https://github.com/fraunhoferhhi/vvenc/blob/v1.14.0/source/Lib/apputils/VVEncAppCfg.h)
at tag `v1.14.0` (SHA `9428ea8636ae7f443ecde89999d16b2dfc421524`,
accessed 2026-05-09). Every knob defaults to `None` (library default
preserved); the search loop opts into a non-default value only when
the corpus row records it.

| Knob | VVenC key | Default | Effect / typical use |
| --- | --- | --- | --- |
| `perceptual_qpa` | `PerceptQPA` | library default | XPSNR-driven perceptual QPA. Materially shifts the rate-distortion curve; recorded per-row in `encoder_extra_params` for predictor conditioning. |
| `internal_bitdepth` | `InternalBitDepth` | library default (10 for VVC) | Force 8- or 10-bit internal precision; necessary for HDR profiles. |
| `tier` | `Tier` | library default (`main`) | `main` or `high`. Caps signalled max bitrate / resolution. |
| `tiles` | `Tiles` | single tile | `(cols, rows)` partitioning, emitted as `NxM`. Useful for parallel encode of high-resolution content. |
| `max_parallel_frames` | `MaxParallelFrames` | library default (auto) | Parallel-frames perf knob; `0` disables, `>=2` enables. |
| `rpr` | `RPR` | library default | VVC reference-picture-resampling. `0` off / `1` on / `2` RPR-ready. |
| `sao` | `SAO` | library default (on) | Sample Adaptive Offset loop filter. Useful for ablation studies. |
| `alf` | `ALF` | library default | Adaptive Loop Filter; useful for ablation. |
| `ccalf` | `CCALF` | library default | Cross-Component ALF (only meaningful when `alf` is on). |

Toggles are emitted in field-declaration order so the argv stays
byte-stable for cache-key hashing (per
[ADR-0298](../adr/0298-vmaf-tune-cache-key.md)). The `adapter_version`
field bumps to `"2"` for the 2026-05-09 surface — stale cached
results are invalidated automatically.

### NN-VC status (deferred)

VVC the standard defines NN-VC tool-points (NN-based intra prediction,
NN-based loop filter, NN-based super-resolution), but **VVenC 1.14.0
does not ship implementations of any of them**. An earlier draft of
this adapter exposed an `nnvc_intra` toggle that emitted
`-vvenc-params IntraNN=1`; that key has never existed in any released
VVenC and has been removed (see
[ADR-0285](../adr/0285-vmaf-tune-vvenc-nnvc.md) §"Status update
2026-05-09"). If upstream VVenC ever lands NN-VC tools the adapter
will pick them up via the placeholder pattern from
[ADR-0294](../adr/0294-vmaf-tune-codec-dispatcher.md)'s
self-activating adapter set.

### External binary requirements

Running the VVenC adapter end-to-end requires:

- `ffmpeg` compiled with `--enable-libvvenc` on `PATH` (or
  `--ffmpeg-bin`).
- The `libvvenc` shared library and headers from
  <https://github.com/fraunhoferhhi/vvenc>.

The shipped unit tests mock `subprocess.run` so the adapter can be
exercised without either binary present; integration smoke is gated
to a CI runner that has a `libvvenc`-enabled FFmpeg.

## Phase D — per-shot CRF tuning

The `tune-per-shot` subcommand drives the Netflix-style per-shot
encoding path. It cuts the source into shots
(via the C-side [`vmaf-perShot`](vmaf-perShot.md) binary, which wraps
TransNet V2 — see [ADR-0223](../adr/0223-transnet-v2-shot-detector.md)),
extracts each shot to a temporary raw-YUV reference, runs the Phase-B
target-VMAF bisect for that shot, and emits an FFmpeg encoding plan
that produces one segment per shot plus a final concat-demuxer command.

The target-VMAF predicate remains pluggable for advanced callers:
`--predicate-module MODULE:CALLABLE` bypasses the default bisect path,
and the Python API still accepts `tune_per_shot(..., predicate=...)`.
Codec emission is still portable segment-and-concat output rather than
native per-shot mechanisms (`--qpfile` for x264, `--zones` for x265, the
SVT-AV1 segment table).

Design rationale and the decision matrix live in
[ADR-0392](../adr/0392-vmaf-tune-phase-d-per-shot.md).

### Quick start

```shell
vmaf-tune tune-per-shot \
    --src ref.mp4 \
    --width 1920 --height 1080 \
    --framerate 24 \
    --target-vmaf 92 \
    --encoder libx264 \
    --output per_shot_encode.mp4 \
    --plan-out plan.json
```

The plan is emitted to stdout as JSON unless `--plan-out` is
specified. Pass `--script-out plan.sh` to also receive a copy-paste
shell script of the per-segment + concat commands.

### CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--src PATH` | — | Required. Source video (raw YUV or container). |
| `--width / --height` | — | Required. Source resolution. |
| `--pix-fmt PFMT` | `yuv420p` | Forwarded to `vmaf-perShot`. |
| `--framerate F` | `24.0` | Used to translate frame counts to `-ss` seek seconds. |
| `--target-vmaf V` | `92.0` | Per-shot quality target. |
| `--encoder NAME` | `libx264` | Any registered codec adapter accepted by the Phase-B bisect backend. |
| `--bitdepth N` | `8` | Forwarded to `vmaf-perShot` (`8`, `10`, or `12`). |
| `--total-frames N` | `0` | Frame count for the single-shot fallback when `vmaf-perShot` is unavailable. |
| `--per-shot-bin PATH` | `vmaf-perShot` | Override the shot detector binary. |
| `--ffmpeg-bin PATH` | `ffmpeg` | Override the FFmpeg binary. |
| `--vmaf-bin PATH` | `vmaf` | Override the libvmaf CLI used by the per-shot scorer. |
| `--preset NAME` | adapter default | Codec preset forwarded to Phase-B bisect. |
| `--crf-min / --crf-max` | adapter range | Optional inclusive CRF search bounds; pass both or neither. |
| `--max-iterations N` | `8` | Maximum encode+score iterations per detected shot. |
| `--vmaf-model NAME` | `vmaf_v0.6.1` | VMAF model forwarded to the per-shot scorer. |
| `--score-backend NAME` | `auto` | libvmaf scoring backend for the per-shot scorer (`auto`, `cpu`, `cuda`, `sycl`, `vulkan`). |
| `--predicate-module SPEC` | — | Advanced hook `MODULE:CALLABLE` matching `(shot, target_vmaf, encoder) -> (crf, measured_vmaf)`; bypasses real bisect. |
| `--output PATH` | `per_shot_encode.mp4` | Final concatenated encode destination. |
| `--segment-dir PATH` | `<output>.parent/segments` | Directory for per-shot segment files. |
| `--plan-out PATH` | stdout | Write the JSON plan here instead of stdout. |
| `--script-out PATH` | — | Optional: also emit a copy-paste shell script. |

### Plan JSON schema

```json
{
  "encoder": "libx264",
  "framerate": 24.0,
  "predicate": "bisect",
  "target_vmaf": 92.0,
  "shots": [
    {"start_frame": 0, "end_frame": 24, "crf": 22, "predicted_vmaf": 93.0},
    {"start_frame": 24, "end_frame": 72, "crf": 26, "predicted_vmaf": 92.5}
  ],
  "segment_commands": [
    ["ffmpeg", "-y", "-hide_banner", "-ss", "0.000000", "-i", "ref.mp4",
     "-frames:v", "24", "-c:v", "libx264", "-crf", "22",
     "/tmp/segments/shot_0000.mp4"]
  ],
  "concat_command": [
    "ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0",
    "-i", "/tmp/segments/concat.txt", "-c", "copy", "out.mp4"
  ]
}
```

`start_frame` is inclusive, `end_frame` is exclusive (Python-slice
convention). The `vmaf-perShot` CSV/JSON sidecar uses inclusive
`end_frame`; the planner normalises into the half-open form and the
segment commands honour the half-open semantics via `-frames:v`.

### Single-shot fallback

If the `vmaf-perShot` binary is not on PATH, or it exits non-zero, the
planner falls back to a single shot covering the whole clip
(`[0, --total-frames)`). This keeps `tune-per-shot` usable as a smoke
test on machines that have not built the shot-detector binary yet.

### What Phase D does **not** do

- Does not run the encodes — only emits the plan. Pipe
  `--script-out plan.sh` through `sh` to execute it manually.
- Does not emit native per-codec per-shot mechanisms (x264 `--qpfile`,
  x265 `--zones`, SVT-AV1 segment tables). Per-segment encode plus
  concat-demuxer is the portable fallback.
- Does not handle GOP-aligned shot boundaries — the per-segment
  approach side-steps this by re-encoding each shot from frame 0.

## auto

`vmaf-tune auto` is the Phase F entry point (ADR-0325). One CLI verb
composes the per-phase subcommands (`corpus`, `recommend`, `predict`,
`tune-per-shot`, `recommend-saliency`, `ladder`, `compare`) plus the
orthogonal modes (HDR auto-detect, sample-clip, resolution-aware) into
a deterministic decision tree. F.1 shipped the sequential composition;
F.2-F.5 added the short-circuits, confidence-aware fallbacks, and
per-content recipes.

### Synopsis

```shell
vmaf-tune auto \
    --src reference.mp4 \
    --target-vmaf 93 \
    --max-budget-bitrate 5000 \
    --allow-codecs libx264,libx265 \
    [--codec libx265] \
    [--sample-clip-seconds 10] \
    [--smoke] \
    [--output plan.json]
```

The non-smoke path probes source geometry, source duration, and HDR
metadata through the same ffprobe/HDR helpers used by the corpus path.
Probe failures degrade to conservative 1920x1080 SDR defaults so the
planner can still emit a JSON plan and expose which later stages need
real evidence. `--smoke` still exercises the composition end-to-end
with mocked sub-phases (no ffmpeg, no ONNX). The JSON plan emitted under
`metadata.short_circuits` records which short-circuits fired; post-hoc
analysis uses this to measure the speedup contribution of each one.
For each non-smoke cell, `auto` now feeds the probed metadata into the
existing `Predictor` path, picks a codec-specific CRF for
`metadata.effective_predictor_target_vmaf`, and records predictor
estimates for `estimated_vmaf` and `estimated_bitrate_kbps`. These are
planner estimates, not measured encode results, until the future
realise/encode step scores the chosen cells. The per-cell
`prediction_source` key distinguishes the production estimate path
(`"predictor"`) from the `--smoke` composition placeholder
(`"smoke-placeholder"`).

### Short-circuits

The ten short-circuits below are the F.2 surface. Each one is a
guarded fast-path with one trigger condition; the predicates are
exposed as `_should_short_circuit_<N>` helpers in
`tools/vmaf-tune/src/vmaftune/auto.py` so they can be unit-tested in
isolation.

| # | Identifier | Trigger | Skips |
|---|------------|---------|-------|
| 1 | `ladder-single-rung` | `meta.height < 2160` | Multi-rung ABR ladder evaluation (ADR-0289 / ADR-0295). |
| 2 | `codec-pinned` | `--codec` set or `--allow-codecs` resolves to one entry | The `compare.shortlist` stage. |
| 3 | `predictor-gospel` | `predict.crf_for_target` returns `GOSPEL` (ADR-0306) | The `recommend.coarse_to_fine` fallback for that cell. |
| 4 | `skip-saliency` | `meta.content_class` is photographic / live-action (not animation / screen content) | The `recommend_saliency.maybe_apply` stage (ADR-0293). |
| 5 | `sdr-skip` | `not meta.is_hdr` (per ADR-0300 detector) | The HDR resolution + model-selection branch. |
| 6 | `sample-clip-propagate` | `--sample-clip-seconds > 0` | Re-deciding clip length per stage; the user-supplied value propagates verbatim (ADR-0301). |
| 7 | `skip-per-shot` | `duration < 5min` AND `shot_variance < 0.15` | The `tune_per_shot.refine` pass (ADR-0392). |
| 8 | `low-complexity` | `meta.complexity_score < 200 kbps` (probe-encode bitrate) | The `recommend.coarse_to_fine` sweep — the predictor's point estimate is already tight on simple content. `0.0`/`NaN` does not fire (no probe run yet). |
| 9 | `baseline-meets-target` | `meta.baseline_vmaf >= target_vmaf` | The full predictor sweep — the default-CRF encode already satisfies the quality target. `0.0`/`NaN` does not fire (no baseline scored yet). |
| 10 | `no-two-pass` | `adapter.supports_two_pass == False` (ADR-0333) | The two-pass calibration stage. Hardware encoders (`*_nvenc`, `*_amf`, `*_qsv`, `*_videotoolbox`) and most software encoders fire this. `libx264` and `libx265` currently set `supports_two_pass = True`. |

The 5-min and 0.15 thresholds in short-circuit #7 are placeholders;
F.3 fits them empirically once Phase F has emitted enough labelled
compositions to make the fit statistically defensible. The constants
live at the top of `auto.py` (`PHASE_D_DURATION_GATE_S`,
`PHASE_D_SHOT_VARIANCE_GATE`) so the eventual fit lands as a one-line
edit. The 200 kbps threshold in short-circuit #8 is likewise a
placeholder: `LOW_COMPLEXITY_PROBE_BITRATE_THRESHOLD_KBPS` at the top
of `auto.py`.

The evaluation order in `SHORT_CIRCUIT_PREDICATES` is part of the
public contract: tests assert that an earlier-firing predicate doesn't
shadow a later one whose result would have been different. Adding a
new short-circuit means appending; never reordering.

`auto` does **not** dispatch the `fast` subcommand from inside its
tree. `fast` (ADR-0276 fast-path) is a different operator surface
(proxy + Bayesian over a single codec) and remains a sibling, not a
child, of `auto`.

For HDR sources, every emitted cell records the codec-specific
`hdr_args` produced by `vmaftune.hdr.hdr_codec_args(codec, info)`.
That means x264 gets only container-level `-color_*` flags, x265 gets
`-x265-params` SEI signalling, SVT-AV1 gets `-svtav1-params`, and SDR
cells record an empty list after the `sdr-skip` short-circuit fires.

### Winner selection

`auto` now finishes the planning pass by selecting one estimated cell.
The chosen cell is marked with `"selected": true` in `cells[]`; every
other cell has `"selected": false`. The same decision is copied to
`metadata.winner` so scripts can read a stable object without scanning
the cell array.

Winner statuses:

| Status | Meaning |
| --- | --- |
| `budget_and_quality_met` | At least one cell met `--target-vmaf` and `--max-budget-bitrate`; the lowest estimated bitrate wins. |
| `quality_met_budget_exceeded` | No cell was inside budget, but at least one cell met the quality target; the smallest budget overage wins. |
| `target_unmet` | No cell met the quality target; the closest estimated VMAF miss wins so the caller still gets a concrete next encode. |
| `no_eligible_cells` | No cell carried finite `estimated_vmaf` and `estimated_bitrate_kbps`; this is an input/planner evidence failure. |

`metadata.winner` records `cell_index`, `rung`, `codec`, `crf`,
`estimated_vmaf`, `estimated_bitrate_kbps`, `quality_margin`, and
`budget_margin_kbps`. This is still a planning result: the selected
cell is the next encode target, not a substitute for the final
encode/score verification pass.

### Execute mode (ADR-0454)

After the planning pass, `--execute` drives real FFmpeg encodes and libvmaf scores
for the selected cell(s):

```shell
vmaf-tune auto \
    --src reference.mp4 \
    --target-vmaf 93 \
    --max-budget-bitrate 5000 \
    --allow-codecs libx264,libx265 \
    --output plan.json \
    --execute \
    --runs-dir runs/
```

`--execute` flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--execute` | off | Enable execute mode; plan-only when absent. |
| `--runs-dir PATH` | `runs/` | Destination for encoded files and `tune_results.jsonl`. |
| `--execute-all` | off | Run every plan cell instead of only the `selected` winner. |

Results are appended to `<runs-dir>/tune_results.jsonl` one row per executed
cell. Each row carries the cell metadata (codec, preset, CRF, estimated
VMAF/bitrate) merged with encode outcomes (size, encode time, FFmpeg version)
and score outcomes (measured VMAF, per-feature means/stds, vmaf-binary version).
The file appends on each run so partial runs and incremental re-runs do not
overwrite previous results.

The CLI exits with status 1 if at least one cell was executed but none scored
successfully (encode failures, vmaf binary absent, etc.); it exits 0 on plan-only
runs regardless of plan content.

#### Per-shot execution (ADR-0468)

`run_plan_per_shot` splits the source into shot boundaries (via `vmaf-perShot` /
TransNet V2, [ADR-0222](../adr/0222-vmaf-pershot-transnet-v2.md)) and scores each
segment independently. Results land in `<runs-dir>/tune_results_per_shot.jsonl`:

```python
from vmaftune.executor import run_plan_per_shot

per_shot_results = run_plan_per_shot(
    plan, src=Path("reference.mp4"), out_dir=Path("runs/"),
    vmaf_model="vmaf_v0.6.1",
)
for r in per_shot_results:
    print(f"cell {r.row['cell_index']}: "
          f"{r.row['shot_count']} shots, "
          f"weighted VMAF={r.weighted_vmaf:.2f}")
```

Each top-level row carries `shot_count` and `weighted_vmaf` (frame-length-weighted
mean of per-shot scores). When `vmaf-perShot` is absent, the call falls back to a
single-shot range covering the whole clip; `shot_count == 1` signals this.

#### Saliency execution (ADR-0468)

`run_plan_saliency` applies saliency-aware encoding via
`saliency_aware_encode` (see [§ Saliency-aware encoding](#saliency-aware-encoding-recommend-saliency---saliency-aware))
before scoring. Results land in `<runs-dir>/tune_results_saliency.jsonl`:

```python
from vmaftune.executor import run_plan_saliency

sal_results = run_plan_saliency(
    plan, src=Path("reference.mp4"), out_dir=Path("runs/"),
    saliency_model_path=Path("model/tiny/saliency_student_v1.onnx"),
    duration_frames=total_frame_count,
)
for r in sal_results:
    print(f"cell {r.row['cell_index']}: "
          f"saliency_available={r.saliency_available}, "
          f"VMAF={r.row['vmaf_score']}")
```

`saliency_available` in the result row is `True` when the ONNX model ran; `False`
means the encoder fell back to a plain encode (model file missing or onnxruntime not
installed). The encode and score always proceed regardless.

### Confidence-aware fallbacks (F.3)

F.2 treats the predictor's verdict as a binary GOSPEL / FALL_BACK gate
(short-circuit #3). F.3 makes the gate **continuous** by consulting
the conformal interval half-width returned by
`Predictor.predict_vmaf_with_uncertainty`
([ADR-0279](../adr/0279-fr-regressor-v2-probabilistic.md)). Two width
gates carve the half-width axis into three regions:

| Interval width | Outcome | Effect on F.2 |
|----------------|---------|---------------|
| `width <= tight_interval_max_width` | `SKIP_ESCALATION` | Predictor is confident; trust the point estimate even when the native verdict said `FALL_BACK`. |
| `tight < width < wide` | `RECOMMEND_ESCALATION` (on `FALL_BACK` / unknown) or `SKIP_ESCALATION` (on `GOSPEL` / `LIKELY`) | Defer to the native verdict — exactly the F.2 behaviour. |
| `width >= wide_interval_min_width` | `FORCE_ESCALATION` | Predictor is uncertain; escalate to `recommend.coarse_to_fine` even when the native verdict said `GOSPEL`. |

The two thresholds are corpus-derived. The conformal-VQA calibration
pipeline ships a JSON sidecar with the canonical keys
`tight_interval_max_width` and `wide_interval_min_width`; the loader
honours per-corpus overrides transparently. When no sidecar is found
the loader falls back to the Research-0067 emergency floor (`2.0` /
`5.0` VMAF) and emits a one-line warning; the floor is documented
behaviour, not a magic constant.

Per-cell decisions are recorded in
`plan.metadata.confidence_aware_escalations[]` (one entry per cell,
keyed by `rung`, `codec`, `verdict`, `interval_width`, `decision`),
and each cell in `plan.cells[]` carries its own
`confidence_decision` + `interval_width` keys so JSON consumers don't
need to cross-reference the metadata array index.

Per CLAUDE.md `feedback_no_test_weakening`: the thresholds are
calibration outputs. If a sidecar value triggers surprising cell
escalations on real data, the fix is a recalibration PR — not a
loosening of the F.3 gate here.

The helper `_confidence_aware_escalation(verdict, interval_width,
thresholds)` in `tools/vmaf-tune/src/vmaftune/auto.py` is exposed for
unit testing and direct embedding by downstream tools (the MCP
server's `auto` proxy, the CI corpus collector). It is a pure
function of its three inputs.

### Per-content-type recipes (F.4)

F.4 layers per-content-type **recipe overrides** on top of F.1+F.2+F.3.
When the upstream classifier (`per_shot.detect_shots` plus the
fork-local content-class heuristics) tags a source as `animation`,
`screen_content`, `live_action_hdr`, or `ugc`, the auto driver applies
a small override dict **before** the F.2 short-circuits evaluate so a
recipe can flip `force_single_rung` and have the ladder stage honour it.
Any source whose `meta.content_class` doesn't match a named recipe
falls through to the empty `default` recipe.

The override keys consumed by the driver are:

| Key | Type | Effect |
| --- | ---- | ------ |
| `tight_interval_max_width` | `float` | Narrows / widens the F.3 conformal-tight gate. |
| `force_single_rung` | `bool` | Arms short-circuit #1 (`ladder-single-rung`) even on >= 2160p sources. |
| `saliency_intensity` | `str` | Passed through to the saliency stage when not skipped. One of `default`, `aggressive`, `very_aggressive`. |
| `target_vmaf_offset` | `float` | Additive offset applied to the *predictor's* effective target VMAF. The input `--target-vmaf` (the gate that ships models) is **never** shifted by this value — see the no-test-weakening note below. |

The five recipe classes ship the following overrides. The values below
are the **F.5-calibrated** thresholds emitted by
`ai/scripts/calibrate_phase_f_recipes.py` and shipped in
`ai/data/phase_f_recipes_calibrated.json`. The calibration was run on
2026-05-09 against the K150K corpus
(`.workingdir2/konvid-150k/konvid_150k.jsonl`, 148 543 rows out of an
expected 153 841 — the ingestion was ~96.6 % complete; a re-run on the
full corpus is a follow-up PR). Threshold rationale and the per-class
proxy-vs-corpus provenance break-down live in
[Research-0067 §"F.4 recipe-override placeholders"](../research/0067-vmaf-tune-phase-f-feasibility-2026-05-08.md)
plus the JSON `metadata` block.

| Class | `tight_interval_max_width` | `force_single_rung` | `saliency_intensity` | `target_vmaf_offset` | Source |
| ----- | -------------------------- | ------------------- | -------------------- | -------------------- | ------ |
| `animation` | `1.75` | `true` | `aggressive` | `+2.0` | proxy (UGC-anchored) |
| `screen_content` | *(unset)* | *(unset)* | `very_aggressive` | `+1.0` | proxy (UGC-anchored) |
| `live_action_hdr` | `1.4` | *(unset)* | *(default)* | `0.0` | proxy (UGC-anchored) |
| `ugc` | `3.5` | `false` | `default` | `+1.5` | corpus (K150K) |
| `default` | *(unset)* | *(unset)* | *(default)* | `0.0` | n/a |

K150K is a UGC-only corpus and carries no per-source `content_class`
column; only the `ugc` row is corpus-derived. The other three rows are
calibrated as documented absolute offsets ("proxy") anchored on the F.4
envelope until PR #477's TransNet shot-metadata columns plus a
class-labelled subset land. The JSON `recipes.<class>._provenance`
sub-dict records the source per row so future re-calibrations can
distinguish corpus-derived from proxy-derived values. UGC's
`target_vmaf_offset` came out empirically positive (`+1.5`) on K150K
because the corpus's MOS distribution has a heavier upper tail than
lower tail; the calibration script clamps every offset to the F.4
documented envelope of `[-2.0, +2.0]` so a pathological corpus cannot
push the predictor target outside the regime the planner has been
exercised against.

The `auto.py` runtime loads the JSON at module import via
`_load_calibrated_recipes`; if the JSON file is missing or malformed,
the F.4 placeholder constants in `_F4_PLACEHOLDER_RECIPES` apply as a
graceful fallback. To regenerate the calibration after the corpus
ingestion completes (or when a class-labelled corpus replaces K150K),
run:

```shell
python ai/scripts/calibrate_phase_f_recipes.py \
    --corpus .workingdir2/konvid-150k/konvid_150k.jsonl \
    --out ai/data/phase_f_recipes_calibrated.json
```

Recipe rationale (each cited threshold is provisional pending F.5
calibration):

- **Animation** — predictor residuals are tighter on flat colour
  fields, single-rung ladder is sufficient, and saliency is more
  aggressive on cel-line edges. Animation is intrinsically more
  compressible at a given perceptual quality, so the predictor aims
  ~2 VMAF higher.
- **Screen content** — split-frame structure (low-entropy
  background + high-detail text/icon regions) benefits from
  `very_aggressive` saliency that raises QP on the background while
  keeping text near-lossless. Predictor target nudged +1.
- **Live-action HDR** — per [ADR-0300](../adr/0300-vmaf-tune-hdr-aware.md)
  the HDR pipeline already runs; the F.3 conformal-tight gate is
  narrowed to `1.4` because a wide predictor interval on HDR is more
  suspect than on SDR (the predictor was largely trained on SDR per
  [ADR-0279](../adr/0279-fr-regressor-v2-probabilistic.md)).
- **UGC** — user-generated content carries higher upstream-encode
  noise, inconsistent grading, and resolution mismatches; predictor
  uncertainty is the baseline. Widening the F.3 tight gate to `3.5`
  avoids over-flagging UGC cells as "needs escalation" simply because
  the interval is wider than a Netflix-grade reference. The K150K
  calibration nudges the predictor target **up** 1.5 because the
  corpus MOS distribution has a heavier upper tail.

The recipe class is recorded in `plan.metadata.recipe_applied` (one
of `animation`, `screen_content`, `live_action_hdr`, `ugc`, or
`default`) and the override dict in `plan.metadata.recipe_overrides`.
Each cell in `plan.cells[]` also carries the resolved
`saliency_intensity` and `effective_predictor_target_vmaf` so JSON
consumers don't need to cross-reference the metadata block.

Per `CLAUDE.md` memory `feedback_no_test_weakening`: recipe overrides
**MUST NOT** silently widen the production-flip gate that ships
models. They affect predictor thresholds (the `effective_predictor_target_vmaf`
that the predictor aims for; the F.3 width gate that decides per-cell
escalation), not the input `--target-vmaf` that downstream consumers
treat as the contract. The driver records the input `target_vmaf`
verbatim in `plan.metadata.target_vmaf` and the offset target in
`plan.metadata.effective_predictor_target_vmaf`; the two are kept
distinct.

The helper `_apply_recipe_override(meta, plan_state, thresholds)` in
`tools/vmaf-tune/src/vmaftune/auto.py` resolves the recipe and returns
a `(recipe_class, recipe, effective_thresholds)` triple; the
`get_recipe_for_class(content_class)` helper returns a fresh override
dict for any of the five canonical class strings. Both are pure
functions; the table at module scope (`_CONTENT_RECIPE_TABLE`) holds
factory callables so each call returns a fresh dict that callers may
mutate without affecting subsequent runs.

References:
[ADR-0325](../adr/0325-vmaf-tune-phase-f-auto.md) §F.4,
[ADR-0279](../adr/0279-fr-regressor-v2-probabilistic.md),
[ADR-0300](../adr/0300-vmaf-tune-hdr-aware.md),
[Research-0067](../research/0067-vmaf-tune-phase-f-feasibility-2026-05-08.md).

## Tests

```shell
pytest tools/vmaf-tune/tests/
```

The shipped suite mocks `subprocess.run` so it neither requires
`ffmpeg` nor a built `vmaf`. Real-binary integration coverage will land
when the codec adapter set widens.
