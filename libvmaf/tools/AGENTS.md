# AGENTS.md — libvmaf/tools

Orientation for agents working on the CLI binaries. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Three C binaries built by libvmaf's Meson tree:

- `vmaf` — the end-user scoring CLI
- `vmaf_bench` — micro-benchmark harness for extractors and backends
- `vmaf-perShot` — per-shot CRF predictor sidecar (T6-3b / ADR-0222)
- `vmaf_roi` — saliency-driven ROI sidecar emitter for x265 / SVT-AV1 (T6-2b)

```text
tools/
  vmaf.c              # main() + option dispatch for the vmaf CLI
  vmaf_bench.c        # main() + benchmark harness
  vmaf_per_shot.c     # main() + scan/predict for the perShot sidecar
  cli_parse.c/.h      # shared option parser (--precision, --tiny-model, …)
  vmaf_roi.c          # main() + sidecar pipeline for vmaf-roi
  vmaf_roi_core.h     # pure helpers (per-CTU mean reduce, saliency->QP)
```

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Default numeric precision is `%.6f`** (Netflix-compatible — required by
  CLAUDE.md §8 golden gate). `--precision=max` (alias `full`) opts in to
  `%.17g` (IEEE-754 round-trip lossless). `--precision=N` overrides with
  `"%.<N>g"`; `--precision=legacy` is preserved as a synonym for the default.
  See [ADR-0119](../../docs/adr/0119-cli-precision-default-revert.md)
  (supersedes [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md)).
  This applies to both stderr and file outputs (XML / JSON / CSV / sub-XML).
- **`--tiny-model PATH`** loads an ONNX checkpoint via
  [src/dnn/](../src/dnn/AGENTS.md). Path is resolved via `realpath` inside
  the loader; the CLI passes the string through unchanged. See
  [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md).
- **No new hard dependencies** — the CLI must still build when `enable_dnn=disabled`.
- **`--frame_skip_ref` / `--frame_skip_dist`** pre-loops in
  [vmaf.c](vmaf.c) MUST `vmaf_picture_unref()` each fetched picture
  immediately. The picture pool is always-on (see ADR-0104 below) and
  fixed-size; without unref the pool exhausts after N skips and the next
  fetch blocks indefinitely. Re-test with
  `python -m pytest python/test/command_line_test.py
  ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping` — if
  it hangs (timeout, no output), the unref is missing or wrong.
- **`vmaf_roi` sidecar contract** (T6-2b / ADR-0247) is
  **rebase-sensitive** — encoder drivers depend on the exact byte
  layouts:
  - `--encoder x265` emits ASCII per-row grid with two `#`-prefixed
    header lines (`# vmaf-roi qpfile (x265, --qpfile-style)` then
    `# frame=N ctu=S cols=C rows=R strength=F.FFF`), space-separated
    signed integers, one row per CTU row, `\n` terminator.
  - `--encoder svt-av1` emits exactly `cols * rows` bytes of `int8_t`,
    row-major, **no header**.
  - QP-offset clamp is `+-12` (`VMAF_ROI_CORE_QP_OFFSET_MAX`).
  - Reduction is per-CTU **mean** (not max — see ADR-0247 alternatives).
  - Pure helpers (`vmaf_roi_reduce_per_ctu`, `vmaf_roi_saliency_to_qp`)
    live in `vmaf_roi_core.h` so the smoke test compiles them
    without dragging libvmaf's link surface in. **Do not** move them
    into a `.c` TU without revisiting the test wiring.
  - The placeholder saliency map (when `--saliency-model` is absent)
    is for smoke-test plumbing only and explicitly documented as
    not-for-real-encodes in `docs/usage/vmaf-roi.md`.
  - `--bitdepth 8|10|12|16` is part of the input contract. High-bit-depth
    planar YUV uses little-endian 16-bit containers; frame seeking must
    count the chroma planes and sample width even though only luma enters
    the saliency path. The DNN-facing tensor remains luma8.
- **Long-only options must not pass synthesised short-option
  chars to `error()`** (rebase-sensitive). Handlers for
  `ARG_THREADS`, `ARG_SUBSAMPLE`, `ARG_CPUMASK`, and any
  future `ARG_*` enum value `>= 256` MUST pass that enum value
  (not a fabricated `'t'` / `'s'` / `'c'`) into
  `parse_unsigned()` / `parse_bitdepth()` / `error()`. The
  `error()` table-walk over `long_opts[]` for a non-existent
  short-option char trips `assert(long_opts[n].name)` and
  takes the binary down with `SIGABRT`. The
  `error()` `< 256` branch already handles long-only options
  via the `--name` path; passing the real enum value is
  required to reach it. See
  [ADR-0316](../../docs/adr/0316-cli-parse-long-only-error-fix.md);
  the parked-then-promoted reproducer
  `libvmaf/test/fuzz/cli_parse_corpus/cli_threads_abbrev_assert.argv`
  protects the rebase, and
  `libvmaf/test/test_cli_parse_long_only_args.c` protects
  the unit-test path.
- **`y4m_convert_411_422jpeg` chroma-row write guards are
  load-bearing** (rebase-sensitive). The 4:1:1 → 4:2:2-jpeg upsample
  in [y4m_input.c](y4m_input.c) writes both even and odd output
  sub-pixels per loop iteration. The destination chroma row width
  `dst_c_w` can be 1 (e.g. a width-2 frame: `dst_c_w = (2 + 2 - 1) /
  2 = 1`), in which case writing `_dst[(x << 1) | 1]` is a 1-byte
  heap-buffer-overflow. **All three sub-loops** in this routine must
  guard the secondary write with `(x << 1 | 1) < dst_c_w`. Upstream
  Daala / Netflix carry the same code shape; if `/sync-upstream`
  reintroduces the unguarded write, re-apply the fix. Regression
  test: `libvmaf/test/test_y4m_411_oob.c` (ASan-required to catch
  the regression deterministically).

## Governing ADRs

- [ADR-0119](../../docs/adr/0119-cli-precision-default-revert.md) — `%.6f`
  default (Netflix-compat) + `--precision=max` for round-trip lossless.
  Supersedes ADR-0006.
- [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md) — *Superseded.*
  Original `%.17g`-default decision; kept for history.
- [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md) — `--tiny-model`
  as one of four tiny-AI surfaces.
- [ADR-0222](../../docs/adr/0222-vmaf-per-shot-tool.md) — `vmaf-perShot`
  per-shot CRF predictor sidecar (T6-3b).
  - **Sidecar invariant**: this binary is **standalone** —
    it does not link the libvmaf metric path; its output is
    an encoder hint, not a quality score. Any future
    integration must keep the per-shot prediction outside
    `vmaf_score_*` to preserve roadmap §2.4's separation.
  - **Schema invariant**: CSV / JSON columns
    (`shot_id`, `start_frame`, `end_frame`, `frames`,
    `mean_complexity`, `mean_motion`, `predicted_crf`)
    are stable across v1; v2's trained MLP must reuse
    them to avoid downstream encoder churn.
  - **Input invariant**: `--pixel_format 420|422|444` only changes
    planar chroma-byte skipping. The per-shot detector and predictor
    remain luma-only, and high-bit-depth inputs use little-endian
    16-bit sample containers for `--bitdepth 10|12|16`.
- [ADR-0104](../../docs/adr/0104-picture-pool-always-on.md) — picture
  pool is always compiled in and sized for the live-picture set; this
  is what makes the `--frame_skip_*` unref invariant load-bearing.
- [ADR-0247](../../docs/adr/0247-vmaf-roi-tool.md) — `vmaf-roi`
  sidecar (per-CTU QP offsets for x265 / SVT-AV1). Encoder format
  contract + per-CTU-mean reduction are rebase-sensitive.
- [ADR-0461](../../docs/adr/0461-cli-validate-dimensions-chroma.md) —
  CLI rejects non-positive and chroma-misaligned input dimensions.
  **Validation invariant**: `validate_video_info()` and
  `validate_chroma_alignment()` are the canonical per-stream and
  chroma-alignment gates; if upstream Netflix adds similar checks to
  `validate_videos()` in a sync, merge rather than duplicate — keep the
  fork's helpers and call them from the merged body.
