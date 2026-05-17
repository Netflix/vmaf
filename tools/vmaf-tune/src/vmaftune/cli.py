# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""argparse entry-point for ``vmaf-tune``.

Phase A exposes one subcommand: ``corpus``. It expands a (preset, crf)
grid against one or more reference YUVs and emits a JSONL row per
encode. Phase B (``bisect``) and Phase C (``predict``) will register
sibling subcommands here.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import __version__
from .bisect import bisect_target_vmaf
from .codec_adapters import get_adapter, known_codecs
from .corpus import CorpusJob, CorpusOptions, coarse_to_fine_search, iter_rows, write_jsonl
from .encode import iter_grid
from .fast import (
    DEFAULT_CRF_HI,
    DEFAULT_CRF_LO,
    DEFAULT_PROXY_TOLERANCE,
    PROD_N_TRIALS,
    SMOKE_N_TRIALS,
    fast_recommend,
)
from .per_shot import PredicateFn as PerShotPredicateFn
from .per_shot import (
    Shot,
    detect_shots,
    merge_shots,
    plan_to_shell_script,
    tune_per_shot,
    write_concat_listing,
)
from .score_backend import ALL_BACKENDS, BackendUnavailableError, select_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmaf-tune",
        description=(
            "Quality-aware encode automation harness. Phase A drives a "
            "(preset, crf) grid through libx264 + libvmaf and emits a JSONL "
            "corpus."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    corpus = sub.add_parser("corpus", help="run the Phase A grid sweep + emit JSONL")
    corpus.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="raw YUV reference (repeat for multiple sources)",
    )
    corpus.add_argument("--width", type=int, required=True)
    corpus.add_argument("--height", type=int, required=True)
    corpus.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pix_fmt (default yuv420p)")
    corpus.add_argument("--framerate", type=float, default=24.0, help="reference framerate")
    corpus.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="reference duration in seconds (used for bitrate calc)",
    )
    corpus.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (default libx264; any registered adapter is accepted)",
    )
    corpus.add_argument(
        "--preset",
        action="append",
        required=True,
        help="x264 preset (repeatable)",
    )
    corpus.add_argument(
        "--crf",
        type=int,
        action="append",
        default=None,
        help=(
            "x264 CRF value (repeatable). Required unless "
            "--coarse-to-fine selects the CRF axis automatically."
        ),
    )
    corpus.add_argument(
        "--output",
        type=Path,
        default=Path("corpus.jsonl"),
        help="JSONL output path (default corpus.jsonl)",
    )
    corpus.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/encodes"),
        help="scratch dir for encodes (default .workingdir2/encodes, gitignored)",
    )
    corpus.add_argument(
        "--keep-encodes",
        action="store_true",
        help="retain encoded outputs after scoring (default: delete)",
    )
    corpus.add_argument(
        "--vmaf-model",
        default="vmaf_v0.6.1",
        help="vmaf model version string (default vmaf_v0.6.1)",
    )
    corpus.add_argument("--ffmpeg-bin", default="ffmpeg")
    corpus.add_argument("--vmaf-bin", default="vmaf")
    corpus.add_argument(
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help=(
            "libvmaf scoring backend (default: auto). 'auto' picks the "
            "fastest available (cuda > vulkan > sycl > cpu); a specific "
            "name is honoured strictly and errors out if unavailable. "
            "Use 'vulkan' on AMD / Intel Arc / Apple-MoltenVK hosts "
            "(ADR-0314)."
        ),
    )
    corpus.add_argument(
        "--no-source-hash",
        action="store_true",
        help="skip src_sha256 (faster on huge YUVs; loses provenance)",
    )
    corpus.add_argument(
        "--two-pass",
        action="store_true",
        help=(
            "Phase F (ADR-0333): run a 2-pass encode for codecs that "
            "support it (libx264 / libx265 today; libsvtav1 / libvvenc "
            "follow as sibling PRs). Default off; single-pass remains "
            "the canonical path. Adapters where supports_two_pass = "
            "False fall back to single-pass with a stderr warning."
        ),
    )
    corpus.add_argument(
        "--sample-clip-seconds",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "encode/score only the centre N-second slice of each source "
            "(default 0 = full source). Encode time scales linearly with "
            "the slice length, so e.g. 10s of a 60s source is a ~6x "
            "speedup; expect a 1-2 VMAF-point delta vs full-clip on "
            "diverse content. See ADR-0297."
        ),
    )
    _add_coarse_to_fine_flags(corpus)

    # HDR mode (Bucket #9 / ADR-0300). Mutually-exclusive group on the
    # corpus subparser; default ``--auto-hdr`` keeps the SDR path
    # untouched until ffprobe detects PQ / HLG signaling on a source.
    hdr = corpus.add_mutually_exclusive_group()
    hdr.add_argument(
        "--auto-hdr",
        dest="hdr_mode",
        action="store_const",
        const="auto",
        help=(
            "(default) probe each source via ffprobe and inject HDR "
            "codec args + the HDR-VMAF model when PQ / HLG signaling "
            "is detected"
        ),
    )
    hdr.add_argument(
        "--force-sdr",
        dest="hdr_mode",
        action="store_const",
        const="force-sdr",
        help="treat all sources as SDR; skip HDR detection and flag injection",
    )
    hdr.add_argument(
        "--force-hdr-pq",
        dest="hdr_mode",
        action="store_const",
        const="force-hdr-pq",
        help="treat all sources as HDR PQ (SMPTE-2084) regardless of probe",
    )
    hdr.add_argument(
        "--force-hdr-hlg",
        dest="hdr_mode",
        action="store_const",
        const="force-hdr-hlg",
        help="treat all sources as HDR HLG (ARIB STD-B67) regardless of probe",
    )
    corpus.set_defaults(hdr_mode="auto")
    corpus.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="path to the ffprobe binary (default: ffprobe on PATH)",
    )

    recommend = sub.add_parser(
        "recommend",
        help=(
            "find the smallest CRF whose VMAF >= --target-vmaf "
            "(coarse-to-fine, ~3.5x fewer encodes than the full grid)"
        ),
    )
    _add_recommend_args(recommend)

    predict = sub.add_parser(
        "predict",
        help=(
            "Phase C — predict per-shot VMAF without running it. Probes-encode "
            "each shot, runs a learned ONNX predictor (or analytical fallback), "
            "validates against real VMAF on K shots, then emits the verdict."
        ),
    )
    predict.add_argument(
        "--source",
        type=Path,
        required=True,
        help="reference video (any FFmpeg-readable container)",
    )
    predict.add_argument(
        "--codec",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (default libx264)",
    )
    predict.add_argument(
        "--target-vmaf",
        type=float,
        default=93.0,
        help="target pooled-mean VMAF (default 93)",
    )
    predict.add_argument(
        "--validate-k",
        type=int,
        default=8,
        help="number of shots to verify against real libvmaf (default 8)",
    )
    predict.add_argument(
        "--residual-threshold",
        type=float,
        default=1.5,
        help="max abs(predicted - measured) VMAF before falling back (default 1.5)",
    )
    predict.add_argument(
        "--use-saliency",
        action="store_true",
        help="layer the saliency QP-offset map on top of the picked CRF "
        "(libx264 only for now; other codecs warn and skip)",
    )
    predict.add_argument(
        "--model",
        type=Path,
        default=None,
        help="path to predictor_<codec>.onnx (default: analytical fallback)",
    )
    predict.add_argument(
        "--per-shot-bin",
        default="vmaf-perShot",
        help="path to the vmaf-perShot binary (default vmaf-perShot on PATH)",
    )
    predict.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="path to the ffmpeg binary (default ffmpeg on PATH)",
    )
    predict.add_argument(
        "--bitdepth",
        type=int,
        default=8,
        choices=(8, 10, 12),
        help="source bit depth (forwarded to vmaf-perShot)",
    )
    predict.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="frame count for the single-shot fallback (when vmaf-perShot is unavailable)",
    )
    predict.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="emit the validation report (verdict + residuals) to this path; default: stdout",
    )
    predict.add_argument(
        "--with-uncertainty",
        action="store_true",
        help=(
            "emit conformal prediction intervals alongside each "
            "predicted VMAF point estimate (per ADR-0279). Each "
            "residual row gains an ``interval`` field with "
            "``{low, high, alpha}``. Requires a calibration sidecar "
            "(``--calibration-sidecar``) to produce a non-trivial "
            "interval; without one the wrapper degrades to "
            "``low == high == point`` and the report is flagged "
            "uncalibrated."
        ),
    )
    predict.add_argument(
        "--calibration-sidecar",
        type=Path,
        default=None,
        help=(
            "path to a split-conformal calibration JSON produced by "
            "``vmaftune.conformal.save_split_calibration``. Loaded only "
            "when ``--with-uncertainty`` is set."
        ),
    )
    predict.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "override the calibration sidecar's nominal miscoverage "
            "level (default: the value baked into the sidecar; "
            "0.05 = 95%% coverage). Ignored without "
            "``--with-uncertainty``."
        ),
    )

    per_shot = sub.add_parser(
        "tune-per-shot",
        help=(
            "Phase D — detect shots via vmaf-perShot/TransNet V2, run "
            "Phase-B bisect per shot, and emit an FFmpeg encoding plan."
        ),
    )
    per_shot.add_argument(
        "--src",
        type=Path,
        required=True,
        help="reference video (raw YUV or any FFmpeg-readable container)",
    )
    per_shot.add_argument("--width", type=int, required=True)
    per_shot.add_argument("--height", type=int, required=True)
    per_shot.add_argument("--pix-fmt", default="yuv420p")
    per_shot.add_argument("--framerate", type=float, default=24.0)
    per_shot.add_argument(
        "--target-vmaf",
        type=float,
        default=92.0,
        help="target pooled-mean VMAF for the per-shot predicate (default 92)",
    )
    per_shot.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (default libx264; any registered adapter is accepted)",
    )
    per_shot.add_argument(
        "--bitdepth",
        type=int,
        default=8,
        choices=(8, 10, 12),
        help="source YUV bit depth (forwarded to vmaf-perShot)",
    )
    per_shot.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help=(
            "frame count for the single-shot fallback (used when " "vmaf-perShot is unavailable)"
        ),
    )
    per_shot.add_argument(
        "--per-shot-bin",
        default="vmaf-perShot",
        help="path to the vmaf-perShot binary (default vmaf-perShot on PATH)",
    )
    per_shot.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="path to the ffmpeg binary (default ffmpeg on PATH)",
    )
    per_shot.add_argument(
        "--vmaf-bin",
        default="vmaf",
        help="path to the vmaf binary used by the per-shot bisect scorer",
    )
    per_shot.add_argument(
        "--preset",
        default=None,
        help="codec preset forwarded to the per-shot bisect backend",
    )
    per_shot.add_argument("--crf-min", type=int, default=None, help="inclusive lower CRF bound")
    per_shot.add_argument("--crf-max", type=int, default=None, help="inclusive upper CRF bound")
    per_shot.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="maximum encode+score iterations per detected shot",
    )
    per_shot.add_argument(
        "--vmaf-model",
        default="vmaf_v0.6.1",
        help="VMAF model name forwarded to the per-shot bisect scorer",
    )
    per_shot.add_argument(
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help="libvmaf score backend for the per-shot bisect scorer",
    )
    per_shot.add_argument(
        "--predicate-module",
        default=None,
        help=(
            "advanced hook MODULE:CALLABLE matching "
            "(shot, target_vmaf, encoder) -> (crf, measured_vmaf); "
            "bypasses real bisect"
        ),
    )
    per_shot.add_argument(
        "--output",
        type=Path,
        default=Path("per_shot_encode.mp4"),
        help="final concatenated encode destination (default per_shot_encode.mp4)",
    )
    per_shot.add_argument(
        "--segment-dir",
        type=Path,
        default=None,
        help="directory for per-shot segment files (default <output>.parent/segments)",
    )
    per_shot.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="emit the JSON plan to this path; default: stdout",
    )
    per_shot.add_argument(
        "--script-out",
        type=Path,
        default=None,
        help="optional: write a copy-paste shell script of the plan",
    )

    rec_sal = sub.add_parser(
        "recommend-saliency",
        help=(
            "saliency-aware ROI encode — biases bits toward salient regions "
            "via the fork-trained ``saliency_student_v1`` ONNX model "
            "(Bucket #2 / ADR-0287)"
        ),
    )
    rec_sal.add_argument("--src", type=Path, required=True, help="raw YUV reference")
    rec_sal.add_argument("--width", type=int, required=True)
    rec_sal.add_argument("--height", type=int, required=True)
    rec_sal.add_argument("--pix-fmt", default="yuv420p")
    rec_sal.add_argument("--framerate", type=float, default=24.0)
    rec_sal.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help=(
            "codec adapter; saliency ROI supports libx264, libaom-av1, "
            "libx265, libsvtav1, and libvvenc"
        ),
    )
    rec_sal.add_argument("--preset", default="medium", help="encoder preset")
    rec_sal.add_argument(
        "--crf",
        type=int,
        default=None,
        help="explicit CRF; defaults to the codec adapter's quality_default",
    )
    rec_sal.add_argument(
        "--duration-frames",
        type=int,
        required=True,
        help="frame count to score saliency over (typical: full clip length)",
    )
    rec_sal.add_argument(
        "--saliency-aware",
        action="store_true",
        help="enable saliency biasing (no-op when off; falls back to plain encode)",
    )
    rec_sal.add_argument(
        "--saliency-offset",
        type=int,
        default=-4,
        help="QP delta applied to salient blocks (default -4; clamped to ±12)",
    )
    rec_sal.add_argument(
        "--saliency-model",
        type=Path,
        default=None,
        help="path to saliency_student_v1.onnx (default: shipped fork model)",
    )
    rec_sal.add_argument(
        "--saliency-aggregator",
        choices=("mean", "ema", "max", "motion-weighted"),
        default="mean",
        help=(
            "temporal reducer for sampled saliency masks: mean preserves "
            "the historical behaviour; ema/max/motion-weighted are "
            "video-saliency baselines"
        ),
    )
    rec_sal.add_argument(
        "--saliency-ema-alpha",
        type=float,
        default=0.6,
        help="current-frame weight for --saliency-aggregator=ema (default 0.6)",
    )
    rec_sal.add_argument("--ffmpeg-bin", default="ffmpeg")
    rec_sal.add_argument(
        "--output",
        type=Path,
        required=True,
        help="encode destination (mp4 / mkv / ...)",
    )

    ladder = sub.add_parser(
        "ladder",
        help=(
            "Phase E — build a per-title bitrate ladder (convex-hull "
            "sweep over (resolution × target-VMAF), pick K knees, "
            "emit HLS / DASH / JSON manifest)"
        ),
    )
    ladder.add_argument(
        "--src",
        type=Path,
        required=True,
        help="source video (raw YUV or any FFmpeg-readable container)",
    )
    ladder.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (default libx264)",
    )
    ladder.add_argument(
        "--resolutions",
        required=True,
        help="comma-separated WxH list, e.g. ``1920x1080,1280x720,854x480``",
    )
    ladder.add_argument(
        "--target-vmafs",
        required=True,
        help="comma-separated VMAF target list, e.g. ``95,90,85``",
    )
    ladder.add_argument(
        "--quality-tiers",
        type=int,
        default=5,
        help="number of ladder rungs to select from the convex hull (default 5)",
    )
    ladder.add_argument(
        "--format",
        default="hls",
        choices=("hls", "dash", "json"),
        help="manifest format (default hls)",
    )
    ladder.add_argument(
        "--spacing",
        default="log_bitrate",
        choices=("log_bitrate", "vmaf", "uniform"),
        help=(
            "knee spacing strategy on the hull: log_bitrate or vmaf "
            "(legacy alias: uniform). Default log_bitrate"
        ),
    )
    ladder.add_argument(
        "--output",
        type=Path,
        default=None,
        help="manifest destination (default: stdout)",
    )
    ladder.add_argument(
        "--with-uncertainty",
        action="store_true",
        help=(
            "apply ADR-0279 uncertainty-aware rung selection: prune "
            "adjacent rungs whose conformal intervals overlap above "
            "the threshold, then insert mid-rungs in wide-interval "
            "regions. No-op without per-rung intervals from the "
            "sampler — see vmaftune.ladder.UncertaintyLadderPoint."
        ),
    )
    ladder.add_argument(
        "--uncertainty-sidecar",
        type=Path,
        default=None,
        help=(
            "calibration sidecar JSON (same schema as "
            "``recommend --uncertainty-sidecar``). Defaults to the "
            "Research-0067 floor (tight=2.0, wide=5.0 VMAF)."
        ),
    )
    ladder.add_argument(
        "--rung-overlap-threshold",
        type=float,
        default=None,
        help=(
            "fraction of the wider rung's conformal-interval width "
            "above which two adjacent rungs are treated as "
            "indistinguishable and the lower-bitrate one is dropped. "
            "Default 0.5 per Research-0067."
        ),
    )

    compare = sub.add_parser(
        "compare",
        help=(
            "compare codec adapters at a target VMAF — runs the "
            "Phase B-lite predicate per encoder, ranks by smallest "
            "bitrate, emits a markdown / JSON / CSV report"
        ),
    )
    compare.add_argument(
        "--src",
        type=Path,
        required=True,
        help="reference video (raw YUV or any FFmpeg-readable container)",
    )
    compare.add_argument(
        "--target-vmaf",
        type=float,
        default=92.0,
        help="VMAF target each codec aims for (default 92)",
    )
    compare.add_argument(
        "--encoders",
        required=True,
        help="comma-separated list of encoders to compare (e.g. ``libx264,libx265,libsvtav1``)",
    )
    compare.add_argument(
        "--format",
        default="markdown",
        choices=("markdown", "json", "csv"),
        help="report format (default markdown)",
    )
    compare.add_argument(
        "--no-parallel",
        action="store_true",
        help="dispatch encoders sequentially (default: thread pool, one worker per encoder)",
    )
    compare.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="override thread-pool size (default: len(encoders))",
    )
    compare.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report destination (default: stdout)",
    )
    compare.add_argument("--width", type=int, default=None, help="source width for real bisect")
    compare.add_argument("--height", type=int, default=None, help="source height for real bisect")
    compare.add_argument("--pix-fmt", default="yuv420p", help="source pixel format")
    compare.add_argument("--framerate", type=float, default=24.0, help="source framerate")
    compare.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="source duration in seconds, used for bitrate math",
    )
    compare.add_argument(
        "--sample-clip-seconds",
        type=float,
        default=0.0,
        help=(
            "score a centered N-second source window per bisect iteration "
            "(ADR-0301). 0 = full source."
        ),
    )
    compare.add_argument("--preset", default=None, help="codec preset for the bisect backend")
    compare.add_argument("--crf-min", type=int, default=None, help="inclusive lower CRF bound")
    compare.add_argument("--crf-max", type=int, default=None, help="inclusive upper CRF bound")
    compare.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="maximum encode+score iterations per codec",
    )
    compare.add_argument(
        "--vmaf-model",
        default="vmaf_v0.6.1",
        help="VMAF model name forwarded to the bisect scorer",
    )
    compare.add_argument(
        "--score-backend",
        default=None,
        choices=(*ALL_BACKENDS, "auto"),
        help="libvmaf score backend for the bisect scorer",
    )
    compare.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary")
    compare.add_argument("--vmaf-bin", default="vmaf", help="vmaf binary")
    compare.add_argument(
        "--predicate-module",
        default=None,
        help=(
            "advanced hook MODULE:CALLABLE matching "
            "(codec, src, target_vmaf) -> RecommendResult; bypasses real bisect"
        ),
    )

    benchmark = sub.add_parser(
        "benchmark",
        help=(
            "Phase G — rank encoders from an existing corpus JSONL at a "
            "matched target VMAF, without running new encodes"
        ),
    )
    benchmark.add_argument(
        "--from-corpus",
        type=Path,
        required=True,
        metavar="JSONL",
        help="Phase-A corpus JSONL to benchmark",
    )
    benchmark.add_argument(
        "--target-vmaf",
        type=float,
        default=92.0,
        help="matched-quality threshold each encoder must clear (default 92)",
    )
    benchmark.add_argument(
        "--baseline-encoder",
        default=None,
        help=(
            "encoder used for bitrate-delta percentages. Default: lowest-bitrate "
            "encoder that clears the target."
        ),
    )
    benchmark.add_argument(
        "--format",
        default="markdown",
        choices=("markdown", "json", "csv"),
        help="report format (default markdown)",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report destination (default: stdout)",
    )

    auto = sub.add_parser(
        "auto",
        help=(
            "Phase F — adaptive recipe-aware tuning entry point "
            "(ADR-0364). Composes the per-phase subcommands into one "
            "deterministic decision tree with seven short-circuits "
            "and non-smoke source metadata probing."
        ),
    )
    auto.add_argument(
        "--src",
        type=Path,
        required=True,
        help="reference video (raw YUV or any FFmpeg-readable container)",
    )
    auto.add_argument(
        "--target-vmaf",
        type=float,
        default=93.0,
        help="target pooled-mean VMAF (default 93)",
    )
    auto.add_argument(
        "--max-budget-bitrate",
        type=float,
        default=8000.0,
        help="upper bound on the picked rendition's bitrate in kbps (default 8000)",
    )
    auto.add_argument(
        "--allow-codecs",
        default="libx264",
        help=(
            "comma-separated list of codecs the tree may pick from "
            "(default libx264). When the list resolves to a single "
            "codec the compare-shortlist stage short-circuits."
        ),
    )
    auto.add_argument(
        "--codec",
        default=None,
        help=(
            "pin the codec choice (overrides --allow-codecs ranking). "
            "When set the compare-shortlist stage short-circuits."
        ),
    )
    auto.add_argument(
        "--sample-clip-seconds",
        type=float,
        default=0.0,
        help=(
            "propagate this clip length to internal sweeps rather than "
            "re-deciding per stage (ADR-0301). 0 = full source."
        ),
    )
    auto.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "exercise the composition end-to-end with mocked sub-phases "
            "(no ffmpeg, no ONNX); non-smoke probes source metadata."
        ),
    )
    auto.add_argument(
        "--output",
        type=Path,
        default=None,
        help="emit the JSON plan to this path (default: stdout)",
    )
    auto.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Phase F execute mode (ADR-0454): after planning, run real FFmpeg "
            "encodes and libvmaf scores for the selected cell(s). Results are "
            "written to --runs-dir/tune_results.jsonl. Default: plan-only."
        ),
    )
    auto.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help=(
            "output directory for encoded files and tune_results.jsonl "
            "(used with --execute; default: runs/)"
        ),
    )
    auto.add_argument(
        "--execute-all",
        action="store_true",
        help=(
            "with --execute: run every plan cell, not just the selected winner "
            "(useful for post-hoc A/B comparison)."
        ),
    )

    fast = sub.add_parser(
        "fast",
        help=(
            "Phase A.5 fast-path — proxy + Bayesian + GPU-verify recommend "
            "(ADR-0276 + ADR-0304). Seconds-to-minutes alternative to the "
            "Phase A grid for the recommendation use case."
        ),
    )
    _add_fast_args(fast)

    sidecar = sub.add_parser(
        "sidecar",
        help=(
            "train and inspect the local on-host predictor sidecar "
            "(ADR-0394 bias-correction model)"
        ),
    )
    _add_sidecar_args(sidecar)

    return parser


def _add_sidecar_common_args(p: argparse.ArgumentParser) -> None:
    """Wire the shared local-sidecar configuration flags."""
    from .sidecar import DEFAULT_PREDICTOR_VERSION

    p.add_argument(
        "--codec",
        default="libx264",
        choices=list(known_codecs()),
        help="codec bucket for the sidecar state (default libx264)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="sidecar cache root (default ${XDG_CACHE_HOME:-~/.cache}/vmaf-tune/sidecar)",
    )
    p.add_argument(
        "--predictor-version",
        default=DEFAULT_PREDICTOR_VERSION,
        help=f"predictor version namespace (default {DEFAULT_PREDICTOR_VERSION})",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="optional predictor_<codec>.onnx path; default uses analytical fallback",
    )


def _add_sidecar_args(p: argparse.ArgumentParser) -> None:
    """Wire ``vmaf-tune sidecar`` nested subcommands."""
    sub = p.add_subparsers(dest="sidecar_cmd", required=True)

    status = sub.add_parser("status", help="print sidecar state metadata")
    _add_sidecar_common_args(status)
    status.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    predict = sub.add_parser(
        "predict",
        help="predict VMAF with the sidecar correction folded in",
    )
    _add_sidecar_common_args(predict)
    predict.add_argument("--features-json", type=Path, required=True)
    predict.add_argument("--crf", type=int, required=True)
    predict.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    record = sub.add_parser(
        "record",
        help="record one observed encode result into the sidecar fit",
    )
    _add_sidecar_common_args(record)
    record.add_argument("--features-json", type=Path, required=True)
    record.add_argument("--crf", type=int, required=True)
    record.add_argument("--observed-vmaf", type=float, required=True)
    record.add_argument(
        "--no-persist",
        action="store_true",
        help="update in memory only; mainly useful for tests",
    )
    record.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    batch = sub.add_parser(
        "batch-record",
        help="record a JSONL capture file with one encode observation per row",
    )
    _add_sidecar_common_args(batch)
    batch.add_argument("--captures-jsonl", type=Path, required=True)
    batch.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _add_coarse_to_fine_flags(p: argparse.ArgumentParser) -> None:
    """Wire ``--coarse-to-fine`` + tunables onto a subparser.

    Used by both ``corpus`` (opt-in) and ``recommend`` (always on).
    """
    p.add_argument(
        "--coarse-to-fine",
        action="store_true",
        help=(
            "run a 2-pass coarse-then-fine CRF search instead of the "
            "full grid (ADR-0296). With defaults: 5 coarse + up to 10 "
            "fine = 15 encodes vs 52 for a full 0..51 sweep."
        ),
    )
    p.add_argument(
        "--coarse-step",
        type=int,
        default=10,
        help="CRF step for the coarse pass (default 10 -> [10,20,30,40,50])",
    )
    p.add_argument(
        "--fine-radius",
        type=int,
        default=5,
        help="±radius around best-coarse CRF for the fine pass (default 5)",
    )
    p.add_argument(
        "--fine-step",
        type=int,
        default=1,
        help="CRF step for the fine pass (default 1)",
    )
    p.add_argument(
        "--target-vmaf",
        type=float,
        default=None,
        help=(
            "target VMAF score; the orchestrator picks the smallest "
            "CRF whose score >= target. Optional for `corpus`, "
            "required for `recommend`."
        ),
    )


def _add_recommend_args(p: argparse.ArgumentParser) -> None:
    """Mirror the corpus subparser's source/encode flags for ``recommend``.

    ``recommend`` always runs coarse-to-fine — keeping the flag surface
    aligned with ``corpus`` means downstream scripts can swap one for
    the other without re-learning the CLI.
    """
    # --source / --width / --height / --preset are only required when
    # not using --from-corpus. Validation happens in the handler.
    p.add_argument("--source", type=Path, action="append", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--framerate", type=float, default=24.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--encoder", default="libx264", choices=list(known_codecs()))
    p.add_argument("--preset", action="append", default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("corpus.jsonl"),
        help="JSONL destination for the visited points",
    )
    p.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/encodes"),
    )
    p.add_argument("--keep-encodes", action="store_true")
    p.add_argument("--vmaf-model", default="vmaf_v0.6.1")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--vmaf-bin", default="vmaf")
    p.add_argument(
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help=(
            "libvmaf scoring backend (default: auto; cuda > vulkan > "
            "sycl > cpu). See `vmaf-tune corpus --help`."
        ),
    )
    p.add_argument("--no-source-hash", action="store_true")
    p.add_argument(
        "--two-pass",
        action="store_true",
        help=(
            "Phase F (ADR-0333): run a 2-pass encode for codecs that "
            "support it. Default off; see `vmaf-tune corpus --help`."
        ),
    )
    _add_coarse_to_fine_flags(p)
    _add_recommend_uncertainty_flags(p)

    p.add_argument(
        "--from-corpus",
        type=Path,
        default=None,
        metavar="JSONL",
        help=(
            "pick from an existing corpus JSONL instead of running new "
            "encodes. When set, --source / --width / --height / --preset "
            "are not required. Use --target-vmaf or --target-bitrate to "
            "select the recommendation strategy."
        ),
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--target-bitrate",
        type=float,
        default=None,
        metavar="KBPS",
        help=(
            "when using --from-corpus: pick the row whose bitrate is "
            "closest to this target (in kbps)."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the recommendation as a single JSON object to stdout.",
    )


def _add_recommend_uncertainty_flags(p: argparse.ArgumentParser) -> None:
    """Wire the ADR-0279 conformal-interval flags onto ``recommend``.

    These flags are passive when ``--with-uncertainty`` is omitted —
    the existing point-estimate recipe runs unchanged. When set, the
    recommend search loop reads conformal intervals from the
    coarse-to-fine row stream's ``vmaf_interval`` blocks (or, for
    tests, from a JSON sidecar) and short-circuits / widens search
    according to :func:`vmaftune.recommend.pick_target_vmaf_with_uncertainty`.
    """
    p.add_argument(
        "--with-uncertainty",
        action="store_true",
        help=(
            "consume conformal prediction intervals (per ADR-0279 / "
            "PR #488) when picking the recommended CRF. Tight "
            "intervals short-circuit the search early; wide "
            "intervals fall back to the full point-estimate scan "
            "with the result tagged ``(UNCERTAIN)``."
        ),
    )
    p.add_argument(
        "--uncertainty-sidecar",
        type=Path,
        default=None,
        help=(
            "path to a calibration sidecar (JSON, schema documented "
            "in vmaftune.uncertainty.load_confidence_thresholds). "
            "Defaults to the documented Research-0067 floor "
            "(tight=2.0, wide=5.0 VMAF) when absent."
        ),
    )


def _build_opts(args: argparse.Namespace) -> CorpusOptions:
    # ADR-0299 / ADR-0314: resolve --score-backend up-front so an
    # unavailable backend errors out before we burn cycles on encodes.
    # `select_backend` raises `BackendUnavailableError` (caught by the
    # caller) when a non-auto backend is requested but the host can't
    # provide it.
    selected = select_backend(prefer=args.score_backend, vmaf_bin=args.vmaf_bin)
    sys.stderr.write(f"vmaf-tune: scoring backend = {selected}\n")
    return CorpusOptions(
        encoder=args.encoder,
        output=args.output,
        encode_dir=args.encode_dir,
        vmaf_model=args.vmaf_model,
        ffmpeg_bin=args.ffmpeg_bin,
        vmaf_bin=args.vmaf_bin,
        keep_encodes=args.keep_encodes,
        src_sha256=not args.no_source_hash,
        sample_clip_seconds=getattr(args, "sample_clip_seconds", 0.0),
        score_backend=selected,
        hdr_mode=getattr(args, "hdr_mode", "auto"),
        ffprobe_bin=getattr(args, "ffprobe_bin", "ffprobe"),
        two_pass=getattr(args, "two_pass", False),
    )


def _build_job(args: argparse.Namespace, src: Path, cells: tuple) -> CorpusJob:
    return CorpusJob(
        source=src,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        framerate=args.framerate,
        duration_s=args.duration,
        cells=cells,
    )


def _run_corpus(args: argparse.Namespace) -> int:
    try:
        opts = _build_opts(args)
    except BackendUnavailableError as exc:
        sys.stderr.write(f"vmaf-tune: {exc}\n")
        return 2

    if args.coarse_to_fine:
        # Coarse-to-fine ignores --crf and uses the configured grid.
        # Use a sentinel preset-only cell list so coarse_to_fine_search
        # can extract the preset axis.
        if not args.preset:
            sys.stderr.write("--preset is required\n")
            return 2
        sentinel_cells = tuple((p, 0) for p in args.preset)

        def _all_rows():
            for src in args.source:
                job = _build_job(args, src, sentinel_cells)
                yield from coarse_to_fine_search(
                    job,
                    opts,
                    target_vmaf=args.target_vmaf,
                    coarse_step=args.coarse_step,
                    fine_radius=args.fine_radius,
                    fine_step=args.fine_step,
                )

        n = write_jsonl(_all_rows(), opts.output)
        sys.stderr.write(f"coarse-to-fine: wrote {n} rows -> {opts.output}\n")
        return 0

    if not args.crf:
        sys.stderr.write("--crf is required (or use --coarse-to-fine)\n")
        return 2
    cells = tuple(iter_grid(args.preset, args.crf))

    def _all_rows():
        for src in args.source:
            job = _build_job(args, src, cells)
            yield from iter_rows(job, opts)

    n = write_jsonl(_all_rows(), opts.output)
    sys.stderr.write(f"wrote {n} rows -> {opts.output}\n")
    return 0


def _run_recommend_from_corpus(args: argparse.Namespace) -> int:
    """Pick a recommendation from a pre-built corpus JSONL (no new encodes)."""
    import json as _json  # noqa: PLC0415

    from .recommend import RecommendRequest, recommend  # noqa: PLC0415

    corpus_path: Path = args.from_corpus
    if not corpus_path.exists():
        sys.stderr.write(f"recommend: corpus file not found: {corpus_path}\n")
        return 2

    rows: list[dict] = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(_json.loads(line))

    target_vmaf: float | None = getattr(args, "target_vmaf", None)
    target_bitrate: float | None = getattr(args, "target_bitrate", None)

    if target_vmaf is not None and target_bitrate is not None:
        sys.stderr.write("recommend: --target-vmaf and --target-bitrate are mutually exclusive\n")
        return 2

    try:
        pick = recommend(
            rows,
            RecommendRequest(
                target_vmaf=target_vmaf,
                target_bitrate_kbps=target_bitrate,
                encoder=args.encoder,
                preset=args.preset[0] if args.preset else None,
            ),
        )
    except ValueError as exc:
        sys.stderr.write(f"recommend: {exc}\n")
        return 2

    json_output: bool = getattr(args, "json_output", False)
    if json_output:
        sys.stdout.write(_json.dumps(pick.row) + "\n")
    else:
        crf = pick.row.get("crf", "?")
        vmaf = pick.row.get("vmaf_score", float("nan"))
        kbps = pick.row.get("bitrate_kbps", float("nan"))
        predicate = pick.predicate
        status = "UNMET" if pick.margin < 0 else "OK"
        sys.stdout.write(
            f"crf={crf}  vmaf={vmaf:.3f}  kbps={kbps:.0f}" f"  predicate={predicate}  [{status}]\n"
        )
    return 0


def _run_recommend(args: argparse.Namespace) -> int:
    if getattr(args, "from_corpus", None) is not None:
        return _run_recommend_from_corpus(args)

    # Encode-driven path: validate required args.
    if not args.source or not args.width or not args.height or not args.preset:
        sys.stderr.write(
            "recommend: --source, --width, --height, --preset are required "
            "unless --from-corpus is used\n"
        )
        return 2

    if args.target_vmaf is None:
        sys.stderr.write("recommend requires --target-vmaf\n")
        return 2

    try:
        opts = _build_opts(args)
    except BackendUnavailableError as exc:
        sys.stderr.write(f"vmaf-tune: {exc}\n")
        return 2
    sentinel_cells = tuple((p, 0) for p in args.preset)

    visited: list[dict] = []

    def _capture():
        for src in args.source:
            job = _build_job(args, src, sentinel_cells)
            for row in coarse_to_fine_search(
                job,
                opts,
                target_vmaf=args.target_vmaf,
                coarse_step=args.coarse_step,
                fine_radius=args.fine_radius,
                fine_step=args.fine_step,
            ):
                visited.append(row)
                yield row

    write_jsonl(_capture(), opts.output)
    if getattr(args, "with_uncertainty", False):
        from .recommend import UncertaintyAwareRequest, pick_target_vmaf_with_uncertainty
        from .uncertainty import load_confidence_thresholds

        thresholds = load_confidence_thresholds(getattr(args, "uncertainty_sidecar", None))
        try:
            ua_result = pick_target_vmaf_with_uncertainty(
                visited,
                UncertaintyAwareRequest(
                    target_vmaf=args.target_vmaf,
                    thresholds=thresholds,
                ),
            )
        except ValueError as exc:
            sys.stderr.write(
                f"recommend: uncertainty pick failed ({exc}); "
                f"visited {len(visited)} encodes -> {opts.output}\n"
            )
            return 1
        row = ua_result.row
        sys.stdout.write(
            f"src={row.get('src')} preset={row.get('preset')} "
            f"crf={row.get('crf')} vmaf={float(row['vmaf_score']):.3f} "
            f"decision={ua_result.decision.value} "
            f"visited={ua_result.visited}/{len(visited)} "
            f"predicate={ua_result.predicate}\n"
        )
        return 0
    pick = _smallest_passing_crf(visited, args.target_vmaf)
    if pick is None:
        sys.stderr.write(
            f"recommend: no CRF meets target VMAF >= {args.target_vmaf}; "
            f"visited {len(visited)} encodes -> {opts.output}\n"
        )
        return 1
    src, preset, crf, score = pick
    sys.stdout.write(
        f"src={src} preset={preset} crf={crf} vmaf={score:.3f} "
        f"(visited {len(visited)} encodes)\n"
    )
    return 0


def _smallest_passing_crf(
    rows: list[dict], target_vmaf: float
) -> tuple[str, str, int, float] | None:
    """Return (src, preset, crf, vmaf) for the cheapest passing encode.

    "Cheapest" here means the LARGEST CRF whose ``vmaf_score`` still
    meets ``target_vmaf`` — for libx264 a larger CRF means a smaller
    bitrate, so the largest passing CRF is the smallest bitrate that
    clears the quality gate. Grouped per (src, preset); we return the
    first such (src, preset) pair in the natural row order.
    """
    best: dict[tuple[str, str], tuple[int, float]] = {}
    for r in rows:
        try:
            score = float(r.get("vmaf_score"))
        except (TypeError, ValueError):
            continue
        if score < target_vmaf:
            continue
        key = (str(r["src"]), str(r["preset"]))
        crf = int(r["crf"])
        cur = best.get(key)
        # We want the LARGEST CRF that still meets the target — that's
        # the smallest bitrate at acceptable quality. Tie-break on the
        # higher VMAF score for determinism.
        if cur is None or crf > cur[0] or (crf == cur[0] and score > cur[1]):
            best[key] = (crf, score)
    if not best:
        return None
    # Return the first key in row order.
    for r in rows:
        key = (str(r["src"]), str(r["preset"]))
        if key in best:
            crf, score = best[key]
            return key[0], key[1], crf, score
    return None


def _run_predict(args: argparse.Namespace) -> int:
    """Phase C — per-shot VMAF prediction + validation harness.

    Pipeline:

    1.  Detect shots via :func:`per_shot.detect_shots` (TransNet V2
        binary if available; one-shot fallback otherwise).
    2.  Build a :class:`predictor.Predictor` (ONNX or analytical
        fallback).
    3.  Validate the predictor on K stratified shots — for each, run
        the real ffmpeg encode at the predictor-picked CRF + libvmaf
        score, compute residuals.
    4.  Emit the verdict + residuals + recommended per-shot CRFs as a
        JSON report.
    """
    import subprocess
    import tempfile

    from .encode import EncodeRequest, run_encode
    from .per_shot import detect_shots
    from .predictor import Predictor
    from .predictor_features import FeatureExtractorConfig, _probe_video_geometry, extract_features
    from .predictor_validate import Verdict, validate_predictor
    from .score import ScoreRequest, run_score

    shots = detect_shots(
        source=args.source,
        width=0,
        height=0,
        bitdepth=args.bitdepth,
        framerate=0.0,
        total_frames=args.total_frames or 0,
        per_shot_bin=args.per_shot_bin,
    )
    if not shots:
        print("predict: no shots detected; nothing to do", file=sys.stderr)
        return 1

    feat_cfg = FeatureExtractorConfig(
        ffmpeg_bin=args.ffmpeg_bin,
        use_saliency=args.use_saliency,
    )

    # Probe geometry once — every validation shot reuses the same
    # width/height/fps/pix_fmt for both reference extraction and the
    # encode dispatch.
    width, height, fps = _probe_video_geometry(args.source, feat_cfg, subprocess.run)
    if width <= 0 or height <= 0:
        print(
            "predict: ffprobe could not read source geometry "
            "(width/height); falling back is not safe — aborting.",
            file=sys.stderr,
        )
        return 1
    pix_fmt = "yuv420p"  # canonical reference format; matches saliency.py + the corpus loop

    predictor = Predictor(model_path=args.model)

    def _features(shot):
        return extract_features(
            shot=shot,
            source=args.source,
            codec=args.codec,
            config=feat_cfg,
        )

    # Validation work-area lives for the lifetime of _run_predict so
    # ``run_score``'s lazy decode of the distorted output finds the
    # encoded file still on disk. Cleaned at function exit.
    workdir = Path(tempfile.mkdtemp(prefix="vmaf-tune-predict-"))

    def _real_encode_and_score(shot: Shot, crf: int, codec: str) -> tuple[Path, float]:
        """Run the actual encode + libvmaf score for one validation shot.

        Workflow: extract the shot range from ``args.source`` to a raw
        YUV reference, encode that reference at the predictor-picked CRF
        via :func:`encode.run_encode`, score with
        :func:`score.run_score` (which handles the distorted-side
        decode internally), and return ``(encoded_path, vmaf_score)``.
        """
        ref_yuv = workdir / f"ref_{shot.start_frame}_{shot.end_frame}.yuv"
        dist_path = workdir / f"dist_{shot.start_frame}_{shot.end_frame}.mp4"

        if fps > 0.0:
            ss_arg = f"{shot.start_frame / fps:.6f}"
        else:
            ss_arg = str(shot.start_frame)
        extract_cmd = [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            ss_arg,
            "-i",
            str(args.source),
            "-frames:v",
            str(shot.length),
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            str(ref_yuv),
        ]
        completed = subprocess.run(extract_cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not ref_yuv.exists():
            return dist_path, float("nan")

        encode_req = EncodeRequest(
            source=ref_yuv,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            framerate=fps if fps > 0.0 else 24.0,
            encoder=codec,
            preset="medium",
            crf=crf,
            output=dist_path,
        )
        encode_result = run_encode(encode_req, ffmpeg_bin=args.ffmpeg_bin)
        if encode_result.exit_status != 0 or not dist_path.exists():
            return dist_path, float("nan")

        score_req = ScoreRequest(
            reference=ref_yuv,
            distorted=dist_path,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
        )
        score_result = run_score(score_req, ffmpeg_bin=args.ffmpeg_bin)
        return dist_path, float(score_result.vmaf_score)

    try:
        report = validate_predictor(
            predictor=predictor,
            shots=shots,
            target_vmaf=args.target_vmaf,
            codec=args.codec,
            feature_extractor=_features,
            real_encode_and_score=_real_encode_and_score,
            k=args.validate_k,
            residual_threshold_vmaf=args.residual_threshold,
        )
    finally:
        # Clean the per-run scratch dir even on interrupt — the encoded
        # distorted files can run to gigabytes for long shots.
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    # Optional conformal calibration: load the sidecar once and reuse
    # the same calibration for every per-shot interval. ``None`` falls
    # through to ``ConformalPredictor``'s degraded path
    # (``low == high == point``) so the JSON schema is stable whether
    # or not the operator shipped a sidecar.
    calibration = None
    uncalibrated = False
    if args.with_uncertainty:
        from .conformal import load_split_calibration

        if args.calibration_sidecar is not None:
            calibration = load_split_calibration(args.calibration_sidecar)
        else:
            uncalibrated = True

    def _interval_for(predicted_vmaf: float) -> dict | None:
        if not args.with_uncertainty:
            return None
        if calibration is None:
            return {"low": predicted_vmaf, "high": predicted_vmaf, "alpha": None}
        from .conformal import ConformalPredictor

        # Acknowledge the wrapper class — we use ``cal`` directly here
        # so the hot path doesn't re-run the predictor.
        _ = ConformalPredictor  # noqa: F841 — referenced for type stability
        cal = calibration
        if args.alpha is not None:
            import dataclasses as _dc

            cal = _dc.replace(cal, alpha=args.alpha)
        # Re-derive the interval purely from the residual quantile —
        # we don't need to re-run the predictor since we already have
        # the point estimate. Construct a synthetic ConformalInterval.
        q = cal.quantile()
        low = max(0.0, min(100.0, predicted_vmaf - q))
        high = max(0.0, min(100.0, predicted_vmaf + q))
        return {"low": low, "high": high, "alpha": cal.alpha}

    payload = {
        "verdict": report.verdict.value,
        "target_vmaf": report.target_vmaf,
        "residual_threshold": report.threshold_vmaf,
        "max_abs_residual": report.max_abs_residual,
        "mean_residual": report.mean_residual,
        "bias_correction": report.bias_correction,
        "k_validated": len(report.residuals),
        "uncertainty": {
            "enabled": bool(args.with_uncertainty),
            "calibrated": args.with_uncertainty and not uncalibrated,
            "alpha": (
                (args.alpha if args.alpha is not None else calibration.alpha)
                if calibration is not None
                else None
            ),
        },
        "residuals": [
            {
                "shot_start": r.shot.start_frame,
                "shot_end": r.shot.end_frame,
                "crf": r.crf_picked,
                "predicted_vmaf": r.predicted_vmaf,
                "measured_vmaf": r.measured_vmaf,
                "residual": r.residual,
                **({"interval": _interval_for(r.predicted_vmaf)} if args.with_uncertainty else {}),
            }
            for r in report.residuals
        ],
    }
    rendered = json.dumps(payload, indent=2)
    if args.report_out is not None:
        args.report_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return 0 if report.verdict != Verdict.FALL_BACK else 2


def _run_tune_per_shot(args: argparse.Namespace) -> int:
    total_frames = args.total_frames if args.total_frames > 0 else None
    shots = detect_shots(
        args.src,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        bitdepth=args.bitdepth,
        total_frames=total_frames,
        per_shot_bin=args.per_shot_bin,
    )
    predicate_label = "bisect"
    scratch_ctx = None
    try:
        if args.predicate_module:
            predicate = _load_per_shot_predicate(args.predicate_module)
            predicate_label = args.predicate_module
        else:
            crf_range = _parse_optional_crf_range(
                args.crf_min,
                args.crf_max,
            )
            scratch_ctx = tempfile.TemporaryDirectory(prefix="vmaf-tune-per-shot-")
            predicate = _build_per_shot_bisect_predicate(
                args,
                scratch=Path(scratch_ctx.name),
                crf_range=crf_range,
            )
        recs = tune_per_shot(
            shots,
            target_vmaf=args.target_vmaf,
            encoder=args.encoder,
            predicate=predicate,
        )
    except (AttributeError, ImportError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"vmaf-tune tune-per-shot: {exc}\n")
        return 2
    finally:
        if scratch_ctx is not None:
            scratch_ctx.cleanup()

    plan = merge_shots(
        recs,
        source=args.src,
        output=args.output,
        framerate=args.framerate,
        encoder=args.encoder,
        segment_dir=args.segment_dir,
        ffmpeg_bin=args.ffmpeg_bin,
    )

    plan_doc = {
        "encoder": plan.encoder,
        "framerate": plan.framerate,
        "predicate": predicate_label,
        "target_vmaf": args.target_vmaf,
        "shots": [
            {
                "start_frame": r.shot.start_frame,
                "end_frame": r.shot.end_frame,
                "crf": r.crf,
                "predicted_vmaf": r.predicted_vmaf,
            }
            for r in plan.recommendations
        ],
        "segment_commands": [list(c) for c in plan.segment_commands],
        "concat_command": list(plan.concat_command),
    }
    rendered = json.dumps(plan_doc, indent=2, sort_keys=True)
    if args.plan_out is None:
        sys.stdout.write(rendered)
        sys.stdout.write("\n")
    else:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(rendered + "\n", encoding="utf-8")
        sys.stderr.write(f"wrote plan -> {args.plan_out}\n")

    if args.script_out is not None:
        args.script_out.parent.mkdir(parents=True, exist_ok=True)
        args.script_out.write_text(plan_to_shell_script(plan), encoding="utf-8")
        sys.stderr.write(f"wrote shell script -> {args.script_out}\n")

    seg_dir = args.segment_dir or args.output.parent / "segments"
    write_concat_listing(plan, seg_dir / "concat.txt")
    return 0


def _parse_optional_crf_range(
    crf_min: int | None,
    crf_max: int | None,
) -> tuple[int, int] | None:
    """Validate optional ``--crf-min`` / ``--crf-max`` pairs."""
    if crf_min is None and crf_max is None:
        return None
    if crf_min is None or crf_max is None:
        raise ValueError("pass both --crf-min and --crf-max")
    if crf_min > crf_max:
        raise ValueError(f"invalid CRF range [{crf_min}, {crf_max}]")
    return (int(crf_min), int(crf_max))


def _build_per_shot_bisect_predicate(
    args: argparse.Namespace,
    *,
    scratch: Path,
    crf_range: tuple[int, int] | None,
) -> PerShotPredicateFn:
    """Build the production Phase-D predicate from Phase-B bisect.

    ``bisect_target_vmaf`` operates on raw YUV references, so the
    per-shot CLI first extracts each detected shot to a temporary raw
    YUV file and then runs the existing encode+score bisect loop over
    that isolated shot.
    """
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive for per-shot bisect")
    if args.framerate <= 0:
        raise ValueError("--framerate must be positive for per-shot bisect")

    scratch.mkdir(parents=True, exist_ok=True)
    refs_dir = scratch / "refs"
    work_dir = scratch / "bisect"
    refs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    score_backend = None if args.score_backend == "auto" else args.score_backend

    def _predicate(shot: Shot, target_vmaf: float, encoder: str) -> tuple[int, float]:
        ref_yuv = refs_dir / f"shot_{shot.start_frame}_{shot.end_frame}.yuv"
        _extract_shot_to_raw_yuv(args, shot=shot, output=ref_yuv)
        result = bisect_target_vmaf(
            ref_yuv,
            encoder,
            float(target_vmaf),
            width=args.width,
            height=args.height,
            pix_fmt=args.pix_fmt,
            framerate=args.framerate,
            duration_s=shot.length / args.framerate,
            preset=args.preset,
            crf_range=crf_range,
            max_iterations=args.max_iterations,
            vmaf_model=args.vmaf_model,
            score_backend=score_backend,
            ffmpeg_bin=args.ffmpeg_bin,
            vmaf_bin=args.vmaf_bin,
            workdir=work_dir / f"shot_{shot.start_frame}_{shot.end_frame}",
        )
        if not result.ok:
            raise RuntimeError(
                "bisect failed for shot " f"[{shot.start_frame}, {shot.end_frame}): {result.error}"
            )
        return (result.best_crf, result.measured_vmaf)

    return _predicate


def _extract_shot_to_raw_yuv(
    args: argparse.Namespace,
    *,
    shot: Shot,
    output: Path,
) -> None:
    """Extract one half-open shot range to raw YUV for Phase-B scoring."""
    output.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = shot.start_frame / args.framerate
    cmd = [
        args.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if _source_needs_rawvideo_demux(args.src):
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                args.pix_fmt,
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                str(args.framerate),
            ]
        )
    cmd.extend(
        [
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(args.src),
            "-frames:v",
            str(shot.length),
            "-pix_fmt",
            args.pix_fmt,
            "-f",
            "rawvideo",
            str(output),
        ]
    )
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.exists():
        tail = (completed.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else "no stderr"
        raise RuntimeError(
            f"ffmpeg shot extraction failed for "
            f"[{shot.start_frame}, {shot.end_frame}) (exit={completed.returncode}): {detail}"
        )


def _source_needs_rawvideo_demux(src: Path) -> bool:
    """Return True for extension-only raw YUV inputs."""
    return src.suffix.lower() in {".yuv", ".raw"}


def _run_recommend_saliency(args: argparse.Namespace) -> int:
    """Bucket #2 — single saliency-aware encode (ADR-0287).

    Builds an :class:`~vmaftune.encode.EncodeRequest` from the CLI
    flags and delegates to :func:`vmaftune.saliency.saliency_aware_encode`,
    which runs the fork's ``saliency_student_v1`` ONNX model over the
    source, materialises the selected encoder's ROI sidecar/argv, and
    runs one encode biased toward salient regions. Falls back to a
    plain encode when onnxruntime / the model are unavailable so the
    caller always gets a result.
    """
    from .encode import EncodeRequest
    from .saliency import SaliencyConfig, saliency_aware_encode

    adapter = get_adapter(args.encoder)
    crf = args.crf if args.crf is not None else adapter.quality_default
    request = EncodeRequest(
        source=args.src,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        framerate=args.framerate,
        encoder=args.encoder,
        preset=args.preset,
        crf=crf,
        output=args.output,
    )
    cfg = (
        SaliencyConfig(
            foreground_offset=args.saliency_offset,
            temporal_aggregator=args.saliency_aggregator,
            ema_alpha=args.saliency_ema_alpha,
        )
        if args.saliency_aware
        else None
    )
    result = saliency_aware_encode(
        request,
        duration_frames=args.duration_frames,
        model_path=args.saliency_model,
        config=cfg,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    payload = {
        "encoder": result.request.encoder,
        "preset": result.request.preset,
        "crf": result.request.crf,
        "output": str(result.request.output),
        "encode_size_bytes": result.encode_size_bytes,
        "encode_time_ms": result.encode_time_ms,
        "ffmpeg_version": result.ffmpeg_version,
        "encoder_version": result.encoder_version,
        "saliency_aware": bool(args.saliency_aware),
        "saliency_aggregator": args.saliency_aggregator,
        "exit_status": result.exit_status,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return result.exit_status


def _parse_resolutions(raw: str) -> list[tuple[int, int]]:
    """Parse ``--resolutions`` ``WxH,WxH,...`` into a list of int pairs."""
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token:
            raise SystemExit(f"vmaf-tune ladder: bad resolution {token!r}; expected WxH")
        w_str, _, h_str = token.partition("x")
        out.append((int(w_str), int(h_str)))
    return out


def _parse_target_vmafs(raw: str) -> list[float]:
    """Parse ``--target-vmafs`` ``95,90,85`` into a list of floats."""
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _run_ladder(args: argparse.Namespace) -> int:
    """Phase E — per-title bitrate ladder (ADR-0295).

    Builds the convex hull of (resolution × target-VMAF) sample points,
    selects ``--quality-tiers`` knees by the chosen spacing strategy,
    and emits an HLS / DASH / JSON manifest. The Phase B sampler is
    used by default; tests can inject a stub via the ``ladder.SamplerFn``
    parameter.

    When ``--with-uncertainty`` is set the production path preserves
    per-row ``vmaf_interval`` payloads from the corpus sampler,
    applies :func:`vmaftune.ladder.apply_uncertainty_recipe`, and
    then selects knees from the adjusted rung set. Point-only sampler
    rows get a conservative centred interval using the active
    ``wide_interval_min_width`` threshold, so they still participate
    in midpoint insertion instead of bypassing the recipe.
    """
    from .ladder import build_and_emit
    from .uncertainty import load_confidence_thresholds

    thresholds = None
    if getattr(args, "with_uncertainty", False):
        thresholds = load_confidence_thresholds(getattr(args, "uncertainty_sidecar", None))
    resolutions = _parse_resolutions(args.resolutions)
    target_vmafs = _parse_target_vmafs(args.target_vmafs)
    manifest = build_and_emit(
        src=args.src,
        encoder=args.encoder,
        resolutions=resolutions,
        target_vmafs=target_vmafs,
        quality_tiers=args.quality_tiers,
        format=args.format,
        spacing=args.spacing,
        with_uncertainty=bool(getattr(args, "with_uncertainty", False)),
        uncertainty_thresholds=thresholds,
        rung_overlap_threshold=getattr(args, "rung_overlap_threshold", None),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest, encoding="utf-8")
        sys.stderr.write(f"wrote ladder manifest -> {args.output}\n")
    else:
        sys.stdout.write(manifest)
        if not manifest.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Compare codec adapters at a target VMAF using Phase B bisect.

    Parses the comma-separated ``--encoders`` list, delegates to
    :func:`vmaftune.compare.compare_codecs` (which runs the per-codec
    predicate in a thread pool and ranks by smallest bitrate), then
    emits a markdown / JSON / CSV report via
    :func:`vmaftune.compare.emit_report`.

    Default CLI behaviour now binds :func:`vmaftune.bisect.make_bisect_predicate`
    from the source geometry flags, so ``compare`` is no longer a
    report-only scaffold. ``--predicate-module`` remains as an advanced
    test/operator hook and bypasses the bisect backend.
    """
    from .bisect import make_bisect_predicate
    from .compare import compare_codecs, emit_report, supported_formats

    encoders = [token.strip() for token in args.encoders.split(",") if token.strip()]
    if not encoders:
        sys.stderr.write("vmaf-tune compare: --encoders is empty\n")
        return 2
    if args.format not in supported_formats():
        sys.stderr.write(
            f"vmaf-tune compare: unsupported --format {args.format!r}; "
            f"expected one of {supported_formats()}\n"
        )
        return 2
    predicate = None
    if args.predicate_module:
        try:
            predicate = _load_compare_predicate(args.predicate_module)
        except (AttributeError, ImportError, ValueError) as exc:
            sys.stderr.write(f"vmaf-tune compare: invalid --predicate-module: {exc}\n")
            return 2
    else:
        if args.width is None or args.height is None:
            sys.stderr.write(
                "vmaf-tune compare: --width and --height are required for the "
                "real bisect backend. Use --predicate-module MODULE:CALLABLE "
                "to provide a custom predicate.\n"
            )
            return 2
        crf_range = None
        if args.crf_min is not None or args.crf_max is not None:
            if args.crf_min is None or args.crf_max is None:
                sys.stderr.write("vmaf-tune compare: pass both --crf-min and --crf-max\n")
                return 2
            crf_range = (args.crf_min, args.crf_max)
        score_backend = None if args.score_backend in (None, "auto") else args.score_backend
        predicate = make_bisect_predicate(
            target_vmaf=args.target_vmaf,
            width=args.width,
            height=args.height,
            pix_fmt=args.pix_fmt,
            framerate=args.framerate,
            duration_s=args.duration,
            sample_clip_seconds=args.sample_clip_seconds,
            preset=args.preset,
            crf_range=crf_range,
            max_iterations=args.max_iterations,
            vmaf_model=args.vmaf_model,
            score_backend=score_backend,
            ffmpeg_bin=args.ffmpeg_bin,
            vmaf_bin=args.vmaf_bin,
        )
    report = compare_codecs(
        src=args.src,
        target_vmaf=args.target_vmaf,
        encoders=encoders,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        predicate=predicate,
    )
    rendered = emit_report(report, format=args.format)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        sys.stderr.write(f"wrote compare report -> {args.output}\n")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
    return 0 if report.best() is not None else 1


def _run_benchmark(args: argparse.Namespace) -> int:
    """Phase G — cross-codec report from an existing corpus JSONL."""
    from .benchmark import render_benchmark, summarize_benchmark
    from .recommend import load_corpus_jsonl

    corpus_path: Path = args.from_corpus
    if not corpus_path.exists():
        sys.stderr.write(f"vmaf-tune benchmark: corpus file not found: {corpus_path}\n")
        return 2
    try:
        summaries = summarize_benchmark(
            load_corpus_jsonl(corpus_path),
            target_vmaf=args.target_vmaf,
            baseline_encoder=args.baseline_encoder,
        )
        rendered = render_benchmark(summaries, fmt=args.format)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"vmaf-tune benchmark: {exc}\n")
        return 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        sys.stderr.write(f"wrote benchmark report -> {args.output}\n")
    else:
        sys.stdout.write(rendered)
    return 0


def _load_compare_predicate(spec: str):
    """Load ``MODULE:CALLABLE`` for ``vmaf-tune compare``."""
    if ":" not in spec:
        raise ValueError("expected MODULE:CALLABLE")
    module_name, attr_name = spec.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError("expected MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    predicate = getattr(module, attr_name)
    if not callable(predicate):
        raise ValueError(f"{spec!r} is not callable")
    return predicate


def _load_per_shot_predicate(spec: str) -> PerShotPredicateFn:
    """Load ``MODULE:CALLABLE`` for ``vmaf-tune tune-per-shot``."""
    if ":" not in spec:
        raise ValueError("expected MODULE:CALLABLE")
    module_name, attr_name = spec.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError("expected MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    predicate = getattr(module, attr_name)
    if not callable(predicate):
        raise ValueError(f"{spec!r} is not callable")
    return predicate


def _run_auto(args: argparse.Namespace) -> int:
    """Phase F — ``vmaf-tune auto`` (ADR-0364 / ADR-0454).

    Runs the Phase F decision tree. Non-smoke mode probes source
    geometry, duration, and HDR metadata before planning; ``--smoke``
    exercises the same composition with synthetic metadata.

    When ``--execute`` is set, the selected plan cell(s) are realised as
    actual FFmpeg encodes followed by libvmaf scores; results land in
    ``--runs-dir/tune_results.jsonl`` (ADR-0454).
    """
    from .auto import emit_plan_json, run_auto

    allow = tuple(token.strip() for token in args.allow_codecs.split(",") if token.strip())
    if not allow:
        sys.stderr.write("vmaf-tune auto: --allow-codecs is empty\n")
        return 2
    try:
        plan = run_auto(
            src=args.src,
            target_vmaf=args.target_vmaf,
            max_budget_kbps=args.max_budget_bitrate,
            allow_codecs=allow,
            user_pinned_codec=args.codec,
            sample_clip_seconds=args.sample_clip_seconds,
            smoke=args.smoke,
        )
    except NotImplementedError as exc:
        sys.stderr.write(f"vmaf-tune auto: {exc}\n")
        return 2
    rendered = emit_plan_json(plan)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        sys.stderr.write(f"wrote auto plan -> {args.output}\n")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")

    execute = getattr(args, "execute", False)
    if execute:
        from .executor import run_plan

        runs_dir: Path = getattr(args, "runs_dir", Path("runs"))
        execute_all: bool = getattr(args, "execute_all", False)
        sys.stderr.write(f"vmaf-tune auto: execute mode — runs dir: {runs_dir}\n")
        results = run_plan(
            plan,
            args.src,
            runs_dir,
            execute_all=execute_all,
        )
        n_ok = sum(1 for r in results if r.score is not None and r.score.exit_status == 0)
        sys.stderr.write(
            f"vmaf-tune auto: executed {len(results)} cell(s), "
            f"{n_ok} scored successfully → {runs_dir / 'tune_results.jsonl'}\n"
        )
        if n_ok == 0 and results:
            return 1

    return 0


def _add_fast_args(p: argparse.ArgumentParser) -> None:
    """Wire ``vmaf-tune fast`` user-facing flags onto ``p``.

    The fast-path replaces the grid sweep with a single short probe
    encode per TPE trial plus one final real-encode verify pass at the
    chosen CRF. Flags mirror ``recommend`` where the semantics overlap
    (``--target-vmaf``, ``--encoder``, ``--preset``, source geometry)
    so operators can swap between subcommands without re-learning the
    surface; fast-path-specific knobs (``--n-trials``, ``--crf-min`` /
    ``--crf-max``, ``--proxy-tolerance``, ``--smoke``) sit alongside.
    """
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help=(
            "source video (raw YUV or any FFmpeg-readable container). "
            "Required for production mode; optional for ``--smoke``."
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=0,
        help="raw-YUV reference width (required when ``--src`` is a raw YUV)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=0,
        help="raw-YUV reference height (required when ``--src`` is a raw YUV)",
    )
    p.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pix_fmt (default yuv420p)")
    p.add_argument("--framerate", type=float, default=24.0, help="reference framerate")
    p.add_argument(
        "--target-vmaf",
        type=float,
        required=True,
        help="quality target on the standard VMAF [0, 100] scale",
    )
    p.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (must be in ENCODER_VOCAB_V2 for production mode)",
    )
    p.add_argument(
        "--preset",
        default="medium",
        help="encoder preset for the probe + verify encodes (default medium)",
    )
    p.add_argument(
        "--crf-min",
        type=int,
        default=DEFAULT_CRF_LO,
        help=f"minimum CRF in the TPE search range (default {DEFAULT_CRF_LO})",
    )
    p.add_argument(
        "--crf-max",
        type=int,
        default=DEFAULT_CRF_HI,
        help=f"maximum CRF in the TPE search range (default {DEFAULT_CRF_HI})",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            f"TPE trial budget. Default: {PROD_N_TRIALS} in production mode, "
            f"{SMOKE_N_TRIALS} in --smoke mode."
        ),
    )
    p.add_argument(
        "--time-budget-s",
        type=int,
        default=300,
        help=(
            "soft wall-clock cap in seconds for the Optuna TPE loop "
            "(default 300; in-flight trials are allowed to finish)"
        ),
    )
    p.add_argument(
        "--proxy-tolerance",
        type=float,
        default=DEFAULT_PROXY_TOLERANCE,
        help=(
            "max absolute proxy/verify VMAF gap before the result is flagged "
            f"out-of-distribution (default {DEFAULT_PROXY_TOLERANCE}). When "
            "exceeded the CLI exits non-zero so callers can fall back to "
            "the slow Phase A grid."
        ),
    )
    p.add_argument(
        "--sample-chunk-seconds",
        type=float,
        default=5.0,
        help=(
            "duration in seconds of the proxy probe-encode slice per TPE trial "
            "(default 5.0). Shorter = faster TPE iterations, longer = more "
            "stable canonical-6 features."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "use the deterministic synthetic CRF->VMAF curve; no ffmpeg, no "
            "ONNX, no GPU verify. Intended for CI on hosts without the "
            "[fast] extras."
        ),
    )
    p.add_argument(
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help=(
            "libvmaf scoring backend for the verify pass (default: auto; "
            "cuda > vulkan > sycl > cpu). See ``vmaf-tune corpus --help``."
        ),
    )
    p.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="path to the ffmpeg binary (default ffmpeg on PATH)",
    )
    p.add_argument(
        "--vmaf-bin",
        default="vmaf",
        help="path to the libvmaf CLI binary (default vmaf on PATH)",
    )
    p.add_argument(
        "--vmaf-model",
        default="vmaf_v0.6.1",
        help="vmaf model version string (default vmaf_v0.6.1)",
    )
    p.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/fast"),
        help="scratch dir for probe + verify encodes (default .workingdir2/fast, gitignored)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON destination for the recommendation payload (default: stdout)",
    )


def _build_fast_sample_extractor(
    args: argparse.Namespace,
    workdir: Path,
) -> "Callable[[Path, int, str], tuple[list[float], float]]":
    """Build the production ``sample_extractor`` callable for fast-path.

    The seam encodes a short ``--sample-chunk-seconds`` slice of the
    source at the trial CRF, scores it with libvmaf, and parses the
    canonical-6 (``adm2``, ``vif_scale0..3``, ``motion2``) per-feature
    means out of the libvmaf JSON output. Proxy normalisation
    (StandardScaler) is the proxy module's responsibility — this
    helper returns the raw libvmaf means.
    """
    import json as _json
    import subprocess as _sub
    import tempfile as _tempfile

    from .encode import EncodeRequest, run_encode
    from .score import build_vmaf_command

    workdir.mkdir(parents=True, exist_ok=True)

    def _extract(src: Path, crf: int, encoder: str) -> tuple[list[float], float]:
        # Encode a short probe slice at this CRF.
        slot = workdir / f"probe_{encoder}_crf{crf}.mp4"
        req = EncodeRequest(
            source=src,
            width=args.width,
            height=args.height,
            pix_fmt=args.pix_fmt,
            framerate=args.framerate,
            encoder=encoder,
            preset=args.preset,
            crf=crf,
            output=slot,
            sample_clip_seconds=args.sample_chunk_seconds,
            sample_clip_start_s=0.0,
        )
        encode_result = run_encode(req, ffmpeg_bin=args.ffmpeg_bin)
        if encode_result.exit_status != 0 or not slot.exists():
            return ([0.0] * 6, 0.0)

        size_bytes = encode_result.encode_size_bytes
        observed_kbps = (
            (size_bytes * 8.0 / 1000.0) / max(args.sample_chunk_seconds, 1e-3)
            if size_bytes > 0
            else 0.0
        )

        # Score the slice and parse canonical-6 per-feature means out
        # of libvmaf's per-frame JSON. We bypass score.run_score's
        # pooled-only parser because we need adm2 / vif_scale0..3 /
        # motion2 means rather than the headline VMAF score.
        with _tempfile.TemporaryDirectory(prefix="fast-score-") as score_tmp:
            json_path = Path(score_tmp) / "vmaf.json"
            score_cmd = build_vmaf_command(
                _ScoreReq(
                    reference=src,
                    distorted=slot,
                    width=args.width,
                    height=args.height,
                    pix_fmt=args.pix_fmt,
                    model=args.vmaf_model,
                ),
                json_path,
                vmaf_bin=args.vmaf_bin,
                backend=None,  # fast-path proxy is encoder-side; pooled CPU is fine
            )
            completed = _sub.run(score_cmd, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not json_path.exists():
                return ([0.0] * 6, observed_kbps)
            payload = _json.loads(json_path.read_text(encoding="utf-8"))
            features = _parse_canonical6_means(payload)
        return (features, observed_kbps)

    return _extract


@dataclasses.dataclass(frozen=True)
class _ScoreReq:
    """Minimal duck-typed ScoreRequest for ``build_vmaf_command``.

    ``score.run_score`` is the wrong seam here — we need the per-feature
    means, not just the pooled VMAF score. Reusing ``build_vmaf_command``
    keeps us on the canonical CLI invocation; this duck-type avoids
    importing extras the helper does not need.
    """

    reference: Path
    distorted: Path
    width: int
    height: int
    pix_fmt: str
    model: str
    frame_skip_ref: int = 0
    frame_cnt: int = 0


_CANONICAL_6_KEYS: tuple[str, ...] = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)


def _parse_canonical6_means(payload: dict) -> list[float]:
    """Pull canonical-6 per-feature means from libvmaf JSON output.

    Tries ``pooled_metrics.<feature>.mean`` first (modern libvmaf shape),
    falls back to averaging ``frames[].metrics.<feature>`` when only the
    per-frame surface is present. Missing features fill 0.0 — the
    fr_regressor_v2 proxy sees a zero feature rather than NaN, which is
    in-distribution for content where libvmaf's model omits a metric.
    """
    pooled = payload.get("pooled_metrics") or {}
    out: list[float] = []
    frames = payload.get("frames") or []
    for key in _CANONICAL_6_KEYS:
        block = pooled.get(key) or {}
        if "mean" in block:
            out.append(float(block["mean"]))
            continue
        # Per-frame fallback.
        vals: list[float] = []
        for fr in frames:
            metrics = fr.get("metrics") or {}
            if key in metrics:
                vals.append(float(metrics[key]))
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def _build_fast_encode_runner(
    args: argparse.Namespace,
    workdir: Path,
    backend: str,
) -> "Callable[[Path, str, int, str], tuple[float, float]]":
    """Build the production ``encode_runner`` callable for the verify pass.

    Runs a single full-clip encode at the recommended CRF and scores it
    via :func:`score.run_score`, returning ``(observed_kbps, vmaf_score)``.
    The verify pass is mandatory — proxy alone never wins
    (ADR-0304 invariant).
    """
    from .encode import EncodeRequest, run_encode
    from .score import ScoreRequest, run_score

    workdir.mkdir(parents=True, exist_ok=True)

    def _runner(src: Path, encoder: str, crf: int, _backend_advisory: str) -> tuple[float, float]:
        slot = workdir / f"verify_{encoder}_crf{crf}.mp4"
        req = EncodeRequest(
            source=src,
            width=args.width,
            height=args.height,
            pix_fmt=args.pix_fmt,
            framerate=args.framerate,
            encoder=encoder,
            preset=args.preset,
            crf=crf,
            output=slot,
        )
        encode_result = run_encode(req, ffmpeg_bin=args.ffmpeg_bin)
        if encode_result.exit_status != 0 or not slot.exists():
            return (0.0, float("nan"))
        # Estimate kbps from size + clip duration; the verify pass has
        # no slice so we use whatever framerate × frame-count the source
        # actually produced. Falls back to file-size / 1 second when the
        # encode metadata is missing — kbps is advisory, not the gate.
        size_bytes = encode_result.encode_size_bytes
        elapsed_s = max(encode_result.encode_time_ms / 1000.0, 1e-3)
        observed_kbps = size_bytes * 8.0 / 1000.0 / elapsed_s if size_bytes > 0 else 0.0

        score_req = ScoreRequest(
            reference=src,
            distorted=slot,
            width=args.width,
            height=args.height,
            pix_fmt=args.pix_fmt,
            model=args.vmaf_model,
        )
        score_result = run_score(
            score_req,
            vmaf_bin=args.vmaf_bin,
            backend=backend if backend != "cpu" else None,
        )
        return (observed_kbps, float(score_result.vmaf_score))

    return _runner


def _run_fast(args: argparse.Namespace) -> int:
    """Drive ``vmaf-tune fast`` end to end and emit the JSON payload.

    Smoke mode skips ffmpeg / ONNX / GPU entirely and runs the
    synthetic curve so CI on bare hosts still exercises the search
    loop. Production mode wires the canonical-6 sample extractor and
    the real-encode verify runner through the existing
    :mod:`vmaftune.encode` + :mod:`vmaftune.score` pipeline.
    """
    if args.crf_min < 0 or args.crf_max < args.crf_min:
        sys.stderr.write(f"vmaf-tune fast: invalid CRF range [{args.crf_min}, {args.crf_max}]\n")
        return 2

    sample_extractor = None
    encode_runner = None
    backend_for_payload: str | None = None

    if not args.smoke:
        if args.src is None:
            sys.stderr.write("vmaf-tune fast: --src is required in production mode\n")
            return 2
        if args.width <= 0 or args.height <= 0:
            sys.stderr.write(
                "vmaf-tune fast: --width / --height are required in production mode "
                "(raw-YUV geometry)\n"
            )
            return 2
        try:
            backend_for_payload = select_backend(prefer=args.score_backend, vmaf_bin=args.vmaf_bin)
        except BackendUnavailableError as exc:
            sys.stderr.write(f"vmaf-tune fast: {exc}\n")
            return 2
        sys.stderr.write(f"vmaf-tune fast: scoring backend = {backend_for_payload}\n")
        workdir = args.encode_dir
        sample_extractor = _build_fast_sample_extractor(args, workdir / "probes")
        encode_runner = _build_fast_encode_runner(args, workdir / "verify", backend_for_payload)

    try:
        result = fast_recommend(
            src=args.src,
            target_vmaf=args.target_vmaf,
            encoder=args.encoder,
            time_budget_s=args.time_budget_s,
            crf_range=(args.crf_min, args.crf_max),
            n_trials=args.n_trials,
            smoke=args.smoke,
            sample_extractor=sample_extractor,
            encode_runner=encode_runner,
            proxy_tolerance=args.proxy_tolerance,
        )
    except (RuntimeError, ValueError) as exc:
        # fast.fast_recommend raises RuntimeError when Optuna is missing
        # and ValueError for invalid in-process arguments.
        sys.stderr.write(f"vmaf-tune fast: {exc}\n")
        return 2
    except NotImplementedError as exc:
        sys.stderr.write(f"vmaf-tune fast: {exc}\n")
        return 2

    if backend_for_payload is not None:
        result["score_backend"] = backend_for_payload

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        sys.stderr.write(f"wrote fast recommendation -> {args.output}\n")
    else:
        sys.stdout.write(rendered + "\n")

    gap = result.get("proxy_verify_gap")
    if gap is not None and gap > args.proxy_tolerance:
        # OOD signal — caller should fall back to the slow grid.
        return 3
    return 0


_SIDECAR_REQUIRED_FEATURE_KEYS: tuple[str, ...] = (
    "probe_bitrate_kbps",
    "probe_i_frame_avg_bytes",
    "probe_p_frame_avg_bytes",
    "probe_b_frame_avg_bytes",
)


def _read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object from ``path`` or raise ``ValueError``."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return doc


def _sidecar_features_from_mapping(row: dict[str, object]):
    """Build ``ShotFeatures`` from a JSON object or a ``features`` wrapper."""
    from .predictor import ShotFeatures

    raw = row.get("features", row)
    if not isinstance(raw, dict):
        raise ValueError("'features' must be a JSON object")
    missing = [key for key in _SIDECAR_REQUIRED_FEATURE_KEYS if key not in raw]
    if missing:
        raise ValueError(f"features missing required keys: {', '.join(missing)}")

    kwargs: dict[str, object] = {}
    for field in dataclasses.fields(ShotFeatures):
        if field.name in raw:
            kwargs[field.name] = raw[field.name]
    try:
        return ShotFeatures(
            probe_bitrate_kbps=float(kwargs["probe_bitrate_kbps"]),
            probe_i_frame_avg_bytes=float(kwargs["probe_i_frame_avg_bytes"]),
            probe_p_frame_avg_bytes=float(kwargs["probe_p_frame_avg_bytes"]),
            probe_b_frame_avg_bytes=float(kwargs["probe_b_frame_avg_bytes"]),
            saliency_mean=float(kwargs.get("saliency_mean", 0.0)),
            saliency_var=float(kwargs.get("saliency_var", 0.0)),
            frame_diff_mean=float(kwargs.get("frame_diff_mean", 0.0)),
            y_avg=float(kwargs.get("y_avg", 0.0)),
            y_var=float(kwargs.get("y_var", 0.0)),
            shot_length_frames=int(kwargs.get("shot_length_frames", 0)),
            fps=float(kwargs.get("fps", 0.0)),
            width=int(kwargs.get("width", 0)),
            height=int(kwargs.get("height", 0)),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid sidecar feature value: {exc}") from exc


def _build_sidecar_predictor(args: argparse.Namespace):
    """Construct the configured ``SidecarPredictor`` for CLI handlers."""
    from .predictor import Predictor
    from .sidecar import SidecarConfig, SidecarPredictor

    cfg_kwargs: dict[str, object] = {
        "predictor_version": args.predictor_version,
    }
    if args.cache_dir is not None:
        cfg_kwargs["cache_dir"] = args.cache_dir
    cfg = SidecarConfig(**cfg_kwargs)
    predictor = Predictor(model_path=args.model)
    return SidecarPredictor.for_codec(predictor, codec=args.codec, config=cfg)


def _sidecar_status_payload(sp) -> dict[str, object]:
    """Return the machine-readable status payload for a sidecar."""
    return {
        "schema": "vmaf-tune-sidecar-status/v1",
        "codec": sp.codec,
        "host_uuid": sp.host_uuid,
        "state_path": str(sp.state_path),
        "predictor_version": sp.model.config.predictor_version,
        "schema_version": sp.model.to_dict()["schema_version"],
        "n_updates": sp.model.n_updates,
        "recent_residual_rms": sp.model.recent_residual_rms,
    }


def _emit_sidecar_status(payload: dict[str, object], as_json: bool) -> None:
    """Write a sidecar status payload to stdout."""
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    sys.stdout.write(
        "codec={codec} predictor_version={predictor_version} "
        "updates={n_updates} residual_rms={recent_residual_rms:.6f} "
        "state={state_path}\n".format(**payload)
    )


def _run_sidecar(args: argparse.Namespace) -> int:
    """Run the ``vmaf-tune sidecar`` operator surface."""
    try:
        sp = _build_sidecar_predictor(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"vmaf-tune sidecar: {exc}\n")
        return 2

    if args.sidecar_cmd == "status":
        _emit_sidecar_status(_sidecar_status_payload(sp), args.json)
        return 0

    if args.sidecar_cmd == "predict":
        try:
            features = _sidecar_features_from_mapping(_read_json_object(args.features_json))
        except ValueError as exc:
            sys.stderr.write(f"vmaf-tune sidecar predict: {exc}\n")
            return 2
        base = sp.predictor.predict_vmaf(features, args.crf, args.codec)
        correction = sp.model.predict_correction(features, args.crf)
        payload = {
            "schema": "vmaf-tune-sidecar-predict/v1",
            "codec": args.codec,
            "crf": args.crf,
            "base_vmaf": base,
            "correction": correction,
            "sidecar_vmaf": sp.predict_vmaf(features, args.crf),
            "n_updates": sp.model.n_updates,
        }
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                "base={base_vmaf:.6f} correction={correction:.6f} "
                "sidecar={sidecar_vmaf:.6f} updates={n_updates}\n".format(**payload)
            )
        return 0

    if args.sidecar_cmd == "record":
        try:
            features = _sidecar_features_from_mapping(_read_json_object(args.features_json))
        except ValueError as exc:
            sys.stderr.write(f"vmaf-tune sidecar record: {exc}\n")
            return 2
        base = sp.predictor.predict_vmaf(features, args.crf, args.codec)
        sp.record_capture(
            features,
            crf=args.crf,
            observed_vmaf=args.observed_vmaf,
            persist=not args.no_persist,
        )
        payload = _sidecar_status_payload(sp)
        payload.update(
            {
                "schema": "vmaf-tune-sidecar-record/v1",
                "crf": args.crf,
                "observed_vmaf": args.observed_vmaf,
                "base_vmaf": base,
                "residual": args.observed_vmaf - base,
            }
        )
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                "recorded updates={n_updates} residual={residual:.6f} "
                "state={state_path}\n".format(**payload)
            )
        return 0

    if args.sidecar_cmd == "batch-record":
        rows = 0
        skipped = 0
        try:
            with args.captures_jsonl.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("row is not an object")
                        features = _sidecar_features_from_mapping(row)
                        crf = int(row["crf"])
                        observed = float(row["observed_vmaf"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        skipped += 1
                        sys.stderr.write(
                            f"vmaf-tune sidecar batch-record: skip line {lineno}: {exc}\n"
                        )
                        continue
                    sp.record_capture(features, crf=crf, observed_vmaf=observed, persist=False)
                    rows += 1
        except OSError as exc:
            sys.stderr.write(f"vmaf-tune sidecar batch-record: cannot read input: {exc}\n")
            return 2
        if rows:
            sp.save()
        payload = _sidecar_status_payload(sp)
        payload.update(
            {
                "schema": "vmaf-tune-sidecar-batch-record/v1",
                "rows_recorded": rows,
                "rows_skipped": skipped,
            }
        )
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                "recorded={rows_recorded} skipped={rows_skipped} "
                "updates={n_updates} state={state_path}\n".format(**payload)
            )
        return 0

    sys.stderr.write(f"vmaf-tune sidecar: unknown subcommand {args.sidecar_cmd!r}\n")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _run_corpus(args)
    if args.cmd == "recommend":
        return _run_recommend(args)
    if args.cmd == "predict":
        return _run_predict(args)
    if args.cmd == "tune-per-shot":
        return _run_tune_per_shot(args)
    if args.cmd == "recommend-saliency":
        return _run_recommend_saliency(args)
    if args.cmd == "ladder":
        return _run_ladder(args)
    if args.cmd == "compare":
        return _run_compare(args)
    if args.cmd == "benchmark":
        return _run_benchmark(args)
    if args.cmd == "auto":
        return _run_auto(args)
    if args.cmd == "fast":
        return _run_fast(args)
    if args.cmd == "sidecar":
        return _run_sidecar(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
