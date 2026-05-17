#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Extract FULL_FEATURES (Research-0026) from KoNViD-150k-A using FR-from-NR adapter.

KoNViD-150k-A (K150K-A) is a no-reference corpus: each clip carries a human MOS
label but no reference video.  To run full-reference libvmaf extractors we use the
FR-from-NR adapter pattern (ADR-0346): the same decoded YUV is fed as *both*
reference and distorted.  This makes all difference-based metrics (ciede2000,
psnr_hvs, ADM, VIF, SSIM) measure "identity" — they return null / floor at their
trivial value — while content-sensitive metrics (cambi, motion, vmaf teacher) remain
informative.  The NaN columns are expected and documented in ADR-0362.

``vmaf`` column: computed via the ``vmaf_v0.6.1`` model (SDR-trained).  The model
is mis-calibrated on PQ HDR clips because it was trained on SDR content; scores from
HDR inputs should be interpreted as a relative comparison baseline only, not as an
absolute quality prediction.  Replace with the Netflix HDR model when it ships.  The
~5–10 % CUDA wall-clock overhead of the model dispatch is acknowledged in
Research-0135 as an acceptable trade-off for preserving the vmaf relationship across
bitrate-ladder rungs.  See Research-0135 for the full analysis and the Option A vs
Option B decision matrix (supersedes PR #898 Option A / Research-0135 draft).

Output: ``runs/full_features_k150k.parquet`` (one row per clip, gitignored).

Schema (46 columns, parquet schema version v2):

    clip_name, mos,
    <21 features>_mean, <21 features>_std    (42 feature columns)

Feature columns follow the FEATURE_NAMES tuple order exactly (column-order-locked;
see ai/AGENTS.md §K150K-A corpus extraction invariants before reordering).

Restartability: a ``.done`` checkpoint file (one clip name per line, append-only)
lets interrupted runs resume without re-processing already-extracted clips.

I/O strategy (perf win — Research-0135):
  Rows are accumulated in memory throughout the run and written to a JSONL staging
  file (``<out>.rows.jsonl``) on every completion.  The parquet is written **once**
  at the end.  This eliminates the O(N²) read-concat-write pattern that the
  per-``--flush-every``-clips flush incurred on long runs.  The ``.done`` checkpoint
  remains the primary restartability signal; the JSONL staging file handles recovery
  of in-memory rows after an unclean exit.

ffprobe skip (Win 2 — Research-0135):
  When ``--metadata-jsonl`` is provided and the sidecar contains
  ``chug_width_manifest``, ``chug_height_manifest``, and
  ``chug_framerate_manifest`` for a clip, ffprobe is skipped for that clip.
  The pixel format is inferred from ``chug_bit_depth`` (10 → ``yuv420p10le``,
  else ``yuv420p``) or defaults to ``yuv420p``.  ffprobe remains necessary for
  clips not covered by the sidecar.

Parallelism (ADR-0382): clips are dispatched to a
``concurrent.futures.ProcessPoolExecutor`` with ``--threads-cuda`` workers
(default 8).  Each worker independently decodes one clip to a worker-private YUV
scratch file, scores it via the selected fork binary, aggregates frames, removes
the YUV immediately, and returns the row dict.  The main process collects
results, writes the ``.done`` checkpoint, and flushes the parquet.  Worker
isolation ensures no shared mutable state and avoids backend context conflicts.

Usage::

    python ai/scripts/extract_k150k_features.py \\
        --clips-dir .workingdir2/konvid-150k/k150ka_extracted \\
        --scores   .workingdir2/konvid-150k/k150ka_scores.csv  \\
        --out      runs/full_features_k150k.parquet

Smoke-test (100 clips, 8 workers)::

    python ai/scripts/extract_k150k_features.py --limit 100 --threads-cuda 8

Resume command (same invocation; already-done clips are skipped)::

    python ai/scripts/extract_k150k_features.py

Hardware: the default path remains CPU-oriented because K150K-A 540p 5s clips are
CPU-bound in aggregate and the CUDA binary provides no per-clip speedup for that
geometry (ADR-0382).  Larger local corpora such as CHUG can opt into a CUDA-capable
``--vmaf-bin``; in that mode the script uses explicit CUDA feature names for the
stable GPU pass and ``--cpu-vmaf-bin`` for residual CPU-only extractors.  The
system ``/usr/local/bin/vmaf`` v3.0.0 lacks ssimulacra2 and motion_v2 — it must NOT
be used for this pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUG_SPLIT_SEED = "chug-hdr-v1"

# ---------------------------------------------------------------------------
# Feature / extractor configuration (column-order-locked per ai/AGENTS.md)
# ---------------------------------------------------------------------------

# Extractor names passed via --feature to the vmaf CLI.
EXTRACTOR_NAMES: tuple[str, ...] = (
    "adm",
    "vif",
    "motion",
    "motion_v2",
    "psnr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede",
    "psnr_hvs",
    "ssimulacra2",
)

CUDA_EXTRACTOR_NAMES: tuple[str, ...] = (
    "adm_cuda",
    "vif_cuda",
    "motion_cuda",
    "motion_v2_cuda",
    "psnr_cuda",
    "ciede_cuda",
    "float_ssim_cuda=scale=1",
    "float_ms_ssim_cuda",
    "psnr_hvs_cuda",
    # cambi_cuda intentionally not promoted: the CUDA extractor
    # segfaults on every input on the rebuilt 2026-05-15 binary
    # (Issue #857). cambi stays on the CPU residual pass below
    # until that's fixed.
    # float_ssim_cuda needs an explicit scale=1: libvmaf v1 supports
    # scale=1 only and refuses on auto-detected scale=4 at 1080p
    # ("libvmaf ERROR ssim_cuda: v1 supports scale=1 only").
)
# ssimulacra2 omitted from K150K/CHUG self-vs-self extraction — produces ~100 constant
# for identity pairs (ref == distorted), yielding zero training signal while consuming
# ~30-50% of GPU time per clip. Use CPU ssimulacra2 extractor for FR pairs where it
# remains informative (ADR-0431).

# 2026-05-15: float_ssim promoted from the CPU residual pass to the
# CUDA primary pass (it has shipped a CUDA implementation since
# libvmaf/src/feature/cuda/float_ssim_cuda.c landed). cambi stays
# on this CPU residual pass — its CUDA twin segfaults (Issue #857).
CUDA_CPU_RESIDUAL_EXTRACTOR_NAMES: tuple[str, ...] = (
    "cambi",
    # speed_temporal + speed_chroma added 2026-05-15. Lawrence's HDR
    # recipe (`hdr_custom_features.py`, Slack) called for both signals
    # in the K150K/CHUG feature set; they are CPU-only extractors
    # (no CUDA twin yet) so they ride the residual pass.
    "speed_temporal",
    "speed_chroma",
)

# Canonical 25-feature output columns (Research-0026, parquet schema v2 +
# 2026-05-15 speed-feature addition).
# WARNING: column order is locked — do not reorder without incrementing the
# parquet schema version and updating ai/AGENTS.md. New columns may only be
# appended at the END of the tuple, never inserted.
# ssimulacra2 dropped: in self-vs-self (FR-from-NR) mode it returns ~100 for every
# frame regardless of input (zero training signal); see ADR-0431 and the docstring
# near CUDA_EXTRACTOR_NAMES above.
FEATURE_NAMES: tuple[str, ...] = (
    "adm2",
    "adm_scale0",
    "adm_scale1",
    "adm_scale2",
    "adm_scale3",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion",
    "motion2",
    "motion3",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede2000",
    "psnr_hvs",
    "vmaf",
    # 2026-05-15 additions — appended at end to preserve column order.
    # Source: lawrence's hdr_custom_features.py recipe (Slack).
    "speed_temporal",
    "speed_chroma_u",
    "speed_chroma_v",
    "speed_chroma_uv",
)

# Map feature names to their JSON key(s) in libvmaf output.  libvmaf may emit
# ``integer_<name>`` for fixed-point kernels; try both in order.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "adm2": ("adm2", "integer_adm2"),
    "adm_scale0": ("adm_scale0", "integer_adm_scale0"),
    "adm_scale1": ("adm_scale1", "integer_adm_scale1"),
    "adm_scale2": ("adm_scale2", "integer_adm_scale2"),
    "adm_scale3": ("adm_scale3", "integer_adm_scale3"),
    "vif_scale0": ("vif_scale0", "integer_vif_scale0"),
    "vif_scale1": ("vif_scale1", "integer_vif_scale1"),
    "vif_scale2": ("vif_scale2", "integer_vif_scale2"),
    "vif_scale3": ("vif_scale3", "integer_vif_scale3"),
    "motion": ("motion", "integer_motion"),
    "motion2": ("motion2", "integer_motion2"),
    "motion3": ("motion3", "integer_motion3"),
    "psnr_y": ("psnr_y", "integer_psnr_y"),
    "psnr_cb": ("psnr_cb", "integer_psnr_cb"),
    "psnr_cr": ("psnr_cr", "integer_psnr_cr"),
    "float_ssim": ("float_ssim",),
    "float_ms_ssim": ("float_ms_ssim",),
    "cambi": ("cambi",),
    "ciede2000": ("ciede2000",),
    "psnr_hvs": ("psnr_hvs",),
    "vmaf": ("vmaf",),
    # 2026-05-15 additions — short aliases registered in
    # libvmaf/src/feature/alias.c.
    "speed_temporal": (
        "speed_temporal",
        "Speed_temporal_feature_speed_temporal_score",
    ),
    "speed_chroma_u": (
        "speed_chroma_u",
        "Speed_chroma_feature_speed_chroma_u_score",
    ),
    "speed_chroma_v": (
        "speed_chroma_v",
        "Speed_chroma_feature_speed_chroma_v_score",
    ),
    "speed_chroma_uv": (
        "speed_chroma_uv",
        "Speed_chroma_feature_speed_chroma_uv_score",
    ),
}

# ---------------------------------------------------------------------------
# YUV decode / geometry helpers
# ---------------------------------------------------------------------------


def _geometry_from_sidecar(meta: dict | None) -> tuple[int, int, str, str] | None:
    """Extract (width, height, pix_fmt, fps) from a CHUG JSONL sidecar row.

    Returns ``None`` if ``meta`` is ``None`` or if any required geometry field
    is absent, so the caller can fall back to ffprobe.  Required fields:
    ``chug_width_manifest``, ``chug_height_manifest``,
    ``chug_framerate_manifest``.  The pixel format is inferred from
    ``chug_bit_depth`` (10 → ``yuv420p10le``, else ``yuv420p``).
    """
    if meta is None:
        return None
    w = meta.get("chug_width_manifest")
    h = meta.get("chug_height_manifest")
    fps = meta.get("chug_framerate_manifest")
    if w is None or h is None or fps is None:
        return None
    bit_depth = meta.get("chug_bit_depth")
    pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    return int(w), int(h), pix_fmt, str(fps)


def _probe_geometry(mp4: Path) -> tuple[int, int, str, str, dict[str, str]]:
    """Return (width, height, pix_fmt, fps, color_meta) for the first video stream.

    The 5th element ``color_meta`` is a dict with keys
    ``color_primaries`` / ``color_transfer`` / ``color_space`` (each
    optional, populated when ffprobe surfaces the field).  Callers use
    this to decide HDR-aware feature options (CAMBI ``eotf=pq``,
    motion ``motion_fps_weight``).
    """
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate,color_primaries,color_transfer,color_space",
            "-of",
            "json",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = json.loads(proc.stdout)["streams"][0]
    pix_fmt: str = s.get("pix_fmt", "yuv420p")
    # Normalise to libvmaf-safe pixel formats.
    pix_fmt = "yuv420p10le" if "10" in pix_fmt else "yuv420p"
    color_meta = {
        "color_primaries": s.get("color_primaries", "") or "",
        "color_transfer": s.get("color_transfer", "") or "",
        "color_space": s.get("color_space", "") or "",
    }
    return (
        int(s["width"]),
        int(s["height"]),
        pix_fmt,
        s.get("r_frame_rate", "25/1"),
        color_meta,
    )


def _is_hdr_source(pix_fmt: str, color_meta: dict[str, str]) -> bool:
    """True when the source is HDR (PQ or HLG transfer characteristics).

    A source needs both:
    1. 10-bit (or higher) pix_fmt — SDR 8-bit can't be HDR.
    2. PQ (``smpte2084``) or HLG (``arib-std-b67``) transfer characteristics
       OR BT.2020 primaries (a weaker fallback when transfer is absent).

    Returns ``False`` on missing metadata to fail-safe to SDR defaults
    rather than mis-applying HDR options to an SDR source.
    """
    if "10" not in pix_fmt and "12" not in pix_fmt and "16" not in pix_fmt:
        return False
    transfer = color_meta.get("color_transfer", "").lower()
    if transfer in ("smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12"):
        return True
    primaries = color_meta.get("color_primaries", "").lower()
    return primaries in ("bt2020", "bt2020nc", "bt2020c")


def _parse_fps(fps_str: str) -> float:
    """Parse the ffprobe ``r_frame_rate`` string ``"num/den"`` into a float.

    Returns 0.0 on parse failure so callers can fall back to a default.
    """
    if "/" in fps_str:
        try:
            num, den = fps_str.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(fps_str)
    except ValueError:
        return 0.0


def _motion_fps_weight(fps: float) -> float:
    """Compute the libvmaf ``motion_fps_weight`` for a given source fps.

    Motion features measure per-frame absolute differences. At 50/60 fps
    the per-frame delta on the same physical motion is ~half what it is
    at 25/30 fps; at 120 fps it's ~quarter. The libvmaf
    ``motion[_v2]_fps_weight`` knob multiplies the score to compensate.

    The reference fps for motion features is 30 (per Netflix golden
    fixtures).  Returns 1.0 (no correction) for any fps in [24, 32];
    otherwise returns ``30 / fps`` clamped to ``[0.25, 4.0]``.
    """
    if fps <= 0:
        return 1.0
    if 24.0 <= fps <= 32.0:
        return 1.0
    weight = 30.0 / fps
    return max(0.25, min(4.0, weight))


# Per-extractor option support. CUDA twins ship a reduced VmafOption
# table vs their CPU counterparts (verified against
# libvmaf/src/feature/cuda/{integer_cambi_cuda,integer_ms_ssim_cuda,
# integer_motion_cuda}.c); options not present here are silently
# dropped from the --feature arg rather than triggering
# "problem loading feature extractor" at runtime.
_FEATURE_OPTION_SUPPORT: dict[str, frozenset[str]] = {
    "cambi": frozenset({"eotf", "cambi_eotf", "full_ref"}),
    "cambi_cuda": frozenset({"eotf", "cambi_eotf"}),
    "float_ms_ssim": frozenset({"enable_db", "clip_db", "enable_lcs"}),
    "float_ms_ssim_cuda": frozenset({"enable_lcs"}),
    "motion": frozenset({"motion_fps_weight"}),
    "motion_cuda": frozenset({"motion_fps_weight"}),
    "motion_v2": frozenset({"motion_fps_weight"}),
    "motion_v2_cuda": frozenset({"motion_fps_weight"}),
}


def _feature_arg(extractor: str, is_hdr: bool, motion_fps_weight: float) -> str:
    """Build the ``--feature`` argument value for one extractor.

    Per libvmaf/tools/cli_parse.c the CLI grammar is
    ``EXTRACTOR=key1=val1:key2=val2``: ``strsep(&optarg, "=")`` consumes
    the extractor name first, then ``:`` separates the ``key=value``
    pairs.  The leading literal ``name=`` token is NOT part of the
    grammar — it parses as a feature called "name" with bad options
    and trips ``problem loading feature extractor: name``.

    Returns ``"<extractor>=k1=v1:k2=v2"`` when HDR-aware options apply
    AND the extractor advertises support for them; returns the bare
    ``<extractor>`` name otherwise (preserving pre-fix behaviour for
    SDR sources and silently dropping CUDA-unsupported options).

    HDR-aware options follow lawrence's 2026-05-15 guidance:
    - CAMBI: ``eotf=pq`` (HDR PQ visibility thresholds, not SDR);
      ``full_ref=true`` (FR-CAMBI matches the script's ref==dis
      topology; the CUDA twin doesn't expose this option, so the
      whitelist drops it for ``cambi_cuda``).
    - MS_SSIM: ``enable_db=false`` (linear scale per recipe);
      the CUDA twin doesn't expose this option either.
    - motion / motion_v2 (both CPU and CUDA): ``motion_fps_weight``
      when fps != 30.
    """
    desired: list[tuple[str, str]] = []
    base = extractor

    if is_hdr and base in ("cambi", "cambi_cuda"):
        desired.append(("eotf", "pq"))
        desired.append(("full_ref", "true"))
    if is_hdr and base in ("float_ms_ssim", "float_ms_ssim_cuda"):
        desired.append(("enable_db", "false"))
    if motion_fps_weight != 1.0 and base in (
        "motion",
        "motion_cuda",
        "motion_v2",
        "motion_v2_cuda",
    ):
        desired.append(("motion_fps_weight", f"{motion_fps_weight:.4f}"))

    supported = _FEATURE_OPTION_SUPPORT.get(base, frozenset())
    opts = [f"{k}={v}" for k, v in desired if k in supported]
    if not opts:
        return base
    return f"{base}=" + ":".join(opts)


def _decode_to_yuv(mp4: Path, yuv_path: Path, pix_fmt: str) -> None:
    """Decode ``mp4`` to raw YUV.  Writes atomically via a ``.tmp`` sibling."""
    tmp = yuv_path.with_suffix(".tmp")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-pix_fmt",
                pix_fmt,
                "-f",
                "rawvideo",
                str(tmp),
            ],
            check=True,
        )
        tmp.rename(yuv_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# vmaf invocation
# ---------------------------------------------------------------------------


def _build_vmaf_cmd(
    vmaf_bin: Path,
    yuv_path: Path,
    width: int,
    height: int,
    pix_fmt: str,
    out_json: Path,
    threads: int,
    extractor_names: tuple[str, ...],
    backend_args: list[str],
    is_hdr: bool = False,
    motion_fps_weight_value: float = 1.0,
) -> list[str]:
    bitdepth = "10" if "10" in pix_fmt else "8"
    feat_args: list[str] = []
    for ex in extractor_names:
        feat_args += ["--feature", _feature_arg(ex, is_hdr, motion_fps_weight_value)]

    return [
        str(vmaf_bin),
        "--reference",
        str(yuv_path),
        "--distorted",
        str(yuv_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--pixel_format",
        "420",
        "--bitdepth",
        bitdepth,
        *feat_args,
        "--threads",
        str(threads),
        *backend_args,
        "--output",
        str(out_json),
        "--json",
        "-q",
    ]


def _run_vmaf_json(
    vmaf_bin: Path,
    yuv_path: Path,
    width: int,
    height: int,
    pix_fmt: str,
    out_json: Path,
    threads: int,
    extractor_names: tuple[str, ...],
    backend_args: list[str],
    is_hdr: bool = False,
    motion_fps_weight_value: float = 1.0,
) -> list[dict]:
    """Run vmaf once and return a list of per-frame metric dicts."""
    cmd = _build_vmaf_cmd(
        vmaf_bin,
        yuv_path,
        width,
        height,
        pix_fmt,
        out_json,
        threads,
        extractor_names,
        backend_args,
        is_hdr=is_hdr,
        motion_fps_weight_value=motion_fps_weight_value,
    )
    subprocess.run(cmd, check=True, capture_output=True)
    with out_json.open() as f:
        data = json.load(f)
    return [fr["metrics"] for fr in data.get("frames", [])]


def _merge_frame_metrics(primary: list[dict], residual: list[dict]) -> list[dict]:
    """Merge per-frame metric dictionaries from two vmaf invocations."""
    frame_count = min(len(primary), len(residual))
    merged: list[dict] = []
    for idx in range(frame_count):
        row = dict(primary[idx])
        row.update(residual[idx])
        merged.append(row)
    return merged


def _run_feature_passes(
    vmaf_bin: Path,
    cpu_vmaf_bin: Path,
    yuv_path: Path,
    width: int,
    height: int,
    pix_fmt: str,
    out_json: Path,
    threads: int,
    use_cuda: bool,
    is_hdr: bool = False,
    motion_fps_weight_value: float = 1.0,
) -> list[dict]:
    """Run vmaf feature extraction, splitting CUDA mode where required.

    The ``vmaf_v0.6.1`` model is dispatched on every invocation so that the
    ``vmaf`` JSON key is populated in the output.  The model is SDR-trained and
    is mis-calibrated on PQ HDR clips; its score is kept as a relative
    comparison baseline only.  See the module docstring and Research-0135 for
    the rationale (Option B, supersedes PR #898 Option A).
    """
    # The --model arg causes libvmaf to run the vmaf_v0.6.1 model dispatch and
    # emit a per-frame "vmaf" key.  Without this, the JSON contains only the
    # raw feature values and no composite score.
    _MODEL_ARGS: list[str] = ["--model", "version=vmaf_v0.6.1"]

    if not use_cuda:
        return _run_vmaf_json(
            vmaf_bin,
            yuv_path,
            width,
            height,
            pix_fmt,
            out_json,
            threads,
            EXTRACTOR_NAMES,
            ["--no_cuda", "--no_sycl", "--no_vulkan", *_MODEL_ARGS],
            is_hdr=is_hdr,
            motion_fps_weight_value=motion_fps_weight_value,
        )

    cuda_json = out_json.with_name(out_json.stem + ".cuda.json")
    cpu_json = out_json.with_name(out_json.stem + ".cpu.json")
    try:
        cuda_frames = _run_vmaf_json(
            vmaf_bin,
            yuv_path,
            width,
            height,
            pix_fmt,
            cuda_json,
            threads,
            CUDA_EXTRACTOR_NAMES,
            ["--backend", "cuda", *_MODEL_ARGS],
            is_hdr=is_hdr,
            motion_fps_weight_value=motion_fps_weight_value,
        )
        # CPU residual pass — kept structurally for future feature
        # additions that lack a CUDA implementation. As of 2026-05-15
        # the residual is empty (CAMBI + float_ssim got promoted to
        # the CUDA pass); the call short-circuits to an empty frames
        # list without spawning a subprocess.
        if CUDA_CPU_RESIDUAL_EXTRACTOR_NAMES:
            cpu_frames = _run_vmaf_json(
                cpu_vmaf_bin,
                yuv_path,
                width,
                height,
                pix_fmt,
                cpu_json,
                threads,
                CUDA_CPU_RESIDUAL_EXTRACTOR_NAMES,
                ["--no_cuda", "--no_sycl", "--no_vulkan"],
                is_hdr=is_hdr,
                motion_fps_weight_value=motion_fps_weight_value,
            )
            frames = _merge_frame_metrics(cuda_frames, cpu_frames)
        else:
            frames = cuda_frames
        out_json.write_text(
            json.dumps({"frames": [{"metrics": row} for row in frames]}),
            encoding="utf-8",
        )
        return frames
    finally:
        cuda_json.unlink(missing_ok=True)
        cpu_json.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Metric lookup and per-clip aggregation
# ---------------------------------------------------------------------------


def _lookup_metric(metrics: dict, feature: str) -> float:
    """Return the float value for ``feature`` from a libvmaf metrics dict.

    Tries each alias in order; returns NaN if none match or value is None.
    """
    for alias in _METRIC_ALIASES.get(feature, (feature,)):
        v = metrics.get(alias)
        if v is not None:
            return float(v)
    return float("nan")


def _aggregate_frames(frames: list[dict]) -> dict[str, float]:
    """Return nanmean and nanstd per feature across all frames."""
    if not frames:
        result: dict[str, float] = {}
        for feat in FEATURE_NAMES:
            result[f"{feat}_mean"] = float("nan")
            result[f"{feat}_std"] = float("nan")
        return result

    data: dict[str, list[float]] = {feat: [] for feat in FEATURE_NAMES}
    for m in frames:
        for feat in FEATURE_NAMES:
            data[feat].append(_lookup_metric(m, feat))

    result = {}
    for feat in FEATURE_NAMES:
        arr = np.array(data[feat], dtype=np.float64)
        # Suppress all-NaN warnings — ciede2000 and psnr_hvs are all-NaN
        # for identity pairs (ref == dis, ADR-0362 §Negative consequences).
        # numpy emits RuntimeWarning via warnings, not the FP error machinery,
        # so errstate alone does not suppress it — use warnings.catch_warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[f"{feat}_mean"] = float(np.nanmean(arr))
            result[f"{feat}_std"] = float(np.nanstd(arr))
    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _content_split_for(content_name: str, *, seed: str = DEFAULT_CHUG_SPLIT_SEED) -> str:
    key = f"{seed}\0{content_name}".encode("utf-8")
    digest = hashlib.blake2s(key, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "val"
    return "test"


def _load_jsonl_metadata(path: Path | None, *, split_seed: str) -> dict[str, dict[str, Any]]:
    """Load optional CHUG/K150K JSONL side metadata keyed by clip basename."""
    if path is None or not path.is_file():
        return {}
    keep = (
        "mos_raw_0_100",
        "chug_video_id",
        "chug_ref",
        "chug_name",
        "chug_bitladder",
        "chug_resolution",
        "chug_bitrate_label",
        "chug_orientation",
        "chug_framerate_manifest",
        "chug_content_name",
        "chug_height_manifest",
        "chug_width_manifest",
    )
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = Path(str(row.get("src") or row.get("filename") or "")).name
            if not src:
                continue
            meta = {key: row[key] for key in keep if key in row}
            split = str(row.get("split") or "").strip().lower()
            content = str(row.get("chug_content_name") or "").strip()
            if split not in {"train", "val", "test"} and content:
                split = _content_split_for(content, seed=split_seed)
            if split in {"train", "val", "test"}:
                meta["split"] = split
                meta["chug_split_key"] = content or src
                meta["chug_split_policy"] = "content-name-blake2s-80-10-10"
            out[src] = meta
    return out


def _load_done_set(done_path: Path) -> set[str]:
    """Load the set of already-processed clip names from the checkpoint file."""
    if not done_path.is_file():
        return set()
    with done_path.open() as f:
        return {line.strip() for line in f if line.strip()}


def _append_done(done_path: Path, clip_name: str) -> None:
    """Append a clip name to the checkpoint file (append-only, one per line)."""
    with done_path.open("a") as f:
        f.write(clip_name + "\n")


# ---------------------------------------------------------------------------
# JSONL staging + at-end parquet write (Win 1, Research-0135)
# ---------------------------------------------------------------------------


def _staging_path(out_path: Path) -> Path:
    """Return the JSONL staging file path for ``out_path``.

    The staging file accumulates all completed rows during the run so that a
    crash does not lose rows that are already past the ``.done`` checkpoint.
    Written in append-only mode; converted to parquet once at the end.
    """
    return out_path.with_suffix(".rows.jsonl")


def _append_row_to_staging(staging_path: Path, row: dict) -> None:
    """Append one row to the JSONL staging file (main process only)."""
    with staging_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, allow_nan=True) + "\n")


def _load_staging_rows(staging_path: Path) -> list[dict]:
    """Load all rows from the JSONL staging file, skipping malformed lines."""
    if not staging_path.is_file():
        return []
    rows: list[dict] = []
    with staging_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def _write_parquet_from_rows(rows: list[dict], out_path: Path) -> None:
    """Write ``rows`` to ``out_path`` atomically, deduplicating by clip_name.

    Parquet writes happen exactly once per run.  The per-clip JSONL staging
    file is the in-run durability mechanism; this function is called only at
    the end of ``main()`` (Research-0135 Win 1).
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["clip_name"], keep="last")
    tmp = out_path.with_suffix(".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.rename(out_path)


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _process_clip(
    mp4_str: str,
    mos: float,
    vmaf_bin_str: str,
    cpu_vmaf_bin_str: str,
    scratch_dir_str: str,
    vmaf_threads: int,
    use_cuda: bool,
    worker_id: int,
    sidecar_meta: dict | None = None,
) -> dict:
    """Decode one clip, score it, aggregate, and return the row dict.

    Runs in a subprocess worker.  Uses a worker-private YUV path so parallel
    workers never clobber each other.  Deletes the YUV unconditionally on exit
    (success or failure) to avoid scratch-dir disk saturation.

    When ``sidecar_meta`` contains the required CHUG geometry fields
    (``chug_width_manifest``, ``chug_height_manifest``,
    ``chug_framerate_manifest``), ffprobe is skipped for that clip (Win 2,
    Research-0135).

    Returns a dict with keys: clip_name, mos, width, height, <feat>_mean/std.
    Raises on any failure so the caller can log and skip.
    """
    mp4 = Path(mp4_str)
    vmaf_bin = Path(vmaf_bin_str)
    cpu_vmaf_bin = Path(cpu_vmaf_bin_str)
    scratch_dir = Path(scratch_dir_str)

    # Worker-private paths — include PID + worker_id for full isolation.
    stem = f"{mp4.stem}_w{worker_id}_{os.getpid()}"
    yuv_path = scratch_dir / f"{stem}.yuv"
    out_json = scratch_dir / f"{stem}.json"

    try:
        # Always probe for color_meta (needed for HDR detection).
        # Win 2: override with sidecar geometry when available — ffprobe
        # geometry is less reliable than the manifest fields for CHUG clips,
        # and the sidecar skips an extra ffprobe call for the geometry fields.
        _pw, _ph, _ppf, fps_str, color_meta = _probe_geometry(mp4)
        geom = _geometry_from_sidecar(sidecar_meta) if sidecar_meta else None
        if geom is not None:
            width, height, pix_fmt, fps_str = geom
        else:
            width, height, pix_fmt = _pw, _ph, _ppf
        is_hdr = _is_hdr_source(pix_fmt, color_meta)
        fps = _parse_fps(fps_str)
        motion_w = _motion_fps_weight(fps)
        _decode_to_yuv(mp4, yuv_path, pix_fmt)
        frames = _run_feature_passes(
            vmaf_bin,
            cpu_vmaf_bin,
            yuv_path,
            width,
            height,
            pix_fmt,
            out_json,
            vmaf_threads,
            use_cuda,
            is_hdr=is_hdr,
            motion_fps_weight_value=motion_w,
        )
        agg = _aggregate_frames(frames)
        return {
            "clip_name": mp4.name,
            "mos": mos,
            "width": width,
            "height": height,
            "fps": fps,
            "is_hdr": is_hdr,
            "motion_fps_weight": motion_w,
            **agg,
        }
    finally:
        yuv_path.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="extract_k150k_features.py",
        description="Extract FULL_FEATURES from KoNViD-150k-A via FR-from-NR adapter (ADR-0346).",
    )
    ap.add_argument(
        "--clips-dir",
        type=Path,
        default=Path(".workingdir2/konvid-150k/k150ka_extracted"),
        help="Directory containing K150K-A .mp4 clips.",
    )
    ap.add_argument(
        "--scores",
        type=Path,
        default=Path(".workingdir2/konvid-150k/k150ka_scores.csv"),
        help="CSV with columns video_name, video_score (MOS labels).",
    )
    ap.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional corpus JSONL sidecar. For CHUG, this preserves content, "
            "ladder, raw MOS, and deterministic split metadata in the parquet."
        ),
    )
    ap.add_argument(
        "--split-seed",
        default=DEFAULT_CHUG_SPLIT_SEED,
        help="Seed for CHUG content-level split metadata when --metadata-jsonl has no split.",
    )
    ap.add_argument(
        "--vmaf-bin",
        type=Path,
        default=REPO_ROOT / "libvmaf" / "build-cpu" / "tools" / "vmaf",
        help=(
            "Path to the fork vmaf binary (built with ssimulacra2 + motion_v2).  "
            "Default: libvmaf/build-cpu/tools/vmaf.  Passing a CUDA-capable binary "
            "enables the split CUDA-safe feature pass plus CPU residual pass "
            "(ADR-0431)."
        ),
    )
    ap.add_argument(
        "--cpu-vmaf-bin",
        type=Path,
        default=REPO_ROOT / "libvmaf" / "build-cpu" / "tools" / "vmaf",
        help=(
            "CPU vmaf binary used for residual CPU-only feature passes when "
            "--vmaf-bin points at a CUDA-capable binary. Default: "
            "libvmaf/build-cpu/tools/vmaf."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "full_features_k150k.parquet",
        help="Output parquet path (gitignored).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=2,
        help="vmaf --threads value per worker (inner threading).  Default 2.",
    )
    ap.add_argument(
        "--threads-cuda",
        type=int,
        default=8,
        help=(
            "Number of parallel worker processes (outer parallelism).  Each "
            "worker runs one vmaf invocation concurrently.  Default 8 is tuned "
            "for a 32-thread Zen5 CPU; reduce on machines with fewer cores.  "
            "Named --threads-cuda for historical reasons (ADR-0382); it controls "
            "outer process parallelism for both CPU and split CUDA modes."
        ),
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help=(
            "Print a progress line every N completed clips.  Default 200.  "
            "Previously named --flush-every; the name changed when the per-flush "
            "parquet rewrite was replaced with at-end-only writes (Research-0135)."
        ),
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # Legacy alias; --progress-every takes precedence.
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N clips (smoke-test mode).",
    )
    ap.add_argument(
        "--no-cuda",
        action="store_true",
        help=(
            "Disable CUDA backend flags on the vmaf invocation.  This is the "
            "default when using libvmaf/build-cpu/tools/vmaf (the recommended "
            "binary); only needed if passing a CUDA-capable binary explicitly."
        ),
    )
    ap.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "k150k_yuv_scratch",
        help="Scratch directory for temporary YUV files.  Cleaned per-clip.",
    )
    args = ap.parse_args()

    # --flush-every is a legacy alias; --progress-every takes precedence.
    progress_every: int = args.progress_every

    use_cuda = not args.no_cuda

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not args.clips_dir.is_dir():
        print(f"error: clips-dir not found: {args.clips_dir}", file=sys.stderr)
        return 2
    if not args.scores.is_file():
        print(f"error: scores CSV not found: {args.scores}", file=sys.stderr)
        return 2
    if not args.vmaf_bin.is_file():
        print(
            f"error: vmaf binary not found: {args.vmaf_bin}\n"
            "Build the fork vmaf binary with:\n"
            "  meson setup libvmaf/build-cpu libvmaf -Denable_cuda=false "
            "--buildtype=release && ninja -C libvmaf/build-cpu\n"
            "Then re-run with --vmaf-bin libvmaf/build-cpu/tools/vmaf",
            file=sys.stderr,
        )
        return 2
    if use_cuda and not args.cpu_vmaf_bin.is_file():
        print(f"error: cpu-vmaf-bin not found: {args.cpu_vmaf_bin}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Load MOS labels
    # ------------------------------------------------------------------
    scores_df = pd.read_csv(args.scores)
    scores_df = scores_df.rename(columns={"video_score": "mos"})
    mos_map: dict[str, float] = dict(zip(scores_df["video_name"], scores_df["mos"], strict=True))
    score_meta: dict[str, dict[str, Any]] = {}
    for row in scores_df.to_dict(orient="records"):
        name = str(row.get("video_name") or "")
        if not name:
            continue
        meta: dict[str, Any] = {}
        if "mos_raw_0_100" in row:
            meta["mos_raw_0_100"] = row["mos_raw_0_100"]
        score_meta[name] = meta
    jsonl_meta = _load_jsonl_metadata(args.metadata_jsonl, split_seed=args.split_seed)

    # ------------------------------------------------------------------
    # Enumerate clips and apply checkpoint
    # ------------------------------------------------------------------
    clips = sorted(args.clips_dir.glob("*.mp4"))
    if args.limit is not None:
        clips = clips[: args.limit]

    done_path = args.out.with_suffix(".done")
    done_set = _load_done_set(done_path)
    pending = [c for c in clips if c.name not in done_set]

    print(
        f"[k150k] total={len(clips)} done={len(done_set)} pending={len(pending)} "
        f"cuda={'yes' if use_cuda else 'no'} "
        f"workers={args.threads_cuda} threads/worker={args.threads} "
        f"out={args.out}",
        flush=True,
    )

    if not pending:
        print("[k150k] nothing to do.", flush=True)
        return 0

    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    # JSONL staging file — accumulates rows during the run for crash durability.
    # Converted to parquet exactly once at the end (Research-0135 Win 1).
    staging_path = _staging_path(args.out)
    # Reload any rows from a previous partial run that are in the done set but
    # whose staging rows survived.  This covers the edge-case where the process
    # was killed after writing the staging line but before the final parquet write.
    recovered_rows = _load_staging_rows(staging_path)
    recovered_names = {r.get("clip_name") for r in recovered_rows if r.get("clip_name")}

    # ------------------------------------------------------------------
    # Parallel extraction via ProcessPoolExecutor
    # ------------------------------------------------------------------
    rows: list[dict] = list(recovered_rows)
    ok = 0
    fail = 0
    t0 = time.time()
    completed = 0

    # Build submit order: (future, clip_name) pairs.
    # We use as_completed() so results flow back as soon as workers finish,
    # keeping the checkpoint and staging file up-to-date without waiting for
    # the whole batch.  Parquet is written once at the end.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads_cuda) as executor:
        future_to_clip: dict[concurrent.futures.Future, str] = {}
        for idx, mp4 in enumerate(pending):
            clip_name = mp4.name
            mos = mos_map.get(clip_name, float("nan"))
            # Pass sidecar metadata to the worker for ffprobe skip (Win 2).
            clip_sidecar = jsonl_meta.get(clip_name)
            fut = executor.submit(
                _process_clip,
                str(mp4),
                mos,
                str(args.vmaf_bin),
                str(args.cpu_vmaf_bin),
                str(args.scratch_dir),
                args.threads,
                use_cuda,
                idx % args.threads_cuda,
                clip_sidecar,
            )
            future_to_clip[fut] = clip_name

        for fut in concurrent.futures.as_completed(future_to_clip):
            clip_name = future_to_clip[fut]
            completed += 1
            try:
                row = fut.result()
                row.update(score_meta.get(clip_name, {}))
                row.update(jsonl_meta.get(clip_name, {}))
                # Append to JSONL staging immediately for crash durability.
                if clip_name not in recovered_names:
                    _append_row_to_staging(staging_path, row)
                rows.append(row)
                _append_done(done_path, clip_name)
                ok += 1
            except Exception as exc:
                print(f"[k150k] FAIL {clip_name}: {exc}", file=sys.stderr, flush=True)
                fail += 1

            # Periodic progress log (no parquet write — that happens at the end).
            if completed % progress_every == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (len(pending) - completed) / rate / 3600.0 if rate > 0 else float("nan")
                print(
                    f"[k150k] {completed}/{len(pending)} ok={ok} fail={fail} "
                    f"{rate:.2f} clip/s eta={remaining:.1f}h",
                    flush=True,
                )

    # Write parquet exactly once at the end (Research-0135 Win 1).
    # Include both newly-processed rows and any rows recovered from the staging file.
    if rows:
        _write_parquet_from_rows(rows, args.out)
        # Clean up the staging file now that the parquet is durable.
        staging_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    rate = ok / elapsed if elapsed > 0 else 0.0
    print(
        f"[k150k] done. ok={ok} fail={fail} total_time={elapsed:.1f}s "
        f"rate={rate:.2f} clip/s out={args.out}",
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
