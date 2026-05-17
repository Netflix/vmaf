#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Cross-backend GPU-parity CI gate (T6-8 / ADR-0214).

Generalisation of ``cross_backend_vif_diff.py``: iterates every
configured ``(feature, backend_pair)`` cell, runs the ``vmaf`` binary
once per backend, then diffs the per-frame metrics with a feature-
specific absolute tolerance.

The single-feature script gates one cell per CLI invocation; this gate
runs the whole matrix in one process and emits two artefacts that
downstream consumers can read without re-parsing stdout:

* ``--json-out`` — machine-readable summary, one record per cell
  (status, max_abs_diff per metric, frame count, tolerance, command).
* ``--md-out``   — human-readable Markdown table, suitable for pasting
  into a PR comment or rendering in CI logs.

Tolerance policy (T6-8 / ADR-0214 § "Tolerance schema"):

* Most integer-pipeline features (``vif``, ``adm``, ``motion``,
  ``motion_v2``, ``psnr``, ``float_moment``) lock places=4 — already
  the production contract from ADR-0125 / ADR-0138 / ADR-0140.
* Float-pipeline features that hit transcendentals (``ciede``,
  ``psnr_hvs``, ``ssimulacra2``) carry a per-feature relaxation
  declared inline in ``FEATURE_TOLERANCE`` with the ADR justifying it.
* The ``--fp16-features`` flag forces the FP16 contract (1e-2 absolute)
  on the listed feature names, used by the future ONNX tiny-AI lane
  once T7-39 lands.

Exit code: 0 if every cell within tolerance, 1 if any cell exceeds
its tolerance, 2 on a binary / fixture failure.

The gate **never modifies** any feature implementation — it only
verifies. Per CLAUDE.md §12 r1 (Netflix golden assertions are
untouchable), any tightening of tolerance must come from a
measurement-driven ADR, not from the CI lane.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

# Calibration loader sits next to this script. Same sys.path tweak
# as cross_backend_vif_diff.py so direct ``python3 scripts/ci/<this>.py``
# invocations resolve the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_backend_calibration import (
    DEFAULT_CALIBRATION_PATH,
    CalibrationTable,
    load_calibration_table,
)

# ---------------------------------------------------------------------------
# Feature → metric-name list. Mirror of ``FEATURE_METRICS`` in
# ``cross_backend_vif_diff.py`` (single source of truth for the
# extractor name → emitted-metric mapping). When a new feature gets a
# GPU twin, add it here.
# ---------------------------------------------------------------------------

FEATURE_METRICS: dict[str, tuple[str, ...]] = {
    "vif": (
        "integer_vif_scale0",
        "integer_vif_scale1",
        "integer_vif_scale2",
        "integer_vif_scale3",
    ),
    # T3-15(c) / ADR-0219: GPU motion now emits motion3_score in
    # 3-frame window mode. The 5-frame window mode
    # (motion_five_frame_window=true) remains deferred — the GPU
    # extractors reject it with -ENOTSUP at init().
    "motion": (
        "integer_motion",
        "integer_motion2",
        "integer_motion3",
    ),
    "motion_v2": (
        "VMAF_integer_feature_motion_v2_sad_score",
        "VMAF_integer_feature_motion2_v2_score",
    ),
    "adm": (
        "integer_adm2",
        "integer_adm_scale0",
        "integer_adm_scale1",
        "integer_adm_scale2",
        "integer_adm_scale3",
    ),
    "psnr": ("psnr_y",),
    "float_moment": (
        "float_moment_ref1st",
        "float_moment_dis1st",
        "float_moment_ref2nd",
        "float_moment_dis2nd",
    ),
    "ciede": ("ciede2000",),
    "float_ssim": ("float_ssim",),
    "float_ms_ssim": ("float_ms_ssim",),
    # T7-35 / ADR-0215: enable_lcs adds 15 per-scale L/C/S triples on
    # top of the combined float_ms_ssim score. The Vulkan/CUDA kernels
    # already produce the per-scale L/C/S means; gating only the
    # feature_collector_append calls keeps default-path output
    # bit-identical. Cell uses extractor `float_ms_ssim` with the
    # `enable_lcs=true` option pass-through (resolved by
    # `cross_backend_vif_diff.py`'s FEATURE_ALIASES on the per-
    # feature lane). The matrix gate runs the LCS variant only when
    # opted in via `--features` (skipped by default to keep
    # vulkan-parity-matrix-gate cheap).
    "float_ms_ssim_lcs": (
        "float_ms_ssim",
        "float_ms_ssim_l_scale0",
        "float_ms_ssim_l_scale1",
        "float_ms_ssim_l_scale2",
        "float_ms_ssim_l_scale3",
        "float_ms_ssim_l_scale4",
        "float_ms_ssim_c_scale0",
        "float_ms_ssim_c_scale1",
        "float_ms_ssim_c_scale2",
        "float_ms_ssim_c_scale3",
        "float_ms_ssim_c_scale4",
        "float_ms_ssim_s_scale0",
        "float_ms_ssim_s_scale1",
        "float_ms_ssim_s_scale2",
        "float_ms_ssim_s_scale3",
        "float_ms_ssim_s_scale4",
    ),
    "float_ansnr": (
        "float_ansnr",
        "float_anpsnr",
    ),
    "float_psnr": ("float_psnr",),
    "float_motion": (
        "motion",
        "motion2",
    ),
    "float_vif": (
        "vif_scale0",
        "vif_scale1",
        "vif_scale2",
        "vif_scale3",
    ),
    "psnr_hvs": (
        "psnr_hvs_y",
        "psnr_hvs_cb",
        "psnr_hvs_cr",
        "psnr_hvs",
    ),
    "float_adm": (
        "adm2",
        "adm_scale0",
        "adm_scale1",
        "adm_scale2",
        "adm_scale3",
    ),
    "ssimulacra2": ("ssimulacra2",),
    "cambi": ("Cambi_feature_cambi_score",),
}

# ---------------------------------------------------------------------------
# Per-feature tolerance contract. Absolute |a - b| ceiling. Mirrors
# the ``--places`` flags wired into the existing per-feature lanes
# (places=4 → 5e-5, places=3 → 5e-4, places=2 → 5e-3); kept as raw
# floats so the gate can express future FP16 tiers (1e-2) without
# overloading the places vocabulary.
#
# Default for any feature not listed: ``DEFAULT_FP32_TOLERANCE``
# (5e-5, equivalent to places=4) — same contract the per-feature
# lane defaults to in `cross_backend_vif_diff.py`.
# ---------------------------------------------------------------------------

DEFAULT_FP32_TOLERANCE = 5e-5  # places=4
DEFAULT_FP16_TOLERANCE = 1e-2  # T6-8 FP16 contract (future tiny-AI lane)

FEATURE_TOLERANCE: dict[str, float] = {
    # Integer pipeline — places=4 (5e-5). ADR-0138 / ADR-0140.
    "vif": 5e-5,
    "adm": 5e-5,
    "motion": 5e-5,
    "motion_v2": 5e-5,
    "psnr": 5e-5,
    "float_moment": 5e-5,
    # Float pipeline, well-conditioned. places=4.
    "float_ssim": 5e-5,
    "float_ms_ssim": 5e-5,
    # T7-35 / ADR-0215: LCS triples are the same float reductions
    # that feed the Wang combine — same conditioning, same places=4.
    "float_ms_ssim_lcs": 5e-5,
    "float_ansnr": 5e-5,
    "float_psnr": 5e-5,
    "float_motion": 5e-5,
    "float_vif": 5e-5,
    "float_adm": 5e-5,
    # Transcendentals / DCT — relaxed contract per ADR-0187 / ADR-0188.
    "ciede": 5e-3,  # per-pixel pow/sqrt/sin/atan2 — places=2.
    "psnr_hvs": 5e-4,  # DCT + per-block float reductions — places=3.
    # XYB cube root + IIR blur reassociation — places=2 per ADR-0192.
    "ssimulacra2": 5e-3,
    # Integer pipeline — places=4 (5e-5). ADR-0360.
    "cambi": 5e-5,
}

# Backend → extractor-name suffix and CLI device-selection flag.
BACKEND_SUFFIX: dict[str, str] = {
    "cpu": "",
    "cuda": "_cuda",
    "sycl": "_sycl",
    "vulkan": "_vulkan",
}
BACKEND_DEVICE_FLAG: dict[str, str] = {
    "cuda": "--gpumask",
    "sycl": "--sycl_device",
    "vulkan": "--vulkan_device",
}

# Default device index per backend. CUDA gpumask=1 picks the first GPU;
# Vulkan / SYCL device 0 is the first compute-capable.
BACKEND_DEFAULT_DEVICE: dict[str, int] = {
    "cuda": 1,
    "sycl": 0,
    "vulkan": 0,
}


@dataclasses.dataclass(frozen=True)
class Cell:
    """One (feature, backend_a, backend_b) cell of the parity matrix."""

    feature: str
    backend_a: str
    backend_b: str


@dataclasses.dataclass
class CellResult:
    """Outcome of running one cell of the matrix."""

    feature: str
    backend_a: str
    backend_b: str
    tolerance: float
    n_frames: int
    per_metric_max: dict[str, float]
    per_metric_mismatches: dict[str, int]
    status: str  # "OK" | "FAIL" | "SKIP" | "ERROR"
    note: str = ""
    # ADR-0234 calibration provenance — which calibration entry
    # supplied the tolerance, or "default" when the gate fell back
    # to FEATURE_TOLERANCE / DEFAULT_FP32_TOLERANCE. Surfaced in the
    # JSON / Markdown artefacts so reviewers can audit per-arch
    # tolerance decisions without re-reading the YAML.
    tolerance_source: str = "default"


# ---------------------------------------------------------------------------
# Matrix construction helpers. The gate covers every pairwise comparison
# between the user-selected backend list, for every user-selected
# feature. CPU is always included as the canonical reference; if the
# user selects only one extra backend, the matrix degenerates to
# (CPU, that_backend) per feature — same shape as the legacy gate.
# ---------------------------------------------------------------------------


def build_matrix(features: Iterable[str], backends: Iterable[str]) -> list[Cell]:
    backend_list = list(backends)
    cells: list[Cell] = []
    for feature in features:
        for i, a in enumerate(backend_list):
            for b in backend_list[i + 1 :]:
                cells.append(Cell(feature=feature, backend_a=a, backend_b=b))
    return cells


# Pseudo-features that map to a real extractor + a `:opt=val` option
# pass-through. T7-35 / ADR-0215: float_ms_ssim_lcs reuses the
# `float_ms_ssim` extractor with `enable_lcs=true` to gate the 15
# extra L/C/S metrics.
FEATURE_ALIASES: dict[str, tuple[str, str]] = {
    "float_ms_ssim_lcs": ("float_ms_ssim", "enable_lcs=true"),
}


def feature_extractor_name(feature: str, backend: str) -> str:
    """Map (feature, backend) to the extractor name `--feature` accepts.

    Pseudo-features in ``FEATURE_ALIASES`` are resolved to
    ``base_extractor=opt=val`` so libvmaf's option parser flips the
    underlying extractor into the right mode (e.g. ``enable_lcs``).
    """

    suffix = BACKEND_SUFFIX[backend]
    if feature in FEATURE_ALIASES:
        base_name, opt_string = FEATURE_ALIASES[feature]
        return f"{base_name}{suffix}={opt_string}"
    return f"{feature}{suffix}"


def build_command(
    binary: Path,
    ref: Path,
    dist: Path,
    width: int,
    height: int,
    pix_fmt: str,
    bitdepth: int,
    feature: str,
    backend: str,
    device: int | None,
    output: Path,
) -> list[str]:
    extractor = feature_extractor_name(feature, backend)
    cmd: list[str] = [
        str(binary),
        "--reference",
        str(ref),
        "--distorted",
        str(dist),
        "--width",
        str(width),
        "--height",
        str(height),
        "--pixel_format",
        pix_fmt,
        "--bitdepth",
        str(bitdepth),
        "--feature",
        extractor,
        "--no_prediction",
        "--output",
        str(output),
        "--json",
        "--backend",
        backend,
    ]
    if backend != "cpu" and device is not None:
        cmd += [BACKEND_DEVICE_FLAG[backend], str(device)]
    return cmd


def run_one(
    binary: Path,
    ref: Path,
    dist: Path,
    width: int,
    height: int,
    pix_fmt: str,
    bitdepth: int,
    feature: str,
    backend: str,
    device: int | None,
    output: Path,
) -> tuple[int, str]:
    """Run a single ``vmaf`` invocation. Returns (returncode, stderr)."""

    cmd = build_command(
        binary,
        ref,
        dist,
        width,
        height,
        pix_fmt,
        bitdepth,
        feature,
        backend,
        device,
        output,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        return proc.returncode, (proc.stderr or proc.stdout)
    return 0, ""


def load_frames(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["frames"]


def diff_frames(
    a_frames: list[dict],
    b_frames: list[dict],
    metrics: tuple[str, ...],
    tolerance: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-metric max abs diff and mismatch count across two runs."""

    per_max = dict.fromkeys(metrics, 0.0)
    per_mismatch = dict.fromkeys(metrics, 0)
    for fa, fb in zip(a_frames, b_frames, strict=True):
        for m in metrics:
            va, vb = fa["metrics"][m], fb["metrics"][m]
            d = abs(va - vb)
            per_max[m] = max(per_max[m], d)
            if d > tolerance:
                per_mismatch[m] += 1
    return per_max, per_mismatch


def resolve_cell_tolerance(
    feature: str,
    *,
    fp16_features: Iterable[str],
    calibration: CalibrationTable | None,
    gpu_id: str | None,
) -> tuple[float, str]:
    """Resolve ``(tolerance_abs, source_label)`` for one cell.

    Source-label vocabulary (recorded on every CellResult):

    * ``"fp16"``      — feature opted into the FP16 contract.
    * ``"calibrated:<pattern>"`` — calibration table matched and
      supplied a per-feature override; ``status: calibrated`` row.
    * ``"placeholder:<pattern>"`` — calibration table matched but
      the row is a placeholder (no per-feature override); fell back
      to ``FEATURE_TOLERANCE``.
    * ``"placeholder-default:<pattern>"`` — match was a placeholder
      and the matched row also lacked the feature; same numeric
      fallback as ``placeholder:`` but kept distinct so future
      audits can tell whether the row was deliberately silent.
    * ``"no-calibration:<gpu_id>"`` — ``--gpu-id`` supplied but no
      row matched.
    * ``"default"``   — neither FP16 nor calibration applied.
    """

    if feature in fp16_features:
        return DEFAULT_FP16_TOLERANCE, "fp16"

    feature_default = FEATURE_TOLERANCE.get(feature, DEFAULT_FP32_TOLERANCE)
    if calibration is None or gpu_id is None:
        return feature_default, "default"

    entry = calibration.lookup(gpu_id)
    if entry is None:
        return feature_default, f"no-calibration:{gpu_id}"
    if feature in entry.features:
        label_kind = "calibrated" if entry.status == "calibrated" else "placeholder"
        return float(entry.features[feature]), f"{label_kind}:{entry.gpu_id_pattern}"
    # Matched arch row, no per-feature override — typical of placeholder
    # rows whose ``features:`` block is empty until measured.
    return feature_default, f"placeholder-default:{entry.gpu_id_pattern}"


def run_cell(
    cell: Cell,
    *,
    binary: Path,
    ref: Path,
    dist: Path,
    width: int,
    height: int,
    pix_fmt: str,
    bitdepth: int,
    workdir: Path,
    devices: dict[str, int],
    tolerance: float,
    tolerance_source: str = "default",
) -> CellResult:
    """Execute one cell of the parity matrix and diff it."""

    metrics = FEATURE_METRICS[cell.feature]
    out_a = workdir / f"{cell.feature}_{cell.backend_a}.json"
    out_b = workdir / f"{cell.feature}_{cell.backend_b}.json"

    rc_a, err_a = run_one(
        binary,
        ref,
        dist,
        width,
        height,
        pix_fmt,
        bitdepth,
        cell.feature,
        cell.backend_a,
        devices.get(cell.backend_a),
        out_a,
    )
    if rc_a != 0:
        return CellResult(
            feature=cell.feature,
            backend_a=cell.backend_a,
            backend_b=cell.backend_b,
            tolerance=tolerance,
            n_frames=0,
            per_metric_max=dict.fromkeys(metrics, 0.0),
            per_metric_mismatches=dict.fromkeys(metrics, 0),
            status="ERROR",
            note=f"backend_a {cell.backend_a} failed: {err_a.strip()[:200]}",
            tolerance_source=tolerance_source,
        )

    rc_b, err_b = run_one(
        binary,
        ref,
        dist,
        width,
        height,
        pix_fmt,
        bitdepth,
        cell.feature,
        cell.backend_b,
        devices.get(cell.backend_b),
        out_b,
    )
    if rc_b != 0:
        return CellResult(
            feature=cell.feature,
            backend_a=cell.backend_a,
            backend_b=cell.backend_b,
            tolerance=tolerance,
            n_frames=0,
            per_metric_max=dict.fromkeys(metrics, 0.0),
            per_metric_mismatches=dict.fromkeys(metrics, 0),
            status="ERROR",
            note=f"backend_b {cell.backend_b} failed: {err_b.strip()[:200]}",
            tolerance_source=tolerance_source,
        )

    a_frames = load_frames(out_a)
    b_frames = load_frames(out_b)
    if len(a_frames) != len(b_frames):
        return CellResult(
            feature=cell.feature,
            backend_a=cell.backend_a,
            backend_b=cell.backend_b,
            tolerance=tolerance,
            n_frames=0,
            per_metric_max=dict.fromkeys(metrics, 0.0),
            per_metric_mismatches=dict.fromkeys(metrics, 0),
            status="ERROR",
            note=f"frame-count mismatch a={len(a_frames)} b={len(b_frames)}",
            tolerance_source=tolerance_source,
        )

    per_max, per_mismatch = diff_frames(a_frames, b_frames, metrics, tolerance)
    fail = any(c > 0 for c in per_mismatch.values())
    return CellResult(
        feature=cell.feature,
        backend_a=cell.backend_a,
        backend_b=cell.backend_b,
        tolerance=tolerance,
        n_frames=len(a_frames),
        per_metric_max=per_max,
        per_metric_mismatches=per_mismatch,
        status="FAIL" if fail else "OK",
        tolerance_source=tolerance_source,
    )


def emit_json(results: list[CellResult], path: Path) -> None:
    payload = {
        "schema_version": 1,
        "cells": [
            {
                "feature": r.feature,
                "backend_a": r.backend_a,
                "backend_b": r.backend_b,
                "tolerance_abs": r.tolerance,
                "tolerance_source": r.tolerance_source,
                "n_frames": r.n_frames,
                "status": r.status,
                "note": r.note,
                "per_metric_max_abs_diff": r.per_metric_max,
                "per_metric_mismatches": r.per_metric_mismatches,
            }
            for r in results
        ],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def emit_md(results: list[CellResult], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Cross-backend parity gate (T6-8)")
    lines.append("")
    lines.append("| feature | backend pair | tolerance | source | frames | max abs diff | status |")
    lines.append("|---|---|---:|---|---:|---:|---|")
    for r in results:
        max_diff = max(r.per_metric_max.values()) if r.per_metric_max else 0.0
        pair = f"{r.backend_a} ↔ {r.backend_b}"
        lines.append(
            f"| `{r.feature}` | {pair} | {r.tolerance:.1e} | "
            f"`{r.tolerance_source}` | {r.n_frames} | "
            f"{max_diff:.3e} | {r.status} |"
        )
    lines.append("")
    fails = [r for r in results if r.status in ("FAIL", "ERROR")]
    if fails:
        lines.append("## Failures detail")
        lines.append("")
        for r in fails:
            lines.append(f"### `{r.feature}` ({r.backend_a} ↔ {r.backend_b}) — {r.status}")
            if r.note:
                lines.append(f"- note: {r.note}")
            for m, d in r.per_metric_max.items():
                miss = r.per_metric_mismatches.get(m, 0)
                lines.append(f"- `{m}`: max_abs_diff={d:.3e}, mismatches={miss}")
            lines.append("")
    with path.open("w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vmaf-binary",
        type=Path,
        required=True,
        help="path to libvmaf/build/tools/vmaf",
    )
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--distorted", type=Path, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--pixel-format", default="420")
    ap.add_argument("--bitdepth", type=int, default=8)
    ap.add_argument(
        "--features",
        nargs="+",
        default=sorted(FEATURE_METRICS.keys()),
        choices=sorted(FEATURE_METRICS.keys()),
        help="features to include in the matrix (default: all registered)",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["cpu", "vulkan"],
        choices=sorted(BACKEND_SUFFIX.keys()),
        help="backends to pair (default: cpu + vulkan, lavapipe-friendly)",
    )
    ap.add_argument(
        "--fp16-features",
        nargs="*",
        default=[],
        help=(
            "feature names that should use the FP16 tolerance "
            f"({DEFAULT_FP16_TOLERANCE:.1e}) instead of their FP32 default"
        ),
    )
    ap.add_argument(
        "--cuda-device",
        type=int,
        default=BACKEND_DEFAULT_DEVICE["cuda"],
    )
    ap.add_argument(
        "--sycl-device",
        type=int,
        default=BACKEND_DEFAULT_DEVICE["sycl"],
    )
    ap.add_argument(
        "--vulkan-device",
        type=int,
        default=BACKEND_DEFAULT_DEVICE["vulkan"],
    )
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vmaf_parity_gate",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write machine-readable summary to this path",
    )
    ap.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="write Markdown summary to this path",
    )
    ap.add_argument(
        "--gpu-id",
        type=str,
        default=None,
        help=(
            "runtime GPU identifier (Research-0041 schema, e.g. "
            "'vulkan:0x10005:0x0' for lavapipe, 'cuda:8.6' for Ampere "
            "RTX 30). Used to look up per-arch tolerances in the "
            "ADR-0234 calibration table; falls back to "
            "FEATURE_TOLERANCE when omitted."
        ),
    )
    ap.add_argument(
        "--calibration-table",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=(f"path to the ADR-0234 calibration YAML (default: {DEFAULT_CALIBRATION_PATH})"),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.vmaf_binary.exists():
        sys.stderr.write(f"vmaf binary not found: {args.vmaf_binary}\n")
        return 2
    for p in (args.reference, args.distorted):
        if not p.exists():
            sys.stderr.write(f"fixture not found: {p}\n")
            return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    devices = {
        "cuda": args.cuda_device,
        "sycl": args.sycl_device,
        "vulkan": args.vulkan_device,
    }

    cells = build_matrix(args.features, args.backends)
    if not cells:
        sys.stderr.write("empty matrix — supply at least two backends or one feature\n")
        return 2

    # ADR-0234: load the calibration table once. ``None`` is the
    # backward-compatible signal (pyyaml missing, file absent, or
    # ``--gpu-id`` not supplied) and forces the per-feature default
    # path everywhere downstream.
    calibration: CalibrationTable | None = None
    if args.gpu_id is not None:
        calibration = load_calibration_table(args.calibration_table)
        if calibration is not None:
            entry = calibration.lookup(args.gpu_id)
            if entry is None:
                print(
                    f"calibration: no row matches gpu_id={args.gpu_id}; "
                    f"using FEATURE_TOLERANCE defaults"
                )
            else:
                print(
                    f"calibration: matched '{entry.gpu_id_pattern}' "
                    f"({entry.label}, status={entry.status})"
                )

    results: list[CellResult] = []
    for cell in cells:
        tolerance, tolerance_source = resolve_cell_tolerance(
            cell.feature,
            fp16_features=args.fp16_features,
            calibration=calibration,
            gpu_id=args.gpu_id,
        )
        result = run_cell(
            cell,
            binary=args.vmaf_binary,
            ref=args.reference,
            dist=args.distorted,
            width=args.width,
            height=args.height,
            pix_fmt=args.pixel_format,
            bitdepth=args.bitdepth,
            workdir=args.workdir,
            devices=devices,
            tolerance=tolerance,
            tolerance_source=tolerance_source,
        )
        results.append(result)
        max_diff = max(result.per_metric_max.values()) if result.per_metric_max else 0.0
        print(
            f"{result.feature:<14} "
            f"{result.backend_a:<6} ↔ {result.backend_b:<6}  "
            f"tol={result.tolerance:.1e} ({result.tolerance_source})  "
            f"max_abs_diff={max_diff:.3e}  "
            f"{result.status}" + (f"  ({result.note})" if result.note else "")
        )

    if args.json_out is not None:
        emit_json(results, args.json_out)
    if args.md_out is not None:
        emit_md(results, args.md_out)

    fail = any(r.status in ("FAIL", "ERROR") for r in results)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
