#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Per-frame CPU/GPU score collector for the calibration corpus (T7-GPU-ULP-CAL).

Companion to :mod:`scripts.ci.cross_backend_parity_gate`. Where the
parity gate emits a per-cell *summary* (max abs diff, mismatch count),
this script emits the **raw per-frame scores** for every
``(feature, backend, frame)`` triple — the labelled training data the
proposed GPU-generation calibration head (`ADR-0234`_,
`Research-0041`_) needs.

Usage modes:

* ``--smoke``: 100 frames × 5 features × Vulkan-only (lavapipe-friendly)
  — verifies the pipeline end-to-end on hosted CI without GPU hardware.
* full (no flag): caller-controlled feature / backend / frame selection.

Output: a parquet at ``--output`` with columns

    arch_id, feature_name, metric_name, frame_idx, raw_score_cpu, raw_score_gpu

This is a **proposal-stage** script (ADR-0234 is `Status: Proposed`).
It does not train or ship a calibration head; it produces the corpus
the training PR will consume. See `Research-0041`_ for the corpus
design rationale and the smallest viable training set.

.. _ADR-0234: ../docs/adr/0234-gpu-gen-ulp-calibration.md
.. _Research-0041: ../docs/research/0041-gpu-gen-ulp-calibration.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Reuse the parity gate's authoritative feature ↔ metric-name table and
# backend ↔ extractor-suffix table. Importing from a sibling top-level
# package would require packaging shenanigans, so we duplicate the
# constants here with a comment pinning the single-source-of-truth.

# Single-source-of-truth: ``scripts/ci/cross_backend_parity_gate.py``.
# When that file's ``FEATURE_METRICS`` / ``BACKEND_SUFFIX`` /
# ``BACKEND_DEVICE_FLAG`` change, this file changes in lockstep. The
# parity gate is the canonical owner; this is a deliberate copy to
# avoid Python sys.path gymnastics in a script that ships under
# ``ai/scripts/``.

FEATURE_METRICS: dict[str, tuple[str, ...]] = {
    "vif": (
        "integer_vif_scale0",
        "integer_vif_scale1",
        "integer_vif_scale2",
        "integer_vif_scale3",
    ),
    "motion": ("integer_motion", "integer_motion2"),
    "adm": (
        "integer_adm2",
        "integer_adm_scale0",
        "integer_adm_scale1",
        "integer_adm_scale2",
        "integer_adm_scale3",
    ),
    "psnr": ("psnr_y",),
    "float_ssim": ("float_ssim",),
    "float_ms_ssim": ("float_ms_ssim",),
    "psnr_hvs": (
        "psnr_hvs_y",
        "psnr_hvs_cb",
        "psnr_hvs_cr",
        "psnr_hvs",
    ),
    "ciede": ("ciede2000",),
    "ssimulacra2": ("ssimulacra2",),
}

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

# ``--smoke`` mode: smallest viable training set per Research-0041
# § "Smallest viable training set (proof-of-concept)". 5 features
# covering integer pipeline + float pipeline + DCT-heavy, Vulkan-only
# (lavapipe-friendly), 100 frames.

SMOKE_FEATURES: tuple[str, ...] = (
    "vif",
    "motion",
    "psnr",
    "float_ssim",
    "psnr_hvs",
)
SMOKE_BACKENDS: tuple[str, ...] = ("vulkan",)
SMOKE_FRAME_LIMIT: int = 100


@dataclass(frozen=True)
class Row:
    """One row of the calibration corpus."""

    arch_id: str
    feature_name: str
    metric_name: str
    frame_idx: int
    raw_score_cpu: float
    raw_score_gpu: float


def feature_extractor_name(feature: str, backend: str) -> str:
    """Map ``(feature, backend)`` to the extractor name ``--feature`` accepts."""

    suffix = BACKEND_SUFFIX[backend]
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
    frame_limit: int | None,
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
    if frame_limit is not None:
        cmd += ["--frame_cnt", str(frame_limit)]
    return cmd


def run_one(cmd: list[str]) -> tuple[int, str]:
    """Run one ``vmaf`` invocation. Returns ``(returncode, stderr_or_stdout)``."""

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return proc.returncode, (proc.stderr or proc.stdout)
    return 0, ""


def load_frames(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["frames"]


def collect_for_cell(
    *,
    feature: str,
    backend: str,
    arch_id: str,
    binary: Path,
    ref: Path,
    dist: Path,
    width: int,
    height: int,
    pix_fmt: str,
    bitdepth: int,
    device: int | None,
    workdir: Path,
    frame_limit: int | None,
) -> list[Row]:
    """Run ``feature`` on CPU and on ``backend``; emit per-frame paired rows."""

    metrics = FEATURE_METRICS[feature]
    out_cpu = workdir / f"{feature}_cpu.json"
    out_gpu = workdir / f"{feature}_{backend}.json"

    cpu_cmd = build_command(
        binary,
        ref,
        dist,
        width,
        height,
        pix_fmt,
        bitdepth,
        feature,
        "cpu",
        None,
        out_cpu,
        frame_limit,
    )
    gpu_cmd = build_command(
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
        out_gpu,
        frame_limit,
    )

    rc_cpu, err_cpu = run_one(cpu_cmd)
    if rc_cpu != 0:
        sys.stderr.write(f"[{feature}/cpu] vmaf failed (rc={rc_cpu}): {err_cpu.strip()[:200]}\n")
        return []
    rc_gpu, err_gpu = run_one(gpu_cmd)
    if rc_gpu != 0:
        sys.stderr.write(
            f"[{feature}/{backend}] vmaf failed (rc={rc_gpu}): {err_gpu.strip()[:200]}\n"
        )
        return []

    cpu_frames = load_frames(out_cpu)
    gpu_frames = load_frames(out_gpu)
    if len(cpu_frames) != len(gpu_frames):
        sys.stderr.write(
            f"[{feature}/{backend}] frame-count mismatch "
            f"cpu={len(cpu_frames)} gpu={len(gpu_frames)} — skipping cell\n"
        )
        return []

    rows: list[Row] = []
    for f_cpu, f_gpu in zip(cpu_frames, gpu_frames, strict=True):
        idx = int(f_cpu.get("frameNum", f_gpu.get("frameNum", -1)))
        for m in metrics:
            rows.append(
                Row(
                    arch_id=arch_id,
                    feature_name=feature,
                    metric_name=m,
                    frame_idx=idx,
                    raw_score_cpu=float(f_cpu["metrics"][m]),
                    raw_score_gpu=float(f_gpu["metrics"][m]),
                )
            )
    return rows


def write_parquet(rows: Iterable[Row], path: Path) -> int:
    """Write ``rows`` to ``path`` as parquet. Returns the row count."""

    rows_list = list(rows)
    if not rows_list:
        sys.stderr.write("no rows collected — nothing to write\n")
        return 0

    # Lazy import keeps the script importable on environments without
    # pandas/pyarrow installed (e.g. when only --help is invoked).
    try:
        import pandas as pd
    except ImportError as exc:
        sys.stderr.write(
            f"pandas required for parquet output ({exc}); "
            "install via `pip install pandas pyarrow`\n"
        )
        sys.exit(2)

    df = pd.DataFrame(
        [
            {
                "arch_id": r.arch_id,
                "feature_name": r.feature_name,
                "metric_name": r.metric_name,
                "frame_idx": r.frame_idx,
                "raw_score_cpu": r.raw_score_cpu,
                "raw_score_gpu": r.raw_score_gpu,
            }
            for r in rows_list
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(rows_list)


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
        "--arch-id",
        default="vulkan:0x10005:lavapipe",
        help=(
            "stable per-(backend,device) identifier — see Research-0041 "
            '§ "Per-arch detection mechanism". Default targets Mesa lavapipe.'
        ),
    )
    ap.add_argument(
        "--features",
        nargs="+",
        default=None,
        choices=sorted(FEATURE_METRICS.keys()),
        help="features to collect (default: all registered)",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=[b for b in BACKEND_SUFFIX if b != "cpu"],
        help="GPU backends to pair against CPU (default: vulkan only)",
    )
    ap.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="cap frames per (feature, backend) cell via --frame_cnt",
    )
    ap.add_argument("--cuda-device", type=int, default=1)
    ap.add_argument("--sycl-device", type=int, default=0)
    ap.add_argument("--vulkan-device", type=int, default=0)
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vmaf_calib_collect",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="parquet output path (e.g. runs/gpu_calibration_v0.parquet)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "shortcut: 5 features × Vulkan only × 100 frames "
            "(per Research-0041 smallest-viable-training-set)"
        ),
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

    # Resolve smoke shortcut (overrides feature / backend / frame defaults
    # in one go to guarantee a deterministic, hosted-CI-friendly run).
    if args.smoke:
        features = list(SMOKE_FEATURES)
        backends = list(SMOKE_BACKENDS)
        frame_limit = SMOKE_FRAME_LIMIT
    else:
        features = args.features or sorted(FEATURE_METRICS.keys())
        backends = args.backends or ["vulkan"]
        frame_limit = args.frame_limit

    args.workdir.mkdir(parents=True, exist_ok=True)
    devices = {
        "cuda": args.cuda_device,
        "sycl": args.sycl_device,
        "vulkan": args.vulkan_device,
    }

    all_rows: list[Row] = []
    for feature in features:
        for backend in backends:
            print(
                f"[collect] feature={feature} backend={backend} "
                f"arch_id={args.arch_id} frames={frame_limit or 'all'}"
            )
            rows = collect_for_cell(
                feature=feature,
                backend=backend,
                arch_id=args.arch_id,
                binary=args.vmaf_binary,
                ref=args.reference,
                dist=args.distorted,
                width=args.width,
                height=args.height,
                pix_fmt=args.pixel_format,
                bitdepth=args.bitdepth,
                device=devices.get(backend),
                workdir=args.workdir,
                frame_limit=frame_limit,
            )
            print(f"[collect]   → {len(rows)} rows")
            all_rows.extend(rows)

    n = write_parquet(all_rows, args.output)
    print(f"[collect] wrote {n} rows to {args.output}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
