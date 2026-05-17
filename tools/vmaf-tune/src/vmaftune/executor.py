# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase F execute mode — drive real encodes + scores for a ``vmaf-tune auto`` plan.

``run_plan`` iterates the ``selected`` cell(s) from an :class:`~vmaftune.auto.AutoPlan`,
runs FFmpeg (via :func:`~vmaftune.encode.run_encode`) per rung, scores each output with
the libvmaf CLI (via :func:`~vmaftune.score.run_score`), and appends rows to a JSONL
results file under ``out_dir/tune_results.jsonl``.

Two additional execution modes extend the base executor (ADR-0468):

* **Per-shot** (``run_plan_per_shot``): detects shot boundaries via
  :func:`~vmaftune.per_shot.detect_shots`, scores each segment independently,
  and reports per-shot VMAF alongside an aggregate weighted by shot length.
* **Saliency** (``run_plan_saliency``): applies saliency-aware encoding via
  :func:`~vmaftune.saliency.saliency_aware_encode` so salient regions receive
  preferential bit allocation; the resulting encode is scored in the same
  encode → score pipeline as the base mode.

Design notes (ADR-0454, ADR-0468):

* Zero new mandatory dependencies — results are written as JSONL, matching the corpus
  path (``corpus.py``). A future polars/pyarrow layer can convert on demand.
* The subprocess boundary is the seam: ``encode_runner`` and ``score_runner`` kwargs
  accept the same mock-runner pattern used throughout the harness so the executor is
  fully testable without FFmpeg or ``vmaf`` binaries.
* Only the ``selected`` cell is executed by default; pass ``execute_all=True`` to run
  every cell in the plan (useful for a post-hoc A/B comparison).
* Per-shot and saliency modes are opt-in via dedicated entry-points rather than flags
  on ``run_plan`` — the call-site intent is explicit and test isolation is simpler.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .encode import EncodeRequest, EncodeResult, run_encode
from .score import ScoreRequest, ScoreResult, run_score


@dataclasses.dataclass(frozen=True)
class ExecuteResult:
    """Outcome of one (encode + score) pair for a single plan cell.

    ``cell`` is the original plan dict (a reference to the ``cells`` entry).
    ``encode`` and ``score`` are ``None`` when the respective step was skipped
    (e.g. encode failed and ``score`` was not attempted).
    ``row`` is the flat dict written to the JSONL results file — it is a
    merged view of the cell metadata and the encode/score outcomes.
    """

    cell: dict[str, Any]
    encode: EncodeResult | None
    score: ScoreResult | None
    row: dict[str, Any]


def _cell_to_encode_request(
    cell: dict[str, Any],
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str,
    width: int,
    height: int,
    framerate: float,
    source_is_container: bool,
) -> EncodeRequest:
    """Build an :class:`EncodeRequest` from an ``AutoPlan`` cell dict."""
    codec = str(cell.get("codec", "libx264"))
    preset = str(cell.get("preset", "medium"))
    crf = int(cell.get("crf", 23))
    cell_index = int(cell.get("cell_index", 0))
    output = out_dir / f"encode_{cell_index:03d}_{codec}_{preset}_crf{crf}.mkv"
    return EncodeRequest(
        source=src,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        framerate=framerate,
        encoder=codec,
        preset=preset,
        crf=crf,
        output=output,
        source_is_container=source_is_container,
    )


def _make_row(
    cell: dict[str, Any],
    enc: EncodeResult | None,
    sc: ScoreResult | None,
) -> dict[str, Any]:
    """Flatten cell + encode + score outcome into a single results dict."""
    row: dict[str, Any] = {
        "cell_index": cell.get("cell_index"),
        "codec": cell.get("codec"),
        "preset": cell.get("preset"),
        "crf": cell.get("crf"),
        "selected": bool(cell.get("selected", False)),
        "estimated_vmaf": cell.get("estimated_vmaf"),
        "estimated_bitrate_kbps": cell.get("estimated_bitrate_kbps"),
        "prediction_source": cell.get("prediction_source"),
        # Encode outcomes
        "encode_size_bytes": enc.encode_size_bytes if enc else None,
        "encode_time_ms": enc.encode_time_ms if enc else None,
        "encode_exit_status": enc.exit_status if enc else None,
        "ffmpeg_version": enc.ffmpeg_version if enc else None,
        "encoder_version": enc.encoder_version if enc else None,
        "encode_path": str(enc.request.output) if enc else None,
        # Score outcomes
        "vmaf_score": sc.vmaf_score if sc else None,
        "score_time_ms": sc.score_time_ms if sc else None,
        "score_exit_status": sc.exit_status if sc else None,
        "vmaf_binary_version": sc.vmaf_binary_version if sc else None,
    }
    if sc is not None:
        for feat, val in sc.feature_means.items():
            row[f"feature_{feat}_mean"] = val
        for feat, val in sc.feature_stds.items():
            row[f"feature_{feat}_std"] = val
    return row


def run_plan(
    plan: "AutoPlan",  # type: ignore[name-defined]  # noqa: F821
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str = "yuv420p",
    width: int = 1920,
    height: int = 1080,
    framerate: float = 25.0,
    source_is_container: bool = True,
    execute_all: bool = False,
    vmaf_model: str = "vmaf_v0.6.1",
    vmaf_bin: str = "vmaf",
    ffmpeg_bin: str = "ffmpeg",
    encode_runner: Callable[..., Any] | None = None,
    score_runner: Callable[..., Any] | None = None,
) -> list[ExecuteResult]:
    """Realise an ``AutoPlan`` by running real encodes and scores.

    Parameters
    ----------
    plan:
        The :class:`~vmaftune.auto.AutoPlan` returned by :func:`~vmaftune.auto.run_auto`.
    src:
        Reference source path (forwarded to the encoder as input).
    out_dir:
        Directory for encoded files and the ``tune_results.jsonl`` log.
        Created if absent.
    pix_fmt:
        FFmpeg pixel format string (default ``yuv420p``).  When
        ``source_is_container=True`` the encoder driver reads format from the
        container; ``pix_fmt`` is still stored in :class:`EncodeRequest` for
        the score driver.
    width, height:
        Frame geometry; taken from plan ``metadata.source_meta`` when not
        overridden (the CLI wrapper does this automatically).
    framerate:
        Frame rate; same override semantics as ``width``/``height``.
    source_is_container:
        When ``True`` (default) the encoder driver omits raw-YUV input flags
        and lets FFmpeg detect the format from the container.
    execute_all:
        When ``True`` run every cell; otherwise only cells with
        ``selected=True`` are executed (default).
    vmaf_model:
        libvmaf model identifier forwarded to :class:`~vmaftune.score.ScoreRequest`.
    vmaf_bin, ffmpeg_bin:
        Binary names / paths for the ``vmaf`` and ``ffmpeg`` executables.
    encode_runner, score_runner:
        Optional ``subprocess.run``-compatible callables used as test seams.
        Pass ``None`` in production (the drivers call ``subprocess.run``
        directly).

    Returns
    -------
    list[ExecuteResult]
        One entry per executed cell, in plan order.  Always written to
        ``out_dir/tune_results.jsonl`` even on partial failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "tune_results.jsonl"

    # Pull geometry from plan metadata when caller did not override.
    source_meta = plan.metadata.get("source_meta", {})
    eff_width = int(source_meta.get("width", width)) if width == 1920 else width
    eff_height = int(source_meta.get("height", height)) if height == 1080 else height

    cells_to_run = [cell for cell in plan.cells if execute_all or bool(cell.get("selected", False))]

    results: list[ExecuteResult] = []

    with results_path.open("a", encoding="utf-8") as fh:
        for cell in cells_to_run:
            enc_req = _cell_to_encode_request(
                cell,
                src,
                out_dir,
                pix_fmt=pix_fmt,
                width=eff_width,
                height=eff_height,
                framerate=framerate,
                source_is_container=source_is_container,
            )

            enc: EncodeResult | None = None
            sc: ScoreResult | None = None

            try:
                enc = run_encode(enc_req, ffmpeg_bin=ffmpeg_bin, runner=encode_runner)
            except Exception as exc:  # noqa: BLE001
                # Encode failure is recorded in the row; scoring is skipped.
                _log(f"executor: encode failed for cell {cell.get('cell_index')}: {exc}")

            if enc is not None and enc.exit_status == 0:
                with tempfile.TemporaryDirectory() as td:
                    score_req = ScoreRequest(
                        reference=src,
                        distorted=enc_req.output,
                        width=eff_width,
                        height=eff_height,
                        pix_fmt=pix_fmt,
                        model=vmaf_model,
                    )
                    try:
                        sc = run_score(
                            score_req,
                            vmaf_bin=vmaf_bin,
                            runner=score_runner,
                            workdir=Path(td),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _log(f"executor: score failed for cell " f"{cell.get('cell_index')}: {exc}")

            row = _make_row(cell, enc, sc)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            results.append(ExecuteResult(cell=cell, encode=enc, score=sc, row=row))

    return results


def _log(msg: str) -> None:
    """Write a timestamped line to stderr (no logging dep)."""
    import sys

    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------------------------
# Per-shot execution mode (ADR-0468)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ShotExecuteResult:
    """Outcome of scoring one shot segment in per-shot execution mode.

    ``shot_index`` is the 0-based position in the shot list returned by
    :func:`~vmaftune.per_shot.detect_shots`. ``score`` is ``None`` when
    the shot's temporary encode or score step failed. ``length_frames``
    drives the weighted aggregate in :class:`PerShotPlanResult`.
    """

    shot_index: int
    length_frames: int
    score: "ScoreResult | None"
    row: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PerShotPlanResult:
    """Aggregate result of a per-shot execution run for one plan cell.

    ``shot_results`` contains one :class:`ShotExecuteResult` per detected
    shot. ``weighted_vmaf`` is the frame-length-weighted mean of per-shot
    VMAF scores (shots with ``score=None`` are excluded). When all shots
    fail, ``weighted_vmaf`` is ``float('nan')``.
    """

    cell: dict[str, Any]
    shot_results: tuple[ShotExecuteResult, ...]
    weighted_vmaf: float
    row: dict[str, Any]


def run_plan_per_shot(
    plan: "AutoPlan",  # type: ignore[name-defined]  # noqa: F821
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str = "yuv420p",
    width: int = 1920,
    height: int = 1080,
    framerate: float = 25.0,
    execute_all: bool = False,
    vmaf_model: str = "vmaf_v0.6.1",
    vmaf_bin: str = "vmaf",
    ffmpeg_bin: str = "ffmpeg",
    per_shot_bin: str = "vmaf-perShot",
    encode_runner: Callable[..., Any] | None = None,
    score_runner: Callable[..., Any] | None = None,
    shot_runner: object | None = None,
) -> list[PerShotPlanResult]:
    """Execute an ``AutoPlan`` with per-shot VMAF scoring (ADR-0468).

    For each selected plan cell:

    1. Detect shot boundaries using :func:`~vmaftune.per_shot.detect_shots`
       (falls back to a single-shot range when ``vmaf-perShot`` is absent).
    2. For each shot, encode the source segment and score it independently.
    3. Aggregate per-shot VMAF scores into a frame-length-weighted mean.

    Results are appended to ``out_dir/tune_results_per_shot.jsonl``.

    Parameters
    ----------
    plan:
        The :class:`~vmaftune.auto.AutoPlan` from :func:`~vmaftune.auto.run_auto`.
    src:
        Reference source (container or raw YUV; shot detection works on
        containers via ``vmaf-perShot``).
    out_dir:
        Directory for encoded segments and the JSONL log. Created if absent.
    per_shot_bin:
        Binary name for the ``vmaf-perShot`` shot-detection tool.
    shot_runner:
        Test seam for the ``vmaf-perShot`` subprocess call (same pattern as
        ``encode_runner`` / ``score_runner``).
    """
    from .per_shot import Shot, detect_shots  # local import to avoid cycles

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "tune_results_per_shot.jsonl"

    source_meta = plan.metadata.get("source_meta", {})
    eff_width = int(source_meta.get("width", width)) if width == 1920 else width
    eff_height = int(source_meta.get("height", height)) if height == 1080 else height

    cells_to_run = [cell for cell in plan.cells if execute_all or bool(cell.get("selected", False))]

    all_results: list[PerShotPlanResult] = []

    with results_path.open("a", encoding="utf-8") as fh:
        for cell in cells_to_run:
            codec = str(cell.get("codec", "libx264"))
            crf = int(cell.get("crf", 23))
            cell_index = int(cell.get("cell_index", 0))

            shots: list[Shot] = detect_shots(
                src,
                width=eff_width,
                height=eff_height,
                pix_fmt=pix_fmt,
                per_shot_bin=per_shot_bin,
                runner=shot_runner,
            )

            shot_results: list[ShotExecuteResult] = []
            for si, shot in enumerate(shots):
                seg_out = out_dir / f"shot_{cell_index:03d}_{si:04d}_{codec}_crf{crf}.mkv"
                enc_req = EncodeRequest(
                    source=src,
                    width=eff_width,
                    height=eff_height,
                    pix_fmt=pix_fmt,
                    framerate=framerate,
                    encoder=codec,
                    preset=str(cell.get("preset", "medium")),
                    crf=crf,
                    output=seg_out,
                    source_is_container=True,
                )

                sc: ScoreResult | None = None
                try:
                    enc = run_encode(enc_req, ffmpeg_bin=ffmpeg_bin, runner=encode_runner)
                    if enc.exit_status == 0:
                        score_req = ScoreRequest(
                            reference=src,
                            distorted=seg_out,
                            width=eff_width,
                            height=eff_height,
                            pix_fmt=pix_fmt,
                            model=vmaf_model,
                            frame_skip_ref=shot.start_frame,
                            frame_cnt=shot.length,
                        )
                        with tempfile.TemporaryDirectory() as td:
                            sc = run_score(
                                score_req,
                                vmaf_bin=vmaf_bin,
                                runner=score_runner,
                                workdir=Path(td),
                            )
                except Exception as exc:  # noqa: BLE001
                    _log(f"executor per-shot: cell {cell_index} shot {si} failed: {exc}")

                shot_row: dict[str, Any] = {
                    "cell_index": cell_index,
                    "shot_index": si,
                    "shot_start_frame": shot.start_frame,
                    "shot_end_frame": shot.end_frame,
                    "shot_length_frames": shot.length,
                    "codec": codec,
                    "crf": crf,
                    "vmaf_score": sc.vmaf_score if sc else None,
                    "score_exit_status": sc.exit_status if sc else None,
                }
                shot_results.append(
                    ShotExecuteResult(
                        shot_index=si,
                        length_frames=shot.length,
                        score=sc,
                        row=shot_row,
                    )
                )

            # Frame-length-weighted mean VMAF across shots that succeeded.
            total_frames = 0
            weighted_sum = 0.0
            for sr in shot_results:
                if sr.score is not None and not _is_nan(sr.score.vmaf_score):
                    total_frames += sr.length_frames
                    weighted_sum += sr.score.vmaf_score * sr.length_frames
            weighted_vmaf = weighted_sum / total_frames if total_frames > 0 else float("nan")

            plan_row: dict[str, Any] = {
                "cell_index": cell_index,
                "codec": codec,
                "crf": crf,
                "selected": bool(cell.get("selected", False)),
                "shot_count": len(shots),
                "weighted_vmaf": weighted_vmaf,
            }
            fh.write(json.dumps(plan_row, sort_keys=True) + "\n")
            fh.flush()

            all_results.append(
                PerShotPlanResult(
                    cell=cell,
                    shot_results=tuple(shot_results),
                    weighted_vmaf=weighted_vmaf,
                    row=plan_row,
                )
            )

    return all_results


# ---------------------------------------------------------------------------
# Saliency execution mode (ADR-0468)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SaliencyExecuteResult:
    """Outcome of one saliency-aware (encode + score) pair for a plan cell.

    ``saliency_available`` is ``True`` when the saliency model ran
    successfully. When ``False``, the encoder fell back to a plain encode
    (the :func:`~vmaftune.saliency.saliency_aware_encode` graceful-fallback
    path). ``score`` and ``row`` follow the same shape as
    :class:`ExecuteResult`.
    """

    cell: dict[str, Any]
    encode: "EncodeResult | None"
    score: "ScoreResult | None"
    saliency_available: bool
    row: dict[str, Any]


def run_plan_saliency(
    plan: "AutoPlan",  # type: ignore[name-defined]  # noqa: F821
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str = "yuv420p",
    width: int = 1920,
    height: int = 1080,
    framerate: float = 25.0,
    execute_all: bool = False,
    vmaf_model: str = "vmaf_v0.6.1",
    vmaf_bin: str = "vmaf",
    ffmpeg_bin: str = "ffmpeg",
    saliency_model_path: "Path | None" = None,
    duration_frames: int = 1,
    encode_runner: Callable[..., Any] | None = None,
    score_runner: Callable[..., Any] | None = None,
    session_factory: Any = None,
) -> list[SaliencyExecuteResult]:
    """Execute an ``AutoPlan`` with saliency-weighted encoding (ADR-0468).

    For each selected plan cell, the source is encoded using
    :func:`~vmaftune.saliency.saliency_aware_encode` which biases bits
    toward salient regions via per-codec ROI/qpfile injection, then the
    output is scored in the standard way.

    When saliency is unavailable (onnxruntime or model file missing) the
    encode falls back silently to a plain encode — ``saliency_available``
    in the result records which path was taken.

    Results are appended to ``out_dir/tune_results_saliency.jsonl``.

    Parameters
    ----------
    saliency_model_path:
        Path to the ``saliency_student_v1.onnx`` model. When ``None``,
        :func:`~vmaftune.saliency.compute_saliency_map` uses its default
        (``model/tiny/saliency_student_v1.onnx`` relative to repo root).
    duration_frames:
        Number of frames to write into the per-codec ROI sidecar. Forwarded
        to the saliency augment helpers.
    session_factory:
        Test seam for the ONNX Runtime session factory — same pattern as in
        :func:`~vmaftune.saliency.compute_saliency_map`.
    """
    from .saliency import SaliencyConfig, SaliencyUnavailableError, saliency_aware_encode

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "tune_results_saliency.jsonl"

    source_meta = plan.metadata.get("source_meta", {})
    eff_width = int(source_meta.get("width", width)) if width == 1920 else width
    eff_height = int(source_meta.get("height", height)) if height == 1080 else height

    cells_to_run = [cell for cell in plan.cells if execute_all or bool(cell.get("selected", False))]

    results: list[SaliencyExecuteResult] = []

    with results_path.open("a", encoding="utf-8") as fh:
        for cell in cells_to_run:
            cell_index = int(cell.get("cell_index", 0))
            codec = str(cell.get("codec", "libx264"))
            preset = str(cell.get("preset", "medium"))
            crf = int(cell.get("crf", 23))
            output = out_dir / f"sal_{cell_index:03d}_{codec}_{preset}_crf{crf}.mkv"

            enc_req = EncodeRequest(
                source=src,
                width=eff_width,
                height=eff_height,
                pix_fmt=pix_fmt,
                framerate=framerate,
                encoder=codec,
                preset=preset,
                crf=crf,
                output=output,
                source_is_container=True,
            )

            enc: "EncodeResult | None" = None
            sc: ScoreResult | None = None
            sal_available = False

            try:
                enc = saliency_aware_encode(
                    enc_req,
                    duration_frames=duration_frames,
                    model_path=saliency_model_path,
                    config=SaliencyConfig(),
                    encode_runner=encode_runner,
                    session_factory=session_factory,
                    ffmpeg_bin=ffmpeg_bin,
                )
                # ``saliency_aware_encode`` always returns an EncodeResult even
                # on fallback. Detect whether saliency actually ran by checking
                # for known ROI-related flags in extra_params on the request the
                # encoder saw — the saliency helpers mutate extra_params.
                sal_available = _saliency_was_applied(enc_req, enc)
            except SaliencyUnavailableError as exc:
                _log(f"executor saliency: unavailable for cell {cell_index}: {exc}")
            except Exception as exc:  # noqa: BLE001
                _log(f"executor saliency: encode failed for cell {cell_index}: {exc}")

            if enc is not None and enc.exit_status == 0:
                score_req = ScoreRequest(
                    reference=src,
                    distorted=output,
                    width=eff_width,
                    height=eff_height,
                    pix_fmt=pix_fmt,
                    model=vmaf_model,
                )
                with tempfile.TemporaryDirectory() as td:
                    try:
                        sc = run_score(
                            score_req,
                            vmaf_bin=vmaf_bin,
                            runner=score_runner,
                            workdir=Path(td),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _log(f"executor saliency: score failed for cell {cell_index}: {exc}")

            row: dict[str, Any] = {
                "cell_index": cell_index,
                "codec": codec,
                "preset": preset,
                "crf": crf,
                "selected": bool(cell.get("selected", False)),
                "saliency_available": sal_available,
                "encode_exit_status": enc.exit_status if enc else None,
                "vmaf_score": sc.vmaf_score if sc else None,
                "score_exit_status": sc.exit_status if sc else None,
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            results.append(
                SaliencyExecuteResult(
                    cell=cell,
                    encode=enc,
                    score=sc,
                    saliency_available=sal_available,
                    row=row,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_nan(v: float) -> bool:
    """Return ``True`` if ``v`` is ``float('nan')`` (avoids math import)."""
    return v != v  # noqa: PLR0124  (identity NaN check; intentional, not a bug)


def _saliency_was_applied(original_req: "EncodeRequest", enc: "EncodeResult") -> bool:
    """Heuristic: true when the encode request the driver saw has ROI params.

    The saliency augment helpers patch ``extra_params`` on a copy of the
    original request.  Because the executor only holds the pre-augmentation
    request, we inspect the result's ``request`` (which is the augmented one
    that ``run_encode`` received) for known ROI token prefixes.
    """
    augmented_params = enc.request.extra_params if enc is not None else ()
    roi_prefixes = ("-x264-params", "-x265-params", "-svtav1-params", "-vvenc-params", "-qpfile")
    return any(p in augmented_params for p in roi_prefixes)


__all__ = [
    "ExecuteResult",
    "PerShotPlanResult",
    "SaliencyExecuteResult",
    "ShotExecuteResult",
    "run_plan",
    "run_plan_per_shot",
    "run_plan_saliency",
]
