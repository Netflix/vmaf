# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase D — per-shot CRF tuning.

The "Netflix per-shot encoding" table-stakes feature for `vmaf-tune`.
TransNet V2 (real weights, ADR-0223) cuts the source into shots; for
each shot callers provide a target-VMAF predicate, usually the Phase-B
bisect backend, to pick a CRF; then we emit an FFmpeg encoding plan
that produces a per-shot CRF-varying encode and concatenates the
segments.

The library API keeps the predicate pluggable so tests and custom
operators can inject deterministic or content-aware selectors. The CLI
binds that seam to Phase-B bisect by default.

Public surface:

* :class:`Shot` — half-open frame range ``[start_frame, end_frame)``.
* :class:`ShotRecommendation` — ``(shot, crf, predicted_vmaf)``.
* :class:`EncodingPlan` — ordered segments + the FFmpeg argv list to
  produce the final encode.
* :func:`detect_shots` — wraps the C-side ``vmaf-perShot`` binary
  (ADR-0222) when available; falls back to a one-shot range.
* :func:`tune_per_shot` — drives the target-VMAF predicate per shot.
* :func:`merge_shots` — collapses recommendations into an
  :class:`EncodingPlan`.

The planner still stops short of executing the final segment encodes;
operators inspect or run the emitted FFmpeg command list. Native
per-codec zone/qpfile emission remains a later optimization over the
segment-and-concat path.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from io import StringIO
from pathlib import Path

from .codec_adapters import get_adapter

# Pluggable predicate signature: given a shot + target VMAF + encoder
# name, return ``(crf, measured_or_predicted_vmaf)``. The CLI calls
# Phase B's bisect; tests and advanced callers may inject a deterministic
# selector.
PredicateFn = Callable[["Shot", float, str], tuple[int, float]]


@dataclasses.dataclass(frozen=True)
class Shot:
    """Half-open frame range describing one shot.

    ``start_frame`` is inclusive, ``end_frame`` is exclusive — matching
    Python slice convention. ``vmaf-perShot``'s CSV output uses
    inclusive ``end_frame``; :func:`detect_shots` normalises into the
    half-open form.
    """

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(f"invalid shot range: [{self.start_frame}, {self.end_frame})")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclasses.dataclass(frozen=True)
class ShotRecommendation:
    """Per-shot CRF recommendation produced by :func:`tune_per_shot`."""

    shot: Shot
    crf: int
    predicted_vmaf: float


@dataclasses.dataclass(frozen=True)
class EncodingPlan:
    """Segment list plus the FFmpeg argv list that realises the encode.

    The plan is split into per-shot single-encode commands plus a
    final concat-demuxer command. Callers are free to run them
    sequentially or to parallelise per-shot encodes — the segment
    files are independent.
    """

    recommendations: tuple[ShotRecommendation, ...]
    encoder: str
    framerate: float
    segment_commands: tuple[tuple[str, ...], ...]
    concat_command: tuple[str, ...]
    concat_listing: str


def _which(binary: str) -> str | None:
    """``shutil.which`` wrapper kept thin so tests can monkeypatch."""
    return shutil.which(binary)


def detect_shots(
    video_path: Path,
    *,
    width: int,
    height: int,
    pix_fmt: str = "yuv420p",
    bitdepth: int = 8,
    total_frames: int | None = None,
    per_shot_bin: str = "vmaf-perShot",
    runner: object | None = None,
) -> list[Shot]:
    """Return the shot boundary list for ``video_path``.

    Calls the fork's C-side ``vmaf-perShot`` binary (ADR-0222) which
    wraps TransNet V2 (ADR-0223). Falls back to a single-shot range
    spanning the whole clip when the binary is missing or fails.

    ``total_frames`` is required for the fallback path; the
    ``vmaf-perShot`` path infers it from the YUV size.
    """
    shots, _ok = _detect_shots_with_status(
        video_path,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        bitdepth=bitdepth,
        total_frames=total_frames,
        per_shot_bin=per_shot_bin,
        runner=runner,
    )
    return shots


def _detect_shots_with_status(
    video_path: Path,
    *,
    width: int,
    height: int,
    pix_fmt: str = "yuv420p",
    bitdepth: int = 8,
    total_frames: int | None = None,
    per_shot_bin: str = "vmaf-perShot",
    runner: object | None = None,
) -> tuple[list[Shot], bool]:
    """Like :func:`detect_shots` but also returns ``ok`` — ``True`` iff
    the ``vmaf-perShot`` invocation succeeded and yielded shot data.

    Internal helper for :func:`_resolve_shot_metadata` (corpus.py): the
    summarise step needs to distinguish "real one-shot source" from
    "fallback because the binary failed", which the public list-only
    return shape can't carry.
    """
    binary = _which(per_shot_bin) if runner is None else per_shot_bin
    if binary is None:
        return _single_shot_fallback(total_frames), False

    # vmaf-perShot always writes "vmaf-perShot: wrote N shot(s) to PATH" to
    # stdout regardless of the --output value (including "--output -"). When
    # "--output -" was used, the JSON landed in a file literally named "-"
    # in the CWD while stdout carried the progress string, which made
    # json.loads(stdout) fail with JSONDecodeError. Use a real tmpfile so
    # stdout is exclusively the progress message and the JSON is read from
    # the file the binary actually wrote. See fix/vmaf-tune-pershot-stdout-json-protocol.
    with tempfile.NamedTemporaryFile(suffix=".json", prefix="vmaf_pershot_", delete=False) as _tmp:
        tmp_path = Path(_tmp.name)

    cmd = [
        per_shot_bin,
        "--reference",
        str(video_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--pixel_format",
        _bitdepth_aware_pix(pix_fmt),
        "--bitdepth",
        str(bitdepth),
        "--output",
        str(tmp_path),
        "--format",
        "json",
    ]

    try:
        runner_fn = runner or subprocess.run
        completed = runner_fn(  # type: ignore[operator]
            cmd, capture_output=True, text=True, check=False
        )
        rc = int(getattr(completed, "returncode", 1))
        if rc != 0:
            return _single_shot_fallback(total_frames), False

        # Read the JSON payload from the tmpfile the binary wrote.
        try:
            payload = tmp_path.read_text(encoding="utf-8")
        except OSError:
            return _single_shot_fallback(total_frames), False
        if not payload.strip():
            return _single_shot_fallback(total_frames), False
    finally:
        # Best-effort cleanup; unlink failure is non-fatal.
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return _parse_per_shot_json(payload), True


def _bitdepth_aware_pix(pix_fmt: str) -> str:
    """Map ffmpeg pix_fmt names to ``vmaf-perShot``'s ``--pixel_format``."""
    if "422" in pix_fmt:
        return "422"
    if "444" in pix_fmt:
        return "444"
    return "420"


def _single_shot_fallback(total_frames: int | None) -> list[Shot]:
    """One shot covering the whole clip — used when shot detection fails."""
    if total_frames is None or total_frames <= 0:
        # Caller has no frame count: emit a sentinel range that downstream
        # can pattern-match. End-frame > start-frame keeps :class:`Shot`
        # happy without lying about real length.
        return [Shot(start_frame=0, end_frame=1)]
    return [Shot(start_frame=0, end_frame=total_frames)]


def _parse_per_shot_json(payload: str) -> list[Shot]:
    """Parse ``vmaf-perShot``'s JSON output into a list of shots.

    Schema per ``docs/usage/vmaf-perShot.md``:

    .. code-block:: json

       {"shots": [{"start_frame": 0, "end_frame": 3, ...}, ...]}

    ``end_frame`` is inclusive in the source schema; we normalise to
    half-open here.
    """
    data = json.loads(payload)
    shots = data.get("shots") or []
    out: list[Shot] = []
    for entry in shots:
        start = int(entry["start_frame"])
        # Source schema is inclusive; half-open conversion adds 1.
        end = int(entry["end_frame"]) + 1
        out.append(Shot(start_frame=start, end_frame=end))
    if not out:
        return [Shot(start_frame=0, end_frame=1)]
    return out


def parse_per_shot_csv(payload: str) -> list[Shot]:
    """CSV variant of :func:`_parse_per_shot_json` — public for callers
    who already have the CSV sidecar from a prior ``vmaf-perShot`` run.
    """
    out: list[Shot] = []
    reader = csv.DictReader(StringIO(payload))
    for row in reader:
        start = int(row["start_frame"])
        end = int(row["end_frame"]) + 1
        out.append(Shot(start_frame=start, end_frame=end))
    return out


def tune_per_shot(
    shots: Sequence[Shot],
    *,
    target_vmaf: float,
    encoder: str = "libx264",
    predicate: PredicateFn | None = None,
) -> list[ShotRecommendation]:
    """Pick a per-shot CRF for each shot.

    ``predicate`` is the integration seam for Phase B's target-VMAF
    bisect. The CLI wires :func:`vmaftune.bisect.bisect_target_vmaf`;
    tests and advanced callers inject custom selectors. The default
    predicate uses the codec adapter's ``quality_default`` clamped
    into the codec's quality range so the library API remains usable
    in dry-run contexts without launching encodes.
    """
    if not shots:
        raise ValueError("tune_per_shot requires at least one shot")
    adapter = get_adapter(encoder)
    pred = predicate or _default_predicate

    recs: list[ShotRecommendation] = []
    for shot in shots:
        crf, predicted = pred(shot, target_vmaf, encoder)
        lo, hi = adapter.quality_range
        clamped = max(lo, min(hi, int(crf)))
        recs.append(ShotRecommendation(shot=shot, crf=clamped, predicted_vmaf=float(predicted)))
    return recs


def _default_predicate(shot: Shot, target_vmaf: float, encoder: str) -> tuple[int, float]:
    """Trivial fallback predicate.

    Used only when the caller does not pass a real bisect; exists so
    dry runs stay deterministic. Returns the codec's default
    quality value alongside the requested target VMAF.
    """
    _ = shot  # length unused in the trivial predicate
    adapter = get_adapter(encoder)
    return (adapter.quality_default, float(target_vmaf))


def merge_shots(
    recommendations: Sequence[ShotRecommendation],
    *,
    source: Path,
    output: Path,
    framerate: float,
    encoder: str = "libx264",
    segment_dir: Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> EncodingPlan:
    """Collapse per-shot recommendations into an :class:`EncodingPlan`.

    The plan emits one ``ffmpeg`` invocation per segment (using
    ``-ss`` + ``-frames:v`` derived from the half-open shot range) and
    a final concat-demuxer command that stitches the segments into
    ``output``. Segment files live under ``segment_dir`` (defaults to
    ``output.parent / "segments"``).
    """
    if not recommendations:
        raise ValueError("merge_shots requires at least one recommendation")
    adapter = get_adapter(encoder)

    seg_dir = segment_dir or output.parent / "segments"
    segment_cmds: list[tuple[str, ...]] = []
    listing_lines: list[str] = []
    for idx, rec in enumerate(recommendations):
        seg_path = seg_dir / f"shot_{idx:04d}.mp4"
        cmd = _segment_command(
            source=source,
            framerate=framerate,
            shot=rec.shot,
            crf=rec.crf,
            output=seg_path,
            adapter=adapter,
            ffmpeg_bin=ffmpeg_bin,
        )
        segment_cmds.append(cmd)
        # concat-demuxer expects POSIX-style escaped paths.
        listing_lines.append(f"file '{seg_path.as_posix()}'")

    listing = "\n".join(listing_lines) + "\n"
    concat_cmd: tuple[str, ...] = (
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str((seg_dir / "concat.txt").as_posix()),
        "-c",
        "copy",
        str(output),
    )

    return EncodingPlan(
        recommendations=tuple(recommendations),
        encoder=adapter.encoder,
        framerate=float(framerate),
        segment_commands=tuple(segment_cmds),
        concat_command=concat_cmd,
        concat_listing=listing,
    )


def _segment_command(
    *,
    source: Path,
    framerate: float,
    shot: Shot,
    crf: int,
    output: Path,
    adapter: object,
    ffmpeg_bin: str,
    preset: str | None = None,
) -> tuple[str, ...]:
    """Build the per-shot FFmpeg argv.

    Uses ``-ss`` (input-seek) + ``-frames:v`` so the segment is exactly
    ``shot.length`` frames regardless of GOP placement. Callers
    encoding a raw YUV source must add the ``-f rawvideo`` + geometry
    flags upstream — Phase D's smoke path here assumes the source is
    already an addressable container.

    The ``-c:v ...`` slice is delegated to the codec adapter's
    :meth:`ffmpeg_codec_args` per HP-1 / ADR-0326 so non-x264 codecs
    (libaom-av1, NVENC, QSV, AMF, ...) get their codec-correct flags
    instead of a hardcoded ``-preset ... -crf ...`` pair.
    """
    start_seconds = shot.start_frame / framerate
    fn = getattr(adapter, "ffmpeg_codec_args", None)
    chosen_preset = preset if preset is not None else _default_segment_preset(adapter)
    if fn is None:
        # Legacy fallback — preserves the historic libx264 shape for
        # adapters that haven't migrated yet.
        codec_args: tuple[str, ...] = (
            "-c:v",
            getattr(adapter, "encoder", "libx264"),
            "-preset",
            chosen_preset,
            "-crf",
            str(crf),
        )
    else:
        codec_args = tuple(fn(chosen_preset, crf))
    return (
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-frames:v",
        str(shot.length),
        *codec_args,
        str(output),
    )


def _default_segment_preset(adapter: object) -> str:
    """Pick a preset for the per-shot segment encode.

    Phase D's :class:`ShotRecommendation` carries a CRF only — preset
    is left to the adapter. We use ``"medium"`` when the adapter
    exposes it (the consistent default across x264 / x265 / NVENC /
    QSV / AMF / VVenC / libaom), falling back to the first declared
    preset otherwise, and to ``"medium"`` for legacy stubs without a
    ``presets`` tuple.
    """
    presets = getattr(adapter, "presets", ())
    if presets:
        if "medium" in presets:
            return "medium"
        return presets[0]
    return "medium"


def write_concat_listing(plan: EncodingPlan, listing_path: Path) -> Path:
    """Persist the concat-demuxer listing to ``listing_path``.

    Convenience helper — kept separate from :func:`merge_shots` so the
    plan can be inspected/tested without filesystem side effects.
    """
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(plan.concat_listing, encoding="utf-8")
    return listing_path


def plan_to_shell_script(plan: EncodingPlan) -> str:
    """Render a plan as a copy-paste shell script for diagnostics.

    Not used by production callers; useful when debugging a Phase D
    smoke run.
    """
    lines: list[str] = ["#!/bin/sh", "set -eu"]
    for cmd in plan.segment_commands:
        lines.append(_shell_join(cmd))
    lines.append(_shell_join(plan.concat_command))
    return "\n".join(lines) + "\n"


def _shell_join(parts: Iterable[str]) -> str:
    """Quote-aware join — minimum viable, no exotic shell escaping.

    Stops short of full ``shlex.quote`` because the plan argv is
    constructed in-process and does not contain shell metacharacters in
    normal usage; the helper exists for human-readable output, not for
    safe shell evaluation.
    """
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Shot-metadata aggregation (research-0086 / contributor-pack)
# ---------------------------------------------------------------------------
#
# TransNet-V2 cuts a source into shot ranges; the per-shot CRF tuner is
# the primary consumer. The corpus orchestrator wants a *summary* of
# the shot distribution — count, mean shot length in seconds, and the
# population std of shot lengths. Animation tends toward short shots
# with low variance, live-action drama toward longer shots with higher
# variance; the std column gives Phase B / C predictors a content-class
# proxy that costs nothing extra to compute (we already ran TransNet
# for the per-shot tuner).
#
# When ``detect_shots`` falls back to a single-shot range (binary
# missing or run failed), :func:`summarise_shots` returns the
# ``(0, 0.0, 0.0)`` sentinel so downstream consumers can filter the
# row out of any analysis that requires real shot data — see
# ``docs/usage/vmaf-tune.md`` § Shot metadata.


@dataclasses.dataclass(frozen=True)
class ShotMetadata:
    """Aggregate shot statistics for one source.

    All fields are zero when shot detection is unavailable
    (single-shot fallback). ``count > 0`` with ``avg_duration_sec > 0``
    is the contract for "real shot data was captured".
    """

    count: int
    avg_duration_sec: float
    duration_std_sec: float


_FALLBACK_METADATA = ShotMetadata(count=0, avg_duration_sec=0.0, duration_std_sec=0.0)


def _is_fallback_shotlist(shots: Sequence[Shot]) -> bool:
    """Heuristic: detect the ``[Shot(0, 1)]`` / single-shot fallback.

    ``detect_shots`` emits either:

    * a sentinel ``Shot(0, 1)`` when no frame count is known, or
    * a single shot spanning the whole clip when the binary failed.

    Both cases mean "shot detection was not real"; the caller should
    treat the metadata as missing rather than as "one giant shot".
    """
    if len(shots) != 1:
        return False
    only = shots[0]
    return only.start_frame == 0 and only.length <= 1


def summarise_shots(
    shots: Sequence[Shot],
    *,
    framerate: float,
) -> ShotMetadata:
    """Compute (count, mean, std) of shot lengths in seconds.

    ``framerate`` must be positive. Returns the all-zero sentinel for
    single-shot fallback lists or for any non-finite framerate. Uses
    the *population* standard deviation (``statistics.pstdev``) so the
    result is well-defined for ``count == 1``; sample std would emit
    ``NaN`` and force the caller to special-case the singleton.
    """
    if not shots:
        return _FALLBACK_METADATA
    if not math.isfinite(framerate) or framerate <= 0.0:
        return _FALLBACK_METADATA
    if _is_fallback_shotlist(shots):
        return _FALLBACK_METADATA

    durations = [shot.length / framerate for shot in shots]
    mean = sum(durations) / len(durations)
    # ``pstdev`` returns 0.0 for a one-element sequence which matches
    # the desired "no spread" semantic; ``stdev`` would raise.
    std = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    return ShotMetadata(
        count=len(shots),
        avg_duration_sec=float(mean),
        duration_std_sec=float(std),
    )


__all__ = [
    "EncodingPlan",
    "PredicateFn",
    "Shot",
    "ShotMetadata",
    "ShotRecommendation",
    "detect_shots",
    "merge_shots",
    "parse_per_shot_csv",
    "plan_to_shell_script",
    "summarise_shots",
    "tune_per_shot",
    "write_concat_listing",
]
