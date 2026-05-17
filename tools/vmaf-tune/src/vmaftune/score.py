# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""vmaf binary driver — Phase A.

Spawns the libvmaf CLI (`vmaf`) against a (reference YUV, distorted
encode) pair and parses the pooled VMAF score from the JSON output.

Subprocess boundary is the integration seam — tests mock subprocess.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import CANONICAL6_FEATURES


@dataclasses.dataclass(frozen=True)
class ScoreRequest:
    """Pair to score: reference YUV vs distorted encode.

    ``frame_skip_ref`` / ``frame_cnt`` mirror the libvmaf CLI flags
    (``--frame_skip_ref`` / ``--frame_cnt``). Sample-clip mode (ADR-0301)
    sets these so VMAF compares the same time window of the reference
    that was fed to the encoder, instead of slicing the reference YUV
    on disk. Both ``0`` (default) keeps the legacy full-source scoring.
    """

    reference: Path
    distorted: Path
    width: int
    height: int
    pix_fmt: str
    model: str = "vmaf_v0.6.1"
    frame_skip_ref: int = 0
    frame_cnt: int = 0


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    """Outcome of one scoring call.

    ``feature_means`` / ``feature_stds`` carry the canonical-6 libvmaf
    per-feature pooled aggregates parsed out of
    ``pooled_metrics.<feature>``: ``adm2``, ``vif_scale0..3``,
    ``motion2`` (see ``vmaftune.CANONICAL6_FEATURES``). Each feature key
    that libvmaf does not emit for the run (e.g. when a cambi-only
    model is selected) is absent from the dict — the corpus row writer
    fills the missing column with ``NaN`` rather than inventing a zero
    (ADR-0366).
    """

    request: ScoreRequest
    vmaf_score: float
    score_time_ms: float
    vmaf_binary_version: str
    exit_status: int
    stderr_tail: str
    feature_means: dict[str, float] = dataclasses.field(default_factory=dict)
    feature_stds: dict[str, float] = dataclasses.field(default_factory=dict)


_VMAF_VERSION_RE = re.compile(r"VMAF version[: ]+(\S+)")


def build_vmaf_command(
    req: ScoreRequest,
    json_output: Path,
    *,
    vmaf_bin: str = "vmaf",
    backend: str | None = None,
) -> list[str]:
    """Compose the libvmaf CLI argv. Pure function for test pinning.

    ``backend`` (when set) is forwarded as the libvmaf CLI's
    ``--backend NAME`` selector — values ``cpu`` / ``cuda`` / ``sycl``
    / ``vulkan`` per ADR-0127 / ADR-0175 / ADR-0186 / ADR-0299. When
    ``None`` the flag is omitted so the libvmaf binary picks its own
    default (CPU on a stock build).
    """
    cmd = [
        vmaf_bin,
        "--reference",
        str(req.reference),
        "--distorted",
        str(req.distorted),
        "--width",
        str(req.width),
        "--height",
        str(req.height),
        "--pixel_format",
        _pixfmt_to_vmaf(req.pix_fmt),
        "--bitdepth",
        str(_bitdepth_for(req.pix_fmt)),
        "--model",
        _model_arg(req.model),
        "--json",
        "--output",
        str(json_output),
    ]
    if backend:
        cmd.extend(["--backend", backend])
    # Sample-clip mode (ADR-0301): align reference window with the
    # encoded slice so VMAF compares matching frames. The distorted is
    # already a clip-length encode, so no --frame_skip_dist is needed.
    if req.frame_skip_ref > 0:
        cmd.extend(["--frame_skip_ref", str(req.frame_skip_ref)])
    if req.frame_cnt > 0:
        cmd.extend(["--frame_cnt", str(req.frame_cnt)])
    return cmd


def _model_arg(model: str) -> str:
    """Format the ``--model`` argument for the libvmaf CLI.

    Accepts either a bare version identifier (``"vmaf_v0.6.1"``) or a
    pre-formatted ``key=value`` string (``"path=/abs/model.json"``,
    ``"version=vmaf_v0.6.1"``). Bare identifiers are wrapped as
    ``version=...``; pre-formatted strings pass through. Used by
    ``corpus.py`` to inject HDR-model paths (see ADR-0295).
    """
    if "=" in model:
        return model
    return f"version={model}"


def _pixfmt_to_vmaf(pix_fmt: str) -> str:
    """Map ffmpeg pix_fmt to libvmaf's --pixel_format vocabulary.

    Only the subset Phase A actually drives. Falls back to ``420``.
    """
    if pix_fmt.startswith("yuv422"):
        return "422"
    if pix_fmt.startswith("yuv444"):
        return "444"
    return "420"


def _bitdepth_for(pix_fmt: str) -> int:
    if "10le" in pix_fmt or "p10" in pix_fmt:
        return 10
    if "12le" in pix_fmt or "p12" in pix_fmt:
        return 12
    return 8


def parse_vmaf_json(payload: dict) -> float:
    """Pull the pooled VMAF score from libvmaf's JSON output.

    Tries the modern ``pooled_metrics.vmaf.mean`` shape first, falls
    back to the older top-level ``VMAF score``. Raises ``ValueError``
    if neither is present.
    """
    pooled = payload.get("pooled_metrics") or {}
    vmaf = pooled.get("vmaf") or {}
    if "mean" in vmaf:
        return float(vmaf["mean"])
    if "VMAF score" in payload:
        return float(payload["VMAF score"])
    raise ValueError("vmaf JSON missing pooled_metrics.vmaf.mean")


def parse_feature_aggregates(
    payload: dict, feature_names: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Pull per-feature ``mean`` / ``stddev`` aggregates from libvmaf JSON.

    Modern libvmaf emits ``pooled_metrics.<feature> = {"min", "max",
    "mean", "stddev"}`` for every registered feature extractor. We
    surface ``mean`` and ``stddev`` because the canonical-6 trainers
    (``train_fr_regressor_v[23].py``) consume both. Features not
    present in ``pooled_metrics`` (model-dependent — e.g. a cambi-only
    fixture won't carry ``adm2``) are simply absent from the returned
    dicts; the corpus row writer translates absence into ``NaN``.

    The legacy top-level ``VMAF score`` shape predates per-feature
    pooling and is silently treated as an empty aggregate set.
    """
    pooled = payload.get("pooled_metrics") or {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in feature_names:
        block = pooled.get(name)
        if not isinstance(block, dict):
            continue
        if "mean" in block:
            try:
                means[name] = float(block["mean"])
            except (TypeError, ValueError):
                pass
        if "stddev" in block:
            try:
                stds[name] = float(block["stddev"])
            except (TypeError, ValueError):
                pass
    return means, stds


def _decode_to_raw_yuv(
    src: Path,
    dst: Path,
    *,
    pix_fmt: str,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
) -> int:
    """Decode a container (mp4/mkv/…) to a raw planar YUV file for the vmaf CLI.

    The vmaf CLI only accepts ``.yuv`` / ``.y4m`` inputs. When the distorted
    encode is a container file (e.g. ``.mp4``) the caller must decode it first.
    Returns the ffmpeg exit code — non-zero signals a decode failure.
    """
    runner_fn = runner or subprocess.run
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        str(dst),
    ]
    completed = runner_fn(cmd, capture_output=True, text=True, check=False)  # type: ignore[operator]
    return int(getattr(completed, "returncode", 1))


def run_score(
    req: ScoreRequest,
    *,
    vmaf_bin: str = "vmaf",
    runner: object | None = None,
    workdir: Path | None = None,
    backend: str | None = None,
) -> ScoreResult:
    """Drive the vmaf CLI for a single (ref, dist) pair.

    ``backend`` is forwarded to :func:`build_vmaf_command`; when ``None``
    no ``--backend`` flag is emitted (libvmaf picks its own default).

    The vmaf CLI only accepts raw ``.yuv`` / ``.y4m`` inputs. Callers that
    pass a container path (``mp4``, ``mkv``, etc.) as ``req.distorted`` must
    decode it to a raw YUV file first — see :func:`decode_distorted_container`
    in corpus.py for the corpus pipeline's decode step.
    """
    runner_fn = runner or subprocess.run

    if workdir is None:
        workdir_ctx = tempfile.TemporaryDirectory()
        workdir_path = Path(workdir_ctx.name)
    else:
        workdir_ctx = None
        workdir_path = workdir
        workdir_path.mkdir(parents=True, exist_ok=True)

    json_path = workdir_path / "vmaf.json"
    cmd = build_vmaf_command(req, json_path, vmaf_bin=vmaf_bin, backend=backend)

    try:
        started = time.monotonic()
        completed = runner_fn(  # type: ignore[operator]
            cmd, capture_output=True, text=True, check=False
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0

        stderr = getattr(completed, "stderr", "") or ""
        rc = int(getattr(completed, "returncode", 1))

        score = float("nan")
        feature_means: dict[str, float] = {}
        feature_stds: dict[str, float] = {}
        if rc == 0 and json_path.exists():
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            try:
                score = parse_vmaf_json(payload)
            except ValueError:
                rc = rc or 65
            # Per-feature aggregates are best-effort — a cambi-only
            # model won't expose ``adm2`` etc.; the corpus row writer
            # fills missing entries with NaN.
            feature_means, feature_stds = parse_feature_aggregates(payload, CANONICAL6_FEATURES)

        match = _VMAF_VERSION_RE.search(stderr)
        version = match.group(1) if match else "unknown"

        return ScoreResult(
            request=req,
            vmaf_score=score,
            score_time_ms=elapsed_ms,
            vmaf_binary_version=version,
            exit_status=rc,
            stderr_tail=stderr[-2048:],
            feature_means=feature_means,
            feature_stds=feature_stds,
        )
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()
