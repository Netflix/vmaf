# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Tests for the TransNet-V2 shot-metadata corpus columns (research-0086).

Mocks the ``vmaf-perShot`` invocation so the suite runs without the
binary on PATH. Two scenarios exercise the column wiring end-to-end:
a low-shot-count fixture (live-action drama proxy: 2 long shots) and
a high-shot-count fixture (animation / news proxy: 6 short shots).
A third test covers the "shot detection unavailable" sentinel path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune import CORPUS_ROW_KEYS  # noqa: E402
from vmaftune.corpus import CorpusJob, CorpusOptions, iter_rows  # noqa: E402
from vmaftune.per_shot import Shot, ShotMetadata, summarise_shots  # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _shot_runner_for(shot_ranges: list[tuple[int, int]]):
    """Return a fake subprocess runner that emits TransNet JSON with
    inclusive end-frames matching ``shot_ranges`` (half-open input).

    The new protocol writes JSON to the ``--output`` tmpfile (not stdout);
    the runner finds the path in ``cmd`` and writes the fixture there.
    """
    inclusive = [{"start_frame": s, "end_frame": e - 1} for s, e in shot_ranges]
    payload = json.dumps({"shots": inclusive})

    def runner(cmd, capture_output, text, check):  # noqa: ARG001
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(payload, encoding="utf-8")
        n = len(inclusive)
        return _FakeCompleted(
            returncode=0,
            stdout=f"vmaf-perShot: wrote {n} shot(s) to {out_path}\n",
        )

    return runner


def _failing_shot_runner():
    """Runner stub that mimics ``vmaf-perShot`` returning non-zero."""

    def runner(cmd, capture_output, text, check):  # noqa: ARG001
        return _FakeCompleted(returncode=1, stdout="")

    return runner


def _fake_encode_run(cmd, capture_output, text, check):  # noqa: ARG001
    out_path = Path(cmd[-1])
    out_path.write_bytes(b"\x00" * 4096)
    return _FakeCompleted(returncode=0, stderr="ffmpeg version 6.1.1\nx264 - core 164\n")


def _fake_score_run(cmd, capture_output, text, check):  # noqa: ARG001
    out_idx = cmd.index("--output") + 1
    out = Path(cmd[out_idx])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.5}}}))
    return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0-lusoris\n")


# ---------------------------------------------------------------------------
# summarise_shots — pure-helper coverage
# ---------------------------------------------------------------------------


def test_summarise_shots_population_std_singleton_is_zero():
    md = summarise_shots([Shot(0, 240)], framerate=24.0)
    assert md.count == 1
    assert md.avg_duration_sec == pytest.approx(10.0)
    assert md.duration_std_sec == pytest.approx(0.0)


def test_summarise_shots_animation_proxy_has_low_mean_and_some_spread():
    # 4 shots @ 24 fps: 12, 24, 36, 48 frames -> 0.5, 1.0, 1.5, 2.0 s
    shots = [Shot(0, 12), Shot(12, 36), Shot(36, 72), Shot(72, 120)]
    md = summarise_shots(shots, framerate=24.0)
    assert md.count == 4
    assert md.avg_duration_sec == pytest.approx(1.25)
    # Population std of [0.5, 1.0, 1.5, 2.0] = sqrt(5/16) ~= 0.559
    assert md.duration_std_sec == pytest.approx(0.5590169943749475, rel=1e-6)


def test_summarise_shots_falls_back_on_sentinel():
    assert summarise_shots([Shot(0, 1)], framerate=24.0) == ShotMetadata(0, 0.0, 0.0)


def test_summarise_shots_falls_back_on_empty_or_bad_framerate():
    assert summarise_shots([], framerate=24.0).count == 0
    assert summarise_shots([Shot(0, 240)], framerate=0.0).count == 0
    assert summarise_shots([Shot(0, 240)], framerate=float("nan")).count == 0


# ---------------------------------------------------------------------------
# iter_rows — column wiring
# ---------------------------------------------------------------------------


def _make_job(tmp_path: Path) -> CorpusJob:
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x80" * 1024)
    return CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=10.0,
        cells=(("medium", 23),),
    )


def _make_opts(tmp_path: Path) -> CorpusOptions:
    return CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        keep_encodes=False,
        src_sha256=False,
    )


def test_iter_rows_low_shot_count_fixture_populates_columns(tmp_path: Path):
    """Live-action drama proxy: 2 long shots over 10 s (mean ~5 s)."""
    job = _make_job(tmp_path)
    opts = _make_opts(tmp_path)
    # 240 frames total at 24 fps; split 120 / 120 (5 s each).
    shot_runner = _shot_runner_for([(0, 120), (120, 240)])

    rows = list(
        iter_rows(
            job,
            opts,
            encode_runner=_fake_encode_run,
            score_runner=_fake_score_run,
            shot_runner=shot_runner,
        )
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(CORPUS_ROW_KEYS) == set(row.keys())
    assert row["shot_count"] == 2
    assert row["shot_avg_duration_sec"] == pytest.approx(5.0)
    # Population std of [5.0, 5.0] = 0.0
    assert row["shot_duration_std_sec"] == pytest.approx(0.0)


def test_iter_rows_high_shot_count_fixture_populates_columns(tmp_path: Path):
    """Animation / news proxy: 6 shots, varied lengths -> non-zero std."""
    job = _make_job(tmp_path)
    opts = _make_opts(tmp_path)
    # 240 frames at 24 fps split into 6 shots of varied length:
    # 24, 36, 36, 48, 48, 48 frames -> 1.0, 1.5, 1.5, 2.0, 2.0, 2.0 s
    ranges = [(0, 24), (24, 60), (60, 96), (96, 144), (144, 192), (192, 240)]
    shot_runner = _shot_runner_for(ranges)

    rows = list(
        iter_rows(
            job,
            opts,
            encode_runner=_fake_encode_run,
            score_runner=_fake_score_run,
            shot_runner=shot_runner,
        )
    )
    row = rows[0]
    assert row["shot_count"] == 6
    assert row["shot_avg_duration_sec"] == pytest.approx(10.0 / 6.0, rel=1e-6)
    assert row["shot_duration_std_sec"] > 0.0


def test_iter_rows_emits_zero_metadata_when_shot_detection_fails(tmp_path: Path):
    """Sentinel path: ``vmaf-perShot`` non-zero exit -> all zeros."""
    job = _make_job(tmp_path)
    opts = _make_opts(tmp_path)

    rows = list(
        iter_rows(
            job,
            opts,
            encode_runner=_fake_encode_run,
            score_runner=_fake_score_run,
            shot_runner=_failing_shot_runner(),
        )
    )
    row = rows[0]
    assert row["shot_count"] == 0
    assert row["shot_avg_duration_sec"] == 0.0
    assert row["shot_duration_std_sec"] == 0.0


def test_corpus_row_keys_includes_shot_metadata_trio():
    assert "shot_count" in CORPUS_ROW_KEYS
    assert "shot_avg_duration_sec" in CORPUS_ROW_KEYS
    assert "shot_duration_std_sec" in CORPUS_ROW_KEYS
