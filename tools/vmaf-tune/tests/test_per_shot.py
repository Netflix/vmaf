# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase D smoke tests — mocks vmaf-perShot + Phase B predicate.

Covers:
* Three-shot detection routes through complexity-aware predicate and
  yields three different CRFs.
* Single-shot fallback fires when ``vmaf-perShot`` is unavailable.
* :func:`merge_shots` emits a well-formed FFmpeg per-segment + concat
  command pair.
* CLI entrypoint extracts shots and binds the real Phase-B predicate
  seam, with subprocess + bisect monkeypatched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make src/ importable without an editable install (Phase A pattern).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune import cli  # noqa: E402
from vmaftune.per_shot import (  # noqa: E402
    EncodingPlan,
    Shot,
    ShotRecommendation,
    detect_shots,
    merge_shots,
    parse_per_shot_csv,
    plan_to_shell_script,
    tune_per_shot,
    write_concat_listing,
)


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------- #
# Shot dataclass                                                              #
# --------------------------------------------------------------------------- #


def test_shot_rejects_invalid_range():
    with pytest.raises(ValueError):
        Shot(start_frame=10, end_frame=5)
    with pytest.raises(ValueError):
        Shot(start_frame=0, end_frame=0)
    with pytest.raises(ValueError):
        Shot(start_frame=-1, end_frame=10)


def test_shot_length_is_half_open():
    assert Shot(start_frame=4, end_frame=48).length == 44


# --------------------------------------------------------------------------- #
# detect_shots                                                                #
# --------------------------------------------------------------------------- #


def test_detect_shots_falls_back_to_single_shot_when_binary_missing(monkeypatch, tmp_path):
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)
    monkeypatch.setattr("vmaftune.per_shot._which", lambda _b: None)

    shots = detect_shots(src, width=64, height=64, total_frames=120)
    assert shots == [Shot(start_frame=0, end_frame=120)]


def test_detect_shots_fallback_no_total_frames(monkeypatch, tmp_path):
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)
    monkeypatch.setattr("vmaftune.per_shot._which", lambda _b: None)

    shots = detect_shots(src, width=64, height=64)
    assert shots == [Shot(start_frame=0, end_frame=1)]


def test_detect_shots_parses_per_shot_json(tmp_path):
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)
    payload = json.dumps(
        {
            "shots": [
                {"start_frame": 0, "end_frame": 23},
                {"start_frame": 24, "end_frame": 71},
                {"start_frame": 72, "end_frame": 119},
            ]
        }
    )

    def fake_run(cmd, capture_output, text, check):
        # Sanity-check the CLI shape we built.
        assert cmd[0] == "vmaf-perShot"
        assert "--reference" in cmd
        assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "json"
        # The new protocol writes JSON to the --output tmpfile, not stdout.
        # Find the path passed as --output and write the fixture JSON there.
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(payload, encoding="utf-8")
        progress = f"vmaf-perShot: wrote 3 shot(s) to {out_path}\n"
        return _FakeCompleted(returncode=0, stdout=progress)

    shots = detect_shots(
        src,
        width=128,
        height=128,
        per_shot_bin="vmaf-perShot",
        runner=fake_run,
    )
    # Half-open conversion: end_frame in source schema is inclusive.
    assert shots == [
        Shot(0, 24),
        Shot(24, 72),
        Shot(72, 120),
    ]


def test_detect_shots_falls_back_on_runner_failure(tmp_path):
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    def failing_run(cmd, capture_output, text, check):
        return _FakeCompleted(returncode=1, stdout="", stderr="boom")

    shots = detect_shots(
        src,
        width=64,
        height=64,
        total_frames=240,
        per_shot_bin="vmaf-perShot",
        runner=failing_run,
    )
    assert shots == [Shot(0, 240)]


def test_parse_per_shot_csv_round_trip():
    csv_text = (
        "shot_id,start_frame,end_frame,frames,mean_complexity,mean_motion,predicted_crf\n"
        "0,0,3,4,0.000051,0.020046,25.48\n"
        "1,4,47,44,0.019353,0.016716,24.62\n"
    )
    shots = parse_per_shot_csv(csv_text)
    assert shots == [Shot(0, 4), Shot(4, 48)]


# --------------------------------------------------------------------------- #
# tune_per_shot                                                               #
# --------------------------------------------------------------------------- #


def test_tune_per_shot_three_shots_yields_three_distinct_crfs():
    """Mocked complexity-aware predicate — busier shots get lower CRF."""
    shots = [Shot(0, 24), Shot(24, 72), Shot(72, 144)]
    complexity = {
        (0, 24): (0.05, 92.5),  # quiet — relax
        (24, 72): (0.30, 92.0),  # mid
        (72, 144): (0.85, 91.7),  # busy
    }

    def predicate(shot, target_vmaf, encoder):
        c, predicted = complexity[(shot.start_frame, shot.end_frame)]
        # Linear blend: CRF rises as complexity falls.
        crf = round(20 + (1.0 - c) * 12)
        # Sanity: caller passed the target through.
        assert target_vmaf == 92.0
        assert encoder == "libx264"
        return (crf, predicted)

    recs = tune_per_shot(shots, target_vmaf=92.0, encoder="libx264", predicate=predicate)
    assert len(recs) == 3
    crfs = [r.crf for r in recs]
    assert len(set(crfs)) == 3, f"expected three distinct CRFs, got {crfs}"
    # Quiet shot -> highest CRF; busy shot -> lowest.
    assert crfs[0] > crfs[1] > crfs[2]


def test_tune_per_shot_clamps_to_codec_quality_range():
    """Predicate values outside the adapter's `quality_range` clamp."""
    from vmaftune.codec_adapters import get_adapter

    lo, hi = get_adapter("libx264").quality_range

    def predicate(shot, target_vmaf, encoder):
        return (lo - 10, 95.0)  # below libx264's lower clamp

    recs = tune_per_shot([Shot(0, 24)], target_vmaf=92.0, predicate=predicate)
    assert recs[0].crf == lo

    def predicate_high(shot, target_vmaf, encoder):
        return (hi + 50, 50.0)  # above upper clamp

    recs = tune_per_shot([Shot(0, 24)], target_vmaf=92.0, predicate=predicate_high)
    assert recs[0].crf == hi


def test_tune_per_shot_default_predicate_returns_codec_default():
    recs = tune_per_shot([Shot(0, 24)], target_vmaf=92.0)
    assert recs[0].crf == 23  # libx264 default


def test_tune_per_shot_rejects_empty_input():
    with pytest.raises(ValueError):
        tune_per_shot([], target_vmaf=92.0)


# --------------------------------------------------------------------------- #
# merge_shots                                                                 #
# --------------------------------------------------------------------------- #


def test_merge_shots_emits_well_formed_ffmpeg_plan(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    recs = (
        ShotRecommendation(Shot(0, 24), crf=22, predicted_vmaf=93.0),
        ShotRecommendation(Shot(24, 72), crf=26, predicted_vmaf=92.5),
        ShotRecommendation(Shot(72, 144), crf=30, predicted_vmaf=91.8),
    )
    plan = merge_shots(
        recs,
        source=src,
        output=out,
        framerate=24.0,
        encoder="libx264",
    )

    assert isinstance(plan, EncodingPlan)
    assert len(plan.segment_commands) == 3
    # Each segment command has the libx264 + crf wiring at the right index.
    for cmd, rec in zip(plan.segment_commands, recs):
        assert cmd[0] == "ffmpeg"
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
        assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == str(rec.crf)
        # -ss seek + -frames:v are present.
        assert "-ss" in cmd
        assert "-frames:v" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == str(rec.shot.length)

    # Concat command shape.
    assert plan.concat_command[0] == "ffmpeg"
    assert "-f" in plan.concat_command
    assert plan.concat_command[plan.concat_command.index("-f") + 1] == "concat"
    assert plan.concat_command[-1] == str(out)

    # Concat listing has one line per shot.
    listing_lines = [ln for ln in plan.concat_listing.splitlines() if ln.strip()]
    assert len(listing_lines) == 3
    assert all(ln.startswith("file '") for ln in listing_lines)


def test_merge_shots_rejects_empty_recommendations(tmp_path):
    with pytest.raises(ValueError):
        merge_shots(
            (),
            source=tmp_path / "x",
            output=tmp_path / "y",
            framerate=24.0,
        )


def test_write_concat_listing_persists(tmp_path):
    plan = merge_shots(
        (ShotRecommendation(Shot(0, 24), crf=23, predicted_vmaf=92.0),),
        source=tmp_path / "in.mp4",
        output=tmp_path / "out.mp4",
        framerate=24.0,
    )
    listing = tmp_path / "concat.txt"
    written = write_concat_listing(plan, listing)
    assert written == listing
    assert listing.read_text().startswith("file '")


def test_plan_to_shell_script_round_trip(tmp_path):
    plan = merge_shots(
        (ShotRecommendation(Shot(0, 24), crf=23, predicted_vmaf=92.0),),
        source=tmp_path / "in.mp4",
        output=tmp_path / "out.mp4",
        framerate=24.0,
    )
    script = plan_to_shell_script(plan)
    assert script.startswith("#!/bin/sh")
    assert "ffmpeg" in script


# --------------------------------------------------------------------------- #
# CLI smoke                                                                   #
# --------------------------------------------------------------------------- #


def test_cli_tune_per_shot_binds_bisect_predicate(tmp_path, monkeypatch):
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)
    plan_out = tmp_path / "plan.json"
    out = tmp_path / "out.mp4"

    payload = json.dumps(
        {
            "shots": [
                {"start_frame": 0, "end_frame": 23},
                {"start_frame": 24, "end_frame": 71},
            ]
        }
    )

    # Pretend the binary is available + intercept the subprocess call.
    monkeypatch.setattr("vmaftune.per_shot._which", lambda _b: "/fake/vmaf-perShot")

    extracted: list[Path] = []

    def fake_run(cmd, capture_output, text, check):
        if cmd[0] == "vmaf-perShot":
            # New protocol: write JSON to the --output tmpfile, not stdout.
            out_path = Path(cmd[cmd.index("--output") + 1])
            out_path.write_text(payload, encoding="utf-8")
            progress = f"vmaf-perShot: wrote 2 shot(s) to {out_path}\n"
            return _FakeCompleted(returncode=0, stdout=progress)
        assert cmd[0] == "ffmpeg"
        assert "-f" in cmd and "rawvideo" in cmd
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 16)
        extracted.append(out_path)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr("vmaftune.per_shot.subprocess.run", fake_run)

    calls: list[tuple[Path, str, float]] = []

    def fake_bisect(src, codec, target_vmaf, **kwargs):
        calls.append((Path(src), codec, target_vmaf))
        crf = 21 + len(calls)
        return SimpleNamespace(
            ok=True,
            best_crf=crf,
            measured_vmaf=target_vmaf + len(calls) / 10.0,
            error="",
        )

    monkeypatch.setattr("vmaftune.cli.bisect_target_vmaf", fake_bisect)

    rc = cli.main(
        [
            "tune-per-shot",
            "--src",
            str(src),
            "--width",
            "1920",
            "--height",
            "1080",
            "--framerate",
            "24",
            "--target-vmaf",
            "92",
            "--encoder",
            "libx264",
            "--crf-min",
            "18",
            "--crf-max",
            "30",
            "--max-iterations",
            "4",
            "--output",
            str(out),
            "--plan-out",
            str(plan_out),
        ]
    )
    assert rc == 0
    assert plan_out.exists()
    doc = json.loads(plan_out.read_text())
    assert doc["encoder"] == "libx264"
    assert doc["predicate"] == "bisect"
    assert len(doc["shots"]) == 2
    assert [s["crf"] for s in doc["shots"]] == [22, 23]
    assert [s["predicted_vmaf"] for s in doc["shots"]] == [92.1, 92.2]
    assert len(calls) == 2
    assert calls[0][1:] == ("libx264", 92.0)
    assert len(extracted) == 2
    # Concat listing was written next to the segments.
    listing = (out.parent / "segments" / "concat.txt").read_text()
    assert listing.count("file '") == 2
