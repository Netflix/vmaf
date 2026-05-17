# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Smoke tests for ``vmaftune.executor.run_plan`` (ADR-0454).

No real FFmpeg or ``vmaf`` binary is required — both the encode runner and
the score runner are stubbed via the seams the underlying drivers expose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.auto import AutoPlan  # noqa: E402
from vmaftune.executor import ExecuteResult, run_plan  # noqa: E402

# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def _make_plan(*, selected_index: int = 0, n_cells: int = 1) -> AutoPlan:
    """Return a minimal AutoPlan with ``n_cells`` cells."""
    cells = []
    for i in range(n_cells):
        cells.append(
            {
                "cell_index": i,
                "codec": "libx264",
                "preset": "medium",
                "crf": 23,
                "selected": i == selected_index,
                "estimated_vmaf": 93.0,
                "estimated_bitrate_kbps": 3000.0,
                "prediction_source": "smoke-placeholder",
            }
        )
    metadata = {
        "src": "/tmp/ref.yuv",
        "target_vmaf": 93.0,
        "max_budget_kbps": 8000.0,
        "allow_codecs": ["libx264"],
        "user_pinned_codec": None,
        "smoke": True,
        "source_meta": {"width": 1280, "height": 720},
        "short_circuits": [],
        "winner": {"cell_index": selected_index},
    }
    return AutoPlan(cells=cells, metadata=metadata)


def _encode_runner_ok(cmd, *, capture_output=True, text=True, check=False):
    """Stub encode runner that reports success and a 1 000-byte output."""
    # The encode driver reads returncode + stderr from the CompletedProcess.
    return SimpleNamespace(
        returncode=0,
        stdout="",
        stderr=(
            "ffmpeg version 6.1 Copyright ...\n"
            "  built with gcc 12\n"
            "  libx264 core 164\n"
            "frame=   25 fps=0.0 q=-1.0 Lsize=    1000kB time=00:00:01.00\n"
        ),
    )


def _encode_runner_fail(cmd, *, capture_output=True, text=True, check=False):
    """Stub encode runner that reports failure."""
    return SimpleNamespace(returncode=1, stdout="", stderr="error: codec crash\n")


def _score_runner_ok(cmd, *, capture_output=True, text=True, check=False):
    """Stub score runner that writes a minimal vmaf.json and returns success.

    The real :func:`vmaftune.score.run_score` reads ``vmaf.json`` from the
    workdir it controls; we cannot write it here.  Instead, the stub returns a
    returncode of 0 and empty stderr — ``run_score`` will then fail to find the
    JSON and record exit_status=65.  That is acceptable for a seam-level smoke
    test: what we verify is that ``run_plan`` calls the score driver and records
    a row, not that a real VMAF score was produced.
    """
    return SimpleNamespace(returncode=0, stdout="", stderr="VMAF version: 3.0.0\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_plan_single_selected_cell_encode_ok(tmp_path: Path) -> None:
    """run_plan executes the selected cell and writes a row to tune_results.jsonl."""
    plan = _make_plan(selected_index=0, n_cells=1)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan(
        plan,
        src,
        tmp_path / "runs",
        source_is_container=False,
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
    )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ExecuteResult)
    assert r.encode is not None
    assert r.encode.exit_status == 0
    # Score may be NaN (no real vmaf.json) but the row must exist.
    assert r.row["codec"] == "libx264"
    assert r.row["selected"] is True

    jsonl = tmp_path / "runs" / "tune_results.jsonl"
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["crf"] == 23


def test_run_plan_skips_non_selected_cells(tmp_path: Path) -> None:
    """With execute_all=False, only the selected cell runs."""
    plan = _make_plan(selected_index=1, n_cells=3)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan(
        plan,
        src,
        tmp_path / "runs",
        source_is_container=False,
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
    )

    assert len(results) == 1
    assert results[0].cell["cell_index"] == 1


def test_run_plan_execute_all(tmp_path: Path) -> None:
    """execute_all=True runs every cell regardless of selected flag."""
    plan = _make_plan(selected_index=0, n_cells=3)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan(
        plan,
        src,
        tmp_path / "runs",
        source_is_container=False,
        execute_all=True,
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
    )

    assert len(results) == 3
    jsonl = tmp_path / "runs" / "tune_results.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3


def test_run_plan_encode_failure_skips_score(tmp_path: Path) -> None:
    """When encode exits non-zero, scoring is skipped; row still written."""
    plan = _make_plan(selected_index=0, n_cells=1)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan(
        plan,
        src,
        tmp_path / "runs",
        source_is_container=False,
        encode_runner=_encode_runner_fail,
        score_runner=_score_runner_ok,
    )

    assert len(results) == 1
    r = results[0]
    assert r.encode is not None
    assert r.encode.exit_status != 0
    assert r.score is None
    # Row must still land in the JSONL file.
    jsonl = tmp_path / "runs" / "tune_results.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["score_exit_status"] is None


def test_run_plan_geometry_from_metadata(tmp_path: Path) -> None:
    """run_plan reads width/height from plan.metadata.source_meta when defaults."""
    plan = _make_plan(selected_index=0)
    # The plan metadata has 1280x720 (set in _make_plan).
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan(
        plan,
        src,
        tmp_path / "runs",
        source_is_container=False,
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
    )

    assert len(results) == 1
    # Verify the encode request used the plan's geometry.
    assert results[0].encode is not None
    assert results[0].encode.request.width == 1280
    assert results[0].encode.request.height == 720
