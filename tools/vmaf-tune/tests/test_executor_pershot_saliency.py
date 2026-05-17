# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Smoke tests for ``run_plan_per_shot`` and ``run_plan_saliency`` (ADR-0468).

No real FFmpeg, ``vmaf``, or ``vmaf-perShot`` binary is required — all
subprocess boundaries are stubbed via the seams the drivers expose.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.auto import AutoPlan  # noqa: E402
from vmaftune.executor import (  # noqa: E402
    PerShotPlanResult,
    SaliencyExecuteResult,
    ShotExecuteResult,
    run_plan_per_shot,
    run_plan_saliency,
)

# ---------------------------------------------------------------------------
# Shared stub factories
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
        "source_meta": {"width": 320, "height": 240},
        "short_circuits": [],
        "winner": {"cell_index": selected_index},
    }
    return AutoPlan(cells=cells, metadata=metadata)


def _encode_runner_ok(cmd, *, capture_output=True, text=True, check=False):
    return SimpleNamespace(
        returncode=0,
        stdout="",
        stderr="ffmpeg version 6.1\nframe=   25 fps=0.0 Lsize=    500kB\n",
    )


def _encode_runner_fail(cmd, *, capture_output=True, text=True, check=False):
    return SimpleNamespace(returncode=1, stdout="", stderr="error: codec crash\n")


def _score_runner_ok(cmd, *, capture_output=True, text=True, check=False):
    # run_score reads vmaf.json from a workdir it controls; the stub returning
    # rc=0 causes a NaN score (JSON not found) which is acceptable for seam
    # smoke tests — the row must exist and the call must not raise.
    return SimpleNamespace(returncode=0, stdout="", stderr="VMAF version: 3.0.0\n")


# ---------------------------------------------------------------------------
# Per-shot tests
# ---------------------------------------------------------------------------

# vmaf-perShot JSON payload with two 10-frame shots (end_frame is inclusive).
_PERSHOT_JSON_TWO_SHOTS = json.dumps(
    {"shots": [{"start_frame": 0, "end_frame": 9}, {"start_frame": 10, "end_frame": 19}]}
)


def _shot_runner_ok(cmd, *, capture_output=True, text=True, check=False):
    """Stub vmaf-perShot runner returning two 10-frame shots."""
    return SimpleNamespace(returncode=0, stdout=_PERSHOT_JSON_TWO_SHOTS, stderr="")


def _shot_runner_fail(cmd, *, capture_output=True, text=True, check=False):
    """Stub vmaf-perShot runner that fails (triggers single-shot fallback)."""
    return SimpleNamespace(returncode=1, stdout="", stderr="binary not found\n")


def test_run_plan_per_shot_detects_two_shots(tmp_path: Path) -> None:
    """run_plan_per_shot scores two shots when vmaf-perShot reports them."""
    plan = _make_plan(selected_index=0, n_cells=1)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan_per_shot(
        plan,
        src,
        tmp_path / "runs",
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
        shot_runner=_shot_runner_ok,
    )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, PerShotPlanResult)
    assert len(r.shot_results) == 2, "expected 2 shot results from stub"
    for sr in r.shot_results:
        assert isinstance(sr, ShotExecuteResult)
        assert sr.length_frames == 10

    jsonl = tmp_path / "runs" / "tune_results_per_shot.jsonl"
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["shot_count"] == 2


def test_run_plan_per_shot_fallback_to_single_shot(tmp_path: Path) -> None:
    """When vmaf-perShot fails, run_plan_per_shot falls back to one shot."""
    plan = _make_plan(selected_index=0, n_cells=1)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan_per_shot(
        plan,
        src,
        tmp_path / "runs",
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
        shot_runner=_shot_runner_fail,
    )

    assert len(results) == 1
    r = results[0]
    # Single-shot fallback from detect_shots: Shot(0, 1) sentinel or a single
    # range; either way shot_count is 1.
    assert r.row["shot_count"] == 1


def test_run_plan_per_shot_weighted_vmaf_is_nan_when_all_fail(tmp_path: Path) -> None:
    """weighted_vmaf is NaN when all per-shot encodes fail."""
    plan = _make_plan(selected_index=0, n_cells=1)
    src = tmp_path / "ref.yuv"
    src.write_bytes(b"\x00" * 16)

    results = run_plan_per_shot(
        plan,
        src,
        tmp_path / "runs",
        encode_runner=_encode_runner_fail,
        score_runner=_score_runner_ok,
        shot_runner=_shot_runner_ok,
    )

    assert len(results) == 1
    assert math.isnan(results[0].weighted_vmaf)


# ---------------------------------------------------------------------------
# Saliency tests
# ---------------------------------------------------------------------------


def _stub_session_factory(model_path: Path):
    """Fake ONNX session: returns a zero saliency map of the expected shape."""

    class _FakeSession:
        def run(self, output_names, inputs):
            import numpy as np  # noqa: PLC0415

            # Input shape is [1, 3, H, W]; output is [1, 1, H, W] in [0, 1].
            tensor = inputs["input"]
            h, w = tensor.shape[2], tensor.shape[3]
            return [np.zeros((1, 1, h, w), dtype=np.float32)]

    return _FakeSession()


def test_run_plan_saliency_returns_result_row(tmp_path: Path) -> None:
    """run_plan_saliency produces a row in tune_results_saliency.jsonl."""
    plan = _make_plan(selected_index=0, n_cells=1)
    # Write a minimal 320x240 yuv420p stub (1 frame = 320*240*3//2 bytes).
    src = tmp_path / "ref.yuv"
    frame_size = 320 * 240 * 3 // 2
    src.write_bytes(b"\x80" * frame_size)

    results = run_plan_saliency(
        plan,
        src,
        tmp_path / "runs",
        duration_frames=1,
        encode_runner=_encode_runner_ok,
        score_runner=_score_runner_ok,
        session_factory=_stub_session_factory,
    )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SaliencyExecuteResult)
    # Saliency ran (stub session provided); encode succeeded.
    assert r.encode is not None
    assert r.row["cell_index"] == 0

    jsonl = tmp_path / "runs" / "tune_results_saliency.jsonl"
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["codec"] == "libx264"
