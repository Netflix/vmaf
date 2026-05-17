# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Tests for the I/O and geometry optimisations in extract_k150k_features.py.

Covers:
  Win 1 — at-end-only parquet write: JSONL staging accumulates rows and the
           parquet is equivalent to the old per-200-flush strategy.
  Win 2 — ffprobe skip: geometry is resolved from the CHUG JSONL sidecar when
           all required fields are present.

These tests are intentionally vmaf-binary-free and ffprobe-free; they exercise
only the I/O layer (Research-0135) via small synthetic data.
"""

from __future__ import annotations

import math

# Import the module under test.
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "scripts"))

from extract_k150k_features import (
    FEATURE_NAMES,
    _append_row_to_staging,
    _geometry_from_sidecar,
    _load_staging_rows,
    _process_clip,
    _staging_path,
    _write_parquet_from_rows,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(clip_name: str, mos: float = 3.5) -> dict[str, Any]:
    """Build a minimal valid feature row with NaN-filled feature columns."""
    row: dict[str, Any] = {
        "clip_name": clip_name,
        "mos": mos,
        "width": 960,
        "height": 540,
    }
    for feat in FEATURE_NAMES:
        row[f"{feat}_mean"] = float("nan")
        row[f"{feat}_std"] = float("nan")
    return row


# ---------------------------------------------------------------------------
# Win 1: JSONL staging + at-end parquet
# ---------------------------------------------------------------------------


class TestStagingIO:
    """JSONL staging file helpers produce a parquet equivalent to legacy flush."""

    def test_append_and_reload(self, tmp_path: Path) -> None:
        """Rows appended to the staging file are reloaded correctly."""
        staging = tmp_path / "test.rows.jsonl"
        rows = [_make_row(f"clip_{i:04d}.mp4", mos=float(i)) for i in range(10)]
        for r in rows:
            _append_row_to_staging(staging, r)
        loaded = _load_staging_rows(staging)
        assert len(loaded) == 10
        assert loaded[5]["clip_name"] == "clip_0005.mp4"

    def test_staging_path_derivation(self, tmp_path: Path) -> None:
        """_staging_path returns <out>.rows.jsonl."""
        out = tmp_path / "features.parquet"
        assert _staging_path(out) == tmp_path / "features.rows.jsonl"

    def test_load_staging_rows_missing_file(self, tmp_path: Path) -> None:
        """_load_staging_rows returns [] when the file does not exist."""
        assert _load_staging_rows(tmp_path / "nonexistent.rows.jsonl") == []

    def test_load_staging_rows_skips_malformed(self, tmp_path: Path) -> None:
        """Malformed JSONL lines are skipped without raising."""
        staging = tmp_path / "bad.rows.jsonl"
        staging.write_text('{"clip_name": "a.mp4"}\nNOT JSON\n{"clip_name": "b.mp4"}\n')
        loaded = _load_staging_rows(staging)
        assert len(loaded) == 2
        assert {r["clip_name"] for r in loaded} == {"a.mp4", "b.mp4"}

    def test_write_parquet_from_rows_deduplicates(self, tmp_path: Path) -> None:
        """Duplicate clip_name rows are deduplicated, keeping the last occurrence."""
        rows = [_make_row("dup.mp4", mos=1.0), _make_row("dup.mp4", mos=5.0)]
        out = tmp_path / "out.parquet"
        _write_parquet_from_rows(rows, out)
        df = pd.read_parquet(out)
        assert len(df) == 1
        assert float(df["mos"].iloc[0]) == pytest.approx(5.0)

    def test_write_parquet_column_order(self, tmp_path: Path) -> None:
        """Parquet columns follow FEATURE_NAMES order for _mean/_std pairs."""
        rows = [_make_row(f"c{i}.mp4") for i in range(3)]
        out = tmp_path / "order.parquet"
        _write_parquet_from_rows(rows, out)
        df = pd.read_parquet(out)
        feature_cols = [c for c in df.columns if c.endswith("_mean")]
        expected = [f"{feat}_mean" for feat in FEATURE_NAMES]
        assert feature_cols == expected, "Column order must follow FEATURE_NAMES"

    def test_write_parquet_preserves_nan(self, tmp_path: Path) -> None:
        """NaN feature values survive the JSONL round-trip and parquet write."""
        row = _make_row("nan_clip.mp4")
        staging = tmp_path / "nan.rows.jsonl"
        _append_row_to_staging(staging, row)
        loaded = _load_staging_rows(staging)
        out = tmp_path / "nan.parquet"
        _write_parquet_from_rows(loaded, out)
        df = pd.read_parquet(out)
        assert math.isnan(float(df["adm2_mean"].iloc[0]))

    def test_write_parquet_noop_on_empty(self, tmp_path: Path) -> None:
        """_write_parquet_from_rows does not create a file for empty input."""
        out = tmp_path / "empty.parquet"
        _write_parquet_from_rows([], out)
        assert not out.exists()

    def test_parquet_matches_legacy_flush_equivalent(self, tmp_path: Path) -> None:
        """At-end parquet matches an equivalent concat of legacy per-200-clip flushes.

        This is the bit-equivalence assertion for Win 1.  We build 600 rows,
        simulate the old strategy (3 × 200-row concat-write), then compare
        against the new strategy (single _write_parquet_from_rows call).
        """
        n = 600
        all_rows = [_make_row(f"clip_{i:06d}.mp4", mos=float(i % 5) + 1.0) for i in range(n)]

        # --- legacy strategy: three concat-write cycles ---
        legacy_out = tmp_path / "legacy.parquet"
        for batch_start in range(0, n, 200):
            batch = all_rows[batch_start : batch_start + 200]
            new_df = pd.DataFrame(batch)
            if legacy_out.is_file():
                existing = pd.read_parquet(legacy_out)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["clip_name"], keep="last")
            else:
                combined = new_df
            tmp_file = legacy_out.with_suffix(".tmp")
            combined.to_parquet(tmp_file, index=False)
            tmp_file.rename(legacy_out)

        # --- new strategy: single write ---
        new_out = tmp_path / "new.parquet"
        _write_parquet_from_rows(all_rows, new_out)

        legacy_df = pd.read_parquet(legacy_out).sort_values("clip_name").reset_index(drop=True)
        new_df = pd.read_parquet(new_out).sort_values("clip_name").reset_index(drop=True)

        assert list(legacy_df.columns) == list(new_df.columns), "Column order must match"
        assert len(legacy_df) == len(new_df) == n, "Row count must match"
        assert list(legacy_df["clip_name"]) == list(
            new_df["clip_name"]
        ), "clip_name order must match"
        # MOS values (non-NaN) must be numerically identical.
        pd.testing.assert_series_equal(legacy_df["mos"], new_df["mos"], check_names=True)


# ---------------------------------------------------------------------------
# Win 2: ffprobe skip via sidecar geometry
# ---------------------------------------------------------------------------


class TestGeometryFromSidecar:
    """_geometry_from_sidecar resolves geometry from CHUG JSONL fields."""

    def test_full_geometry_resolved(self) -> None:
        meta = {
            "chug_width_manifest": 1920,
            "chug_height_manifest": 1080,
            "chug_framerate_manifest": "30/1",
        }
        result = _geometry_from_sidecar(meta)
        assert result is not None
        width, height, pix_fmt, fps = result
        assert width == 1920
        assert height == 1080
        assert pix_fmt == "yuv420p"
        assert fps == "30/1"

    def test_10bit_pixel_format(self) -> None:
        meta = {
            "chug_width_manifest": 960,
            "chug_height_manifest": 540,
            "chug_framerate_manifest": "25/1",
            "chug_bit_depth": 10,
        }
        result = _geometry_from_sidecar(meta)
        assert result is not None
        _w, _h, pix_fmt, _fps = result
        assert pix_fmt == "yuv420p10le"

    def test_8bit_default_pixel_format(self) -> None:
        meta = {
            "chug_width_manifest": 960,
            "chug_height_manifest": 540,
            "chug_framerate_manifest": "25/1",
            "chug_bit_depth": 8,
        }
        result = _geometry_from_sidecar(meta)
        assert result is not None
        assert result[2] == "yuv420p"

    def test_missing_width_returns_none(self) -> None:
        meta = {
            "chug_height_manifest": 540,
            "chug_framerate_manifest": "25/1",
        }
        assert _geometry_from_sidecar(meta) is None

    def test_missing_height_returns_none(self) -> None:
        meta = {
            "chug_width_manifest": 960,
            "chug_framerate_manifest": "25/1",
        }
        assert _geometry_from_sidecar(meta) is None

    def test_missing_framerate_returns_none(self) -> None:
        meta = {
            "chug_width_manifest": 960,
            "chug_height_manifest": 540,
        }
        assert _geometry_from_sidecar(meta) is None

    def test_none_sidecar_returns_none(self) -> None:
        assert _geometry_from_sidecar(None) is None  # type: ignore[arg-type]

    def test_empty_sidecar_returns_none(self) -> None:
        assert _geometry_from_sidecar({}) is None


class TestProcessClipFfprobeSkip:
    """_process_clip skips ffprobe when a valid sidecar geometry is provided."""

    def test_ffprobe_skipped_when_sidecar_has_geometry(self, tmp_path: Path) -> None:
        """ffprobe must NOT be called when _geometry_from_sidecar returns a result."""
        sidecar = {
            "chug_width_manifest": 960,
            "chug_height_manifest": 540,
            "chug_framerate_manifest": "25/1",
        }
        dummy_frames = [{"vmaf": 80.0} for _ in range(5)]

        with (
            patch(
                "extract_k150k_features._probe_geometry",
                side_effect=AssertionError("ffprobe must not be called"),
            ),
            patch(
                "extract_k150k_features._decode_to_yuv",
                return_value=None,
            ),
            patch(
                "extract_k150k_features._run_feature_passes",
                return_value=dummy_frames,
            ),
            patch(
                "extract_k150k_features._aggregate_frames",
                return_value={f"{feat}_mean": float("nan") for feat in FEATURE_NAMES}
                | {f"{feat}_std": float("nan") for feat in FEATURE_NAMES},
            ),
        ):
            mp4 = tmp_path / "clip.mp4"
            mp4.write_bytes(b"")  # Dummy — never read.
            scratch = tmp_path / "scratch"
            scratch.mkdir()

            row = _process_clip(
                str(mp4),
                3.0,
                "/dev/null",
                "/dev/null",
                str(scratch),
                2,
                False,
                0,
                sidecar,
            )

        assert row["width"] == 960
        assert row["height"] == 540

    def test_ffprobe_called_when_sidecar_lacks_geometry(self, tmp_path: Path) -> None:
        """ffprobe is called when sidecar is missing geometry fields."""
        sidecar: dict = {}  # No geometry fields.
        probe_called = []
        dummy_frames = [{"vmaf": 75.0} for _ in range(5)]

        def mock_probe(mp4: Path) -> tuple:
            probe_called.append(mp4)
            return 1280, 720, "yuv420p", "30/1"

        with (
            patch("extract_k150k_features._probe_geometry", side_effect=mock_probe),
            patch("extract_k150k_features._decode_to_yuv", return_value=None),
            patch("extract_k150k_features._run_feature_passes", return_value=dummy_frames),
            patch(
                "extract_k150k_features._aggregate_frames",
                return_value={f"{feat}_mean": float("nan") for feat in FEATURE_NAMES}
                | {f"{feat}_std": float("nan") for feat in FEATURE_NAMES},
            ),
        ):
            mp4 = tmp_path / "clip2.mp4"
            mp4.write_bytes(b"")
            scratch = tmp_path / "scratch2"
            scratch.mkdir()

            row = _process_clip(
                str(mp4),
                2.5,
                "/dev/null",
                "/dev/null",
                str(scratch),
                2,
                False,
                0,
                sidecar,
            )

        assert len(probe_called) == 1, "ffprobe must be called when sidecar has no geometry"
        assert row["width"] == 1280

    def test_ffprobe_called_when_no_sidecar(self, tmp_path: Path) -> None:
        """ffprobe is called when sidecar_meta is None."""
        probe_called = []
        dummy_frames = [{"vmaf": 70.0} for _ in range(5)]

        def mock_probe(mp4: Path) -> tuple:
            probe_called.append(mp4)
            return 854, 480, "yuv420p", "24/1"

        with (
            patch("extract_k150k_features._probe_geometry", side_effect=mock_probe),
            patch("extract_k150k_features._decode_to_yuv", return_value=None),
            patch("extract_k150k_features._run_feature_passes", return_value=dummy_frames),
            patch(
                "extract_k150k_features._aggregate_frames",
                return_value={f"{feat}_mean": float("nan") for feat in FEATURE_NAMES}
                | {f"{feat}_std": float("nan") for feat in FEATURE_NAMES},
            ),
        ):
            mp4 = tmp_path / "clip3.mp4"
            mp4.write_bytes(b"")
            scratch = tmp_path / "scratch3"
            scratch.mkdir()

            row = _process_clip(
                str(mp4),
                4.0,
                "/dev/null",
                "/dev/null",
                str(scratch),
                2,
                False,
                0,
                None,
            )

        assert len(probe_called) == 1
        assert row["height"] == 480
