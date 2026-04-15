"""Smoke tests for the vmaf-mcp server — no network, no GPU required."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vmaf_mcp import server as srv


REPO = Path(__file__).resolve().parents[3]


def test_repo_root_detects_testdata():
    root = srv._repo_root()
    assert (root / "testdata").is_dir()


def test_validate_path_accepts_golden_yuv():
    yuv = REPO / "python/test/resource/yuv/src01_hrc00_576x324.yuv"
    if not yuv.exists():
        pytest.skip("Netflix golden YUV not present")
    assert srv._validate_path(str(yuv)) == yuv.resolve()


def test_validate_path_rejects_outside_roots(tmp_path):
    bad = tmp_path / "evil.yuv"
    bad.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="not under an allowlisted root"):
        srv._validate_path(str(bad))


def test_validate_path_accepts_custom_allow(tmp_path, monkeypatch):
    f = tmp_path / "ok.yuv"
    f.write_bytes(b"\x00" * 16)
    monkeypatch.setenv("VMAF_MCP_ALLOW", str(tmp_path))
    assert srv._validate_path(str(f)) == f.resolve()


def test_list_models_returns_list():
    models = srv._list_models()
    assert isinstance(models, list)
    for m in models:
        assert "name" in m and "path" in m and "format" in m


def test_list_backends_always_includes_cpu():
    backends = srv._list_backends()
    assert backends["cpu"] is True
