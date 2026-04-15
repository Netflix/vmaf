"""Unit tests for the manifest-scan tooling."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vmaf_train.data import manifest_scan


def _write(p: Path, content: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_scan_finds_yuv_and_pins_sha256(tmp_path: Path) -> None:
    _write(tmp_path / "a.yuv", b"AAA")
    _write(tmp_path / "sub/b.y4m", b"BBB")
    _write(tmp_path / "readme.txt", b"skipme")

    entries = manifest_scan.scan("nflx", tmp_path)
    assert [e.key for e in entries] == ["a", "sub_b"]
    assert entries[0].sha256 == hashlib.sha256(b"AAA").hexdigest()
    assert entries[1].sha256 == hashlib.sha256(b"BBB").hexdigest()
    assert all(e.mos is None for e in entries)


def test_scan_joins_mos_csv(tmp_path: Path) -> None:
    _write(tmp_path / "clip.yuv", b"XY")
    csv = tmp_path / "mos.csv"
    csv.write_text("key,mos\nclip,42.5\n")
    entries = manifest_scan.scan("nflx", tmp_path, csv)
    assert len(entries) == 1
    assert entries[0].mos == pytest.approx(42.5)


def test_scan_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        manifest_scan.scan("does-not-exist", tmp_path)


def test_load_mos_csv_rejects_bad_schema(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("name,score\nx,1\n")
    with pytest.raises(ValueError):
        manifest_scan.load_mos_csv(csv)


def test_write_manifest_roundtrips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    dst = tmp_path / "nflx.yaml"
    monkeypatch.setattr(manifest_scan, "manifest_path", lambda name: dst)

    entries = [
        manifest_scan.ScanEntry(key="a", path="a.yuv", sha256="de" * 32, mos=50.0),
        manifest_scan.ScanEntry(key="b", path="b.yuv", sha256="ad" * 32, mos=None),
    ]
    out = manifest_scan.write_manifest("nflx", entries)
    assert out == dst

    doc = yaml.safe_load(dst.read_text())
    assert doc["name"] == "nflx"
    assert len(doc["entries"]) == 2
    assert doc["entries"][0]["key"] == "a"
    assert doc["entries"][0]["mos"] == 50.0
    assert doc["entries"][1]["mos"] is None
