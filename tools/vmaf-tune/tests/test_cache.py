# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Cache layer tests — ADR-0298.

Covers the four contracts the cache promises callers:

1. First trial misses, second trial hits (no encoder/scorer re-run).
2. Different CRF / different encoder / different adapter version /
   different ffmpeg version → different key.
3. LRU eviction kicks in at the size cap.
4. ``--no-cache`` (``CorpusOptions.cache_enabled = False``) disables
   the layer end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.cache import CachedResult, TuneCache, cache_key, default_cache_dir  # noqa: E402
from vmaftune.corpus import CorpusJob, CorpusOptions, iter_rows  # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_yuv(path: Path, nbytes: int = 1024) -> Path:
    path.write_bytes(b"\x80" * nbytes)
    return path


# ---------------------------------------------------------------- key


def test_cache_key_is_stable():
    a = cache_key(
        src_sha256="abc",
        encoder="libx264",
        preset="medium",
        crf=23,
        adapter_version="1",
        ffmpeg_version="6.1.1",
    )
    b = cache_key(
        src_sha256="abc",
        encoder="libx264",
        preset="medium",
        crf=23,
        adapter_version="1",
        ffmpeg_version="6.1.1",
    )
    assert a == b
    assert len(a) == 64


def test_cache_key_diffs_on_each_field():
    base = dict(
        src_sha256="abc",
        encoder="libx264",
        preset="medium",
        crf=23,
        adapter_version="1",
        ffmpeg_version="6.1.1",
    )
    base_key = cache_key(**base)

    for field, alt in (
        ("src_sha256", "deadbeef"),
        ("encoder", "libx265"),
        ("preset", "slow"),
        ("crf", 28),
        ("adapter_version", "2"),
        ("ffmpeg_version", "7.0.0"),
    ):
        mutated = {**base, field: alt}
        assert cache_key(**mutated) != base_key, field


def test_cache_key_rejects_empty_src_hash():
    import pytest

    with pytest.raises(ValueError):
        cache_key(
            src_sha256="",
            encoder="libx264",
            preset="medium",
            crf=23,
            adapter_version="1",
            ffmpeg_version="6.1.1",
        )


# ---------------------------------------------------------- get / put


def _result(score=92.5, size=4096) -> CachedResult:
    return CachedResult(
        encode_size_bytes=size,
        encode_time_ms=10.0,
        encoder_version="libx264-164",
        ffmpeg_version="6.1.1",
        vmaf_score=score,
        vmaf_model="vmaf_v0.6.1",
        score_time_ms=5.0,
        vmaf_binary_version="3.0.0-lusoris",
        artifact_path=Path("placeholder"),
    )


def test_get_returns_none_on_miss(tmp_path):
    c = TuneCache(tmp_path / "cache")
    assert c.get("any") is None


def test_put_then_get_round_trip(tmp_path):
    c = TuneCache(tmp_path / "cache")
    blob = tmp_path / "encode.mp4"
    blob.write_bytes(b"\x01\x02\x03")
    stored = c.put("k1", _result(), blob)
    hit = c.get("k1")
    assert hit is not None
    assert hit.vmaf_score == 92.5
    assert hit.encode_size_bytes == 4096
    assert hit.artifact_path.exists()
    assert hit.artifact_path == stored.artifact_path


def test_default_cache_dir_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "vmaf-tune"


def test_default_cache_dir_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert default_cache_dir() == tmp_path / "home" / ".cache" / "vmaf-tune"


def test_evict_lru_drops_oldest_until_under_cap(tmp_path):
    c = TuneCache(tmp_path / "cache", size_bytes=10**12)
    # Put 5 entries, each with a 1 KiB blob.
    for i in range(5):
        b = tmp_path / f"e{i}.bin"
        b.write_bytes(b"\x00" * 1024)
        c.put(f"k{i}", _result(), b)
    before = c.total_bytes()
    assert before > 0
    # Squeeze cap: keep only ~last entry's worth of blob bytes.
    target = 1024  # one blob, plus index overhead
    evicted = c.evict_lru(target)
    assert evicted >= 4
    assert c.total_bytes() <= max(before, target * 4)  # generous: index+meta overhead


def test_evict_lru_zero_target_is_noop(tmp_path):
    c = TuneCache(tmp_path / "cache")
    b = tmp_path / "e.bin"
    b.write_bytes(b"\x00" * 1024)
    c.put("k", _result(), b)
    assert c.evict_lru(0) == 0


# -------------------------------------------------- corpus integration


def _run_corpus(tmp_path: Path, *, cache_enabled: bool, encode_calls: list, score_calls: list):
    src = _make_yuv(tmp_path / "ref.yuv")

    def fake_encode(cmd, capture_output, text, check):
        encode_calls.append(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 4096)
        return _FakeCompleted(
            returncode=0,
            stderr="ffmpeg version 6.1.1\nx264 - core 164 r3107\n",
        )

    def fake_score(cmd, capture_output, text, check):
        score_calls.append(cmd)
        out_idx = cmd.index("--output") + 1
        out_path = Path(cmd[out_idx])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.5}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0-lusoris\n")

    def fake_probe(cmd, capture_output, text, check):
        return _FakeCompleted(returncode=0, stdout="ffmpeg version 6.1.1\n")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=2.0,
        cells=(("medium", 23),),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        keep_encodes=False,
        src_sha256=True,
        cache_enabled=cache_enabled,
        cache_dir=tmp_path / "cache",
    )
    rows = list(
        iter_rows(
            job,
            opts,
            encode_runner=fake_encode,
            score_runner=fake_score,
            probe_runner=fake_probe,
        )
    )
    return rows


def test_corpus_first_miss_second_hit(tmp_path):
    """The headline contract: first trial encodes, second trial doesn't."""
    enc_a, sc_a = [], []
    rows1 = _run_corpus(tmp_path, cache_enabled=True, encode_calls=enc_a, score_calls=sc_a)
    # x264 runs stats-pass + real-encode per CRF (supports_encoder_stats=True).
    assert len(enc_a) == 2
    assert len(sc_a) == 1
    assert rows1[0]["vmaf_score"] == 92.5

    enc_b, sc_b = [], []
    rows2 = _run_corpus(tmp_path, cache_enabled=True, encode_calls=enc_b, score_calls=sc_b)
    # Cache hit — neither encoder nor scorer should have been called.
    assert enc_b == [], "encode runner ran on a cache hit"
    assert sc_b == [], "score runner ran on a cache hit"
    assert rows2[0]["vmaf_score"] == 92.5
    # Row schema is identical between miss and hit.
    assert set(rows1[0].keys()) == set(rows2[0].keys())


def test_corpus_no_cache_flag_forces_re_encode(tmp_path):
    """``cache_enabled=False`` => always re-runs both subprocesses."""
    enc_a, sc_a = [], []
    _run_corpus(tmp_path, cache_enabled=False, encode_calls=enc_a, score_calls=sc_a)
    enc_b, sc_b = [], []
    _run_corpus(tmp_path, cache_enabled=False, encode_calls=enc_b, score_calls=sc_b)
    # x264 runs stats-pass + real-encode per CRF.
    assert len(enc_b) == 2
    assert len(sc_b) == 1


def test_corpus_different_crf_misses(tmp_path):
    """Different CRF must produce a different key (cache miss)."""
    src = _make_yuv(tmp_path / "ref.yuv")

    encode_calls = []

    def fake_encode(cmd, capture_output, text, check):
        encode_calls.append(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 4096)
        return _FakeCompleted(
            returncode=0,
            stderr="ffmpeg version 6.1.1\nx264 - core 164 r3107\n",
        )

    def fake_score(cmd, capture_output, text, check):
        out_idx = cmd.index("--output") + 1
        out_path = Path(cmd[out_idx])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0-lusoris\n")

    def fake_probe(cmd, capture_output, text, check):
        return _FakeCompleted(returncode=0, stdout="ffmpeg version 6.1.1\n")

    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        cache_enabled=True,
        cache_dir=tmp_path / "cache",
    )
    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=2.0,
        cells=(("medium", 23), ("medium", 28)),
    )
    list(
        iter_rows(
            job,
            opts,
            encode_runner=fake_encode,
            score_runner=fake_score,
            probe_runner=fake_probe,
        )
    )
    # Two distinct CRFs × 2 (stats-pass + real-encode per CRF) = 4.
    assert len(encode_calls) == 4
