# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Content-addressed cache for vmaf-tune trial results.

Iterative tuning workflows re-run the same `(src, encoder, preset, crf)`
tuples constantly — adjusting one flag and re-launching the corpus
sweep is the dominant interaction. Re-encoding and re-scoring the
unchanged tuples burns minutes-to-hours of wall clock for no new
information. This module turns those cells into a free hit.

Cache key components (all six required to invalidate correctly):

1. ``src_sha256``         — content hash of the reference YUV
2. ``encoder``            — adapter name (``libx264``, …)
3. ``preset``             — encoder preset string
4. ``crf``                — quality knob value (int)
5. ``adapter_version``    — bumps when the adapter's argv shape changes
6. ``ffmpeg_version``     — host ffmpeg version string

The key is the SHA-256 of the canonical-JSON-encoded tuple of those
six fields. Dropping any one of them produces wrong cached scores
when, e.g., the adapter is upgraded or ffmpeg is rebuilt — that's a
bug, not an optimisation, so the key signature is enforced by tests.

Cache layout on disk::

    <cache-dir>/
      meta/<key>.json     — small JSON sidecar with the parsed result tuple
      blobs/<key>.bin     — opaque encoded artifact (atomic put: tmp + rename)

A single ``__index__.json`` carries last-access timestamps for LRU
eviction; it's rewritten whole on `put` / `evict_lru` (cheap because
entries are small and writes are infrequent compared to encodes).

Per ADR-0298, **the cache content is not baked into the JSONL row** —
the row stays the canonical record. The cache is an opaque
encode/score result store keyed by inputs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

# Bumps any time the cache key composition or the on-disk layout
# changes in a way that should invalidate older entries.
CACHE_VERSION = 1

# Default ceiling — generous for typical workflows (~50-100 cells of
# 720p/1080p HEVC sit comfortably under 10 GB), small enough to not
# silently consume a workstation's home dir.
DEFAULT_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB


@dataclasses.dataclass(frozen=True)
class CachedResult:
    """The parsed tuple a hit returns alongside the artifact path.

    Mirrors the subset of ``EncodeResult`` + ``ScoreResult`` fields the
    corpus row needs. We store the parsed values rather than the raw
    stderr blobs — those can balloon and are not load-bearing.
    """

    encode_size_bytes: int
    encode_time_ms: float
    encoder_version: str
    ffmpeg_version: str
    vmaf_score: float
    vmaf_model: str
    score_time_ms: float
    vmaf_binary_version: str
    artifact_path: Path


def _xdg_cache_home() -> Path:
    """Resolve XDG cache root with the documented fallbacks."""
    env = os.environ.get("XDG_CACHE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache"


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/vmaf-tune`` (or ``~/.cache/vmaf-tune``)."""
    return _xdg_cache_home() / "vmaf-tune"


def cache_key(
    *,
    src_sha256: str,
    encoder: str,
    preset: str,
    crf: int,
    adapter_version: str,
    ffmpeg_version: str,
) -> str:
    """Return the SHA-256 hex digest that identifies a trial.

    All six fields are mandatory — see module docstring. Inputs are
    serialised through canonical JSON (``sort_keys=True``,
    ``separators=(",", ":")``) so the digest is stable across Python
    versions and dict orderings.
    """
    if not src_sha256:
        raise ValueError("cache_key requires non-empty src_sha256")
    payload = {
        "v": CACHE_VERSION,
        "src_sha256": src_sha256,
        "encoder": encoder,
        "preset": preset,
        "crf": int(crf),
        "adapter_version": adapter_version,
        "ffmpeg_version": ffmpeg_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class TuneCache:
    """Content-addressed store for vmaf-tune trial results.

    Thread-safety: not thread-safe. The corpus loop is single-threaded
    by design; concurrent runs against the same cache dir from
    separate processes work for reads but races on `put` may overwrite
    each other (last writer wins, both rows are valid by content
    addressing).
    """

    INDEX_NAME = "__index__.json"
    META_DIR = "meta"
    BLOB_DIR = "blobs"

    def __init__(
        self,
        path: Path | None = None,
        *,
        size_bytes: int = DEFAULT_SIZE_BYTES,
    ) -> None:
        self.path = Path(path) if path is not None else default_cache_dir()
        self.size_bytes = int(size_bytes)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / self.META_DIR).mkdir(exist_ok=True)
        (self.path / self.BLOB_DIR).mkdir(exist_ok=True)

    # -- index helpers -----------------------------------------------

    def _index_path(self) -> Path:
        return self.path / self.INDEX_NAME

    def _read_index(self) -> dict[str, float]:
        idx = self._index_path()
        if not idx.exists():
            return {}
        try:
            with idx.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                # values are floats (epoch seconds); coerce defensively
                return {str(k): float(v) for k, v in data.items()}
        except (OSError, ValueError, json.JSONDecodeError):
            # Corrupt index — rebuild from filesystem on next put.
            return {}
        return {}

    def _write_index(self, index: dict[str, float]) -> None:
        idx = self._index_path()
        tmp = idx.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, sort_keys=True)
        os.replace(tmp, idx)

    def _meta_path(self, key: str) -> Path:
        return self.path / self.META_DIR / f"{key}.json"

    def _blob_path(self, key: str) -> Path:
        return self.path / self.BLOB_DIR / f"{key}.bin"

    # -- public API --------------------------------------------------

    def get(self, key: str) -> CachedResult | None:
        """Return the cached result for ``key``, or ``None`` on miss."""
        meta = self._meta_path(key)
        blob = self._blob_path(key)
        if not (meta.exists() and blob.exists()):
            return None
        try:
            with meta.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

        # Refresh LRU access time on hit.
        index = self._read_index()
        index[key] = time.time()
        with contextlib.suppress(OSError):
            self._write_index(index)

        return CachedResult(
            encode_size_bytes=int(payload.get("encode_size_bytes", 0)),
            encode_time_ms=float(payload.get("encode_time_ms", 0.0)),
            encoder_version=str(payload.get("encoder_version", "")),
            ffmpeg_version=str(payload.get("ffmpeg_version", "")),
            vmaf_score=float(payload.get("vmaf_score", float("nan"))),
            vmaf_model=str(payload.get("vmaf_model", "")),
            score_time_ms=float(payload.get("score_time_ms", 0.0)),
            vmaf_binary_version=str(payload.get("vmaf_binary_version", "")),
            artifact_path=blob,
        )

    def put(
        self,
        key: str,
        result: CachedResult,
        artifact_path: Path,
    ) -> CachedResult:
        """Store ``artifact_path`` + parsed ``result`` under ``key``.

        The artifact is copied (not moved) so callers retain ownership
        of the original. Returns a refreshed ``CachedResult`` whose
        ``artifact_path`` points at the cached copy.
        """
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact missing: {artifact_path}")

        meta = self._meta_path(key)
        blob = self._blob_path(key)

        # Atomic blob put: copy to tmp in the same dir, then rename.
        tmp_blob = blob.with_suffix(".bin.tmp")
        shutil.copyfile(artifact_path, tmp_blob)
        os.replace(tmp_blob, blob)

        payload = {
            "encode_size_bytes": int(result.encode_size_bytes),
            "encode_time_ms": float(result.encode_time_ms),
            "encoder_version": result.encoder_version,
            "ffmpeg_version": result.ffmpeg_version,
            "vmaf_score": float(result.vmaf_score),
            "vmaf_model": result.vmaf_model,
            "score_time_ms": float(result.score_time_ms),
            "vmaf_binary_version": result.vmaf_binary_version,
        }
        tmp_meta = meta.with_suffix(".json.tmp")
        with tmp_meta.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        os.replace(tmp_meta, meta)

        index = self._read_index()
        index[key] = time.time()
        self._write_index(index)

        if self.size_bytes > 0:
            self.evict_lru(self.size_bytes)

        return dataclasses.replace(result, artifact_path=blob)

    def total_bytes(self) -> int:
        """Sum of meta + blob bytes across all entries."""
        total = 0
        for sub in (self.META_DIR, self.BLOB_DIR):
            d = self.path / sub
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file():
                    with contextlib.suppress(OSError):
                        total += f.stat().st_size
        return total

    def keys(self) -> Iterable[str]:
        meta_dir = self.path / self.META_DIR
        if not meta_dir.exists():
            return []
        return sorted(p.stem for p in meta_dir.iterdir() if p.suffix == ".json")

    def evict_lru(self, target_bytes: int) -> int:
        """Drop oldest entries until total size ≤ ``target_bytes``.

        Returns the number of entries evicted. Entries with no index
        timestamp are treated as oldest (tie-broken by key).
        """
        if target_bytes <= 0:
            # Caller asked for unbounded → no-op.
            return 0
        index = self._read_index()

        evicted = 0
        # Build sorted (timestamp, key) — oldest first.
        keys_sorted = sorted(self.keys(), key=lambda k: (index.get(k, 0.0), k))

        while self.total_bytes() > target_bytes and keys_sorted:
            victim = keys_sorted.pop(0)
            self._drop(victim)
            index.pop(victim, None)
            evicted += 1

        if evicted:
            with contextlib.suppress(OSError):
                self._write_index(index)
        return evicted

    def _drop(self, key: str) -> None:
        for p in (self._meta_path(key), self._blob_path(key)):
            with contextlib.suppress(OSError):
                p.unlink()

    # -- ergonomic helpers used by `corpus.py` -----------------------

    @staticmethod
    def make_artifact_blob(payload: bytes, dst: Path) -> Path:
        """Helper used by tests to fabricate an opaque artifact byte
        blob in a tmp dir so ``put`` has something to copy in.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=dst.parent, delete=False) as fh:
            fh.write(payload)
            tmp = Path(fh.name)
        os.replace(tmp, dst)
        return dst


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_SIZE_BYTES",
    "CachedResult",
    "TuneCache",
    "cache_key",
    "default_cache_dir",
]
