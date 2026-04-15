"""Populate a dataset manifest from a local cache.

Each manifest shipped in `manifests/` is empty on purpose — the repo cannot
redistribute Netflix / KoNViD / LIVE-VQC / YouTube-UGC / BVI-DVC content or
MOS scores. Operators run `vmaf-train manifest-scan` once they have the
dataset on disk under `VMAF_DATA_ROOT`; this module walks the tree, pins
every YUV / Y4M file by SHA-256, and optionally joins a MOS CSV.

CSV format (if provided): one header row, columns `key,mos`. Unknown keys
in the CSV are ignored; files without a MOS entry simply get `mos: null`.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .datasets import DATASETS, manifest_path

_VIDEO_SUFFIXES = {".yuv", ".y4m", ".mp4", ".mkv", ".webm"}
_CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _key_from_relpath(rel: Path) -> str:
    return rel.with_suffix("").as_posix().replace("/", "_")


@dataclass(frozen=True)
class ScanEntry:
    key: str
    path: str
    sha256: str
    mos: float | None


def load_mos_csv(csv_path: Path) -> dict[str, float]:
    mos: dict[str, float] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "key" not in reader.fieldnames or "mos" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing required 'key' / 'mos' columns")
        for row in reader:
            key = row["key"].strip()
            if not key:
                continue
            try:
                mos[key] = float(row["mos"])
            except ValueError as exc:
                raise ValueError(f"{csv_path}: bad mos for key={key!r}: {exc}") from exc
    return mos


def scan(dataset: str, root: Path, mos_csv: Path | None = None) -> list[ScanEntry]:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset: {dataset}")
    if not root.is_dir():
        raise NotADirectoryError(root)

    mos = load_mos_csv(mos_csv) if mos_csv is not None else {}

    entries: list[ScanEntry] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _VIDEO_SUFFIXES:
            continue
        rel = p.relative_to(root)
        key = _key_from_relpath(rel)
        entries.append(ScanEntry(
            key=key,
            path=str(rel),
            sha256=_sha256(p),
            mos=mos.get(key),
        ))
    return entries


def write_manifest(dataset: str, entries: list[ScanEntry]) -> Path:
    dst = manifest_path(dataset)
    meta = DATASETS[dataset]
    doc = {
        "name": dataset,
        "license": meta["license"],
        "entries": [
            {"key": e.key, "path": e.path, "sha256": e.sha256, "mos": e.mos}
            for e in entries
        ],
    }
    with dst.open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return dst
