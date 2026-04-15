"""Dataset manifests for NFLX / KoNViD / LIVE-VQC / YouTube-UGC / BVI-DVC.

Manifests (`manifests/<name>.yaml`) declare the authoritative file list with
SHA-256 pins. The repo does not redistribute the data — consumers point
`VMAF_DATA_ROOT` at a pre-downloaded cache and populate their manifest with::

    vmaf-train manifest-scan --dataset <name> --root $VMAF_DATA_ROOT/<name>

See `vmaf_train.data.manifest_scan` for the scanner implementation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DATASETS: dict[str, dict[str, str]] = {
    "nflx":         {"capability": "C1/C2", "license": "Netflix research"},
    "konvid-1k":    {"capability": "C2",    "license": "CC BY 4.0"},
    "live-vqc":     {"capability": "C2",    "license": "Academic"},
    "youtube-ugc":  {"capability": "C2",    "license": "CC BY 3.0"},
    "bvi-dvc":      {"capability": "C3",    "license": "Academic"},
}


@dataclass(frozen=True)
class ManifestEntry:
    key: str
    path: str
    sha256: str
    mos: float | None = None


def data_root() -> Path:
    return Path(os.environ.get("VMAF_DATA_ROOT", Path.home() / ".cache" / "vmaf-train"))


def manifest_path(name: str) -> Path:
    here = Path(__file__).resolve().parent / "manifests"
    return here / f"{name}.yaml"


def load_manifest(name: str) -> list[ManifestEntry]:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name}")
    p = manifest_path(name)
    if not p.exists():
        return []
    with p.open() as fh:
        doc = yaml.safe_load(fh) or {}
    return [
        ManifestEntry(
            key=e["key"],
            path=e["path"],
            sha256=e["sha256"],
            mos=e.get("mos"),
        )
        for e in doc.get("entries", [])
    ]
