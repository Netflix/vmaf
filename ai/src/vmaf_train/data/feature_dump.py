"""Extract libvmaf feature vectors → parquet (C1 training input).

Drives the `vmaf` CLI in JSON mode, collects per-frame features listed in the
dataset manifest, and writes a parquet table suitable for pandas / polars
consumption. A feature vector is the union of whatever extractors the runner
configures (adm2, vif_scale0..3, motion2, etc.).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Entry:
    key: str
    ref: Path
    dis: Path
    width: int
    height: int
    pix_fmt: str = "yuv420p"
    mos: float | None = None


DEFAULT_FEATURES = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)


def _run_vmaf(binary: Path, entry: Entry, features: tuple[str, ...]) -> dict:
    feat_args: list[str] = []
    for f in features:
        feat_args += ["--feature", f]
    cmd = [
        str(binary),
        "-r", str(entry.ref),
        "-d", str(entry.dis),
        "-w", str(entry.width),
        "-h", str(entry.height),
        "-p", entry.pix_fmt,
        "--json", "-o", "-",
        *feat_args,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def dump_features(
    entries: list[Entry],
    out_parquet: Path,
    vmaf_binary: Path = Path("vmaf"),
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> Path:
    rows: list[dict] = []
    for e in entries:
        doc = _run_vmaf(vmaf_binary, e, features)
        for frame in doc.get("frames", []):
            row: dict[str, object] = {
                "key": e.key,
                "frame": frame.get("frameNum"),
                "mos": e.mos,
            }
            fmetrics = frame.get("metrics", {})
            for f in features:
                row[f] = fmetrics.get(f)
            rows.append(row)
    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return out_parquet
