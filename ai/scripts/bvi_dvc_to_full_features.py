#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""BVI-DVC → full-feature VMAF parquet (corpus-3 for tiny-AI v2).

Mirrors :mod:`ai.scripts.konvid_to_full_features` (the canonical-21
KoNViD acquisition pipeline shipped in PR #178) but sources clips from
the BVI-DVC Part 1 dataset (Ma, Zhang, Bull 2021) — a 4-tier 4:2:0
10-bit YCbCr reference corpus distributed as a single ``Videos/``
directory inside ``BVI-DVC Part 1.zip``. Tiers are encoded in the
filename prefix:

    A_3840x2176, B_1920x1088, C_960x544, D_480x272

BVI-DVC ships **reference-only** material — no human DMOS — so we
generate the distorted side ourselves at CRF 35 (matching the KoNViD
flow) and treat the libvmaf score as the teacher signal for the
tiny-AI student.

Output schema (one row per (clip, frame) pair):

    key, frame_index, codec, <21 feature columns>, vmaf

The ``codec`` column is the encoder family that produced the distorted
side. BVI-DVC ships reference-only material and this script encodes
internally with ``libx264``, so the column is a constant ``"x264"`` —
captured eagerly so the parquet self-describes for the codec-aware FR
regressor (see [ADR-0235](../../docs/adr/0235-codec-aware-fr-regressor.md)).

Output parquet: ``runs/full_features_bvi_dvc_<tier>.parquet`` (gitignored).
Per-clip JSON cache: ``$XDG_CACHE_HOME/vmaf-tiny-ai-bvi-dvc-full/<key>.json``
(separate from the konvid cache so all three corpora can coexist).

The 10-bit input path is the only structural delta from the konvid
script: ffmpeg gets ``-pix_fmt yuv420p10le`` and libvmaf gets
``--bitdepth 10``; everything else (FULL_FEATURES tuple, EXTRACTORS
tuple, CRF 35, single-thread CPU vmaf, model attached for per-frame
teacher score) is verbatim.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


# Verbatim mirror of FULL_FEATURES in ai/scripts/konvid_to_full_features.py
# (PR #178). Keep these tuples in sync — they define the corpus-portable
# 21-feature pool the Phase-3b sweep consumes.
FULL_FEATURES: tuple[str, ...] = (
    "adm2",
    "adm_scale0",
    "adm_scale1",
    "adm_scale2",
    "adm_scale3",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion",
    "motion2",
    "motion3",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede2000",
    "psnr_hvs",
    "ssimulacra2",
)


EXTRACTORS = (
    "adm",
    "vif",
    "motion",
    "motion_v2",
    "psnr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede",
    "psnr_hvs",
    "ssimulacra2",
)


# Filename pattern: e.g. "DBookcaseBVITexture_480x272_120fps_10bit_420.mp4".
# Group 1 = tier letter (A/B/C/D), 2 = width, 3 = height.
_NAME_RE = re.compile(r"^([ABCD])[A-Za-z0-9]+_(\d+)x(\d+)_\d+fps_10bit_420\.mp4$")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def _decode_yuv_10bit(src_mp4: Path, out_yuv: Path) -> tuple[int, int, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(src_mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    w = int(info["width"])
    h = int(info["height"])
    nb_frames = int(info.get("nb_frames", 0))
    out_yuv.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src_mp4),
            "-pix_fmt",
            "yuv420p10le",
            "-f",
            "rawvideo",
            str(out_yuv),
        ]
    )
    return w, h, nb_frames


def _encode_dis_10bit(src_mp4: Path, out_yuv: Path, crf: int) -> None:
    out_yuv.parent.mkdir(parents=True, exist_ok=True)
    # libx264 10-bit lossy compression at CRF 35 (matches KoNViD flow);
    # matroska pipe container so the moov atom doesn't need seekable
    # output. Decode side restores to yuv420p10le raw for libvmaf.
    cmd = (
        f"ffmpeg -y -loglevel error -i {shlex.quote(str(src_mp4))} "
        f"-c:v libx264 -pix_fmt yuv420p10le -crf {crf} -preset fast -an "
        f"-f matroska pipe:1 | "
        f"ffmpeg -y -loglevel error -i pipe:0 -pix_fmt yuv420p10le "
        f"-f rawvideo {shlex.quote(str(out_yuv))}"
    )
    _run(["bash", "-c", cmd])


def _run_vmaf_full(
    vmaf_bin: Path,
    ref_yuv: Path,
    dis_yuv: Path,
    w: int,
    h: int,
    out_json: Path,
    model_path: Path,
) -> None:
    """Run libvmaf CLI with all FULL_FEATURES extractors + the
    vmaf_v0.6.1 model attached for the per-frame VMAF teacher score.

    BVI-DVC ships 10-bit; ``--bitdepth 10`` is the only structural
    delta vs. the 8-bit konvid invocation.
    """
    feat_args: list[str] = []
    for ex in EXTRACTORS:
        feat_args += ["--feature", ex]
    _run(
        [
            str(vmaf_bin),
            "--reference",
            str(ref_yuv),
            "--distorted",
            str(dis_yuv),
            "--width",
            str(w),
            "--height",
            str(h),
            "--pixel_format",
            "420",
            "--bitdepth",
            "10",
            "--model",
            f"path={model_path}",
            *feat_args,
            "--threads",
            "1",
            "--no_cuda",
            "--no_sycl",
            "--no_vulkan",
            "--output",
            str(out_json),
            "--json",
            "-q",
        ]
    )


def _lookup(metrics: dict, name: str) -> float | None:
    """libvmaf may emit ``integer_<name>`` for fixed-point kernels."""
    if name in metrics:
        return float(metrics[name])
    if f"integer_{name}" in metrics:
        return float(metrics[f"integer_{name}"])
    return None


def _frames_to_rows(key: str, vmaf_json: Path, codec: str) -> list[dict]:
    with vmaf_json.open() as f:
        d = json.load(f)
    rows = []
    for fr in d["frames"]:
        m = fr["metrics"]
        row: dict = {"key": key, "frame_index": int(fr["frameNum"]), "codec": codec}
        for feat in FULL_FEATURES:
            v = _lookup(m, feat)
            row[feat] = float("nan") if v is None else v
        row["vmaf"] = float(m["vmaf"])
        rows.append(row)
    return rows


def _process_clip(
    key: str,
    src_mp4: Path,
    vmaf_bin: Path,
    model_path: Path,
    crf: int,
    cache_dir: Path | None,
    scratch: Path,
    codec: str,
) -> list[dict]:
    if cache_dir is not None:
        cache_path = cache_dir / f"{key}.json"
        if cache_path.is_file():
            return _frames_to_rows(key, cache_path, codec)
    ref_yuv = scratch / f"{key}_ref.yuv"
    dis_yuv = scratch / f"{key}_dis.yuv"
    vmaf_json = scratch / f"{key}_vmaf.json"
    try:
        w, h, _nb = _decode_yuv_10bit(src_mp4, ref_yuv)
        _encode_dis_10bit(src_mp4, dis_yuv, crf)
        _run_vmaf_full(vmaf_bin, ref_yuv, dis_yuv, w, h, vmaf_json, model_path)
        rows = _frames_to_rows(key, vmaf_json, codec)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{key}.json").write_text(vmaf_json.read_text())
        return rows
    finally:
        for p in (ref_yuv, dis_yuv, vmaf_json):
            with contextlib.suppress(FileNotFoundError):
                p.unlink()


def _select_tier_entries(zf: zipfile.ZipFile, tier: str) -> list[zipfile.ZipInfo]:
    """Filter the zip's ``Videos/`` entries to a single tier (or all).

    ``tier == "all"`` returns every clip across A/B/C/D in deterministic
    sorted order so reruns give the same indexing.
    """
    selected: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if "/Videos/" not in name or not name.endswith(".mp4"):
            continue
        base = name.rsplit("/", 1)[-1]
        m = _NAME_RE.match(base)
        if m is None:
            continue
        clip_tier = m.group(1)
        if tier != "all" and clip_tier != tier:
            continue
        selected.append(info)
    selected.sort(key=lambda i: i.filename)
    return selected


def _stream_extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> Path:
    """Stream a single zip entry to ``dest`` without unpacking the
    rest of the archive. Returns the dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, dest.open("wb") as out:
        # 4 MiB chunks — keeps RAM bounded for the 372 MB A-tier clips.
        while True:
            chunk = src.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(prog="bvi_dvc_to_full_features.py")
    ap.add_argument(
        "--bvi-zip",
        type=Path,
        default=Path(
            os.environ.get(
                "VMAF_BVI_DVC_ZIP",
                str(REPO_ROOT.parent / ".workingdir2" / "BVI-DVC Part 1.zip"),
            )
        ),
        help="Path to BVI-DVC Part 1.zip.",
    )
    ap.add_argument(
        "--tier",
        choices=("A", "B", "C", "D", "all"),
        default="D",
        help="Resolution tier to process (A=3840x2176, B=1920x1088, "
        "C=960x544, D=480x272, all=every tier in sorted order).",
    )
    ap.add_argument(
        "--vmaf-bin",
        type=Path,
        default=REPO_ROOT / "build-cpu" / "tools" / "vmaf",
        help="Path to the libvmaf CLI binary.",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "model" / "vmaf_v0.6.1.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet (default: runs/full_features_bvi_dvc_<tier>.parquet).",
    )
    ap.add_argument(
        "--scratch",
        type=Path,
        default=Path(os.environ.get("VMAF_TINY_AI_SCRATCH", "/tmp/bvi_dvc_full_acquire")),
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "VMAF_TINY_AI_CACHE_BVI_DVC_FULL",
                str(Path.home() / ".cache" / "vmaf-tiny-ai-bvi-dvc-full"),
            )
        ),
    )
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--crf", type=int, default=35)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument(
        "--codec",
        type=str,
        default="x264",
        help="Codec label baked into the parquet's `codec` column. Must "
        "match an entry in ai/src/vmaf_train/codec.py CODEC_VOCAB (or it "
        "will bucket to 'unknown' at training time). The script always "
        "encodes via libx264 today; this flag exists so a future "
        "multi-codec sweep can reuse the same harness.",
    )
    args = ap.parse_args()

    if not args.bvi_zip.is_file():
        print(f"error: BVI-DVC zip not found at {args.bvi_zip}", file=sys.stderr)
        return 2
    if not args.vmaf_bin.is_file():
        print(f"error: vmaf binary not found at {args.vmaf_bin}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 2

    out_path = args.out or (REPO_ROOT / "runs" / f"full_features_bvi_dvc_{args.tier}.parquet")
    args.scratch.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.bvi_zip) as zf:
        entries = _select_tier_entries(zf, args.tier)
        if args.max_clips is not None:
            entries = entries[: args.max_clips]
        print(
            f"[bvi-dvc-full] tier={args.tier} processing {len(entries)} clips → {out_path}",
            flush=True,
        )

        rows: list[dict] = []
        t0 = time.time()
        for i, info in enumerate(entries):
            base = info.filename.rsplit("/", 1)[-1]
            key = base[: -len(".mp4")]
            local_mp4 = args.scratch / base
            try:
                # Skip extraction if the cached vmaf JSON already exists —
                # _process_clip will short-circuit before touching the mp4.
                cache_hit = cache_dir is not None and (cache_dir / f"{key}.json").is_file()
                if not cache_hit:
                    _stream_extract(zf, info, local_mp4)
                rows += _process_clip(
                    key,
                    local_mp4,
                    args.vmaf_bin,
                    args.model,
                    args.crf,
                    cache_dir,
                    args.scratch,
                    args.codec,
                )
            finally:
                with contextlib.suppress(FileNotFoundError):
                    local_mp4.unlink()
            if (i + 1) % 5 == 0 or (i + 1) == len(entries):
                wt = time.time() - t0
                print(
                    f"[bvi-dvc-full] {i + 1}/{len(entries)} clips, {len(rows)} frames, {wt:.1f}s",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(
        f"[bvi-dvc-full] wrote {out_path} ({len(df)} frames, "
        f"{len(entries)} clips, {len(df.columns)} cols)",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
