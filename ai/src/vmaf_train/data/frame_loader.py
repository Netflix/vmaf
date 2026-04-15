"""ffmpeg-decoded frame batches for NR / learned-filter training (C2, C3)."""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameSource:
    path: Path
    width: int
    height: int
    pix_fmt: str = "gray"


def iter_frames(source: FrameSource, ffmpeg: str = "ffmpeg") -> Iterator[np.ndarray]:
    """Yield frames from `source` as HxW uint8 numpy arrays (single-channel)."""
    if source.pix_fmt != "gray":
        raise NotImplementedError("only pix_fmt=gray supported for now")
    frame_bytes = source.width * source.height
    proc = subprocess.Popen(
        [
            ffmpeg, "-v", "error", "-i", str(source.path),
            "-f", "rawvideo", "-pix_fmt", source.pix_fmt, "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape(source.height, source.width).copy()
    finally:
        proc.stdout.close()
        proc.wait()
