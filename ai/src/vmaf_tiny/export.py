"""Export a trained tiny-VMAF checkpoint to ONNX.

The exported model must be consumable by the C inference path in
``libvmaf/src/dnn/`` under ONNX Runtime with CPU / CUDA / SYCL (TensorRT
or DirectML) execution providers. Opset is pinned to 17 to match our
ORT baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

from .model import TinyVMAF

OPSET = 17


def export(ckpt: Path, out: Path, in_features: int = 6) -> None:
    model = TinyVMAF.load_from_checkpoint(ckpt).eval()
    dummy = torch.zeros(1, in_features, dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=OPSET,
    )
    onnx.checker.check_model(onnx.load(str(out)))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ref_torch = model(dummy).detach().numpy()
    ref_ort = sess.run(None, {"features": dummy.numpy()})[0]
    max_abs = float(abs(ref_torch - ref_ort).max())
    if max_abs > 1e-4:
        raise RuntimeError(f"torch vs onnxruntime drift too large: {max_abs:g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a tiny-VMAF checkpoint to ONNX.")
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--in-features", type=int, default=6)
    args = ap.parse_args()
    export(args.ckpt, args.out, args.in_features)


if __name__ == "__main__":
    main()
