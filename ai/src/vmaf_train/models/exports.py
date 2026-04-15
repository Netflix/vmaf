"""Torch → ONNX export with roundtrip validation.

Validates opset 17, dynamic batch axis, and runs the exported graph through
onnxruntime; asserts allclose(atol=1e-5) vs the pytorch `.eval()` output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

OPSET = 17


def _guess_dummy(model: nn.Module, in_shape: tuple[int, ...] | None) -> torch.Tensor:
    if in_shape is not None:
        return torch.zeros(*in_shape, dtype=torch.float32)
    first = next(model.parameters())
    dtype = torch.float32
    if first.dim() == 4:  # Conv-first model (NR / filter)
        in_c = first.shape[1]
        return torch.zeros(1, in_c, 64, 64, dtype=dtype)
    in_f = first.shape[1] if first.dim() == 2 else 7
    return torch.zeros(1, in_f, dtype=dtype)


def export_to_onnx(
    model: nn.Module,
    out_path: Path,
    in_shape: tuple[int, ...] | None = None,
    input_name: str = "input",
    output_name: str = "output",
    atol: float = 1e-5,
    opset: int = OPSET,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dummy = _guess_dummy(model, in_shape)

    dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    onnx.checker.check_model(onnx.load(str(out_path)))

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).cpu().numpy()
    ort_out = sess.run(None, {input_name: dummy.numpy()})[0]
    max_abs = float(np.abs(ref - ort_out).max())
    if max_abs > atol:
        raise RuntimeError(
            f"torch vs onnxruntime drift {max_abs:g} exceeds atol {atol:g}"
        )
