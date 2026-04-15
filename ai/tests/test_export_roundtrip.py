"""Export roundtrip — torch.eval() vs onnxruntime must agree to atol=1e-5."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

import numpy as np  # noqa: E402

from vmaf_train.models import FRRegressor, LearnedFilter, NRMetric, export_to_onnx  # noqa: E402


@pytest.mark.parametrize(
    "model_cls,in_shape,input_name",
    [
        (lambda: FRRegressor(in_features=6), (2, 6), "features"),
        (lambda: NRMetric(in_channels=1, width=8), (1, 1, 64, 64), "input"),
        (lambda: LearnedFilter(channels=1, width=8, num_blocks=2), (1, 1, 64, 64), "input"),
    ],
)
def test_export_roundtrip(tmp_path: Path, model_cls, in_shape, input_name) -> None:
    model = model_cls().eval()
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(model, onnx_path, in_shape=in_shape, input_name=input_name, atol=1e-5)
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(*in_shape, dtype=torch.float32)
    with torch.no_grad():
        ref = model(x).cpu().numpy()
    got = sess.run(None, {input_name: x.numpy()})[0]
    assert np.allclose(ref, got, atol=1e-5), f"drift {np.abs(ref - got).max():g}"
