"""Round-trip the sidecar metadata through registry.register / registry.load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from vmaf_train.models import FRRegressor, export_to_onnx  # noqa: E402
from vmaf_train.registry import VALID_KINDS, load, register  # noqa: E402


def test_register_roundtrip(tmp_path: Path) -> None:
    model = FRRegressor(in_features=6).eval()
    onnx_path = tmp_path / "fr.onnx"
    export_to_onnx(model, onnx_path, in_shape=(1, 6), input_name="features")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: fr_regressor\n")

    sidecar = register(
        onnx_path=onnx_path,
        kind="fr",
        dataset="synthetic",
        license_="BSD-3-Clause-Plus-Patent",
        train_commit="deadbeef",
        train_config=cfg,
    )
    assert sidecar.exists()

    meta = load(sidecar)
    assert meta.kind == "fr"
    assert meta.dataset == "synthetic"
    assert meta.train_commit == "deadbeef"
    assert meta.onnx_opset >= 17
    assert "features" in meta.input_names
    assert meta.train_config_hash and len(meta.train_config_hash) == 64

    doc = json.loads(sidecar.read_text())
    assert doc["schema_version"] == 1


def test_register_rejects_unknown_kind(tmp_path: Path) -> None:
    model = FRRegressor(in_features=6).eval()
    onnx_path = tmp_path / "fr.onnx"
    export_to_onnx(model, onnx_path, in_shape=(1, 6), input_name="features")

    with pytest.raises(ValueError):
        register(onnx_path=onnx_path, kind="bogus")


def test_valid_kinds_constant() -> None:
    assert VALID_KINDS == {"fr", "nr", "filter"}
