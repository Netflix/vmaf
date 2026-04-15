"""Model registry — write/read sidecar metadata for shipped tiny models.

Each `.onnx` under `model/tiny/` gets a `<name>.json` sidecar recording:
  * kind (fr | nr | filter)
  * input_names, output_names
  * normalization (mean/std)
  * dataset, train_commit, train_config_hash
  * onnx_opset
  * expected_output_range (for runtime sanity bounds)
  * license
  * cosign_signature (filled in by release workflow)

Keeps a flat schema so the C loader can parse it with a minimal JSON reader.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VALID_KINDS = {"fr", "nr", "filter"}


@dataclass
class ModelMetadata:
    schema_version: int
    name: str
    kind: str
    onnx_opset: int
    input_names: list[str]
    output_names: list[str]
    normalization: dict[str, list[float]] = field(default_factory=dict)
    dataset: str | None = None
    train_commit: str | None = None
    train_config_hash: str | None = None
    parent_dataset_manifest: str | None = None
    expected_output_range: list[float] | None = None
    license: str | None = None
    cosign_signature: str | None = None
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(config_path: Path, manifest_path: Path | None = None) -> str:
    h = hashlib.sha256()
    h.update(config_path.read_bytes())
    if manifest_path is not None and manifest_path.exists():
        h.update(b"\x00")
        h.update(manifest_path.read_bytes())
    return h.hexdigest()


def register(
    onnx_path: Path,
    kind: str,
    dataset: str | None = None,
    license_: str | None = None,
    train_commit: str | None = None,
    train_config: Path | None = None,
    manifest: Path | None = None,
    normalization: dict[str, list[float]] | None = None,
    expected_output_range: tuple[float, float] | None = None,
    notes: str | None = None,
) -> Path:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    opset = max(o.version for o in model.opset_import) if model.opset_import else 0
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    cfg_hash = compute_config_hash(train_config, manifest) if train_config else None

    meta = ModelMetadata(
        schema_version=SCHEMA_VERSION,
        name=onnx_path.stem,
        kind=kind,
        onnx_opset=opset,
        input_names=input_names,
        output_names=output_names,
        normalization=normalization or {},
        dataset=dataset,
        train_commit=train_commit,
        train_config_hash=cfg_hash,
        parent_dataset_manifest=_hash_file(manifest) if manifest else None,
        expected_output_range=list(expected_output_range) if expected_output_range else None,
        license=license_,
        notes=notes,
    )

    sidecar = onnx_path.with_suffix(".json")
    sidecar.write_text(meta.to_json())
    return sidecar


def load(sidecar_path: Path) -> ModelMetadata:
    doc: dict[str, Any] = json.loads(sidecar_path.read_text())
    return ModelMetadata(**doc)
