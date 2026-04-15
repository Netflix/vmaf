"""Main training entry, driven by a YAML config or direct kwargs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from .datamodule import VmafTrainDataModule
from .models import FRRegressor, LearnedFilter, NRMetric

MODEL_REGISTRY: dict[str, type[L.LightningModule]] = {
    "fr_regressor":   FRRegressor,
    "nr_metric":      NRMetric,
    "learned_filter": LearnedFilter,
}


@dataclass
class TrainConfig:
    model: str
    model_args: dict[str, Any]
    cache: Path
    output: Path
    epochs: int = 50
    batch_size: int = 256
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0
    precision: str = "32-true"


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    with path.open() as fh:
        doc = yaml.safe_load(fh) or {}
    if overrides:
        doc.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(
        model=doc["model"],
        model_args=doc.get("model_args", {}),
        cache=Path(doc["cache"]),
        output=Path(doc.get("output", "runs/default")),
        epochs=int(doc.get("epochs", 50)),
        batch_size=int(doc.get("batch_size", 256)),
        val_frac=float(doc.get("val_frac", 0.1)),
        test_frac=float(doc.get("test_frac", 0.1)),
        seed=int(doc.get("seed", 0)),
        precision=str(doc.get("precision", "32-true")),
    )


def train(cfg: TrainConfig) -> Path:
    if cfg.model not in MODEL_REGISTRY:
        raise KeyError(f"unknown model kind: {cfg.model}")
    L.seed_everything(cfg.seed, workers=True)
    model_cls = MODEL_REGISTRY[cfg.model]
    model = model_cls(**cfg.model_args)

    dm = VmafTrainDataModule(
        cfg.cache,
        batch_size=cfg.batch_size,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )

    cfg.output.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.output,
        filename="best",
        monitor="val/mse" if cfg.model != "learned_filter" else "val/l1",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[ckpt_cb],
        default_root_dir=cfg.output,
        log_every_n_steps=10,
        precision=cfg.precision,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    return cfg.output / "last.ckpt"
