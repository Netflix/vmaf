"""Training entry point for vmaf-tiny."""
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from .datamodule import VmafTinyDataModule
from .model import TinyVMAF


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tiny VMAF regressor.")
    ap.add_argument("--cache", type=Path, required=True,
                    help="Path to .npz cache produced by the offline feature extractor.")
    ap.add_argument("--out", type=Path, default=Path("runs/mini"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dm = VmafTinyDataModule(args.cache, batch_size=args.batch_size)
    model = TinyVMAF(lr=args.lr)

    ckpt = ModelCheckpoint(
        dirpath=args.out, filename="best", monitor="val/mse",
        mode="min", save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[ckpt],
        default_root_dir=args.out,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
