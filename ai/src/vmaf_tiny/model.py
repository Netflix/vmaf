"""Lightning module for the tiny VMAF regressor.

Skeleton only — the architecture is intentionally trivial (MLP over a
pre-extracted feature vector) so the scaffolding can be validated end-to-end
before we invest in a proper 2-stream CNN.
"""
from __future__ import annotations

import lightning as L
import torch
from torch import nn


class TinyVMAF(L.LightningModule):
    """Tiny regressor: feature vector → scalar VMAF-aligned score.

    The production version will accept (ref, dis) luma patches and learn its
    own features; this placeholder regresses against hand-picked upstream
    feature extractor outputs (adm2, vif_scale0..3, motion2) to keep the
    export / inference harness honest while the real model is under design.
    """

    def __init__(self, in_features: int = 6, hidden: int = 64, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train/mse", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        self.log("val/mse", nn.functional.mse_loss(y_hat, y), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
