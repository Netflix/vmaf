"""C1 — feature-vector → MOS regressor (replaces / augments SVM)."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn


class FRRegressor(L.LightningModule):
    """Tiny MLP over precomputed libvmaf feature vectors (adm, vif, motion, ...)."""

    def __init__(
        self,
        in_features: int = 7,
        hidden: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], tag: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log(f"{tag}/mse", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, _idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch, _idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
