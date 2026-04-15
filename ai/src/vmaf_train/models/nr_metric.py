"""C2 — no-reference metric: distorted frame → MOS (small CNN)."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn


def _dw_sep(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class NRMetric(L.LightningModule):
    """MobileNet-tiny-ish backbone → global pool → scalar MOS."""

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU6(inplace=True),
        )
        self.body = nn.Sequential(
            _dw_sep(w, w * 2, stride=2),
            _dw_sep(w * 2, w * 2),
            _dw_sep(w * 2, w * 4, stride=2),
            _dw_sep(w * 4, w * 4),
            _dw_sep(w * 4, w * 8, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x))).squeeze(-1)

    def _step(self, batch, tag: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log(f"{tag}/mse", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, _idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
