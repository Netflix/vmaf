"""Data loading for vmaf-tiny.

Expects a YAML manifest of the form:

```yaml
entries:
  - ref:   testdata/ref_001.yuv
    dis:   testdata/dis_001.yuv
    w:     576
    h:     324
    score: 76.668904824
```

The C engine's feature extractor is used offline to materialize per-frame
feature vectors into an .npz cache. This module just loads the cache.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class FeatureScoreDataset(Dataset):
    def __init__(self, cache: Path) -> None:
        data = np.load(cache)
        self.x = torch.from_numpy(data["features"]).float()
        self.y = torch.from_numpy(data["scores"]).float()
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("features/scores length mismatch in cache")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class VmafTinyDataModule(L.LightningDataModule):
    def __init__(self, cache: Path, batch_size: int = 256, val_split: float = 0.1) -> None:
        super().__init__()
        self.cache = cache
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage: str | None = None) -> None:
        ds = FeatureScoreDataset(self.cache)
        n_val = int(len(ds) * self.val_split)
        n_train = len(ds) - n_val
        self.train, self.val = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0)
