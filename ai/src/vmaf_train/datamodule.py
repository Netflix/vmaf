"""Feature-cache datamodule for the C1 (FR regressor) path.

Accepts either a numpy .npz cache (legacy) or a parquet file produced by
`data.feature_dump.dump_features`. Deterministic split derived from
`data.splits.split_keys` on the per-row `key` column when present.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .data.splits import split_keys

FEATURE_COLUMNS = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)


class FeatureScoreDataset(Dataset):
    def __init__(self, cache: Path) -> None:
        cache = Path(cache)
        if cache.suffix == ".npz":
            data = np.load(cache)
            self.x = torch.from_numpy(data["features"]).float()
            self.y = torch.from_numpy(data["scores"]).float()
            self.keys: list[str] | None = None
        elif cache.suffix in (".parquet", ".pq"):
            import pandas as pd

            df = pd.read_parquet(cache)
            cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            if not cols:
                raise ValueError(f"no known feature columns in {cache}")
            self.x = torch.tensor(df[cols].to_numpy(), dtype=torch.float32)
            if "mos" not in df.columns:
                raise ValueError(f"no 'mos' column in {cache}")
            self.y = torch.tensor(df["mos"].to_numpy(), dtype=torch.float32)
            self.keys = df["key"].astype(str).tolist() if "key" in df.columns else None
        else:
            raise ValueError(f"unsupported cache extension: {cache.suffix}")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("features/scores length mismatch")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class VmafTrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        cache: Path,
        batch_size: int = 256,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.cache = Path(cache)
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        ds = FeatureScoreDataset(self.cache)
        keys = ds.keys
        if keys is not None and len(set(keys)) > 1:
            splits = split_keys(sorted(set(keys)), self.val_frac, self.test_frac)
            bag = {k: tag for tag, keys_ in (
                ("train", splits.train),
                ("val", splits.val),
                ("test", splits.test),
            ) for k in keys_}
            train_idx = [i for i, k in enumerate(keys) if bag[k] == "train"]
            val_idx   = [i for i, k in enumerate(keys) if bag[k] == "val"]
            test_idx  = [i for i, k in enumerate(keys) if bag[k] == "test"]
            self.train = Subset(ds, train_idx)
            self.val   = Subset(ds, val_idx)
            self.test  = Subset(ds, test_idx)
        else:
            n = len(ds)
            n_test = int(n * self.test_frac)
            n_val  = int(n * self.val_frac)
            n_tr   = n - n_val - n_test
            gen = torch.Generator().manual_seed(0)
            perm = torch.randperm(n, generator=gen).tolist()
            self.train = Subset(ds, perm[:n_tr])
            self.val   = Subset(ds, perm[n_tr:n_tr + n_val])
            self.test  = Subset(ds, perm[n_tr + n_val:])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
