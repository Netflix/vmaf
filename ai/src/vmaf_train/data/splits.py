"""Deterministic train/val/test splits keyed by stable content hash.

The hash must be stable across runs and across machines — derived from the
source filename (or video ID) rather than list position.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Splits:
    train: list[str]
    val: list[str]
    test: list[str]


def _bucket(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


def split_keys(
    keys: list[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    salt: str = "vmaf-train-splits-v1",
) -> Splits:
    if val_frac + test_frac >= 1.0:
        raise ValueError("val+test must be < 1.0")
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for k in keys:
        b = _bucket(k, salt)
        if b < test_frac:
            test.append(k)
        elif b < test_frac + val_frac:
            val.append(k)
        else:
            train.append(k)
    return Splits(train=sorted(train), val=sorted(val), test=sorted(test))
