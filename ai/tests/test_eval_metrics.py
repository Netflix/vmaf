"""PLCC/SROCC/RMSE sanity checks against synthetic data with known correlation."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from vmaf_train.eval import correlations  # noqa: E402


def test_perfect_correlation() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, size=500).astype(np.float32)
    r = correlations(y.copy(), y)
    assert r.plcc > 0.9999
    assert r.srocc > 0.9999
    assert r.rmse < 1e-4


def test_anticorrelation() -> None:
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 100, size=500).astype(np.float32)
    r = correlations(-y, y)
    assert r.plcc < -0.9999
    assert r.srocc < -0.9999


def test_noise_degrades_correlation() -> None:
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 100, size=500).astype(np.float32)
    noisy = y + rng.normal(scale=30.0, size=500).astype(np.float32)
    r = correlations(noisy, y)
    assert 0.5 < r.plcc < 0.99
