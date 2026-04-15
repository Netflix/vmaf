"""Correlation + error metrics for trained FR/NR regressors.

Reports PLCC (Pearson), SROCC (Spearman), and RMSE. Accepts either:
  * a torch checkpoint (evaluated via lightning), or
  * an ONNX file (evaluated via onnxruntime CPU EP).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalReport:
    plcc: float
    srocc: float
    rmse: float
    n: int

    def pretty(self) -> str:
        return (
            f"n={self.n:>6}  PLCC={self.plcc:+.4f}  "
            f"SROCC={self.srocc:+.4f}  RMSE={self.rmse:.4f}"
        )


def correlations(pred: np.ndarray, target: np.ndarray) -> EvalReport:
    if pred.shape != target.shape or pred.ndim != 1:
        raise ValueError("pred and target must be 1-D arrays of matching shape")
    from scipy.stats import pearsonr, spearmanr  # local import

    plcc = float(pearsonr(pred, target).statistic)
    srocc = float(spearmanr(pred, target).statistic)
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    return EvalReport(plcc=plcc, srocc=srocc, rmse=rmse, n=pred.shape[0])


def evaluate_onnx(
    onnx_path: Path,
    features: np.ndarray,
    targets: np.ndarray,
    input_name: str = "input",
) -> EvalReport:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {input_name: features.astype(np.float32)})[0]
    pred = np.asarray(out).reshape(-1)
    return correlations(pred, targets.astype(np.float32))
