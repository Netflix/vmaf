"""Optional Optuna hyperparameter sweep — imported lazily to keep core deps small."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .train import TrainConfig, train


def sweep(
    base_cfg: TrainConfig,
    suggest: Callable[["optuna.Trial"], dict[str, object]],  # noqa: F821
    n_trials: int = 20,
    study_name: str = "vmaf-train-sweep",
    storage: str | None = None,
) -> "optuna.Study":  # noqa: F821
    import optuna  # local import — optional dep

    def objective(trial: "optuna.Trial") -> float:
        overrides = suggest(trial)
        cfg = TrainConfig(
            model=base_cfg.model,
            model_args={**base_cfg.model_args, **overrides},
            cache=base_cfg.cache,
            output=Path(base_cfg.output) / f"trial_{trial.number:03d}",
            epochs=base_cfg.epochs,
            batch_size=base_cfg.batch_size,
            val_frac=base_cfg.val_frac,
            test_frac=base_cfg.test_frac,
            seed=base_cfg.seed,
            precision=base_cfg.precision,
        )
        train(cfg)
        metrics_csv = cfg.output / "lightning_logs" / "version_0" / "metrics.csv"
        if not metrics_csv.exists():
            return float("inf")
        import pandas as pd

        df = pd.read_csv(metrics_csv)
        if "val/mse" in df.columns:
            return float(df["val/mse"].dropna().min())
        if "val/l1" in df.columns:
            return float(df["val/l1"].dropna().min())
        return float("inf")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
