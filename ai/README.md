# `vmaf-train` — tiny AI training harness

Training, exporting, registering, and evaluating **tiny perceptual-quality
models** for libvmaf. Shipped artefacts are ONNX; runtime loading happens in
[libvmaf/src/dnn/](../libvmaf/src/dnn/) through ONNX Runtime.

Three model families:

| Family            | What it does                                          | File                                             |
|-------------------|-------------------------------------------------------|--------------------------------------------------|
| `fr_regressor`    | feature-vector → MOS (C1, replaces/augments SVM)      | `src/vmaf_train/models/fr_regressor.py`          |
| `nr_metric`       | distorted frame → MOS, no reference (C2)              | `src/vmaf_train/models/nr_metric.py`             |
| `learned_filter`  | frame → frame residual CNN, encoder pre-filter (C3)   | `src/vmaf_train/models/learned_filter.py`        |

## Install

```bash
cd ai
pip install -e ".[dev,viz]"
```

## CLI

```bash
# 1. Dump features for FR training
vmaf-train extract-features --dataset nflx --output ai/data/nflx_features.parquet

# 2. Train
vmaf-train fit --config ai/configs/fr_tiny_v1.yaml

# 3. Export to ONNX (torch ↔ onnxruntime roundtrip atol=1e-5)
vmaf-train export \
  --checkpoint runs/fr_tiny_v1/last.ckpt \
  --output model/tiny/vmaf_tiny_fr_v1.onnx \
  --model fr_regressor --opset 17

# 4. Evaluate (PLCC / SROCC / RMSE)
vmaf-train eval \
  --model model/tiny/vmaf_tiny_fr_v1.onnx \
  --features ai/data/nflx_features.parquet \
  --split test

# 5. Register sidecar metadata
vmaf-train register \
  --model model/tiny/vmaf_tiny_fr_v1.onnx \
  --kind fr --dataset nflx \
  --license CDLA-Permissive-2.0 \
  --train-commit $(git rev-parse HEAD)
```

## Layout

```
ai/
├── pyproject.toml
├── configs/                       # fr_tiny_v1, nr_mobilenet_v1, filter_residual_v1
├── src/vmaf_train/
│   ├── cli.py                     # `vmaf-train` entry point (typer)
│   ├── train.py                   # training loop (Lightning)
│   ├── eval.py                    # PLCC / SROCC / RMSE
│   ├── registry.py                # sidecar metadata writer/reader
│   ├── tune.py                    # optional Optuna sweep
│   ├── datamodule.py              # parquet / npz feature-cache loader
│   ├── data/
│   │   ├── datasets.py            # manifest loaders (NFLX, KoNViD, LIVE-VQC, YouTube-UGC, BVI-DVC)
│   │   ├── feature_dump.py        # drives libvmaf → parquet
│   │   ├── frame_loader.py        # ffmpeg → numpy frames
│   │   ├── splits.py              # deterministic hash-based splits
│   │   └── manifests/             # per-dataset YAML manifests (SHA-256 pinned)
│   └── models/
│       ├── fr_regressor.py        # C1
│       ├── nr_metric.py           # C2
│       ├── learned_filter.py      # C3
│       └── exports.py             # torch → ONNX with roundtrip validation
├── tests/
│   ├── test_export_roundtrip.py   # torch.eval() == onnxruntime (atol=1e-5)
│   ├── test_dataset_splits.py
│   ├── test_registry.py
│   └── test_eval_metrics.py
└── notebooks/                     # exploratory; outputs gitignored
```

## Determinism

`vmaf-train fit` seeds torch / numpy / python / lightning with `--seed` (default 0)
and enables `Trainer(deterministic=True)`. The sidecar records the train commit,
config hash, and dataset manifest hash so a retrain is reproducible to float-
rounding nondeterminism.

## License

BSD-3-Clause-Plus-Patent — matches the rest of the fork.
