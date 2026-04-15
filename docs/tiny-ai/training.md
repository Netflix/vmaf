# Tiny AI — training

Everything happens through `vmaf-train`, the typer CLI in
[`ai/`](../../ai/). Five subcommands: `extract-features`, `fit`, `export`,
`eval`, `register`.

## Install

```bash
pip install -e ai
# optional extras
pip install -e 'ai[tune,viz]'
```

This pulls `torch>=2.4,<2.6` + `lightning>=2.4,<2.5`. If you have a
GPU-capable PyTorch wheel installed separately, the extras will not
reinstall it.

## Dataset acquisition

Datasets are NOT committed. `ai/src/vmaf_train/data/datasets.py` knows
five canonical sources and caches them under
`${VMAF_DATA_ROOT:-~/.cache/vmaf-train}/datasets/<name>/`. Each dataset
ships a `manifests/<name>.yaml` SHA-256 manifest so downloads are
verifiable.

| Dataset | Use | License | Purpose |
| --- | --- | --- | --- |
| Netflix Public (NFLX) | C1, C2 | Netflix research | Same source as upstream `vmaf_v0.6.1` |
| KoNViD-1k | C2 | CC BY 4.0 | NR-friendly UGC clips with MOS |
| LIVE-VQC | C2 | Academic | NR validation |
| YouTube-UGC | C2 | CC BY 3.0 | Large-scale NR |
| BVI-DVC | C3 | Academic | Encoder distortion pairs for learned filters |

You are responsible for complying with each dataset's license. The
manifests only record hashes, not bytes.

## C1 — FR regressor walkthrough

```bash
# 1. Extract feature vectors from the NFLX pairs using the existing
#    libvmaf CPU backend.
vmaf-train extract-features \
    --dataset nflx \
    --output ai/data/nflx_features.parquet

# 2. Train a 2-layer MLP on the extracted features.
vmaf-train fit \
    --config ai/configs/fr_tiny_v1.yaml \
    --features ai/data/nflx_features.parquet \
    --output runs/fr_tiny_v1/

# 3. Export the trained weights to ONNX and validate roundtrip
#    (torch eval vs onnxruntime within atol=1e-5).
vmaf-train export \
    --checkpoint runs/fr_tiny_v1/last.ckpt \
    --output model/tiny/vmaf_tiny_fr_v1.onnx \
    --opset 17

# 4. Hold-out evaluation.
vmaf-train eval \
    --model model/tiny/vmaf_tiny_fr_v1.onnx \
    --features ai/data/nflx_features.parquet \
    --split test
# → PLCC, SROCC, RMSE vs MOS.

# 5. Write a sidecar and register into model/tiny/.
vmaf-train register \
    --model model/tiny/vmaf_tiny_fr_v1.onnx \
    --kind fr \
    --dataset nflx \
    --license CDLA-Permissive-2.0 \
    --train-commit "$(git rev-parse HEAD)"
```

The sidecar `model/tiny/vmaf_tiny_fr_v1.json` pins:

```json
{
  "schema_version": 1,
  "name": "vmaf_tiny_fr_v1",
  "kind": "fr",
  "onnx_opset": 17,
  "input_name": "features",
  "output_name": "score",
  "input_normalization": { "mean": [...], "std": [...] },
  "expected_output_range": [0.0, 100.0],
  "dataset": "nflx",
  "train_commit": "…",
  "train_config_hash": "sha256:…",
  "license": "CDLA-Permissive-2.0"
}
```

## C2 — NR metric

Same flow, different config: [`ai/configs/nr_mobilenet_v1.yaml`](../../ai/configs/nr_mobilenet_v1.yaml).
`extract-features` is replaced by a direct frame loader
([`frame_loader.py`](../../ai/src/vmaf_train/data/frame_loader.py)) that
feeds ffmpeg-decoded tensors into training.

## C3 — Learned filter

[`ai/configs/filter_residual_v1.yaml`](../../ai/configs/filter_residual_v1.yaml)
trains a residual CNN where the model is clamped to `x + residual` in
normalized space. Target is BVI-DVC encoder-distortion pairs.

## Determinism

`vmaf-train fit` seeds Python, NumPy, and PyTorch with the config's `seed`
field and sets Lightning's `deterministic=True`. Combined with
`train_commit + train_config_hash + dataset_manifest_sha + seed` the output
weights are reproducible to within float-rounding nondeterminism (which CI
will flag as a regression when it exceeds a tight allclose).

## Hyperparameter sweeps

```bash
pip install -e 'ai[tune]'
vmaf-train tune \
    --config ai/configs/fr_tiny_v1.yaml \
    --study fr_tiny_v1_sweep
```

Uses Optuna. Results are written under `runs/<study>/`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `extract-features` is slow | libvmaf CPU-only | rebuild with `-Denable_cuda=true` and rerun |
| `fit` OOM | batch size too big for GPU | edit `ai/configs/*.yaml` `batch_size`, or drop `precision` to `16-mixed` |
| Export roundtrip fails atol=1e-5 | op using `float16` with a value near `inf` | retrain in `float32` end-to-end, or tighten clamping |
