# Tiny AI — benchmarks

How to produce comparable numbers for the tiny-AI models and how to read
them. **Placeholder numbers below** — regenerate via the scripts in this
page once a trained model is available.

## Accuracy methodology

For FR (C1) and NR (C2) models, the three canonical regression metrics:

- **PLCC** — Pearson linear correlation with MOS.
- **SROCC** — Spearman rank-order correlation.
- **RMSE** — root mean square error against MOS (0–100 scale).

All three are computed by `vmaf-train eval` on the held-out **test split**
produced by `vmaf_train.data.splits.split_keys` with the fixed salt
`vmaf-train-splits-v1`. Splits are deterministic so baseline and
challenger see the same frames/keys.

```bash
vmaf-train eval \
    --model model/tiny/vmaf_tiny_fr_v1.onnx \
    --features ai/data/nflx_features.parquet \
    --split test
```

### Baseline: upstream `vmaf_v0.6.1` SVM

To compare a new tiny FR model against the upstream SVM, score the same
test pairs through both and run `ai/tests/test_eval_metrics.py` helper
functions. Keep the baseline's version in the committed report for
auditability.

## Runtime methodology

```bash
# Frames/second, end-to-end, single-thread CPU:
./testdata/bench_all.sh --tiny-model model/tiny/vmaf_tiny_fr_v1.onnx --backend=cpu

# GPU throughput:
./testdata/bench_all.sh --tiny-model model/tiny/vmaf_tiny_fr_v1.onnx --backend=cuda
```

The `testdata/bench_all.sh` harness logs into
`testdata/netflix_benchmark_results.json` (never committed — ad-hoc run
artefact). Collect multiple runs and report median + p99.

## Placeholder scoreboard

These rows are illustrative. The `vmaf-train register` step records
actual scores in the sidecar JSON; paste them here on shipping a model.

| Model | Target | PLCC | SROCC | RMSE | ms/frame (CPU) | ms/frame (CUDA) |
| --- | --- | --- | --- | --- | --- | --- |
| `vmaf_v0.6.1` (SVM, upstream) | FR | TBD | TBD | TBD | — | — |
| `vmaf_tiny_fr_v1` | FR | TBD | TBD | TBD | TBD | TBD |
| `vmaf_nr_mobilenet_v1` | NR | TBD | TBD | TBD | TBD | TBD |
| `filter_denoise_residual_v1` | C3 | — | — | — | TBD | TBD |

## Model-size targets

| Model class | Target size | Typical |
| --- | --- | --- |
| C1 (FR MLP) | ≤ 100 KB | ~50 KB |
| C2 (NR CNN) | ≤ 5 MB | ~2 MB |
| C3 (learned filter) | ≤ 2 MB | ~800 KB |

Models larger than `VMAF_DEFAULT_MAX_MODEL_BYTES` (50 MB, overridable via
`VMAF_MAX_MODEL_BYTES`) are rejected at load time. Tiny-AI is tiny by
definition — if a candidate model balloons past the targets, the design
is wrong, not the limit.

## Determinism in benchmarks

Same `--seed` + same `train_commit` + same dataset manifest SHA should
reproduce the reported scores within a tight allclose. CI includes a
float-rounding guard so drift ≥ 1e-3 on the primary FR metric trips a
regression failure.
