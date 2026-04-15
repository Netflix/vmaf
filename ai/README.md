# VMAF Tiny-AI — training harness

Experimental surface for training **tiny** perceptual-quality models that can
replace or complement the stock VMAF SVR in low-compute contexts (embedded
decoders, mobile, bulk preprocessing pipelines).

## Status

Scaffolding only. No trained weights are shipped yet. The
[model/tiny/](../model/tiny/) directory is reserved for ONNX artefacts once
the first milestone model is ready.

## Goals

- **Tiny**: <1 MB ONNX, <5 ms/frame on a laptop CPU, batchable on GPU.
- **Aligned with VMAF**: trained against the stock VMAF score on the Netflix
  public corpus plus our cross-backend snapshots.
- **Deterministic inference**: bit-identical across CPU / CUDA / SYCL via
  ONNX Runtime with the same execution provider config.
- **Round-trippable**: training → ONNX export → validation in
  [libvmaf/src/dnn/](../libvmaf/src/dnn/) inference path.

## Layout

```
ai/
  pyproject.toml        # torch, lightning, onnx, onnxruntime
  src/vmaf_tiny/
    __init__.py
    datamodule.py       # loads (ref, dis, score) triplets
    model.py            # lightning module
    train.py            # CLI entry point
    export.py           # ONNX export with opset + shape checks
  tests/
    test_export.py      # round-trip sanity checks
```

## Training

```bash
cd ai
pip install -e ".[dev]"
vmaf-tiny-train --data /path/to/corpus --epochs 50 --out runs/mini
vmaf-tiny-export runs/mini/best.ckpt --out ../model/tiny/vmaf_tiny_v0.onnx
```

## Inference hook

Once an ONNX artefact exists under `model/tiny/`, the C side in
[libvmaf/src/dnn/](../libvmaf/src/dnn/) loads it via ONNX Runtime and exposes
it as a feature extractor selectable by `--model path=model/tiny/<name>.onnx`.

## License

BSD-3-Clause-Plus-Patent (matches the rest of the fork).
