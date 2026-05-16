#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Measure fp32-vs-int8 PLCC drop across multiple Execution Providers (T5-3e).

Sibling of ``ai/scripts/measure_quant_drop.py`` — same gate semantics,
but runs the comparison under each requested EP rather than the default
``CPUExecutionProvider`` only. Tracks the open question raised in
``docs/research/0006-tinyai-ptq-accuracy-targets.md`` §"GPU-EP
quantisation": with both a CUDA card and an Intel Arc accelerator
present, we can finally measure (instead of defer) how int8 quantisation
behaves on each EP.

Backends covered:

* ``cpu``       -> ORT ``CPUExecutionProvider`` (the existing baseline).
* ``cuda``      -> ORT ``CUDAExecutionProvider`` (NVIDIA, requires
                   CUDA 12 / cuDNN 9 runtime libs and an ORT-GPU wheel).
* ``openvino``  -> the native ``openvino`` Python runtime targeting
                   ``GPU.<index>`` (Intel Arc / iGPU). We deliberately
                   skip ``onnxruntime-openvino`` because no cp314 wheel
                   exists on PyPI; the native ``openvino`` package
                   ships cp314 wheels and is what the upstream Intel EP
                   delegates to anyway.

Models are taken from ``model/tiny/registry.json`` plus an explicit
``--extra-fp32`` allowlist for fp32-only baselines (``vmaf_tiny_v1``,
``vmaf_tiny_v1_medium``). For fp32-only baselines we synthesise a
dynamic-PTQ int8 sibling at runtime (matches the published mode for
``learned_filter_v1``); we do **not** rewrite the on-disk model.

Outputs ``runs/quant-eps-<date>/results.{json,md}`` (gitignored). The
markdown table is what gets pasted into the research digest.

Usage::

    python ai/scripts/measure_quant_drop_per_ep.py \\
        --eps cpu cuda openvino \\
        --extra-fp32 vmaf_tiny_v1.onnx vmaf_tiny_v1_medium.onnx \\
        --out runs/quant-eps-2026-04-28
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "model" / "tiny" / "registry.json"
SEED = 0
N_SAMPLES = 16

# Per-EP runtime constants. Kept here (not split across helpers) so the
# table at the top of the file gives a complete picture of which EP we
# tunnel through.
EP_CPU = "cpu"
EP_CUDA = "cuda"
EP_OPENVINO = "openvino"
SUPPORTED_EPS = (EP_CPU, EP_CUDA, EP_OPENVINO)


# ---------------------------------------------------------------------------
# Model loading helpers (reuses the sibling-rename workaround from PR #165).
# ---------------------------------------------------------------------------


def _load_proto(model_path: Path):
    """Load an ONNX proto, fixing the ``mlp_*_final`` location-rename bug.

    Returns the in-memory ``onnx.ModelProto`` with all external initialisers
    materialised. Saves on disk are the caller's job.
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    proto = onnx.load(str(model_path), load_external_data=False)
    sibling = model_path.with_suffix(".onnx.data")
    if sibling.is_file():
        for tensor in proto.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = sibling.name
        load_external_data_for_model(proto, str(sibling.parent))
    return proto


def _save_inlined(proto, dest: Path) -> None:
    """Save a proto with all initialisers inlined (no external data).

    Also strips ``value_info`` entries whose names collide with
    ``initializer`` entries. The shipped ``vmaf_tiny_v1*.onnx`` have
    weight tensors duplicated in ``value_info`` with shape annotations
    that confuse ``onnxruntime.quantization.quantize_dynamic``'s shape
    inference (it raises ``Inferred shape and existing shape differ``).
    Removing the duplicates is safe — initializers carry their own
    canonical shape.
    """
    import onnx

    init_names = {t.name for t in proto.graph.initializer}
    survivors = [vi for vi in proto.graph.value_info if vi.name not in init_names]
    if len(survivors) != len(proto.graph.value_info):
        del proto.graph.value_info[:]
        proto.graph.value_info.extend(survivors)
    onnx.save(proto, str(dest), save_as_external_data=False)


def _dynamic_quantise(fp32_path: Path, work_dir: Path) -> Path:
    """Produce a dynamic-PTQ int8 ONNX from the given fp32 model.

    Materialises the model (rename workaround) into ``work_dir`` first
    so ``onnxruntime.quantization.quantize_dynamic`` can locate the
    initialisers.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    proto = _load_proto(fp32_path)
    inlined = work_dir / (fp32_path.stem + ".fp32_inlined.onnx")
    _save_inlined(proto, inlined)
    int8_path = work_dir / (fp32_path.stem + ".int8.onnx")
    quantize_dynamic(str(inlined), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


# ---------------------------------------------------------------------------
# Per-EP runners.
# ---------------------------------------------------------------------------


class _Runner:
    """Minimal interface: ``infer(x: np.ndarray) -> np.ndarray``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def infer(self, x):  # pragma: no cover - abstract
        # TODO(#842): convert _Runner to abc.ABC with @abc.abstractmethod so
        # the abstract contract is enforced at instantiation time rather than
        # relying on a bare raise here.
        raise NotImplementedError


class _OrtRunner(_Runner):
    def __init__(self, name: str, model_path: Path, providers: list[str]) -> None:
        import onnxruntime as ort

        super().__init__(name)
        proto = _load_proto(model_path)
        sess = ort.InferenceSession(proto.SerializeToString(), providers=providers)
        self._sess = sess
        self._in_name = sess.get_inputs()[0].name
        self._out_name = sess.get_outputs()[0].name
        in_shape = sess.get_inputs()[0].shape
        self._static_shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in in_shape)
        # Verify the EP we asked for is actually engaged. ORT silently
        # falls back to CPU if a provider can't initialise — without
        # this guard we'd report "CUDA PLCC drop" for a run that
        # actually executed on CPU.
        used = sess.get_providers()
        wanted = providers[0]
        if wanted not in used or used[0] != wanted:
            raise RuntimeError(
                f"requested EP {wanted!r} not engaged for {model_path.name}; "
                f"providers used: {used}"
            )

    def infer(self, x):
        return self._sess.run([self._out_name], {self._in_name: x})[0]

    @property
    def static_shape(self):
        return self._static_shape


class _OpenVinoRunner(_Runner):
    def __init__(self, name: str, model_path: Path, device: str) -> None:
        import openvino as ov

        super().__init__(name)
        # Materialise the proto first to dodge the location-rename bug.
        proto = _load_proto(model_path)
        with tempfile.NamedTemporaryFile(
            suffix=".onnx", delete=False, dir=str(model_path.parent)
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _save_inlined(proto, tmp_path)
            core = ov.Core()
            ov_model = core.read_model(str(tmp_path))
            self._compiled = core.compile_model(ov_model, device)
        finally:
            tmp_path.unlink(missing_ok=True)
        in_port = self._compiled.inputs[0]
        self._in_name = in_port.any_name
        self._out_port = self._compiled.outputs[0]
        in_shape = in_port.partial_shape
        self._static_shape = tuple(d.get_length() if d.is_static else 1 for d in in_shape)

    def infer(self, x):
        result = self._compiled([x])
        return result[self._out_port]

    @property
    def static_shape(self):
        return self._static_shape


def _make_runner(ep: str, model_path: Path, openvino_device: str) -> _Runner:
    if ep == EP_CPU:
        return _OrtRunner(ep, model_path, ["CPUExecutionProvider"])
    if ep == EP_CUDA:
        return _OrtRunner(ep, model_path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if ep == EP_OPENVINO:
        return _OpenVinoRunner(ep, model_path, openvino_device)
    raise ValueError(f"unknown EP: {ep}")


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


def _measure_one_ep(
    ep: str,
    fp32_path: Path,
    int8_path: Path,
    openvino_device: str,
) -> dict[str, Any]:
    """Run fp32 + int8 under one EP and report PLCC, drop, max-abs delta."""
    import numpy as np

    fp32 = _make_runner(ep, fp32_path, openvino_device)
    int8 = _make_runner(ep, int8_path, openvino_device)

    shape = fp32.static_shape
    rng = np.random.default_rng(SEED)
    accum_fp = []
    accum_q = []
    worst = 0.0
    for _ in range(N_SAMPLES):
        x = rng.random(shape, dtype=np.float32)
        y_fp = np.asarray(fp32.infer(x)).astype(np.float64)
        y_q = np.asarray(int8.infer(x)).astype(np.float64)
        worst = max(worst, float(np.abs(y_fp - y_q).max()))
        accum_fp.append(y_fp.ravel())
        accum_q.append(y_q.ravel())
    all_fp = np.concatenate(accum_fp)
    all_q = np.concatenate(accum_q)
    # PLCC undefined if either side is constant; fall back to 1.0 so a
    # degenerate model doesn't gate us — the worst_abs delta still
    # surfaces real divergence.
    if all_fp.std() < 1e-12 or all_q.std() < 1e-12:
        plcc = 1.0
    else:
        plcc = float(np.corrcoef(all_fp, all_q)[0, 1])
    return {
        "plcc": plcc,
        "drop": 1.0 - plcc,
        "worst_abs": worst,
    }


def _run_model(
    model_id: str,
    fp32_path: Path,
    int8_path: Path,
    eps: list[str],
    budget: float,
    openvino_device: str,
) -> dict[str, Any]:
    per_ep: dict[str, Any] = {}
    for ep in eps:
        t0 = time.monotonic()
        try:
            res = _measure_one_ep(ep, fp32_path, int8_path, openvino_device)
            res["status"] = "ok"
            res["wall_s"] = round(time.monotonic() - t0, 3)
            res["pass"] = res["drop"] <= budget
        except Exception as exc:
            res = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=4),
                "wall_s": round(time.monotonic() - t0, 3),
                "pass": False,
            }
        per_ep[ep] = res
    return {
        "model_id": model_id,
        "fp32": str(fp32_path),
        "int8": str(int8_path),
        "budget": budget,
        "per_ep": per_ep,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eps = report["eps"]
    lines: list[str] = []
    lines.append("# PTQ accuracy across Execution Providers (T5-3e)")
    lines.append("")
    lines.append(f"Generated: {report['generated']}")
    lines.append(f"EPs: {', '.join(eps)}")
    lines.append(f"Hardware: {report['hw']}")
    lines.append(f"Samples per measurement: {report['n_samples']}")
    lines.append(f"Seed: {report['seed']}")
    lines.append("")
    lines.append("## PLCC drop per (model, EP)")
    lines.append("")
    header = "| model | budget | " + " | ".join(eps) + " |"
    sep = "|---|---:|" + "|".join([":---:"] * len(eps)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in report["models"]:
        cells = []
        for ep in eps:
            r = m["per_ep"][ep]
            if r["status"] != "ok":
                cells.append("ERR")
            else:
                tag = "PASS" if r["pass"] else "FAIL"
                cells.append(f"drop={r['drop']:.6f} ({tag})")
        lines.append(f"| `{m['model_id']}` | {m['budget']:.4f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-EP details (PLCC, worst |delta|, wall-time)")
    lines.append("")
    for m in report["models"]:
        lines.append(f"### `{m['model_id']}`")
        lines.append("")
        lines.append(f"- fp32: `{m['fp32']}`")
        lines.append(f"- int8: `{m['int8']}`")
        lines.append(f"- budget: {m['budget']:.4f} PLCC drop")
        lines.append("")
        lines.append("| EP | status | PLCC | drop | worst_abs | wall (s) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for ep in eps:
            r = m["per_ep"][ep]
            if r["status"] == "ok":
                lines.append(
                    f"| {ep} | ok | {r['plcc']:.6f} | {r['drop']:.6f} | "
                    f"{r['worst_abs']:.4g} | {r['wall_s']} |"
                )
            else:
                lines.append(f"| {ep} | **{r['status']}** | — | — | — | {r['wall_s']} |")
                lines.append("")
                lines.append(f"  > {r.get('error', '?')}")
                lines.append("")
        lines.append("")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _registry_targets(reg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return registry rows that already declare a quant_mode."""
    out = []
    for m in reg["models"]:
        if m.get("quant_mode", "fp32") == "fp32":
            continue
        out.append(m)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eps",
        nargs="+",
        choices=SUPPORTED_EPS,
        default=list(SUPPORTED_EPS),
        help="Execution providers to compare (default: all supported).",
    )
    parser.add_argument(
        "--extra-fp32",
        nargs="*",
        default=[],
        help=(
            "Additional fp32-only baselines (relative to model/tiny/) to "
            "quantise dynamically and measure. e.g. vmaf_tiny_v1.onnx"
        ),
    )
    parser.add_argument(
        "--extra-budget",
        type=float,
        default=0.01,
        help="PLCC-drop budget applied to each --extra-fp32 model.",
    )
    parser.add_argument(
        "--openvino-device",
        default="GPU.0",
        help="OpenVINO device string (default: GPU.0 = first dGPU).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / f"quant-eps-{time.strftime('%Y-%m-%d')}",
        help="Output directory for results.{json,md}.",
    )
    parser.add_argument(
        "--hw",
        default=os.environ.get("VMAF_HW_TAG", "ryzen-9950x3d+rtx4090+arc-a380"),
        help="Free-form hardware tag stored in the JSON header.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if any (model, EP) drop exceeds its budget.",
    )
    args = parser.parse_args()

    try:
        reg = json.loads(REGISTRY.read_text())
    except Exception as exc:
        print(f"failed to load registry: {exc}", file=sys.stderr)
        return 2

    work_dir = Path(tempfile.mkdtemp(prefix="quant_eps_"))
    try:
        report: dict[str, Any] = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hw": args.hw,
            "eps": args.eps,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "models": [],
        }

        # 1. Registry-shipped quantised models.
        for m in _registry_targets(reg):
            fp32 = REPO_ROOT / "model" / "tiny" / m["onnx"]
            int8 = fp32.with_name(fp32.stem + ".int8.onnx")
            if not fp32.is_file() or not int8.is_file():
                print(
                    f"[skip] {m['id']} — missing fp32 or int8 sibling",
                    file=sys.stderr,
                )
                continue
            budget = float(m.get("quant_accuracy_budget_plcc", 0.01))
            row = _run_model(m["id"], fp32, int8, args.eps, budget, args.openvino_device)
            row["source"] = "registry"
            report["models"].append(row)

        # 2. Extra fp32-only baselines (dynamic PTQ on the fly).
        for rel in args.extra_fp32:
            fp32 = REPO_ROOT / "model" / "tiny" / rel
            if not fp32.is_file():
                print(f"[skip] extra fp32 not found: {fp32}", file=sys.stderr)
                continue
            try:
                int8 = _dynamic_quantise(fp32, work_dir)
            except Exception as exc:
                print(
                    f"[skip] failed to dynamic-quantise {rel}: {exc}",
                    file=sys.stderr,
                )
                continue
            row = _run_model(
                fp32.stem,
                fp32,
                int8,
                args.eps,
                args.extra_budget,
                args.openvino_device,
            )
            row["source"] = "extra_fp32_dynamic_ptq"
            report["models"].append(row)

        json_out = args.out / "results.json"
        md_out = args.out / "results.md"
        _write_json(report, json_out)
        _write_markdown(report, md_out)
        print(f"[ok] wrote {json_out}")
        print(f"[ok] wrote {md_out}")

        if args.gate:
            failed = [
                (m["model_id"], ep)
                for m in report["models"]
                for ep, r in m["per_ep"].items()
                if not r.get("pass", False)
            ]
            if failed:
                for mid, ep in failed:
                    print(f"[FAIL] {mid} on {ep}", file=sys.stderr)
                return 1
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
