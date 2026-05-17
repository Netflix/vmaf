"""MCP server for the Lusoris VMAF fork.

Exposes seven tools over the Model Context Protocol (stdio transport):

- ``vmaf_score``            — score a (reference, distorted) pair.
- ``list_models``           — enumerate the VMAF models registered with the build.
- ``list_backends``         — report which backends (cpu/cuda/sycl) are available.
- ``run_benchmark``         — run the Netflix benchmark harness on a pair.
- ``eval_model_on_split``   — run an ONNX tiny-AI model against a parquet feature
  cache on a deterministic split and report PLCC/SROCC/RMSE.
- ``compare_models``        — rank several ONNX models on the same split.
- ``describe_worst_frames`` — score a pair, pick the N worst-VMAF frames, and
  describe the visible artefacts via SmolVLM / Moondream2 (ADR-0172 / T6-6;
  requires the ``vlm`` extras for actual descriptions, otherwise returns
  frame metadata only).

The server assumes ``build/tools/vmaf`` exists (build first with
``meson compile -C build``). Paths are validated to live under either the
repository's ``testdata/`` / ``python/test/resource/`` / ``model/`` trees
or an explicitly-allowlisted prefix passed via ``VMAF_MCP_ALLOW``. This
prevents callers from coercing the server into reading arbitrary host
paths.
"""

from __future__ import annotations

# NOTE (risk-accept): the `subprocess` import below exec's our own signed
# vmaf binary with an argv list (no shell=True, no user-controlled
# strings in argv[0]); broad exception handlers on the call paths
# convert failures into JSON-RPC errors for the client. If ruff `select`
# is ever widened to include the bandit (`S`) or blind-except (`BLE`)
# rules, re-evaluate these sites deliberately rather than silencing
# with line-level suppression markers.
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Configuration & path validation
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _vmaf_binary() -> Path:
    env = os.environ.get("VMAF_BIN")
    if env:
        return Path(env)
    return _repo_root() / "build" / "tools" / "vmaf"


def _allowed_roots() -> list[Path]:
    roots = [
        _repo_root() / "testdata",
        _repo_root() / "python" / "test" / "resource",
        _repo_root() / "model",
    ]
    extra = os.environ.get("VMAF_MCP_ALLOW")
    if extra:
        roots.extend(Path(p).resolve() for p in extra.split(":") if p)
    return [r.resolve() for r in roots]


def _validate_path(p: str) -> Path:
    path = Path(p).resolve()
    allowed = _allowed_roots()
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError(
            f"path {path} not under an allowlisted root; set VMAF_MCP_ALLOW to extend."
        )
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreRequest:
    ref: Path
    dis: Path
    width: int
    height: int
    pixfmt: str  # "420" | "422" | "444"
    bitdepth: int
    model: str = "version=vmaf_v0.6.1"
    backend: str = "auto"  # "cpu" | "cuda" | "sycl" | "auto"
    precision: str = "17"


async def _run_vmaf_score(req: ScoreRequest) -> dict[str, Any]:
    vmaf = _vmaf_binary()
    if not vmaf.exists():
        raise RuntimeError(
            f"vmaf binary not found at {vmaf}. " "Build first: meson compile -C build."
        )

    output = Path("/tmp") / f"vmaf-mcp-{os.getpid()}-{asyncio.current_task().get_name()}.json"
    try:
        argv = [
            str(vmaf),
            "-r",
            str(req.ref),
            "-d",
            str(req.dis),
            "--width",
            str(req.width),
            "--height",
            str(req.height),
            "-p",
            req.pixfmt,
            "-b",
            str(req.bitdepth),
            "-m",
            req.model,
            "--precision",
            req.precision,
            "-q",
            "-o",
            str(output),
            "--json",
        ]
        if req.backend == "cpu":
            argv.extend(["--no_cuda", "--no_sycl"])
        elif req.backend == "cuda":
            argv.extend(["--no_sycl"])
        elif req.backend == "sycl":
            argv.extend(["--no_cuda"])

        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"vmaf exited {proc.returncode}: {stderr.decode(errors='replace')}")
        return json.loads(output.read_text())
    finally:
        output.unlink(missing_ok=True)


def _list_models() -> list[dict[str, Any]]:
    models_dir = _repo_root() / "model"
    out: list[dict[str, Any]] = []
    for p in sorted(models_dir.rglob("*")):
        if p.suffix in {".json", ".pkl", ".onnx"} and p.is_file():
            out.append(
                {
                    "name": p.stem,
                    "path": str(p.relative_to(_repo_root())),
                    "format": p.suffix.lstrip("."),
                    "size_bytes": p.stat().st_size,
                }
            )
    return out


def _list_backends() -> dict[str, bool]:
    vmaf = _vmaf_binary()
    if not vmaf.exists():
        return {"cpu": False, "cuda": False, "sycl": False, "hip": False}
    try:
        result = subprocess.run(
            [str(vmaf), "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        blob = (result.stdout + result.stderr).lower()
    except (subprocess.TimeoutExpired, OSError):
        blob = ""
    return {
        "cpu": True,
        "cuda": "cuda" in blob,
        "sycl": "sycl" in blob or "oneapi" in blob,
        "hip": "hip" in blob,
    }


_FEATURE_COLUMNS = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)
_VALID_SPLITS = ("train", "val", "test", "all")


def _eval_model_on_split(
    model: Path, features: Path, split: str, input_name: str
) -> dict[str, Any]:
    """Run @p model on @p split of @p features and return PLCC/SROCC/RMSE.

    Imports are lazy so the base mcp-server install (no pandas / onnxruntime
    / scipy) isn't forced to pull in ML deps just to score video.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}; got {split!r}")
    try:
        import numpy as np
        import onnxruntime as ort
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr
    except ImportError as exc:  # pragma: no cover — exercised only without extras
        raise RuntimeError(
            "eval_model_on_split requires the 'eval' extra: " "pip install 'vmaf-mcp[eval]'"
        ) from exc

    df = pd.read_parquet(features)
    if "mos" not in df.columns:
        raise ValueError(f"{features} has no 'mos' column — can't score correlations")
    if split != "all" and "key" in df.columns:
        # Inline the split_keys hashing so we don't depend on vmaf_train.
        import hashlib

        def bucket(key: str) -> float:
            h = hashlib.sha256(f"vmaf-train-splits-v1:{key}".encode()).digest()
            return int.from_bytes(h[:8], "big") / (1 << 64)

        val_frac, test_frac = 0.1, 0.1

        def which(key: str) -> str:
            b = bucket(str(key))
            if b < test_frac:
                return "test"
            if b < test_frac + val_frac:
                return "val"
            return "train"

        keep = df["key"].astype(str).map(which) == split
        df = df[keep]

    cols = [c for c in _FEATURE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(
            f"{features} has none of the expected feature columns "
            f"{_FEATURE_COLUMNS}; got {list(df.columns)}"
        )
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["mos"].to_numpy(dtype=np.float32)
    if len(x) < 2:
        raise ValueError(f"split {split!r} has {len(x)} samples — need ≥2 to compute correlations")

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    pred = np.asarray(sess.run(None, {input_name: x})[0]).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"model output shape {pred.shape} does not match target shape {y.shape}")
    plcc = float(pearsonr(pred, y).statistic)
    srocc = float(spearmanr(pred, y).statistic)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    return {
        "model": str(model),
        "features": str(features),
        "split": split,
        "n": len(x),
        "plcc": plcc,
        "srocc": srocc,
        "rmse": rmse,
        "columns": cols,
    }


def _compare_models(
    models: list[Path], features: Path, split: str, input_name: str
) -> dict[str, Any]:
    """Rank @p models on the same feature split by descending PLCC."""
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for m in models:
        try:
            reports.append(_eval_model_on_split(m, features, split, input_name))
        except Exception as exc:
            errors.append({"model": str(m), "error": str(exc)})
    reports.sort(key=lambda r: r["plcc"], reverse=True)
    return {"ranked": reports, "errors": errors}


# ---------------------------------------------------------------------------
# describe_worst_frames — VLM-assisted artefact triage (ADR-0172 / T6-6)
# ---------------------------------------------------------------------------


_VLM_PROMPT = (
    "Describe what visible compression / encoding artefacts you see in this video "
    "frame in 1-2 sentences. Focus on blocking, ringing, banding, blur, or chroma "
    "distortion if present. Skip aesthetic commentary."
)

# Cached VLM pipeline (model + processor). Populated by `_load_vlm()` on
# first call and then reused. None means "unavailable / disabled".
_vlm_state: dict[str, Any] = {"loaded": False, "pipeline": None, "model_id": None}


def _load_vlm() -> tuple[Any, str] | None:
    """Lazy-import transformers and load the smallest available VLM.

    Returns ``(pipeline, model_id)`` on success, ``None`` if the
    ``vlm`` extras aren't installed or both candidate models fail to
    load. Cached across calls — the first load is slow, subsequent
    calls hit the in-memory state.
    """
    if _vlm_state["loaded"]:
        return (_vlm_state["pipeline"], _vlm_state["model_id"]) if _vlm_state["pipeline"] else None
    _vlm_state["loaded"] = True

    try:
        import torch  # noqa: F401
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return None

    candidates = (
        "HuggingFaceTB/SmolVLM-Instruct",  # ~2 GB, OK on CPU
        "vikhyatk/moondream2",  # ~2 GB, well-known fallback
    )
    for model_id in candidates:
        try:
            pipe = hf_pipeline(
                "image-to-text",
                model=model_id,
                trust_remote_code=True,
            )
            _vlm_state["pipeline"] = pipe
            _vlm_state["model_id"] = model_id
            return (pipe, model_id)
        except Exception:  # pragma: no cover - depends on local env
            continue
    return None


def _describe_image_with_vlm(image_path: Path) -> str:
    """Run the cached VLM on @p image_path. Returns "(VLM unavailable)" when
    the ``vlm`` extras are missing or no candidate model loaded."""
    loaded = _load_vlm()
    if not loaded:
        return "(VLM unavailable — install with `pip install vmaf-mcp[vlm]`)"
    pipe, _model_id = loaded
    try:
        out = pipe(str(image_path), prompt=_VLM_PROMPT)
    except TypeError:
        # Older transformers don't accept `prompt=` for image-to-text;
        # the model defaults to its training caption prompt.
        out = pipe(str(image_path))
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return str(out[0].get("generated_text") or out[0].get("text") or out[0]).strip()
    return str(out).strip()


async def _extract_frame_png(
    yuv: Path,
    *,
    width: int,
    height: int,
    pixfmt: str,
    bitdepth: int,
    frame_index: int,
    out_png: Path,
) -> None:
    """Extract a single distorted frame from a raw YUV file as PNG via
    ffmpeg. We grab a generous slice (frame_index..+1) and select the
    last frame — robust to ffmpeg's seek inaccuracy on raw YUV inputs."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH; install ffmpeg to use describe_worst_frames")
    fmt_map = {
        ("420", 8): "yuv420p",
        ("422", 8): "yuv422p",
        ("444", 8): "yuv444p",
        ("420", 10): "yuv420p10le",
        ("422", 10): "yuv422p10le",
        ("444", 10): "yuv444p10le",
    }
    pix_fmt = fmt_map.get((pixfmt, bitdepth))
    if not pix_fmt:
        raise ValueError(f"unsupported pixfmt/bitdepth combo: {pixfmt}/{bitdepth}")
    argv = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-i",
        str(yuv),
        "-vf",
        f"select='eq(n,{frame_index})'",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-y",
        str(out_png),
    ]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame-extract failed: {stderr.decode(errors='replace')}")


def _pick_worst_frames(score_json: dict[str, Any], n: int) -> list[tuple[int, float]]:
    """Walk the per-frame array in @p score_json and return the @p n
    frames with lowest VMAF, sorted ascending by score."""
    frames = score_json.get("frames") or []
    scored: list[tuple[int, float]] = []
    for f in frames:
        idx = f.get("frameNum")
        metrics = f.get("metrics") or {}
        # libvmaf reports the headline score under "vmaf" or "vmaf_v0.6.1"
        # (model-name-dependent). Try a few common keys.
        score = None
        for key in ("vmaf", "vmaf_v0.6.1", "vmaf_v0.6.1neg", "vmaf_4k_v0.6.1"):
            if key in metrics:
                score = float(metrics[key])
                break
        if idx is None or score is None:
            continue
        scored.append((int(idx), score))
    scored.sort(key=lambda kv: kv[1])
    return scored[: max(0, int(n))]


async def _describe_worst_frames(
    req: "ScoreRequest", *, n: int, describe: Any | None = None
) -> dict[str, Any]:
    """Score the pair, pick the @p n worst-VMAF frames, extract each as
    a PNG, and run the VLM (or @p describe override for tests). Returns
    a JSON-able dict: {model_id, frames: [{frame_index, vmaf, png, description}]}.
    """
    # First — score the pair so we have per-frame VMAF.
    score = await _run_vmaf_score(req)
    worst = _pick_worst_frames(score, n)

    descr_fn = describe if describe is not None else _describe_image_with_vlm
    out_frames: list[dict[str, Any]] = []
    tmp_root = Path("/tmp") / f"vmaf-mcp-worst-{os.getpid()}"
    # Clear stale PNGs left by any previous invocation — the comment in the
    # original code said "clear the dir on next invocation" but never
    # implemented it, causing unbounded disk accumulation on long-running
    # servers (T-ROUND8-MCP-TMPDIR-LEAK). PNGs are only useful for the
    # duration of this response, so purge-before-generate is safe.
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    try:
        for frame_idx, vmaf in worst:
            png_path = tmp_root / f"frame_{frame_idx:06d}.png"
            await _extract_frame_png(
                req.dis,
                width=req.width,
                height=req.height,
                pixfmt=req.pixfmt,
                bitdepth=req.bitdepth,
                frame_index=frame_idx,
                out_png=png_path,
            )
            description = descr_fn(png_path)
            out_frames.append(
                {
                    "frame_index": frame_idx,
                    "vmaf": vmaf,
                    "png": str(png_path),
                    "description": description,
                }
            )
    finally:
        # PNGs remain on disk so that callers who need the file path
        # (e.g. a downstream tool that opens the PNG directly) can access
        # them until the next describe_worst_frames call purges the
        # directory (see rmtree above).  The next call, or process exit,
        # cleans up automatically.
        pass
    return {
        "model_id": _vlm_state.get("model_id"),
        "frames": out_frames,
    }


async def _run_benchmark(ref: Path, dis: Path, width: int, height: int) -> dict[str, Any]:
    script = _repo_root() / "testdata" / "bench_all.sh"
    if not script.exists():
        raise FileNotFoundError(f"benchmark harness not found: {script}")
    proc = await asyncio.create_subprocess_exec(
        str(script),
        "-r",
        str(ref),
        "-d",
        str(dis),
        "--width",
        str(width),
        "--height",
        str(height),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------


server: Server = Server("vmaf-mcp")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name="vmaf_score",
            description="Compute a VMAF score for a (reference, distorted) YUV pair.",
            inputSchema={
                "type": "object",
                "required": ["ref", "dis", "width", "height", "pixfmt", "bitdepth"],
                "properties": {
                    "ref": {"type": "string", "description": "Reference YUV path."},
                    "dis": {"type": "string", "description": "Distorted YUV path."},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "pixfmt": {"type": "string", "enum": ["420", "422", "444"]},
                    "bitdepth": {"type": "integer", "enum": [8, 10, 12, 16]},
                    "model": {"type": "string", "default": "version=vmaf_v0.6.1"},
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "cpu", "cuda", "sycl"],
                        "default": "auto",
                    },
                    "precision": {"type": "string", "default": "17"},
                },
            },
        ),
        Tool(
            name="list_models",
            description="Enumerate VMAF models (JSON / pickle / ONNX) shipped with the repo.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_backends",
            description="Report which runtime backends (cpu / cuda / sycl / hip) the local vmaf binary was built with.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="run_benchmark",
            description="Run the Netflix benchmark harness on a pair and return stdout/stderr.",
            inputSchema={
                "type": "object",
                "required": ["ref", "dis", "width", "height"],
                "properties": {
                    "ref": {"type": "string"},
                    "dis": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
        ),
        Tool(
            name="eval_model_on_split",
            description=(
                "Run an ONNX tiny-AI regressor on a parquet feature cache, "
                "filter to a deterministic train/val/test split (keyed by the "
                "'key' column), and report PLCC / SROCC / RMSE."
            ),
            inputSchema={
                "type": "object",
                "required": ["model", "features"],
                "properties": {
                    "model": {"type": "string", "description": "ONNX model path."},
                    "features": {"type": "string", "description": "Parquet feature cache path."},
                    "split": {"type": "string", "enum": list(_VALID_SPLITS), "default": "test"},
                    "input_name": {"type": "string", "default": "features"},
                },
            },
        ),
        Tool(
            name="compare_models",
            description=(
                "Rank several ONNX models on the same parquet feature split by "
                "descending PLCC. Models that fail to load or score are listed "
                "under 'errors' instead of aborting the whole call."
            ),
            inputSchema={
                "type": "object",
                "required": ["models", "features"],
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "features": {"type": "string"},
                    "split": {"type": "string", "enum": list(_VALID_SPLITS), "default": "test"},
                    "input_name": {"type": "string", "default": "features"},
                },
            },
        ),
        Tool(
            name="describe_worst_frames",
            description=(
                "Score a (ref, dis) pair, pick the N worst-VMAF frames, extract "
                "each as PNG via ffmpeg, and run a vision-language model "
                "(SmolVLM → Moondream2 fallback) to describe the visible "
                "artefacts. Falls back to metadata-only output when the [vlm] "
                "extras are not installed. ADR-0172 / T6-6."
            ),
            inputSchema={
                "type": "object",
                "required": ["ref", "dis", "width", "height", "pixfmt", "bitdepth"],
                "properties": {
                    "ref": {"type": "string"},
                    "dis": {"type": "string"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "pixfmt": {"type": "string", "enum": ["420", "422", "444"]},
                    "bitdepth": {"type": "integer", "enum": [8, 10, 12, 16]},
                    "model": {"type": "string", "default": "version=vmaf_v0.6.1"},
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "cpu", "cuda", "sycl"],
                        "default": "auto",
                    },
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 5,
                        "description": "How many worst-VMAF frames to describe.",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "vmaf_score":
            req = ScoreRequest(
                ref=_validate_path(arguments["ref"]),
                dis=_validate_path(arguments["dis"]),
                width=int(arguments["width"]),
                height=int(arguments["height"]),
                pixfmt=str(arguments["pixfmt"]),
                bitdepth=int(arguments["bitdepth"]),
                model=str(arguments.get("model", "version=vmaf_v0.6.1")),
                backend=str(arguments.get("backend", "auto")),
                precision=str(arguments.get("precision", "17")),
            )
            result = await _run_vmaf_score(req)
        elif name == "list_models":
            result = {"models": _list_models()}
        elif name == "list_backends":
            result = _list_backends()
        elif name == "run_benchmark":
            result = await _run_benchmark(
                ref=_validate_path(arguments["ref"]),
                dis=_validate_path(arguments["dis"]),
                width=int(arguments["width"]),
                height=int(arguments["height"]),
            )
        elif name == "eval_model_on_split":
            result = _eval_model_on_split(
                model=_validate_path(arguments["model"]),
                features=_validate_path(arguments["features"]),
                split=str(arguments.get("split", "test")),
                input_name=str(arguments.get("input_name", "features")),
            )
        elif name == "compare_models":
            models_in = arguments["models"]
            if not isinstance(models_in, list) or not models_in:
                raise ValueError("'models' must be a non-empty list of paths")
            result = _compare_models(
                models=[_validate_path(m) for m in models_in],
                features=_validate_path(arguments["features"]),
                split=str(arguments.get("split", "test")),
                input_name=str(arguments.get("input_name", "features")),
            )
        elif name == "describe_worst_frames":
            req = ScoreRequest(
                ref=_validate_path(arguments["ref"]),
                dis=_validate_path(arguments["dis"]),
                width=int(arguments["width"]),
                height=int(arguments["height"]),
                pixfmt=str(arguments["pixfmt"]),
                bitdepth=int(arguments["bitdepth"]),
                model=str(arguments.get("model", "version=vmaf_v0.6.1")),
                backend=str(arguments.get("backend", "auto")),
            )
            result = await _describe_worst_frames(req, n=int(arguments.get("n", 5)))
        else:
            raise ValueError(f"unknown tool: {name}")
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _run() -> None:
    if not shutil.which("meson"):
        print("warning: meson not on PATH — benchmark tool may fail.", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio_impl = os.environ.get("VMAF_MCP_ASYNC", "asyncio")
    if anyio_impl == "asyncio":
        asyncio.run(_run())
    else:
        import anyio

        anyio.run(_run, backend=anyio_impl)


if __name__ == "__main__":
    main()
