"""MCP server for the Lusoris VMAF fork.

Exposes four tools over the Model Context Protocol (stdio transport):

- ``vmaf_score``       — score a (reference, distorted) pair.
- ``list_models``      — enumerate the VMAF models registered with the build.
- ``list_backends``    — report which backends (cpu/cuda/sycl) are available.
- ``run_benchmark``    — run the Netflix benchmark harness on a pair.

The server assumes ``build/tools/vmaf`` exists (build first with
``meson compile -C build``). Paths are validated to live under either the
repository's ``testdata/`` / ``python/test/resource/`` trees or an
explicitly-allowlisted prefix passed via ``VMAF_MCP_ALLOW``. This prevents
callers from coercing the server into reading arbitrary host paths.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess  # noqa: S404 — we exec our own signed vmaf binary, inputs are validated
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
            f"vmaf binary not found at {vmaf}. "
            "Build first: meson compile -C build."
        )

    output = Path("/tmp") / f"vmaf-mcp-{os.getpid()}-{asyncio.current_task().get_name()}.json"
    try:
        argv = [
            str(vmaf),
            "-r", str(req.ref),
            "-d", str(req.dis),
            "--width", str(req.width),
            "--height", str(req.height),
            "-p", req.pixfmt,
            "-b", str(req.bitdepth),
            "-m", req.model,
            "--precision", req.precision,
            "-q",
            "-o", str(output),
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
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"vmaf exited {proc.returncode}: {stderr.decode(errors='replace')}"
            )
        return json.loads(output.read_text())
    finally:
        output.unlink(missing_ok=True)


def _list_models() -> list[dict[str, Any]]:
    models_dir = _repo_root() / "model"
    out: list[dict[str, Any]] = []
    for p in sorted(models_dir.rglob("*")):
        if p.suffix in {".json", ".pkl", ".onnx"} and p.is_file():
            out.append({
                "name": p.stem,
                "path": str(p.relative_to(_repo_root())),
                "format": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
            })
    return out


def _list_backends() -> dict[str, bool]:
    vmaf = _vmaf_binary()
    if not vmaf.exists():
        return {"cpu": False, "cuda": False, "sycl": False, "hip": False}
    try:
        result = subprocess.run(  # noqa: S603
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


async def _run_benchmark(ref: Path, dis: Path, width: int, height: int) -> dict[str, Any]:
    script = _repo_root() / "testdata" / "bench_all.sh"
    if not script.exists():
        raise FileNotFoundError(f"benchmark harness not found: {script}")
    proc = await asyncio.create_subprocess_exec(
        str(script), "-r", str(ref), "-d", str(dis),
        "--width", str(width), "--height", str(height),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
                    "ref":       {"type": "string", "description": "Reference YUV path."},
                    "dis":       {"type": "string", "description": "Distorted YUV path."},
                    "width":     {"type": "integer", "minimum": 1},
                    "height":    {"type": "integer", "minimum": 1},
                    "pixfmt":    {"type": "string", "enum": ["420", "422", "444"]},
                    "bitdepth":  {"type": "integer", "enum": [8, 10, 12, 16]},
                    "model":     {"type": "string", "default": "version=vmaf_v0.6.1"},
                    "backend":   {"type": "string", "enum": ["auto", "cpu", "cuda", "sycl"], "default": "auto"},
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
                    "ref":    {"type": "string"},
                    "dis":    {"type": "string"},
                    "width":  {"type": "integer"},
                    "height": {"type": "integer"},
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
        else:
            raise ValueError(f"unknown tool: {name}")
    except Exception as exc:  # noqa: BLE001
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
