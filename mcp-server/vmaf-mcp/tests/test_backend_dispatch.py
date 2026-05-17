# Copyright 2026 Lusoris and Claude (Anthropic)
"""Verify ``_run_vmaf_score`` emits the correct ``--no_<backend>`` flag set
for each backend selector — including ``vulkan``, ``hip``, ``metal``, which
were missing before this PR and silently fell through to ``auto``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from vmaf_mcp import server as srv


def _make_score_request(backend: str) -> srv.ScoreRequest:
    return srv.ScoreRequest(
        ref=Path("/tmp/ref.yuv"),
        dis=Path("/tmp/dis.yuv"),
        width=576,
        height=324,
        pixfmt="420",
        bitdepth=8,
        model="version=vmaf_v0.6.1",
        backend=backend,
        precision="17",
    )


def _argv_from_call(mock_exec) -> list[str]:
    # asyncio.create_subprocess_exec was called once; positional args are argv.
    assert mock_exec.call_count == 1
    return list(mock_exec.call_args.args)


@pytest.mark.parametrize(
    "backend,expected_no_flags",
    [
        (
            "cpu",
            {"--no_cuda", "--no_sycl", "--no_vulkan", "--no_hip", "--no_metal"},
        ),
        ("cuda", {"--no_sycl", "--no_vulkan", "--no_hip", "--no_metal"}),
        ("sycl", {"--no_cuda", "--no_vulkan", "--no_hip", "--no_metal"}),
        ("vulkan", {"--no_cuda", "--no_sycl", "--no_hip", "--no_metal"}),
        ("hip", {"--no_cuda", "--no_sycl", "--no_vulkan", "--no_metal"}),
        ("metal", {"--no_cuda", "--no_sycl", "--no_vulkan", "--no_hip"}),
        ("auto", set()),
    ],
)
def test_run_vmaf_score_emits_expected_no_flags(
    backend: str, expected_no_flags: set[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each backend selector disables every sibling backend in argv. ``auto``
    leaves the host probe alone."""

    # Fake binary that exists (skips the early "build first" raise).
    fake_vmaf = tmp_path / "vmaf"
    fake_vmaf.write_text("#!/bin/sh\nexit 0\n")
    fake_vmaf.chmod(0o755)
    monkeypatch.setattr(srv, "_vmaf_binary", lambda: fake_vmaf)

    # Fake output payload — _run_vmaf_score reads the JSON, so write one
    # before the function reads it. We capture argv via the mocked exec.
    fake_output = tmp_path / "vmaf-out.json"
    fake_output.write_text('{"pooled_metrics": {"vmaf": {"mean": 90.0}}}')

    async def _fake_communicate(*args, **kwargs):
        return (b"", b"")

    fake_proc = AsyncMock()
    fake_proc.communicate = _fake_communicate
    fake_proc.returncode = 0

    with (
        patch.object(
            srv.asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc)
        ) as mock_exec,
        patch.object(
            srv.Path, "read_text", return_value='{"pooled_metrics": {"vmaf": {"mean": 90.0}}}'
        ),
        patch.object(srv.Path, "unlink", return_value=None),
    ):
        req = _make_score_request(backend)
        asyncio.run(srv._run_vmaf_score(req))
        argv = _argv_from_call(mock_exec)

    # Every "--no_*" flag in argv must be exactly the expected set.
    observed_no_flags = {tok for tok in argv if tok.startswith("--no_")}
    assert observed_no_flags == expected_no_flags, (
        f"backend={backend!r}: expected {expected_no_flags}, observed {observed_no_flags}, "
        f"argv={argv}"
    )


def test_list_backends_includes_vulkan_hip_metal_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_list_backends`` returns a dict with all 6 backend keys, even when
    the local vmaf binary is missing (defensive default).

    CPU is always True — it requires no GPU driver.  Every GPU backend is
    False when the binary is absent (we cannot probe capabilities).
    """
    monkeypatch.setattr(srv, "_vmaf_binary", lambda: Path("/does/not/exist/vmaf"))
    result = srv._list_backends()
    assert set(result.keys()) == {"cpu", "cuda", "sycl", "vulkan", "hip", "metal"}
    # CPU must be True regardless of binary presence.
    assert result["cpu"] is True, result
    # GPU backends are all False when the binary is missing.
    gpu_backends = {k: v for k, v in result.items() if k != "cpu"}
    assert all(v is False for v in gpu_backends.values()), result
