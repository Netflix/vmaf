"""Tests for B1 (_vmaf_binary path resolution) and B2 (_run_benchmark VMAF_ROOT)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from vmaf_mcp import server as srv

# ---------------------------------------------------------------------------
# B1: _vmaf_binary() path resolution
# ---------------------------------------------------------------------------


def test_vmaf_binary_honours_env_var(monkeypatch, tmp_path):
    """VMAF_BIN env var is the highest-priority override."""
    fake_bin = tmp_path / "my-vmaf"
    fake_bin.touch()
    monkeypatch.setenv("VMAF_BIN", str(fake_bin))
    assert srv._vmaf_binary() == fake_bin


def test_vmaf_binary_prefers_usr_local(monkeypatch, tmp_path):
    """When /usr/local/bin/vmaf exists it is preferred over in-tree builds."""
    monkeypatch.delenv("VMAF_BIN", raising=False)
    if Path("/usr/local/bin/vmaf").exists():
        result = srv._vmaf_binary()
        assert result == Path("/usr/local/bin/vmaf")
    else:
        monkeypatch.setattr(srv, "_repo_root", lambda: tmp_path)
        result = srv._vmaf_binary()
        assert "libvmaf" in str(result)


def test_vmaf_binary_falls_back_to_libvmaf_build(monkeypatch, tmp_path):
    """When /usr/local/bin/vmaf is absent the libvmaf/build path is next."""
    monkeypatch.delenv("VMAF_BIN", raising=False)
    monkeypatch.setattr(srv, "_repo_root", lambda: tmp_path)

    libvmaf_bin = tmp_path / "libvmaf" / "build" / "tools" / "vmaf"
    libvmaf_bin.parent.mkdir(parents=True)
    libvmaf_bin.touch()

    original_exists = Path.exists

    def patched_exists(self):
        if str(self) == "/usr/local/bin/vmaf":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", patched_exists)

    result = srv._vmaf_binary()
    assert result == libvmaf_bin


def test_vmaf_binary_falls_back_to_libvmaf_when_all_absent(monkeypatch, tmp_path):
    """When no candidate exists, returns libvmaf/build path as the best guess."""
    monkeypatch.delenv("VMAF_BIN", raising=False)
    monkeypatch.setattr(srv, "_repo_root", lambda: tmp_path)

    original_exists = Path.exists

    def patched_exists(self):
        if str(self) == "/usr/local/bin/vmaf" or str(self).startswith(str(tmp_path)):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", patched_exists)

    result = srv._vmaf_binary()
    assert "libvmaf" in str(result) and "build" in str(result)


# ---------------------------------------------------------------------------
# B2: _run_benchmark passes VMAF_ROOT to the subprocess
# ---------------------------------------------------------------------------


def test_run_benchmark_injects_vmaf_root(monkeypatch, tmp_path):
    """_run_benchmark must pass VMAF_ROOT=<repo_root> to the subprocess so
    bench_all.sh resolves its cd correctly in environments where git is absent
    or the hard-coded host path does not exist."""
    import anyio

    captured_env: dict[str, str] = {}

    monkeypatch.setattr(srv, "_repo_root", lambda: tmp_path)
    (tmp_path / "testdata").mkdir()
    bench_script = tmp_path / "testdata" / "bench_all.sh"
    bench_script.write_text("#!/bin/sh\nexit 0\n")
    bench_script.chmod(0o755)

    async def fake_cse(*args, **kwargs):
        captured_env.update(kwargs.get("env") or {})

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cse)

    ref = tmp_path / "ref.yuv"
    dis = tmp_path / "dis.yuv"
    ref.write_bytes(b"fakeyuv")
    dis.write_bytes(b"fakeyuv")

    anyio.run(lambda: srv._run_benchmark(ref, dis, 576, 324))

    assert "VMAF_ROOT" in captured_env, (
        "_run_benchmark did not pass VMAF_ROOT to the subprocess env; "
        "bench_all.sh will fail in containers where git is unavailable."
    )
    assert captured_env["VMAF_ROOT"] == str(tmp_path), (
        f"VMAF_ROOT={captured_env['VMAF_ROOT']!r} should equal repo root {tmp_path!r}"
    )


def test_run_benchmark_preserves_inherited_env(monkeypatch, tmp_path):
    """VMAF_ROOT injection must not discard the process environment -- PATH
    and other variables required by the binary must survive."""
    import anyio

    captured_env: dict[str, str] = {}
    monkeypatch.setattr(srv, "_repo_root", lambda: tmp_path)
    (tmp_path / "testdata").mkdir()
    bench_script = tmp_path / "testdata" / "bench_all.sh"
    bench_script.write_text("#!/bin/sh\nexit 0\n")
    bench_script.chmod(0o755)

    monkeypatch.setenv("SENTINEL_VAR", "sentinel_value")

    async def fake_cse(*args, **kwargs):
        captured_env.update(kwargs.get("env") or {})

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cse)

    ref = tmp_path / "ref.yuv"
    dis = tmp_path / "dis.yuv"
    ref.write_bytes(b"fakeyuv")
    dis.write_bytes(b"fakeyuv")

    anyio.run(lambda: srv._run_benchmark(ref, dis, 576, 324))

    assert captured_env.get("SENTINEL_VAR") == "sentinel_value", (
        "_run_benchmark must preserve the inherited environment; "
        "PATH / GPU runtime vars would be lost otherwise."
    )
