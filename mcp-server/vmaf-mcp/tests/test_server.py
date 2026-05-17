"""Smoke tests for the vmaf-mcp server — no network, no GPU required."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from vmaf_mcp import server as srv

REPO = Path(__file__).resolve().parents[3]


def test_repo_root_detects_testdata():
    root = srv._repo_root()
    assert (root / "testdata").is_dir()


def test_validate_path_accepts_golden_yuv():
    yuv = REPO / "python/test/resource/yuv/src01_hrc00_576x324.yuv"
    if not yuv.exists():
        pytest.skip("Netflix golden YUV not present")
    assert srv._validate_path(str(yuv)) == yuv.resolve()


def test_validate_path_rejects_outside_roots(tmp_path):
    bad = tmp_path / "evil.yuv"
    bad.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="not under an allowlisted root"):
        srv._validate_path(str(bad))


def test_validate_path_accepts_custom_allow(tmp_path, monkeypatch):
    f = tmp_path / "ok.yuv"
    f.write_bytes(b"\x00" * 16)
    monkeypatch.setenv("VMAF_MCP_ALLOW", str(tmp_path))
    assert srv._validate_path(str(f)) == f.resolve()


def test_list_models_returns_list():
    models = srv._list_models()
    assert isinstance(models, list)
    for m in models:
        assert "name" in m and "path" in m and "format" in m


def test_list_backends_always_includes_cpu():
    backends = srv._list_backends()
    assert backends["cpu"] is True


# ---------------------------------------------------------------------------
# eval_model_on_split / compare_models — require the 'eval' extra
# ---------------------------------------------------------------------------


def _has_eval_deps() -> bool:
    try:
        import numpy  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark_eval = pytest.mark.skipif(
    not _has_eval_deps(), reason="vmaf-mcp[eval] extras not installed"
)


def _make_tiny_mlp(path: Path, in_features: int = 6) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", in_features])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [in_features, 1], [0.5] * in_features)
    b = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
    node = helper.make_node("Gemm", ["features", "W", "B"], ["score"])
    graph = helper.make_graph([node], "mlp", [x], [y], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def _make_feature_parquet(path: Path, n: int = 64) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = {c: rng.standard_normal(n).astype(np.float32) for c in srv._FEATURE_COLUMNS}
    # MOS is a linear transform of the features so correlations are non-trivial.
    x = np.stack(list(data.values()), axis=1)
    data["mos"] = (x.sum(axis=1) * 0.5 + 10.0 + rng.normal(0, 0.01, n)).astype(np.float32)
    data["key"] = [f"sample_{i:04d}" for i in range(n)]
    pd.DataFrame(data).to_parquet(path)


@pytestmark_eval
def test_eval_model_on_split_reports_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("VMAF_MCP_ALLOW", str(tmp_path))
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats)
    result = srv._eval_model_on_split(model, feats, split="all", input_name="features")
    assert result["n"] == 64
    # The MOS column is a linear function of the features with tiny noise, so
    # the linear model's correlation should be very close to 1.
    assert result["plcc"] > 0.99
    assert result["rmse"] >= 0.0
    assert result["split"] == "all"


@pytestmark_eval
def test_eval_model_on_split_rejects_unknown_split(tmp_path):
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats)
    with pytest.raises(ValueError, match="split must be one of"):
        srv._eval_model_on_split(model, feats, split="nope", input_name="features")


@pytestmark_eval
def test_eval_split_filters_by_key(tmp_path):
    """train+val+test should partition the parquet exactly."""
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats, n=200)
    ns = {
        s: srv._eval_model_on_split(model, feats, split=s, input_name="features")["n"]
        for s in ("train", "val", "test")
    }
    assert sum(ns.values()) == 200
    # Small-sample fractions drift from 10% / 10% / 80% but should stay in-band.
    assert ns["train"] > ns["val"] and ns["train"] > ns["test"]


@pytestmark_eval
def test_compare_models_ranks_by_plcc(tmp_path):
    good = tmp_path / "good.onnx"
    bad = tmp_path / "bad.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(good)
    # A model with opposite-sign weights will have strongly negative PLCC.
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", 6])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [6, 1], [-0.5] * 6)
    b = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
    node = helper.make_node("Gemm", ["features", "W", "B"], ["score"])
    graph = helper.make_graph([node], "mlp", [x], [y], [w, b])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(bad))
    _make_feature_parquet(feats)
    report = srv._compare_models([bad, good], feats, split="all", input_name="features")
    assert report["errors"] == []
    assert report["ranked"][0]["model"] == str(good)
    assert report["ranked"][0]["plcc"] > report["ranked"][1]["plcc"]


@pytestmark_eval
def test_compare_models_captures_errors_without_aborting(tmp_path):
    ok = tmp_path / "ok.onnx"
    _make_tiny_mlp(ok)
    missing = tmp_path / "missing.onnx"
    feats = tmp_path / "f.parquet"
    _make_feature_parquet(feats)
    report = srv._compare_models([ok, missing], feats, split="all", input_name="features")
    # One report, one error — the missing model doesn't take the good one down.
    assert len(report["ranked"]) == 1
    assert len(report["errors"]) == 1
    assert report["errors"][0]["model"] == str(missing)


def test_new_tools_registered_in_list_tools():
    """Schema-level check that doesn't need eval extras installed."""
    import anyio

    async def get_tools():
        return await srv._list_tools()

    tools = anyio.run(get_tools)
    names = {t.name for t in tools}
    assert "eval_model_on_split" in names
    assert "compare_models" in names
    # ADR-0172 / T6-6: VLM-assisted artefact triage tool.
    assert "describe_worst_frames" in names


# ─────────────────────────────────────────────────────────────────────
# describe_worst_frames (T6-6 / ADR-0172)
# ─────────────────────────────────────────────────────────────────────


def test_describe_worst_frames_picks_lowest_vmaf():
    """The picker walks the per-frame array and returns the N with
    smallest VMAF, sorted ascending."""
    score_json = {
        "frames": [
            {"frameNum": 0, "metrics": {"vmaf": 90.0}},
            {"frameNum": 1, "metrics": {"vmaf": 30.0}},
            {"frameNum": 2, "metrics": {"vmaf": 50.0}},
            {"frameNum": 3, "metrics": {"vmaf": 10.0}},
            {"frameNum": 4, "metrics": {"vmaf": 70.0}},
        ],
    }
    worst = srv._pick_worst_frames(score_json, n=3)
    assert worst == [(3, 10.0), (1, 30.0), (2, 50.0)]


def test_describe_worst_frames_handles_alternate_metric_keys():
    """Different VMAF model variants emit different headline-metric
    keys. The picker recognises the common ones."""
    score_json = {
        "frames": [
            {"frameNum": 0, "metrics": {"vmaf_v0.6.1": 80.0}},
            {"frameNum": 1, "metrics": {"vmaf_v0.6.1": 20.0}},
        ],
    }
    worst = srv._pick_worst_frames(score_json, n=1)
    assert worst == [(1, 20.0)]


def test_describe_worst_frames_skips_frames_without_a_score():
    score_json = {
        "frames": [
            {"frameNum": 0, "metrics": {}},  # no headline key
            {"frameNum": 1, "metrics": {"vmaf": 5.0}},
        ],
    }
    worst = srv._pick_worst_frames(score_json, n=2)
    assert worst == [(1, 5.0)]


def test_describe_worst_frames_n_zero_returns_empty_list():
    score_json = {
        "frames": [{"frameNum": 0, "metrics": {"vmaf": 99.0}}],
    }
    assert srv._pick_worst_frames(score_json, n=0) == []


def test_describe_worst_frames_tmpdir_cleared_on_next_call(tmp_path, monkeypatch):
    """T-ROUND8-MCP-TMPDIR-LEAK: stale PNGs from previous invocations must be
    purged at the start of the next call, not accumulated indefinitely.

    The test drives _describe_worst_frames with a synthetic score JSON and a
    no-op describe function so no actual ffmpeg/VLM is required. It plants a
    sentinel file in the tmp_root dir, then calls the helper again and verifies
    the sentinel has been removed (i.e. the dir was cleared before the new
    PNGs were generated)."""
    import anyio

    # Use a fixed pid-based path that mirrors the production code.
    pid_root = Path("/tmp") / f"vmaf-mcp-worst-{os.getpid()}"

    # Cleanup any leftover from a previous test run.
    if pid_root.exists():
        shutil.rmtree(pid_root)

    # Plant a sentinel file as if left by a previous invocation.
    pid_root.mkdir(parents=True)
    sentinel = pid_root / "stale_sentinel.png"
    sentinel.write_bytes(b"stale")
    assert sentinel.exists()

    # A minimal score JSON that looks like a 1-frame result with a known score.
    fake_score = {
        "frames": [{"frameNum": 0, "metrics": {"vmaf": 10.0}}],
    }

    async def run():
        # Monkey-patch _run_vmaf_score to return our fake score so no binary
        # is needed.
        async def fake_score_fn(_req):
            return fake_score

        orig = srv._run_vmaf_score
        srv._run_vmaf_score = fake_score_fn
        try:
            # Use a no-op describe function; _extract_frame_png would normally
            # be called but we also need to skip it.  Replace the entire
            # _describe_worst_frames internals by supplying `describe=` and
            # patching _extract_frame_png.
            async def fake_extract(_yuv, **_kwargs):
                # Create a minimal stub PNG so the loop body has a file to
                # pass to descr_fn.
                import struct
                import zlib

                path = _kwargs["out_png"]

                # Write a 1×1 white PNG.
                def write_chunk(chunk_type, data):
                    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
                    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

                raw = (
                    b"\x89PNG\r\n\x1a\n"
                    + write_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                    + write_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
                    + write_chunk(b"IEND", b"")
                )
                path.write_bytes(raw)

            orig_extract = srv._extract_frame_png
            srv._extract_frame_png = fake_extract
            try:
                result = await srv._describe_worst_frames(
                    # ScoreRequest doesn't matter since _run_vmaf_score is mocked.
                    srv.ScoreRequest(
                        ref=Path("/dev/null"),
                        dis=Path("/dev/null"),
                        width=1,
                        height=1,
                        pixfmt="420",
                        bitdepth=8,
                    ),
                    n=1,
                    describe=lambda _p: "stub description",
                )
            finally:
                srv._extract_frame_png = orig_extract
        finally:
            srv._run_vmaf_score = orig
        return result

    result = anyio.run(run)
    # The sentinel planted before the call must be gone.
    assert not sentinel.exists(), (
        "T-ROUND8-MCP-TMPDIR-LEAK: stale PNG from previous invocation was not "
        "cleared at the start of the next describe_worst_frames call"
    )
    assert len(result["frames"]) == 1
    assert result["frames"][0]["description"] == "stub description"


def test_describe_image_falls_back_to_metadata_only_without_extras(monkeypatch):
    """When the [vlm] extras aren't installed, _load_vlm() returns
    None and _describe_image_with_vlm surfaces a clear hint."""
    # Reset the cache so this test isn't influenced by other tests.
    srv._vlm_state["loaded"] = False
    srv._vlm_state["pipeline"] = None
    srv._vlm_state["model_id"] = None

    # Force the import-failure branch by hiding `transformers`.
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    msg = srv._describe_image_with_vlm(Path("/tmp/nonexistent.png"))
    assert "VLM unavailable" in msg
    assert "vmaf-mcp[vlm]" in msg


# ─────────────────────────────────────────────────────────────────────
# ORT InferenceSession cache (_get_ort_session)
# ─────────────────────────────────────────────────────────────────────


@pytestmark_eval
def test_ort_session_cache_reuses_session_for_same_model(tmp_path):
    """5 sequential calls with the same model path must create exactly
    1 InferenceSession (cache hit on calls 2–5)."""
    from collections import OrderedDict

    import onnxruntime as ort

    model = tmp_path / "m.onnx"
    _make_tiny_mlp(model)

    # Patch the module-level cache so this test is isolated.
    orig_cache = srv._ort_session_cache
    srv._ort_session_cache = OrderedDict()

    creation_count = 0
    real_init = ort.InferenceSession.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal creation_count
        creation_count += 1
        return real_init(self, *args, **kwargs)

    try:
        ort.InferenceSession.__init__ = counting_init
        for _ in range(5):
            srv._get_ort_session(model)
        assert creation_count == 1, (
            f"Expected 1 InferenceSession created for 5 calls with the same model; "
            f"got {creation_count}"
        )
    finally:
        ort.InferenceSession.__init__ = real_init
        srv._ort_session_cache = orig_cache


@pytestmark_eval
def test_ort_session_cache_creates_new_session_for_different_model(tmp_path):
    """A 6th call with a different model path must create a 2nd session."""
    from collections import OrderedDict

    import onnxruntime as ort

    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _make_tiny_mlp(model_a)
    _make_tiny_mlp(model_b)

    orig_cache = srv._ort_session_cache
    srv._ort_session_cache = OrderedDict()

    creation_count = 0
    real_init = ort.InferenceSession.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal creation_count
        creation_count += 1
        return real_init(self, *args, **kwargs)

    try:
        ort.InferenceSession.__init__ = counting_init
        for _ in range(5):
            srv._get_ort_session(model_a)
        srv._get_ort_session(model_b)
        assert creation_count == 2, (
            f"Expected 2 InferenceSessions (5 calls model_a + 1 call model_b); got {creation_count}"
        )
    finally:
        ort.InferenceSession.__init__ = real_init
        srv._ort_session_cache = orig_cache


@pytestmark_eval
def test_ort_session_cache_hit_after_eviction_round_trip(tmp_path):
    """After filling the cache with 4 different models, a 7th call
    back to model_a (evicted as LRU) creates a new session (cache miss,
    not a stale hit)."""
    from collections import OrderedDict

    import onnxruntime as ort

    models = [tmp_path / f"{c}.onnx" for c in "abcde"]
    for m in models:
        _make_tiny_mlp(m)

    orig_cache = srv._ort_session_cache
    orig_maxsize = srv._ORT_CACHE_MAXSIZE
    srv._ort_session_cache = OrderedDict()
    srv._ORT_CACHE_MAXSIZE = 4

    creation_count = 0
    real_init = ort.InferenceSession.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal creation_count
        creation_count += 1
        return real_init(self, *args, **kwargs)

    try:
        ort.InferenceSession.__init__ = counting_init
        # Call a, b, c, d — fills the cache (4 sessions).
        for m in models[:4]:
            srv._get_ort_session(m)
        assert creation_count == 4

        # Call e — evicts a (LRU), inserts e.
        srv._get_ort_session(models[4])
        assert creation_count == 5

        # Call a again — cache miss (a was evicted), creates a 6th session.
        srv._get_ort_session(models[0])
        assert creation_count == 6, (
            f"Expected 6 total sessions after round-trip eviction; got {creation_count}"
        )
        assert len(srv._ort_session_cache) == srv._ORT_CACHE_MAXSIZE
    finally:
        ort.InferenceSession.__init__ = real_init
        srv._ort_session_cache = orig_cache
        srv._ORT_CACHE_MAXSIZE = orig_maxsize


@pytestmark_eval
def test_ort_session_cache_invalidated_on_mtime_change(tmp_path):
    """Replacing the model file on disk (different mtime) must cause a
    cache miss on the next call, not a stale session hit."""
    from collections import OrderedDict

    import onnxruntime as ort

    model = tmp_path / "m.onnx"
    _make_tiny_mlp(model)

    orig_cache = srv._ort_session_cache
    srv._ort_session_cache = OrderedDict()

    creation_count = 0
    real_init = ort.InferenceSession.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal creation_count
        creation_count += 1
        return real_init(self, *args, **kwargs)

    try:
        ort.InferenceSession.__init__ = counting_init

        srv._get_ort_session(model)
        assert creation_count == 1

        # Simulate file replacement: touch the file to bump mtime.
        import time

        time.sleep(0.01)  # ensure mtime advances on filesystems with 10ms resolution
        model.touch()

        srv._get_ort_session(model)
        assert creation_count == 2, (
            f"Expected a new session after mtime change; still got {creation_count}"
        )
    finally:
        ort.InferenceSession.__init__ = real_init
        srv._ort_session_cache = orig_cache
