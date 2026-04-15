#!/usr/bin/env python3
"""
Netflix Official Score Validation & Benchmark

Uses our FFmpeg 8 build (not the CLI tool) to run VMAF via all backends:
  CPU          (libvmaf)        — software
  CUDA         (libvmaf_cuda)   — NVIDIA RTX 4090
  SYCL         (libvmaf_sycl)   — Intel Arc A380 DG2 via QSV import

Test sets (Netflix standard):
  1. src01 576x324 — hrc01 vs hrc00 (48 frames, yuv420p 8bit)
  2. checkerboard 1920x1080 — _1_0 vs _0_0 (3 frames, yuv420p 8bit)  [mild distortion]
  3. checkerboard 1920x1080 — _10_0 vs _0_0 (3 frames, yuv420p 8bit) [heavy distortion]

Expected Netflix reference scores (vmaf_v0.6.1, integer path, from
python/test/quality_runner_test.py):
  src01 576x324:              76.66890519623612
  checkerboard 1080p (_1_0):  35.06866714286451
  checkerboard 1080p (_10_0):  7.985898744818505
"""

import subprocess, json, os, sys, time, tempfile

FFMPEG = "/home/kilian/dev/ffmpeg-8/install/bin/ffmpeg"
BASEDIR = os.path.dirname(os.path.abspath(__file__))
YUVDIR = os.path.join(os.path.dirname(BASEDIR), "python", "test", "resource", "yuv")

# Netflix reference scores (vmaf_v0.6.1, integer path)
# From python/test/quality_runner_test.py
EXPECTED = {
    "src01_576x324":        76.66890519623612,
    "checker_1080p_mild":   35.06866714286451,
    "checker_1080p_heavy":   7.985898744818505,
}

TESTS = [
    {
        "name": "src01_576x324",
        "ref": os.path.join(YUVDIR, "src01_hrc00_576x324.yuv"),
        "dis": os.path.join(YUVDIR, "src01_hrc01_576x324.yuv"),
        "width": 576, "height": 324,
        "pix_fmt": "yuv420p", "frames": 48,
    },
    {
        "name": "checker_1080p_mild",
        "ref": os.path.join(YUVDIR, "checkerboard_1920_1080_10_3_0_0.yuv"),
        "dis": os.path.join(YUVDIR, "checkerboard_1920_1080_10_3_1_0.yuv"),
        "width": 1920, "height": 1080,
        "pix_fmt": "yuv420p", "frames": 3,
    },
    {
        "name": "checker_1080p_heavy",
        "ref": os.path.join(YUVDIR, "checkerboard_1920_1080_10_3_0_0.yuv"),
        "dis": os.path.join(YUVDIR, "checkerboard_1920_1080_10_3_10_0.yuv"),
        "width": 1920, "height": 1080,
        "pix_fmt": "yuv420p", "frames": 3,
    },
]

# Backend definitions
#
# SYCL uses QSV import: yuv420p → QSV hwupload (NV12 VA surface on Intel Arc A380)
#   → DMA-BUF export → Level Zero import → Tile4 de-tile kernel → SYCL extractors
BACKENDS = [
    {
        "name": "cpu",
        "filter": "libvmaf",
        "extra_args": [],
        "lavfi": "[0:v][1:v]libvmaf=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
    },
    {
        "name": "cuda",
        "filter": "libvmaf_cuda",
        "extra_args": ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"],
        "lavfi": "[0:v]hwupload[dis];[1:v]hwupload[ref];[dis][ref]libvmaf_cuda=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
    },
    {
        "name": "sycl",
        "filter": "libvmaf_sycl",
        "extra_args": [
            "-init_hw_device", "vaapi=va:/dev/dri/renderD130",
            "-init_hw_device", "qsv=qsv@va",
            "-filter_hw_device", "qsv",
        ],
        "lavfi": "[0:v]hwupload=extra_hw_frames=128[dis];[1:v]hwupload=extra_hw_frames=128[ref];[dis][ref]libvmaf_sycl=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
        "env_extra": {"LIBVA_DRIVER_NAME": "iHD"},
    },
]

RUNS = 3  # timing runs per backend


def run_vmaf(test, backend, log_path):
    """Run FFmpeg VMAF filter and return (pooled_vmaf, per_frame_scores, elapsed, error)."""
    w, h = test["width"], test["height"]
    size = f"{w}x{h}"

    lavfi = backend["lavfi"].format(log=log_path)

    cmd = [
        FFMPEG, "-y",
        *backend["extra_args"],
        "-f", "rawvideo", "-pix_fmt", test["pix_fmt"], "-s", size, "-i", test["dis"],
        "-f", "rawvideo", "-pix_fmt", test["pix_fmt"], "-s", size, "-i", test["ref"],
        "-lavfi", lavfi,
        "-f", "null", "-"
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib"
    if "env_extra" in backend:
        env.update(backend["env_extra"])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return None, None, elapsed, result.stderr[-500:]

    try:
        with open(log_path) as f:
            data = json.load(f)
        pooled = data["pooled_metrics"]["vmaf"]["mean"]
        frames = [fr["metrics"]["vmaf"] for fr in data["frames"]]
        return pooled, frames, elapsed, None
    except Exception as e:
        return None, None, elapsed, str(e)


def main():
    print(f"FFmpeg: {FFMPEG}")
    print(f"Test data: {YUVDIR}")
    print()

    # Verify files exist
    for t in TESTS:
        for k in ("ref", "dis"):
            if not os.path.exists(t[k]):
                print(f"MISSING: {t[k]}")
                sys.exit(1)

    results = {}  # results[test_name][backend] = {pooled, frames, fps, error}

    for test in TESTS:
        results[test["name"]] = {}
        print(f"{'=' * 90}")
        print(f"TEST: {test['name']}  ({test['width']}x{test['height']}, {test['frames']} frames)")
        print(f"  ref: {os.path.basename(test['ref'])}")
        print(f"  dis: {os.path.basename(test['dis'])}")
        print(f"{'=' * 90}")

        for backend in BACKENDS:
            bname = backend["name"]
            print(f"\n  Backend: {bname} ({backend['filter']})")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                log_path = tmp.name

            # Score run (first run, captures scores)
            pooled, frames, elapsed, err = run_vmaf(test, backend, log_path)

            if err:
                print(f"    FAILED: {err[:200]}")
                results[test["name"]][bname] = {"error": err[:200]}
                try:
                    os.unlink(log_path)
                except:
                    pass
                continue

            fps = test["frames"] / elapsed
            print(f"    Pooled VMAF: {pooled:.14f}")
            print(f"    Time: {elapsed:.3f}s  ({fps:.1f} fps)")

            # Timing runs
            fps_list = [fps]
            for run_i in range(1, RUNS):
                _, _, t, e = run_vmaf(test, backend, log_path)
                if not e:
                    fps_list.append(test["frames"] / t)

            best_fps = max(fps_list)
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"    Best FPS: {best_fps:.1f}  Avg FPS: {avg_fps:.1f}  ({RUNS} runs)")

            results[test["name"]][bname] = {
                "pooled": pooled,
                "frames": frames,
                "best_fps": best_fps,
                "avg_fps": avg_fps,
            }

            try:
                os.unlink(log_path)
            except:
                pass

    # === Summary ===
    print(f"\n\n{'=' * 119}")
    print("SCORE COMPARISON — Netflix Reference (vmaf_v0.6.1)")
    print(f"{'=' * 119}")
    print(f"{'Test':>25} | {'Backend':>12} | {'Pooled VMAF':>18} | {'Expected':>18} | {'Delta':>16} | {'PASS':>6} | {'Best FPS':>10}")
    print(f"{'-' * 119}")

    for test in TESTS:
        expected = EXPECTED.get(test["name"])
        for backend in BACKENDS:
            bname = backend["name"]
            r = results[test["name"]].get(bname)
            if not r or "error" in r:
                print(f"{test['name']:>25} | {bname:>12} | {'FAILED':>18} | {expected:>18.14f} | {'':>16} | {'FAIL':>6} | {'':>10}")
                continue

            delta = r["pooled"] - expected
            # Netflix tests use places=4 tolerance -> 5e-5
            passed = abs(delta) < 5e-5
            tag = "OK" if passed else "DIFF"
            print(f"{test['name']:>25} | {bname:>12} | {r['pooled']:>18.14f} | {expected:>18.14f} | {delta:>+16.14f} | {tag:>6} | {r['best_fps']:>10.1f}")
        print(f"{'-' * 119}")

    # === Per-frame cross-backend comparison ===
    print(f"\n{'=' * 118}")
    print("PER-FRAME CROSS-BACKEND COMPARISON (max absolute difference)")
    print(f"{'=' * 118}")
    print(f"{'Test':>25} | {'A':>12} vs {'B':>12} | {'Max Diff':>16} | {'Mean Diff':>16} | {'Match':>8}")
    print(f"{'-' * 118}")

    for test in TESTS:
        have = [(b["name"], results[test["name"]][b["name"]]) for b in BACKENDS
                if b["name"] in results[test["name"]] and "frames" in results[test["name"]].get(b["name"], {})]

        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                name_a, r_a = have[i]
                name_b, r_b = have[j]
                if len(r_a["frames"]) != len(r_b["frames"]):
                    print(f"{test['name']:>25} | {name_a:>12} vs {name_b:>12} | {'FRAME COUNT MISMATCH':>16}")
                    continue

                diffs = [abs(a - b) for a, b in zip(r_a["frames"], r_b["frames"])]
                max_d = max(diffs)
                mean_d = sum(diffs) / len(diffs)
                exact = max_d == 0.0
                tag = "EXACT" if exact else ("OK" if max_d < 0.01 else "DIFF")
                print(f"{test['name']:>25} | {name_a:>12} vs {name_b:>12} | {max_d:>16.12f} | {mean_d:>16.12f} | {tag:>8}")

    # Save results
    out_path = os.path.join(BASEDIR, "netflix_benchmark_results.json")
    save = {}
    for test in TESTS:
        save[test["name"]] = {}
        for b in BACKENDS:
            bname = b["name"]
            r = results[test["name"]].get(bname)
            if r and "frames" in r:
                save[test["name"]][bname] = {
                    "pooled": r["pooled"],
                    "frames": r["frames"],
                    "best_fps": r["best_fps"],
                    "avg_fps": r["avg_fps"],
                }
            elif r:
                save[test["name"]][bname] = {"error": r.get("error", "unknown")}

    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
