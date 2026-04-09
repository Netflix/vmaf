#!/usr/bin/env python3
"""
Performance benchmark — real resolutions (1080p & 4K) via FFmpeg.

Backends:
  CPU          (libvmaf)        — software (AVX-512 on Zen 5)
  CUDA         (libvmaf_cuda)   — NVIDIA RTX 4090
  SYCL         (libvmaf_sycl)   — Intel Arc A380 DG2 via QSV import

Test sets:
  1. BBB 1920×1080 48 frames  (raw YUV)
  2. BBB 3840×2160 200 frames (raw YUV)
  3. BBB 3840×2160 4K MP4     (H.264 decode + VMAF, real-world pipeline)
"""

import subprocess, json, os, sys, time, tempfile, shutil

FFMPEG = "/home/kilian/dev/ffmpeg-8/install/bin/ffmpeg"
BASEDIR = os.path.dirname(os.path.abspath(__file__))

RUNS = 3  # timing runs per backend per test

# ── Test sets ────────────────────────────────────────────────────────────────

TESTS = [
    {
        "name": "BBB 1080p 48f (YUV)",
        "ref": os.path.join(BASEDIR, "ref_1920x1080_48f.yuv"),
        "dis": os.path.join(BASEDIR, "dis_1920x1080_48f.yuv"),
        "width": 1920, "height": 1080, "frames": 48,
        "pix_fmt": "yuv420p",
        "raw": True,
    },
    {
        "name": "BBB 4K 200f (YUV)",
        "ref": os.path.join(BASEDIR, "bbb", "ref_3840x2160_200f.yuv"),
        "dis": os.path.join(BASEDIR, "bbb", "dis_3840x2160_200f.yuv"),
        "width": 3840, "height": 2160, "frames": 200,
        "pix_fmt": "yuv420p",
        "raw": True,
    },
    {
        "name": "BBB 4K MP4 500f (decode+vmaf)",
        "ref": "/home/kilian/dev/ffmpeg-testing/bbb_sunflower_2160p_30fps_normal.mp4",
        "dis": os.path.join(BASEDIR, "bbb", "dis_crf35.mp4"),
        "width": 3840, "height": 2160, "frames": 500,
        "pix_fmt": None,  # auto
        "raw": False,
    },
]

# ── Backend definitions ──────────────────────────────────────────────────────


def _raw_inputs(test):
    """FFmpeg input args for raw YUV."""
    size = f"{test['width']}x{test['height']}"
    return [
        "-f", "rawvideo", "-pix_fmt", test["pix_fmt"], "-s", size, "-i", test["dis"],
        "-f", "rawvideo", "-pix_fmt", test["pix_fmt"], "-s", size, "-i", test["ref"],
    ]

def _mp4_inputs(test):
    """FFmpeg input args for encoded files. Limits frames if test['frames'] is set."""
    args = []
    if test.get("frames"):
        # -frames:v on the *output* side limits how many frames the filter processes
        pass  # handled in run_vmaf
    return ["-i", test["dis"], "-i", test["ref"]]


BACKENDS = [
    {
        "name": "cpu",
        "init_args": [],
        "lavfi": "[0:v][1:v]libvmaf=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
    },
    {
        "name": "cuda",
        "init_args": ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"],
        "lavfi": "[0:v]hwupload[dis];[1:v]hwupload[ref];[dis][ref]libvmaf_cuda=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
    },
    {
        "name": "sycl",
        "init_args": [
            "-init_hw_device", "vaapi=va:/dev/dri/renderD130",
            "-init_hw_device", "qsv=qsv@va",
            "-filter_hw_device", "qsv",
        ],
        "lavfi": "[0:v]hwupload=extra_hw_frames=128[dis];[1:v]hwupload=extra_hw_frames=128[ref];[dis][ref]libvmaf_sycl=log_path={log}:log_fmt=json:model=version=vmaf_v0.6.1",
        "env_extra": {"LIBVA_DRIVER_NAME": "iHD"},
    },
]


def run_vmaf(test, backend, log_path):
    """Run one FFmpeg VMAF invocation. Returns (pooled, nframes, elapsed, error)."""
    inputs = _raw_inputs(test) if test["raw"] else _mp4_inputs(test)
    lavfi = backend["lavfi"].format(log=log_path)

    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        *backend["init_args"],
        *inputs,
        "-lavfi", lavfi,
    ]
    # Limit output frames for encoded inputs
    if not test["raw"] and test.get("frames"):
        cmd += ["-frames:v", str(test["frames"])]
    cmd += ["-f", "null", "-"]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib"
    if "env_extra" in backend:
        env.update(backend["env_extra"])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    except subprocess.TimeoutExpired:
        return None, 0, time.time() - t0, "TIMEOUT (1200s)"
    elapsed = time.time() - t0

    if result.returncode != 0:
        return None, 0, elapsed, result.stderr[-500:]

    try:
        with open(log_path) as f:
            data = json.load(f)
        pooled = data["pooled_metrics"]["vmaf"]["mean"]
        nframes = len(data["frames"])
        return pooled, nframes, elapsed, None
    except Exception as e:
        return None, 0, elapsed, str(e)


def main():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║              VMAF Performance Benchmark — Real Resolutions             ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print(f"  FFmpeg : {FFMPEG}")
    print(f"  Runs   : {RUNS} per backend")
    print()

    # Verify files
    for t in TESTS:
        for k in ("ref", "dis"):
            if not os.path.exists(t[k]):
                print(f"  MISSING: {t[k]}")
                sys.exit(1)
    print("  All test files found.\n")

    all_results = {}

    for test in TESTS:
        all_results[test["name"]] = {}
        nf = test["frames"] or "auto"
        print(f"{'━' * 78}")
        print(f"  {test['name']}   ({test['width']}×{test['height']}, {nf} frames)")
        print(f"{'━' * 78}")

        for backend in BACKENDS:
            bname = backend["name"]
            sys.stdout.write(f"  {bname:>12} : ")
            sys.stdout.flush()

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                log_path = tmp.name

            # First run — get score + timing
            pooled, nframes, t1, err = run_vmaf(test, backend, log_path)

            if err:
                print(f"FAILED — {err[:120]}")
                all_results[test["name"]][bname] = {"error": err[:200]}
                try: os.unlink(log_path)
                except: pass
                continue

            fps1 = nframes / t1
            times = [t1]

            # Additional timing runs
            for _ in range(RUNS - 1):
                _, _, ti, ei = run_vmaf(test, backend, log_path)
                if not ei:
                    times.append(ti)

            best = min(times)
            avg = sum(times) / len(times)
            best_fps = nframes / best
            avg_fps = nframes / avg

            print(f"VMAF {pooled:8.4f}  |  {best_fps:7.1f} fps best  {avg_fps:7.1f} fps avg  "
                  f"({best:.2f}s best of {len(times)})")

            all_results[test["name"]][bname] = {
                "pooled": pooled,
                "nframes": nframes,
                "best_fps": best_fps,
                "avg_fps": avg_fps,
                "best_time": best,
                "times": times,
            }

            try: os.unlink(log_path)
            except: pass

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                              PERFORMANCE SUMMARY                                   ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════╣")

    for test in TESTS:
        tr = all_results[test["name"]]
        nf = next((v["nframes"] for v in tr.values() if "nframes" in v), "?")
        print(f"║  {test['name']:<40} ({nf} frames){'':>24}║")
        print(f"╟──────────────┬──────────┬───────────┬───────────┬──────────────────────────────╢")
        print(f"║ {'Backend':>12} │ {'VMAF':>8} │ {'Best FPS':>9} │ {'Avg FPS':>9} │ {'Best Time':>9}  {'Speedup':>8}        ║")
        print(f"╟──────────────┼──────────┼───────────┼───────────┼──────────────────────────────╢")

        # Find CPU time for speedup calc
        cpu_time = None
        if "cpu" in tr and "best_time" in tr["cpu"]:
            cpu_time = tr["cpu"]["best_time"]

        for backend in BACKENDS:
            bname = backend["name"]
            r = tr.get(bname)
            if not r or "error" in r:
                print(f"║ {bname:>12} │ {'FAIL':>8} │ {'':>9} │ {'':>9} │ {'':>9}  {'':>8}        ║")
                continue

            speedup = ""
            if cpu_time and "best_time" in r:
                s = cpu_time / r["best_time"]
                speedup = f"{s:.2f}×"

            print(f"║ {bname:>12} │ {r['pooled']:>8.4f} │ {r['best_fps']:>9.1f} │ {r['avg_fps']:>9.1f} │ "
                  f"{r['best_time']:>8.2f}s  {speedup:>8}        ║")

        print(f"╚══════════════╧══════════╧═══════════╧═══════════╧══════════════════════════════╝")
        print()

    # Save JSON
    out_path = os.path.join(BASEDIR, "perf_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
