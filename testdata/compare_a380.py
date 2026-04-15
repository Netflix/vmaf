#!/usr/bin/env python3
"""Compare SYCL A380 scores frame-by-frame against CPU golden.
Shows per-frame diffs for frames with diff > 0.0001.
"""
import json, os, glob

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

resolutions = [
    ('576x324',  '576'),
    ('640x480',  '640'),
    ('1280x720', '720'),
    ('1920x1080','1080'),
    ('3840x2160','4k'),
]

for dims, tag in resolutions:
    cpu_f = f'scores_cpu_{tag}.json'
    sycl_f = f'scores_sycl_a380_{tag}.json'
    if not (os.path.exists(cpu_f) and os.path.exists(sycl_f)):
        continue

    with open(cpu_f) as f:
        cpu = json.load(f)
    with open(sycl_f) as f:
        sycl = json.load(f)

    cpu_s = [fr['metrics']['vmaf'] for fr in cpu['frames']]
    sycl_s = [fr['metrics']['vmaf'] for fr in sycl['frames']]
    diffs = [abs(a - b) for a, b in zip(cpu_s, sycl_s)]
    max_diff = max(diffs)
    avg_diff = sum(diffs) / len(diffs)
    cpu_fps = cpu.get('fps', 0)
    sycl_fps = sycl.get('fps', 0)

    print(f'\n=== {dims} ===')
    print(f'  CPU:  pooled={cpu["pooled_metrics"]["vmaf"]["mean"]:.6f}  fps={cpu_fps:.1f}')
    print(f'  SYCL: pooled={sycl["pooled_metrics"]["vmaf"]["mean"]:.6f}  fps={sycl_fps:.1f}')
    print(f'  Max diff: {max_diff:.9f}  Avg diff: {avg_diff:.9f}')
    print(f'  Speedup: {sycl_fps/cpu_fps:.1f}x' if cpu_fps > 0 else '')

    # Show worst frames
    worst = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:5]
    for i in worst:
        if diffs[i] > 0.0001:
            print(f'    frame {i}: CPU={cpu_s[i]:.6f} SYCL={sycl_s[i]:.6f} diff={diffs[i]:.9f}')
