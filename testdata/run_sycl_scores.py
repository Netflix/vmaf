#!/usr/bin/env python3
"""Generate SYCL scores on the current GPU for all resolutions.
Usage: python3 run_sycl_scores.py <gpu_tag>
Example: python3 run_sycl_scores.py a380
"""
import subprocess, json, sys, os, time

gpu_tag = sys.argv[1] if len(sys.argv) > 1 else 'a380'
basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

vmaf_bin = '/usr/local/bin/vmaf'

resolutions = [
    ('576x324',  '576'),
    ('640x480',  '640'),
    ('1280x720', '720'),
    ('1920x1080','1080'),
    ('3840x2160','4k'),
]

model = 'version=vmaf_v0.6.1'

for dims, tag in resolutions:
    w, h = dims.split('x')
    ref = f'ref_{dims}_48f.yuv'
    dis = f'dis_{dims}_48f.yuv'
    if not os.path.exists(ref) or not os.path.exists(dis):
        print(f'SKIP {dims} — files not found')
        continue

    out = f'scores_sycl_{gpu_tag}_{tag}.json'
    print(f'\n=== {dims} SYCL on {gpu_tag} ===')

    cmd = [
        vmaf_bin,
        '-r', ref, '-d', dis,
        '-w', w, '-h', h,
        '-p', '420', '-b', '8',
        '-m', model,
        '--json', '-o', out,
        '--no_cuda',
    ]
    print('  ' + ' '.join(cmd))

    t0 = time.time()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/lib'
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'  FAILED (exit {result.returncode})')
        print(f'  stderr: {result.stderr[:500]}')
        continue

    with open(out) as f:
        j = json.load(f)
    pooled = j['pooled_metrics']['vmaf']['mean']
    nf = len(j['frames'])
    fps = nf / elapsed if elapsed > 0 else 0
    # Store fps in the JSON
    j['fps'] = round(fps, 2)
    with open(out, 'w') as f:
        json.dump(j, f, indent=2)

    print(f'  {nf} frames, pooled={pooled:.6f}, fps={fps:.2f}, time={elapsed:.2f}s')

    # Compare vs CPU golden
    cpu_file = f'scores_cpu_{tag}.json'
    if os.path.exists(cpu_file):
        with open(cpu_file) as f:
            cpu = json.load(f)
        cpu_scores = [fr['metrics']['vmaf'] for fr in cpu['frames']]
        sycl_scores = [fr['metrics']['vmaf'] for fr in j['frames']]
        max_diff = max(abs(a - b) for a, b in zip(cpu_scores, sycl_scores))
        print(f'  vs CPU max diff: {max_diff:.9f}  {"OK" if max_diff < 0.001 else "WARN"}')

print('\nDone!')
