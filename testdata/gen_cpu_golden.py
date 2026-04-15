#!/usr/bin/env python3
"""Generate CPU golden reference scores for all resolutions on this machine.
Usage: python3 gen_cpu_golden.py
"""
import subprocess, json, os, time

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

vmaf_bin = '/usr/local/bin/vmaf'
model = 'version=vmaf_v0.6.1'

resolutions = [
    ('576x324',  '576'),
    ('640x480',  '640'),
    ('1280x720', '720'),
    ('1920x1080','1080'),
    ('3840x2160','4k'),
]

for dims, tag in resolutions:
    w, h = dims.split('x')
    ref = f'ref_{dims}_48f.yuv'
    dis = f'dis_{dims}_48f.yuv'
    if not os.path.exists(ref) or not os.path.exists(dis):
        print(f'SKIP {dims} — files not found')
        continue

    out = f'scores_cpu_{tag}.json'
    print(f'\n=== {dims} CPU ===')

    cmd = [
        vmaf_bin,
        '-r', ref, '-d', dis,
        '-w', w, '-h', h,
        '-p', '420', '-b', '8',
        '-m', model,
        '--json', '-o', out,
        '--no_cuda', '--no_sycl',
    ]

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
    j['fps'] = round(fps, 2)
    with open(out, 'w') as f:
        json.dump(j, f, indent=2)

    print(f'  {nf} frames, pooled={pooled:.6f}, fps={fps:.2f}')

print('\nDone!')
