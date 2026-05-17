## SYCL GPU: CAMBI queue-sync collapse + SSIM horizontal SLM staging (SY-1/SY-2)

**`integer_cambi_sycl.cpp` (SY-1)**: Eliminate 20 redundant `q.wait()` calls per
frame from the 5-scale CAMBI loop (25 → 5 total, one per scale). The SYCL in-order
queue serialises GPU operations automatically; only the mandatory CPU-reads-from-device
sync points are retained. Estimated 0.5–3 ms/frame saved at 1080p on Intel Arc A380.

**`integer_ssim_sycl.cpp` (SY-2)**: Convert `launch_horiz` (the 11-tap Gaussian
horizontal pass) from bare `parallel_for<range<2>>` to `nd_range` with SLM
(`local_accessor`) tile staging. Each 16×8 work-group now loads a 26-float tile
cooperatively into SLM before computing convolutions, reducing global-memory reads
by ~3.25× per pixel. `places=4` parity with CPU `float_ssim` maintained (max |Δ|
= 1e-6 on 5-frame test, Intel Arc A380).

**`integer_ciede_sycl.cpp`**: No change — CIEDE has no separable convolution kernel;
the "11-tap Gaussian" description in the original audit was incorrect.

See [ADR-0458](../docs/adr/0458-sycl-cambi-ssim-slm-staging.md) and
[research digest 0135](../docs/research/0135-sycl-cambi-ssim-ciede-perf-2026-05-16.md).
