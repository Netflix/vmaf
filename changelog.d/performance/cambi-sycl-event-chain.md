**perf(sycl): replace GPU-to-GPU `q.wait()` with `sycl::event` chains in CAMBI SYCL (SY-1)**

`integer_cambi_sycl.cpp` previously issued up to 30 `q.wait()` calls per
frame — one per GPU kernel transition inside the per-scale loop.  Each
call drained the entire SYCL queue to idle before the next dispatch, adding
0.5–2 ms per barrier on Intel Arc / iGPU hardware (15–60 ms/frame total).

GPU-to-GPU transitions now use `sycl::event` `depends_on` chains: each
`launch_spatial_mask`, `launch_decimate`, and `launch_filter_mode` call
returns an event; the next kernel declares it as a dependency via
`h.depends_on(dep)`.  Only two `wait()` calls remain per scale:

1. `q.wait()` after the H2D row-loop (kernels cannot read partial uploads).
2. `ev_prev.wait()` + `q.wait()` before/after D2H copies (CPU residual
   requires fully-written host staging buffers).

Precision contract (`places=4`, bit-exact with CUDA twin) is unaffected —
all arithmetic is unchanged; only scheduling semantics changed.
