# ADR-0468: Real Metal Compute Kernels for CAMBI

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `metal`, `cambi`, `gpu`, `build`

## Context

The CAMBI banding-detection feature extractor had a CUDA Strategy II hybrid
(ADR-0360): three GPU kernels handle the embarrassingly parallel stages
(spatial mask, 2x decimate, separable 3-tap mode filter), while the
precision-sensitive `calculate_c_values` sliding-histogram pass and top-K
spatial pooling run on the host CPU via `cambi_internal.h` wrappers.

The Metal backend had no real CAMBI kernel. This ADR records the decision to
port the three CUDA kernels to MSL and wire them into the existing Metal
dispatch infrastructure, using the same Strategy II hybrid as CUDA.

## Decision

We implement three MSL compute kernels in `integer_cambi.metal`
(`cambi_spatial_mask_kernel`, `cambi_decimate_kernel`,
`cambi_filter_mode_kernel`) that are bit-exact ports of the CUDA kernels in
`cambi_score.cu`. The host dispatch in `integer_cambi_metal.mm` mirrors the
CUDA host code in `integer_cambi_cuda.c`: submit() is synchronous per scale
(same as the CUDA v1 approach), and collect() emits the pre-computed score.
All integer arithmetic is identical to the CUDA path, so ULP=0 vs CPU scalar
is the precision target (places=4 cross-backend gate per ADR-0214).

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Full GPU (all 5 stages on Metal) | No DtoH round-trips | `calculate_c_values` is a complex sliding-histogram; porting it to MSL is high risk and not needed for places=4 parity | Not chosen for v1; mirrors CUDA Strategy II rationale |
| Strategy I (all CPU, Metal no-op) | Zero risk | No GPU acceleration | Does not deliver the requested real kernel |
| Async per-scale (multiple command buffers in flight) | Better throughput | Requires per-scale MTLBuffer ping-pong and synchronisation complexity | Deferred to v2; CUDA v1 is synchronous for the same reason |

## References

- req: "IMPLEMENT REAL Metal cambi kernel (NOT a stub). 20 min budget." (paraphrased)
- ADR-0360: CUDA CAMBI Strategy II hybrid
- ADR-0214: cross-backend ULP gate (places=4)
- `libvmaf/src/feature/cuda/integer_cambi/cambi_score.cu` — CUDA kernel reference
- `libvmaf/src/feature/metal/integer_cambi.metal` — MSL port
- `libvmaf/src/feature/metal/integer_cambi_metal.mm` — host dispatch
