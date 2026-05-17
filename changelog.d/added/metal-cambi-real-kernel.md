Real Metal compute kernels for the CAMBI banding-detection feature extractor
(`cambi_metal`). Three MSL shaders mirror the CUDA twins bit-exactly:
spatial-mask (7x7 box-sum), 2x decimate, and separable 3-tap mode filter.
Host CPU residual (calculate_c_values + top-K pooling) reuses the shared
`cambi_internal.h` wrappers, matching the CUDA Strategy II hybrid.
macOS-only; CPU-only builds are unaffected.
