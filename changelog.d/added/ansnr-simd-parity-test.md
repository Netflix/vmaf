test(ansnr): add AVX2/AVX-512/NEON MSE line parity unit test

Closes the last coverage gap identified in the 2026-05-15 SIMD audit
(T3-7): `test_ansnr_simd.c` exercises all three SIMD line kernels
against the scalar inner loop with a 1e-6 relative tolerance,
covering aligned, tail, and tiny widths on each ISA.
