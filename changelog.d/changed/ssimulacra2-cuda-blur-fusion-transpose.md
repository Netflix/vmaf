### SSIMULACRA2 CUDA: 3-channel kernel fusion + V-pass transpose (ADR-0456)

The SSIMULACRA2 CUDA blur dispatch now issues 3 kernel launches per blur operation (one fused
3-channel H-pass, one 3-channel transpose, one fused 3-channel V-pass) rather than 6 (one
H-pass + one V-pass per channel, looped over 3 channels). Per-frame blur launch count drops
from 180 to 90. A shared-memory transpose kernel (`ssimulacra2_transpose`) converts the
H-pass output to column-major layout before the V-pass, converting per-thread stride-width
memory access to stride-1 sequential reads for improved L2 cache utilisation. Bit-exactness
is maintained: zero absolute difference vs CPU reference on the 576×324 48-frame test pair.
