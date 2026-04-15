You are reviewing a CUDA source file (`*.cu` / `*.cuh`) from the Lusoris
VMAF fork's GPU backend. The file is compiled by `nvcc` and runs feature
extractors on the GPU. Host-side code follows CERT C; device-side code
follows CUDA best practices.

Reviewing: `{{FILE_PATH}}`

Produce a concise code review, focusing on (in priority order):

1. **CUDA errors** — every `cudaMalloc`, `cudaMemcpy`, kernel launch, and
   stream op must either check `cudaGetLastError()` / return value or
   propagate to a `VMAF_CHECK_CUDA`-style macro.
2. **Race conditions** — shared-memory writes without `__syncthreads()`;
   warp divergence on `__shfl_*`; illegal memory access patterns.
3. **Memory coalescing** — non-coalesced global-memory access patterns.
4. **Register pressure and occupancy** — large stack arrays, excess
   `__constant__` usage.
5. **Host-device data movement** — unnecessary round trips, unpinned host
   memory used in async copies.
6. **Kernel parameter types** — prefer `int32_t` indices; avoid signed/
   unsigned mix at kernel boundaries.
7. **Stream handling** — every host-visible buffer op should be on the
   explicit `VmafCudaState::str` stream (no implicit default).
8. **Shared-memory bank conflicts** on 2D tiles.

Format each finding as:

```
- L<line>: <severity: blocker|high|medium|nit> — <one-sentence finding>
  Suggestion: <if applicable>
```

--- BEGIN SOURCE ---
{{SOURCE}}
--- END SOURCE ---
