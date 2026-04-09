# AMD ROCm HIP - Programming Model & Performance Guidelines

> Sources: 
> - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
> - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

## HIP Programming Model

### Heterogeneous Execution
- Host (CPU) manages: initialization, data preparation, kernel launch, synchronization
- Device (GPU) executes: parallel kernel instances across thousands of threads
- API closely mirrors CUDA: `hipMalloc`, `hipMemcpy`, `hipLaunchKernel`, `hipStreamSynchronize`

### Thread Hierarchy
- **Thread** → **Block (Work-group)** → **Grid**
- **Wavefront (AMD)**: 64 threads (GCN/CDNA) or 32 threads (RDNA wave32)
- **Warp (NVIDIA)**: 32 threads
- Max threads per block: typically 1024
- Use multiples of wavefront size (64) for block dimensions

### Memory Model
- **Local (per-thread)**: Registers, fastest access
- **Shared (LDS)**: Per-block, ~100× faster than global, 32-64KB per CU
- **Global**: Device DRAM, high latency, coalesced access critical
- **Constant**: Read-only, cached, limited size
- **Texture**: Read-only, spatially cached, hardware interpolation

## Performance Guidelines

### Optimization Workflow
1. Profile baseline with `rocprofv3`
2. Identify compute-bound vs memory-bound (roofline model)
3. Apply targeted optimizations
4. Re-profile to verify gains
5. Iterate

### Memory Throughput
- **Data transfers**: Minimize host-device copies, batch small transfers, use pinned memory (`hipHostMalloc`)
- **Mapped memory on APUs**: `hipHostMallocMapped` for zero-copy on integrated GPUs
- **Coalesced global access**: Consecutive threads → consecutive addresses
- **2D array alignment**: Pad width to multiples of wavefront size
- **Shared memory**: Avoid 32-bank conflicts, pad arrays by 1 element
- **Texture memory**: Hardware-accelerated 2D spatial caching

### Instruction Throughput
- Single-precision (`float`) >> double-precision (`double`)
- Fast intrinsics: `__expf`, `__logf`, `__fsqrt_rn`, `__frcp_rn`
- Bitwise ops for power-of-2 arithmetic
- Minimize warp divergence; use `__builtin_expect` for branch hints

### Occupancy
- Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerCU)`
- Block sizes: multiples of 64 for AMD (128, 256 common)
- Reduce register pressure: minimize live variables, use shared memory for temp storage
- Profile with `rocprofv3 --occupancy`

### Register Pressure
- Chain operations instead of storing all intermediates
- Use `__shared__` for per-thread temp arrays
- Check register usage: `hipcc --resource-usage`

### Synchronization
- Minimize `__syncthreads()` — only sync when threads exchange shared data
- Use streams for async execution overlap
- Each sync point stalls all threads in block until slowest arrives

## Porting from CUDA
- Use `hipify-perl` or `hipify-clang` for automatic translation
- Key API mapping: `cu*` → `hip*`, `cuda*` → `hip*`
- Warp size difference (32 vs 64) requires reviewing warp-level primitives
- Shared memory → LDS (same concept, different naming)
- Block sizes of 128 work well on both NVIDIA and AMD
