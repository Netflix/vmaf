# CUDA C++ Best Practices Guide - Key Excerpts

**Source:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
**Fetched:** 2026-02-19

## APOD Cycle (Assess, Parallelize, Optimize, Deploy)

Iterative optimization: initial speedups → test → deploy → identify more opportunities → repeat.

## Key Optimization Strategies

### Memory Optimizations (Highest Priority)
- **Minimize host↔device transfers**: Even if kernels don't speed up, avoid PCIe transfers (16 GB/s PCIe x16 Gen3 vs 898 GB/s HBM2 on V100)
- **Batch small transfers** into larger ones
- **Use pinned (page-locked) memory** for higher bandwidth
- **Keep data on device** between kernel calls — don't transfer intermediate results
- **Coalesced global memory access**: k-th thread accesses k-th word in 32-byte aligned arrays. Non-unit-stride access kills bandwidth (stride of 2 = 50% efficiency)
- **Shared memory**: On-chip, much higher bandwidth than global memory. Use to avoid redundant global loads, enable coalescing, avoid wasted bandwidth
- **Async copies (CUDA 11+)**: `__pipeline_memcpy_async()` copies directly global→shared without intermediate registers

### Asynchronous Execution
- **Streams**: Enable overlapping computation with data transfers
- **Concurrent kernel execution**: Multiple kernels in different non-default streams
- **`cudaMemcpyAsync()`**: Non-blocking, requires pinned memory and non-default streams
- **`cuLaunchHostFunc()`**: Launch CPU callback from GPU stream completion

### Execution Configuration
- **Occupancy**: Ratio of active warps to max possible warps. Higher occupancy helps hide memory latency
- **Thread block size**: Multiples of warp size (32). Min 64, 128-256 often good
- **Register pressure**: `-maxrregcount=N` or `__launch_bounds__()` to control
- **Shared memory partitioning** affects max blocks per SM

### Instruction Optimization
- **Minimize low-throughput instructions**: Use intrinsics (`__sinf`, `__cosf`) instead of standard math
- **Avoid warp divergence**: Different execution paths within same warp are serialized
- **Use FMA**: Fused multiply-add is faster and more precise
- **Single precision over double** when accuracy allows

### Memory Hierarchy
| Memory | Location | Cached | Access | Scope |
|--------|----------|--------|--------|-------|
| Register | On-chip | n/a | R/W | Thread |
| Shared | On-chip | n/a | R/W | Block |
| L2 Cache | On-chip | Yes | R/W | Device |
| Global | Off-chip | L1+L2 | R/W | Device+Host |
| Constant | Off-chip | Yes | R | Device+Host |
| Texture | Off-chip | Yes | R | Device+Host |

### L2 Cache Management (Compute Cap 8.0+)
- `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` — set aside L2 for persistent data
- Access policy window with hitRatio tuning for frequently accessed regions
- Up to 50% performance increase for persistent data that fits in L2 set-aside

### Stream Ordered Memory Allocation
- Use `cudaMallocAsync()` / `cudaFreeAsync()` instead of `cudaMalloc()` / `cudaFree()`
- Pool-based allocation, much lower overhead

## CUDA Runtime API Structure
- Device Management, Stream Management, Event Management
- Memory Management (unified addressing, peer access)
- Graph Management (CUDA Graphs for repeatable workloads)
- External Resource Interop (Vulkan/OpenGL/D3D interop)
- Texture/Surface Object Management

## Compatibility
- CUDA Forward Compatible Upgrade for datacenter GPUs
- Enhanced Compatibility (CUDA 11.1+): Build for one minor release, run on all future minor releases in same major family
- PTX provides forward compatibility with future architectures via JIT compilation
