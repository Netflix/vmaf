# CUDA Best Practices Guide - Key Excerpts

> Source: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## Memory Optimization

### Memory Types (fastest → slowest)
1. **Registers** — per-thread, fastest
2. **Shared Memory** — per-block, ~100× faster than global
3. **L1/L2 Cache** — hardware-managed
4. **Global Memory** — device DRAM, highest bandwidth after shared
5. **Host Memory (Pinned)** — page-locked, DMA-capable
6. **Host Memory (Pageable)** — requires staging copy

### Host-Device Transfers
- Use **pinned (page-locked) memory** via `cudaMallocHost` or `cudaHostAlloc` for 2-3× transfer bandwidth
- **Async transfers**: `cudaMemcpyAsync` + streams for overlap with compute
- **Zero-copy mapped memory**: `cudaHostAllocMapped` for integrated GPUs or small, infrequent accesses
- **Unified Virtual Addressing (UVA)**: Simplifies pointer management across host/device

### Coalesced Memory Access
- Consecutive threads should access consecutive memory addresses
- Alignment to 128-byte cache lines maximizes throughput
- Avoid strided access patterns (stride > 1 between threads)

### Shared Memory Bank Conflicts
- 32 banks, 4-byte stride
- Pad arrays by 1 element to avoid systematic conflicts: `__shared__ float data[32][33]`

### L2 Cache Optimization
- Use `cudaAccessPolicyWindow` to set persistence for frequently accessed data
- L2 set-aside: reserve portion of L2 for specific data streams

## Execution Configuration

### Occupancy
- Balance register usage, shared memory, and thread count per block
- Use `cudaOccupancyMaxPotentialBlockSize()` for automatic tuning
- Target >50% occupancy; higher isn't always better if memory-bound

### Thread/Block Heuristics
- Block size: multiples of 32 (warp size), typically 128-256
- Grid size: enough blocks to fill all SMs (at least 2× SM count)
- Use 1D blocks for simple array processing, 2D for image processing

## Instruction Optimization
- Prefer single-precision (`float`) over double-precision (`double`) — 2-32× throughput difference
- Use intrinsics: `__sinf`, `__cosf`, `__expf` for fast math
- Fused Multiply-Add (FMA): `fmaf(a, b, c)` — single instruction for a*b+c

## Control Flow
- Minimize warp divergence (all threads in warp should take same branch)
- Use predication for short branches
- Loop unrolling with `#pragma unroll`

## Async Execution & Streams
- Multiple streams enable kernel/transfer overlap
- **CUDA Graphs**: Capture kernel launch sequences for reduced CPU overhead
- Stream priorities: `cudaStreamCreateWithPriority` for latency-sensitive work
