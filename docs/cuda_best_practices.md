# CUDA Best Practices Research

> Sources: NVIDIA CUDA C Best Practices Guide, NVIDIA Blog (Data Transfers, Overlap)
> Date: 2026-02-19

## Key Findings for libvmaf CUDA Backend

### Memory Transfer Optimization
- **Pinned memory** gives ~2x bandwidth vs pageable (~12 GB/s vs ~6 GB/s on PCIe Gen3)
- `cudaHostAlloc()` / `cuMemHostAlloc()` for pinned host memory
- **Batch small transfers** into single large transfer — per-transfer overhead is significant
- **Minimize host↔device transfers** — keep data on device as long as possible
- `cudaMemcpyAsync()` / `cuMemcpyDtoHAsync()` for non-blocking transfers

### Stream Overlap (Most Relevant to libvmaf)
- **Overlap kernel execution with data transfers** using non-default streams
- Requirements: (1) device has concurrent copy+execute, (2) non-default streams, (3) pinned memory
- Modern GPUs have 2 copy engines (H2D + D2H) + kernel engine → triple overlap possible
- **Staged concurrent pattern**: loop over chunks, each with H2D→kernel→D2H in its own stream
- Hyper-Q (compute capability 3.5+, RTX 4090 = CC 8.9) eliminates need for specific launch ordering

### Kernel Optimization
- **Occupancy**: ratio of active warps to maximum possible warps per SM
- Use `__launch_bounds__(maxThreadsPerBlock)` to help compiler optimize register allocation
- 128-256 threads per block is a good starting range
- Thread blocks should be multiples of 32 (warp size)
- Higher occupancy ≠ always better performance — tradeoff with register usage

### Memory Access Patterns
- **Coalesced global memory access** is critical — adjacent threads access adjacent memory
- Strided access wastes bandwidth (stride of 2 = 50% efficiency)
- Shared memory for data reuse within a thread block
- L2 cache persistence hints for frequently-accessed regions (CC 8.0+)
- **Stream-ordered memory allocation**: `cudaMallocAsync()` / `cudaFreeAsync()` for pool-based alloc

### Driver API Specifics (libvmaf uses driver API only)
- `cuStreamSynchronize()` instead of `cuCtxSynchronize()` — more targeted sync
- `cuDevicePrimaryCtxRetain()` to share context with FFmpeg host process
- Avoid multiple contexts per GPU — time-sliced, adds overhead
- `cuEventRecord()` + `cuEventSynchronize()` for precise timing

## Applicability to libvmaf

### Already Implemented ✅
- Pinned memory via `cuMemHostAlloc`
- Double-buffer submit/collect pattern (our Phase 3.3)
- Non-default stream per extractor
- Data stays on device between kernel launches within a frame

### Potential New Optimizations
1. **Triple overlap** — overlap H2D transfer of frame N+1 with compute of frame N and D2H of frame N-1
   - Would require 3-deep pipeline instead of current 2-deep double-buffer
   - RTX 4090 has 2 copy engines → H2D and D2H can overlap
2. **Stream-ordered memory** — replace cuMemAlloc/cuMemFree with pool allocation if available
3. **Multi-stream per extractor** — one stream for H2D, one for compute, one for D2H with events
4. **Pinned staging reuse** — check if per-frame host buffers are reused or reallocated
5. **L2 cache persistence** — hint to keep VIF lookup tables in L2 cache

### Constraints
- Only CUDA driver API available (via ffnvcodec dynlink wrappers)
- No `cuOccupancyMaxPotentialBlockSize` in headers
- No `cuGraphLaunch` in headers
- No runtime API (`cudaMallocAsync` etc.)
