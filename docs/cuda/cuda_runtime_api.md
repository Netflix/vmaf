# CUDA Runtime API Reference - Key Excerpts

**Source:** https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
**Fetched:** 2026-02-19
**Version:** CUDA Toolkit 13.1.1

## API Modules

### Core
- **Device Management**: `cudaGetDeviceProperties()`, `cudaSetDevice()`, `cudaGetDeviceCount()`
- **Error Handling**: `cudaGetLastError()`, `cudaGetErrorString()`
- **Stream Management**: `cudaStreamCreate()`, `cudaStreamSynchronize()`, `cudaStreamDestroy()`
- **Event Management**: `cudaEventCreate()`, `cudaEventRecord()`, `cudaEventSynchronize()`, `cudaEventElapsedTime()`

### Memory
- **Memory Management**: `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`, `cudaMemcpyAsync()`, `cudaMemcpy2D()`
- **Stream Ordered Memory Allocator**: `cudaMallocAsync()`, `cudaFreeAsync()` — pool-based, much lower overhead
- **Unified Addressing**: UVA on 64-bit, single virtual address space for host+device
- **Peer Device Memory Access**: Direct P2P transfers over PCIe/NVLink

### Execution
- **Execution Control**: `cudaLaunchKernel()`, `cudaFuncSetCacheConfig()`
- **Occupancy**: `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`, `cudaOccupancyMaxPotentialBlockSize()`

### Advanced
- **Graph Management**: Record a sequence of operations into a graph, instantiate, launch repeatedly with minimal overhead
- **External Resource Interop**: Import Vulkan/OpenGL/D3D memory and semaphores for zero-copy sharing
- **Driver Entry Point Access**: `cuGetProcAddress()` for forward-compatible driver API usage
- **Library Management**: Modular CUDA library loading

### Synchronization Behavior
- Most CUDA calls are asynchronous — control returns before GPU completes
- Blocking transfers: `cudaMemcpy()` blocks until completion
- Non-blocking: `cudaMemcpyAsync()` returns immediately (requires pinned memory)
- `cudaDeviceSynchronize()` blocks until all preceding CUDA calls complete

### Key Data Structures
- `cudaDeviceProp`: Device capabilities (compute capability, memory, SM count, etc.)
- `cudaFuncAttributes`: Per-kernel register/shared memory usage
- `cudaAccessPolicyWindow`: L2 cache access policy configuration
- `cudaLaunchConfig_t`: Extended kernel launch configuration
- `cudaMemPoolProps`: Memory pool configuration

## CUDA Graphs
- Capture API calls into a graph representation
- Instantiate graph into an executable
- Launch with single API call — minimal CPU overhead for repeated workloads
- Ideal for repeated kernel sequences (like VMAF feature extraction across frames)

## External Resource Interoperability
- `cudaImportExternalMemory()` — import Vulkan/D3D memory handles
- `cudaImportExternalSemaphore()` — import Vulkan/D3D semaphores
- `cudaSignalExternalSemaphoresAsync()` / `cudaWaitExternalSemaphoresAsync()`
- Enables zero-copy sharing between CUDA and Vulkan pipelines
