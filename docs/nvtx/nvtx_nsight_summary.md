# NVTX C++ API & Nsight Systems Profiling - Summary

> Sources:
> - https://nvidia.github.io/NVTX/doxygen-cpp/index.html
> - https://docs.nvidia.com/nsight-systems/UserGuide/index.html

## NVTX C++ API

### Quick Start
```cpp
#include <nvtx3/nvtx3.hpp>

void my_function() {
    NVTX3_FUNC_RANGE();  // Automatic scoped range using function name
    // ... work ...
}
```

### Annotation Types

| Type | Use | Example |
|------|-----|---------|
| `scoped_range` | RAII push/pop range | `nvtx3::scoped_range r{"frame"};` |
| `unique_range` | Non-RAII start/end range | `auto id = nvtx3::start_range("upload");` |
| `mark` | Single timestamp event | `nvtx3::mark("checkpoint");` |

### Event Attributes
- **Message**: String label (`nvtx3::event_attributes attr{"my_label"};`)
- **Registered message**: Pre-registered string for low overhead
- **Color**: Visual color in profiler (`nvtx3::rgb{255, 0, 0}`)
- **Category**: Group annotations logically
- **Payload**: Attach numeric data (int32, int64, float, double)

### Domains
- Custom domain tag types for isolating annotations per library
- `struct my_domain { static constexpr char const* name = "libvmaf"; };`

### Convenience Macros
- `NVTX3_FUNC_RANGE()` — range for entire function scope
- `NVTX3_FUNC_RANGE_IN(domain)` — range in custom domain

## Nsight Systems CLI Profiling

### Basic Commands
```bash
# Full trace
nsys profile --trace=cuda,vulkan,nvtx --output=report ./app

# With GPU metrics
nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all ./app

# NVTX-triggered capture
nsys profile --capture-range=nvtx --nvtx-capture="MyRange" ./app

# Stats summary
nsys stats report.nsys-rep
```

### Key CLI Switches
| Switch | Purpose |
|--------|---------|
| `--trace=cuda,vulkan,nvtx` | Enable specific tracing |
| `--cuda-memory-usage=true` | Track CUDA memory allocations |
| `--gpu-metrics-devices=all` | Collect GPU hardware metrics |
| `--nvtx-capture=range_name` | Capture only during NVTX range |
| `--capture-range=nvtx` | Use NVTX for capture control |
| `--backtrace=dwarf` | Enable call stack collection |

### GPU Hardware Metrics
- **GR Active**: Graphics/compute engine utilization
- **SM Active/Cycles**: Per-SM instruction throughput
- **Tensor Active**: Tensor core utilization
- **DRAM Bandwidth**: Memory throughput (read/write)
- **NVENC/OFA**: Video encoder/optical flow utilization
- **PCIe Throughput**: Host-device transfer bandwidth
- **NVLink**: Multi-GPU interconnect bandwidth

### Vulkan Tracing
- `vkQueueSubmit` timeline correlation
- Pipeline creation feedback
- Command buffer recording timeline

### CUDA Tracing
- API trace (all CUDA API calls)
- Workload trace (kernel execution on GPU timeline)
- Memory operation trace (allocations, transfers)
- CUDA Graphs trace
