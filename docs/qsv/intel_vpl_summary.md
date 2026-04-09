# Intel oneAPI Video Processing Library (VPL) - Summary

> Sources:
> - https://github.com/intel/libvpl
> - https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-vpl.html

## Overview

Intel VPL is a GPU-accelerated video processing library providing hardware abstraction 
for video decode, encode, and processing on Intel GPUs. It's the successor to Intel Media SDK.

## Architecture
- **Dispatcher**: Loads appropriate GPU runtime based on detected hardware
- **GPU Runtime**: Platform-specific implementation
- **API**: C-based, similar to Media SDK

## Supported Platforms
| Generation | Codename | Support Level |
|-----------|----------|---------------|
| 5th Gen | Broadwell | Legacy |
| 6th Gen | Skylake | Full |
| 7th Gen | Kaby Lake | Full |
| 8th Gen | Coffee Lake | Full |
| 10th Gen | Ice Lake | Full |
| 11th Gen | Tiger Lake | Full |
| 12th Gen | Alder Lake | Full |
| 13th Gen | Raptor Lake | Full |
| 14th Gen | Meteor Lake | Full |
| Arc | Alchemist (DG2) | Full |
| Xe | Lunar Lake | Full |

## Build Integration
```cmake
find_package(VPL REQUIRED)
target_link_libraries(myapp VPL::dispatcher)
```

## Relevance to libvmaf
1. **Zero-copy frame import**: Accept VPL surfaces directly for VMAF computation
2. **Intel GPU compute**: Use Level Zero / OpenCL on Intel GPUs for feature extraction
3. **Integrated GPU (iGPU)**: Shared memory — zero-copy texture access possible
4. **Would require**: New `src/qsv/` backend with Intel-specific compute shaders
