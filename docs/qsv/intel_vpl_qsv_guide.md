# Intel QSV / VPL (Video Processing Library) - Key Excerpts

**Sources:**
- https://github.com/intel/libvpl (Dispatcher + API headers)
- https://github.com/intel/vpl-gpu-rt (GPU Runtime)
- https://intel.github.io/libvpl/latest/index.html (API Spec v2.16)
- https://www.intel.com/content/www/us/en/developer/tools/vpl/overview.html
**Fetched:** 2026-02-19

## Overview

Intel® Video Processing Library (Intel® VPL) provides access to hardware-accelerated video decode, encode, and processing on Intel GPUs. It is the successor to Intel Media SDK.

### Architecture
```
Application
    ↓
Intel VPL Dispatcher (libvpl.so.2)
    ↓
VPL GPU Runtime (vpl-gpu-rt)
    ↓
LibVA (VA-API)
    ↓
Intel Media Driver (iHD)
    ↓
Intel GPU Hardware
```

### Supported Operations
- **Video Decode**: HEVC, AVC, VP8, VP9, MPEG-2, VC1, JPEG, AV1, VVC
- **Video Encode**: HEVC, AVC, MPEG-2, JPEG, VP9, AV1
- **Video Processing (VPP)**: Color Conversion, Deinterlace, Denoise, Resize, Rotate, Composition

### Supported Hardware
**VPL GPU Runtime (modern):**
- Tiger Lake (TGL) and newer
- Intel Iris Xe, Xe MAX, Arc GPUs
- Intel Data Center GPU Flex Series
- Meteor Lake, Arrow Lake, Lunar Lake, Battlemage, Panther Lake

**Media SDK (legacy):**
- Broadwell through Tiger Lake
- 5th to 11th gen Intel Core

### Platform Support
- Linux x86-64 (fully supported)
- Windows 10/11
- VA-API on Linux, DirectX 11 on Windows

## Programming Model

### Include and Link
```c
#include "mfx.h"    /* Intel VPL include file */
// Link: libvpl.so (Linux) or libvpl.lib (Windows)
```

### CMake Integration
```cmake
find_package(VPL REQUIRED)
target_link_libraries(${TARGET} VPL::dispatcher)
```

### Key API Functions
- `MFXLoad()` — Create loader
- `MFXCreateConfig()` — Create config for filtering implementations
- `MFXCreateSession()` — Create session with selected implementation
- `MFXVideoDECODE_Init()` / `MFXVideoDECODE_DecodeFrameAsync()` — Decode
- `MFXVideoENCODE_Init()` / `MFXVideoENCODE_EncodeFrameAsync()` — Encode  
- `MFXVideoVPP_Init()` / `MFXVideoVPP_RunFrameVPPAsync()` — Video Processing

### Acceleration Modes
- `MFX_ACCEL_MODE_VIA_D3D9` — DirectX 9 (legacy)
- `MFX_ACCEL_MODE_VIA_D3D11` — DirectX 11
- `MFX_ACCEL_MODE_VIA_VAAPI` — VA-API (Linux)

## VPL API Specification v2.16

### Features
- Device discovery and selection
- Zero-copy buffer sharing
- Backwards and cross-architecture compatible
- Async operation model
- Surface pool management
- External memory allocation support

### Appendices
- Configuration Parameter Constraints
- Multiple-segment Encoding
- Streaming and Video Conferencing Features
- Switchable Graphics and Multiple Monitors
- Working Directly with VA-API for Linux
- CQP HRD Mode Encoding

## Dependencies
- **LibVA**: Video Acceleration API (Linux)
- **Intel Media Driver**: VA-API backend (iHD driver)
- **CM Runtime**: Required for some features (part of media driver package)

## Build Dependencies
```bash
# Build from source
git clone https://github.com/intel/vpl-gpu-rt
cd vpl-gpu-rt
mkdir build && cd build
cmake ..
make
make install
```

### Build Options
| Option | Default | Description |
|--------|---------|-------------|
| ENABLE_ITT | OFF | VTune instrumentation |
| ENABLE_TEXTLOG | OFF | Text log trace |
| BUILD_RUNTIME | ON | Build the runtime |
| BUILD_TESTS | OFF | Build unit tests |
| MFX_ENABLE_KERNELS | ON | Media shaders support |

## Relevance to VMAF

### Current State
- **No QSV/VPL integration exists in libvmaf**
- Zero references to QSV, VPL, oneVPL, Media SDK, VAAPI, or Intel GPU in the codebase
- VPL is primarily for encode/decode/VPP, NOT general-purpose compute

### Feasibility Assessment
QSV/VPL is **NOT suitable** for VMAF feature extraction because:

1. **Wrong abstraction level**: VPL provides encode/decode/VPP operations, not general-purpose compute shaders needed for VMAF algorithms (DWT, convolution, statistical accumulation)
2. **No custom kernel support**: Unlike CUDA/SYCL, VPL doesn't support custom compute shaders
3. **VPP limitations**: While VPP has some filters (denoise, resize), they don't map to VMAF's specific mathematical operations (Daubechies wavelets, log2 approximations, contrast metrics)

### Alternative for Intel GPUs
For Intel GPU acceleration of VMAF, the **SYCL backend** is used on Intel GPUs (via Level Zero and DPC++). SYCL provides native Intel GPU compute with optimal performance.

### Potential QSV Use Cases (Adjacent)
- Pre-decode video frames on Intel GPU hardware before VMAF analysis
- VPP color space conversion before feeding to VMAF
- Integration in FFmpeg pipeline: `qsv_decode → vpp → libvmaf_sycl`
