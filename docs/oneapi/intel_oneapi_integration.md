# Intel oneAPI Integration for libvmaf — Research Notes

> Date: 2026-02-22
> Status: **IMPLEMENTED** — The SYCL backend is now live in `src/sycl/`. The Vulkan backend has been removed.
> Original context: Vulkan compute on Intel iGPU (UHD 770) had FPS regression and couldn't compete with CPU. oneAPI was proposed as a better alternative, and this proved correct.

## Executive Summary

Instead of optimizing Vulkan shaders for Intel's iGPU (32 EUs, ~0.4 TFLOPS), use Intel's own compute stack:
- **SYCL/DPC++** for GPU compute kernels (replaces GLSL compute shaders)
- **Level Zero** as the GPU runtime (replaces Vulkan)
- **Intel VPL** for hardware video decode (replaces VAAPI)
- **oneMKL** for optimized math primitives (FFT, vector math) where applicable

This would be a new `src/sycl/` backend parallel to `src/vulkan/` and `src/cuda/`.

## Component Analysis

### 1. SYCL / DPC++ (Compute Kernels) — **PRIMARY TOOL**

**What**: C++-based GPU programming model. Single-source: host + device code in one file. Intel's DPC++ compiler generates optimal code for Intel GPUs via Level Zero.

**Why for VMAF**: Our Vulkan compute shaders (8 GLSL shaders for VIF/ADM/Motion) would become SYCL kernels. Benefits:
- Intel's compiler optimizes specifically for Intel EU architecture (SIMD-8/16/32)
- No shader compilation pipeline (GLSL → SPIR-V → ANV driver). SYCL compiles directly to Intel GPU ISA.
- Unified Shared Memory (USM) is a first-class feature — no manual staging buffer management
- Subgroup operations are Intel-optimized (vs. our manual `subgroupSize=32` forcing)
- ESIMD (Explicit SIMD) extensions available for fine-grained SIMD control
- Single-source C++ is easier to maintain than separate GLSL shader files

**Key APIs**:
```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

// Device selection
queue q{gpu_selector_v};  // Pick Intel GPU

// USM allocation (equivalent to our unified memory detection)
float* data = malloc_shared<float>(size, q);  // Host + device accessible, no staging

// Kernel launch (equivalent to vkCmdDispatch)
q.parallel_for(nd_range<2>({height, width}, {local_y, local_x}), [=](nd_item<2> item) {
    int x = item.get_global_id(1);
    int y = item.get_global_id(0);
    // ... VIF/ADM/Motion computation ...
});
```

**ESIMD for manual SIMD control** (Intel extension):
```cpp
#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

// Explicit SIMD — like writing intrinsics but portable across Intel GPU generations
simd<float, 16> data = block_load<float, 16>(ptr);  // 16-wide SIMD load
simd<float, 16> result = data * coeff;                // SIMD multiply
```

**Repo**: https://github.com/intel/llvm (DPC++ compiler)
**License**: Apache 2.0 / MIT (open source)
**Supported GPUs**: UHD 630+, Iris Xe, Arc A/B-series, Data Center GPU Flex/Max, all via Level Zero
**Also supports**: NVIDIA (via CUDA backend plugin), AMD (via HIP plugin) — from Codeplay

### 2. Level Zero (GPU Runtime) — **UNDERNEATH SYCL**

**What**: Intel's low-level GPU compute API. Equivalent to Vulkan but compute-focused. SYCL/DPC++ targets Level Zero on Intel GPUs.

**Why for VMAF**: We don't need to use Level Zero directly — SYCL abstracts it. But understanding the stack:
```
Application (libvmaf SYCL backend)
    ↓
SYCL/DPC++ runtime
    ↓
Level Zero API (ze_api.h)
    ↓
Intel compute-runtime (NEO driver)
    ↓
Intel GPU hardware (EUs)
```

**Repo**: https://github.com/intel/compute-runtime (NEO — the driver)
**Supported platforms**: Tiger Lake, Alder Lake, Raptor Lake, Meteor Lake, Arrow Lake, Lunar Lake, Arc Alchemist, Arc Battlemage, Panther Lake
**API version**: Level Zero 1.14
**Install**: `apt install intel-opencl-icd` (includes Level Zero support)

### 3. Intel VPL (Video Decode) — **DECODE SIDE**

**What**: Successor to Intel Media SDK. Hardware-accelerated video decode/encode on Intel GPUs. NOT a compute library — decode only.

**Why for VMAF**: Replace VAAPI decode path in FFmpeg with VPL for:
- Hardware H.264/H.265/AV1 decode on Intel GPU
- Direct frame output to Level Zero / SYCL memory (via DMA-BUF interop)
- Zero-copy decode → compute pipeline

**Decode → Compute pipeline**:
```
VPL decode (GPU media engine) → DMA-BUF → Level Zero import → SYCL kernel
```
vs. current:
```
VAAPI decode → hwmap → VkImage → vkCmdCopyImageToBuffer → Vulkan compute
```

**Key difference**: VPL decode outputs to Level Zero-accessible memory natively. No Vulkan interop needed. Single driver stack.

**Platform support** (from github.com/intel/libvpl):
| Platform | VPL Runtime | Media SDK |
|----------|------------|-----------|
| Broadwell–Ice Lake | — | ✔️ |
| Tiger Lake, DG1 | ✔️ | ✔️ |
| Alder Lake+ | ✔️ | — |
| Arc (DG2/Alchemist) | ✔️ | — |
| Battlemage | ✔️ | — |

**Repo**: https://github.com/intel/libvpl (API + dispatcher), https://github.com/intel/vpl-gpu-rt (GPU runtime)
**License**: MIT
**API**: C
**FFmpeg integration**: Already exists — `ffmpeg -hwaccel qsv` uses VPL under the hood

### 4. oneMKL (Math Kernels) — **SUPPLEMENTARY**

**What**: Intel's optimized math library. Has FFT, BLAS, vector math, RNG, sparse LA. Available with C, Fortran, and SYCL interfaces.

**Relevance to VMAF compute**:
- ❌ **BLAS/LAPACK**: VMAF doesn't do matrix multiplication. Not useful.
- ❌ **FFT**: VMAF doesn't use Fourier transforms. Not useful.
- ⚠️ **Vector math**: Has optimized element-wise operations (exp, log, pow). VMAF uses log2, pow in VIF — could use `oneMKL::vm::log2` and `oneMKL::vm::pow` on GPU. Minor benefit.
- ❌ **Sparse LA**: Not applicable.
- ❌ **RNG**: Not applicable.

**Bottom line**: oneMKL is unlikely to help VMAF significantly. Our kernels are custom pixel-processing operations (DWT, gaussian convolution, motion SAD) that don't map to standard BLAS/FFT primitives. We need custom SYCL kernels.

### 5. Intel IPP (Performance Primitives) — **CPU ONLY, NOT GPU**

**What**: 2500+ optimized CPU primitives for signal/image processing. Uses AVX2/AVX-512.

**Important**: IPP is **CPU-only**. No GPU offload. It won't help with GPU compute.

**Relevance**: Could optimize the CPU reference path in libvmaf (the `feature/integer_*.c` files), but that's orthogonal to the GPU compute backend question.

**Has**: DFT/FFT, convolution, image filtering, statistical functions — all CPU-optimized.

## Architecture: SYCL Backend for libvmaf

### File Structure (parallel to existing backends)

```
libvmaf/src/
  sycl/                          ← NEW: SYCL backend infrastructure
    common_sycl.cpp              ← Device init, memory management, queue setup
    common_sycl.h                ← SyclState, SyclBuffer, SyclCapabilities
    picture_sycl.cpp             ← Picture upload/download via USM
    picture_sycl.h

  feature/
    sycl/                        ← NEW: SYCL compute kernels
      integer_vif_sycl.cpp       ← VIF kernels (filter1d_vert/hori + score)
      integer_vif_sycl.h
      integer_adm_sycl.cpp       ← ADM kernels (dwt2, decouple, csf, cm)
      integer_adm_sycl.h
      integer_motion_sycl.cpp    ← Motion kernel (SAD + score)
      integer_motion_sycl.h
```

Compare to existing:
```
src/vulkan/   → common.c, common.h, picture_vulkan.c, vulkan_dynload.c
src/cuda/     → common.c, common.h, picture_cuda.c, ring_buffer.c
feature/vulkan/ → integer_{vif,adm,motion}_vulkan.c + shaders/*.comp
feature/cuda/   → integer_{vif,adm,motion}_cuda.c + integer_{vif,adm,motion}/*.cu
```

### Key Advantages over Vulkan on Intel

| Aspect | Vulkan | SYCL/Level Zero |
|--------|--------|----------------|
| Shader language | GLSL → SPIR-V → ANV | C++ (single source) |
| Memory management | Manual staging, unified detect | USM built-in (`malloc_shared`) |
| Workgroup tuning | Specialization constants | Template parameters, ESIMD |
| Subgroup control | `subgroupSize` extension | `intel::reqd_sub_group_size` attribute |
| Debug | RenderDoc, printf | Intel VTune, GDB, printf |
| Driver optimization | Generic Vulkan path in ANV | Native Level Zero path, Intel-optimized |
| Build complexity | GLSL compile + SPIR-V embed | DPC++ compiler handles everything |
| Interop with VPL decode | Via DMA-BUF + Vulkan import | Native Level Zero memory sharing |

### USM vs. Our Manual Unified Memory Detection

Our Vulkan backend (todo8 Phase 1) manually detects unified memory:
```c
// What we built in db230ff8:
if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
    (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
    (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    caps->has_unified_memory = true;
}
```

SYCL USM does this automatically:
```cpp
// USM shared allocation — runtime handles memory placement optimally
float* buf = sycl::malloc_shared<float>(size, queue);
// Host and device can both access. On iGPU: single allocation, zero copy.
// On dGPU: runtime manages host↔device migration automatically.
```

### VMAF Kernel Mapping

| Vulkan Shader | SYCL Equivalent | Notes |
|--------------|----------------|-------|
| `filter1d_vert.comp` | VIF vertical Gaussian filter kernel | nd_range<2>, shared local memory via `local_accessor` |
| `filter1d_hori.comp` | VIF horizontal filter + score | nd_range<2>, subgroup reductions |
| `motion_score.comp` | Motion SAD + convolution | nd_range<2>, similar to Vulkan version |
| `adm_dwt2.comp` | ADM 2D DWT (CDF 9/7) | nd_range<2>, shared memory for lifting steps |
| `adm_decouple.comp` | ADM band decoupling | Element-wise, trivially parallel |
| `adm_csf.comp` | ADM CSF weighting | Element-wise with LUT |
| `adm_csf_den.comp` | ADM CSF + denominator reduce | Element-wise + subgroup reduction |
| `adm_cm.comp` | ADM contrast masking reduce | Element-wise + subgroup reduction |

## Build System Integration

SYCL requires DPC++ compiler (icpx) or compatible SYCL compiler (e.g., AdaptiveCpp).

### Meson Integration
```meson
# meson_options.txt
option('enable_sycl', type: 'boolean', value: false,
       description: 'Enable Intel SYCL/oneAPI compute backend')

# meson.build
if get_option('enable_sycl')
  sycl_compiler = find_program('icpx', required: true)
  # DPC++ compilation flags
  sycl_args = ['-fsycl', '-fsycl-targets=intel_gpu_*']
  # Link flags
  sycl_link_args = ['-fsycl']
endif
```

### Build Dependencies
- **DPC++ compiler**: `icpx` from Intel oneAPI Base Toolkit, or open-source from github.com/intel/llvm
- **Level Zero loader**: `libze_loader.so` (runtime dependency)
- **Intel compute-runtime**: NEO driver (runtime dependency, usually pre-installed)
- **Optional**: VPL dispatcher + GPU runtime for decode

### Install oneAPI Base Toolkit (Linux)
```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
apt update

# Install components (pick what you need)
apt install intel-oneapi-compiler-dpcpp-cpp  # DPC++ compiler (icpx)
apt install intel-oneapi-mkl-devel            # oneMKL (optional)
apt install libvpl-dev                        # VPL dispatcher
apt install intel-media-va-driver-non-free    # VPL GPU runtime (iHD)

# Or install full Base Toolkit:
apt install intel-basekit
```

## VPL Decode → SYCL Compute Pipeline

The zero-copy decode→compute pipeline would look like:

```
FFmpeg input → VPL decode (GPU media engine)
    ↓
VPL surface (Level Zero memory / DMA-BUF)
    ↓
Import into SYCL USM device pointer
    ↓
SYCL kernel: extract Y-plane, convert P010→uint16
    ↓
SYCL kernels: VIF + ADM + Motion (same as Vulkan shaders but in C++)
    ↓
SYCL reduction: compute final VMAF score
    ↓
Host reads score (USM shared pointer)
```

### VPL ↔ Level Zero Interop

VPL GPU runtime uses Level Zero internally. Frame surfaces can be shared:
```cpp
// VPL allocates frame surface internally using Level Zero
mfxFrameSurface1* surface;
MFXVideoDECODE_DecodeFrameAsync(session, bitstream, NULL, &surface, &sync_point);

// Get the Level Zero handle from VPL surface
mfxHDL handle;
surface->FrameInterface->GetNativeHandle(surface, &handle, MFX_RESOURCE_LEVEL_ZERO_MEMORY);

// Import into SYCL
ze_memory_allocation_properties_t props;
zeMemGetAllocProperties(ze_context, handle, &props, &ze_device);
// Now usable in SYCL kernels via Level Zero interop
```

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| DPC++ compiler not available on all systems | HIGH | Make SYCL backend optional (`-Denable_sycl=true`), keep Vulkan as fallback |
| SYCL build complexity | MEDIUM | Provide Docker/CI with oneAPI pre-installed |
| VPL ↔ SYCL interop not documented well | MEDIUM | Start with CPU upload path, add zero-copy later |
| Maintenance burden (3rd compute backend) | MEDIUM | SYCL kernels are C++ — easier to maintain than GLSL shaders |
| Intel-only benefit | LOW | SYCL also runs on NVIDIA/AMD via Codeplay plugins |
| Performance may not beat Vulkan | LOW | Level Zero is Intel's native stack — should be faster than ANV (Vulkan→SPIR-V→compiler→ISA vs. SYCL→compiler→ISA) |

## Comparison: 3 Backend Approaches

| | Vulkan (current) | CUDA (current) | SYCL (proposed) |
|---|-----------------|-----------------|-----------------|
| Target GPUs | All Vulkan-capable | NVIDIA only | Intel (primary), NVIDIA/AMD (via plugins) |
| Shader/Kernel language | GLSL | CUDA C | SYCL C++ |
| Driver | ANV (Mesa) | NVIDIA proprietary | NEO (Level Zero) |
| Memory model | Manual (staging/unified detect) | CUDA managed memory | USM (automatic) |
| Decode interop | VAAPI → VkImage | NVDEC | VPL → Level Zero |
| Subgroup/warp | Extension-based | Warp intrinsics | Built-in, Intel-optimized |
| Debug tools | RenderDoc | Nsight | VTune, GDB |
| Build dependency | Vulkan SDK | CUDA SDK | oneAPI DPC++ |

## Recommendation

**Phase 1**: Implement basic SYCL backend with USM memory management and port the VIF kernel (most complex) to SYCL. Benchmark against Vulkan on UHD 770 to validate the performance hypothesis.

**Phase 2**: Port ADM and Motion kernels. Add VPL decode interop for zero-copy pipeline.

**Phase 3**: Optimize with ESIMD for critical inner loops. Benchmark full pipeline at all resolutions.

If SYCL beats Vulkan on Intel (expected due to native stack), the Vulkan Intel-specific optimizations from todo8 become unnecessary — those workgroup tuning, unified memory detection, etc. would be superseded by SYCL's built-in equivalents.

## References

- Intel oneAPI Base Toolkit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- oneMKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
- Intel VPL: https://github.com/intel/libvpl
- VPL GPU Runtime: https://github.com/intel/vpl-gpu-rt
- DPC++ Compiler (open source): https://github.com/intel/llvm
- Intel Compute Runtime (NEO/Level Zero): https://github.com/intel/compute-runtime
- Intel IPP (CPU only): https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html
- SYCL Programming Guide: https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/data-parallelism-in-c-using-sycl.html
- Level Zero Spec: https://spec.oneapi.io/level-zero/latest/core/INTRO.html
- UXL Foundation oneMath (open source oneMKL): https://github.com/uxlfoundation/oneMath
