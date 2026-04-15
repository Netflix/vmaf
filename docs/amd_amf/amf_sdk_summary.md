# AMD Advanced Media Framework (AMF) SDK - Summary

> Source: https://github.com/GPUOpen-LibrariesAndSDKs/AMF

## Overview

AMF is a light-weight, portable multimedia framework that abstracts platform and API-specific 
details for multimedia applications. Supports DirectX 11, OpenGL, OpenCL, and Vulkan with 
efficient interop between them.

## Supported Platforms
- **Windows**: Windows 10/11, Visual Studio 2019+
- **Linux**: RHEL 9.6/10, Ubuntu 24.04/22.04, SLED/SLES 15.7/16
- **GPU**: AMD Radeon (Southern Islands and newer), APUs (Kabini, Kaveri, Carrizo+)

## Key Features (v1.5.0)
- Video encode/decode (AVC, HEVC, AV1)
- 4:4:4/4:2:2 chroma subsampling for VideoConverter
- B-frame support for AV1 encoder
- Native DX12 encoding support
- Vulkan HEVC encoder on Navi family
- HQScaler (Bilinear/Bicubic/FSR)
- Frame Rate Conversion (FRC)
- VQEnhancer for video quality
- PSNR/SSIM score feedback (built-in!)
- Multi-HW instance encoder mode
- Split frame encoding

## Relevance to libvmaf
1. **Zero-copy surface integration**: AMF surfaces could be passed directly to HIP compute kernels
2. **Built-in PSNR/SSIM**: AMF already computes these metrics — could cross-validate
3. **AMF Compute**: OpenCL abstraction for custom GPU compute on AMD hardware
4. **Vulkan interop**: AMF supports Vulkan Khronos extensions for decoder
5. **RADV driver support**: Experimental, for open-source Vulkan on AMD
