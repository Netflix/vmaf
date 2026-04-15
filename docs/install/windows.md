# Installing on Windows

Use PowerShell as an administrator:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup\windows.ps1
```

The script uses **winget** (falling back to **Chocolatey**) to install:

- Visual Studio 2022 Build Tools (with the Desktop C++ workload)
- `meson`, `ninja`, `nasm`, `python` (3.11+), `llvm`
- Optional: `CUDA 12.6`, `Intel oneAPI Base Toolkit`

## Environment

After install, open a **x64 Native Tools Command Prompt for VS 2022** so
that `cl.exe` and the Windows SDK are on `PATH`. From there:

```cmd
cd libvmaf
meson setup ..\build --buildtype=release
ninja -C ..\build
```

## CUDA

Download the installer from the
[NVIDIA CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads?target_os=Windows).
Re-run meson with `-Denable_cuda=true` after install.

## oneAPI / SYCL

Install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html),
then initialize in the shell via:

```cmd
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Re-run meson with `-Denable_sycl=true`.

## Notes

- Windows CI is Linux's canary for MSVC quirks (forbidden VLAs, different
  64-bit typedefs, narrowing-conversion strictness). If your change builds
  on Linux but fails on Windows, look for one of those first.
