# Building libvmaf on Windows

This guide describes how to build `libvmaf` natively on Windows.  
The steps mirror the configuration used in the official GitHub Actions workflow and have been tested on Windows 11.  
They work from both `cmd` and PowerShell.

**Note:** This guide covers only the C/C++ library (`libvmaf`).  
The Python components are platform‑independent and follow the same setup process as on Linux or macOS.

---

## Building with Visual Studio (MSVC)

### Prerequisites

1. **Initialize git submodules**

   Before building with MSVC, make sure the git submodules are initialized and updated.  
   The Windows build relies on the bundled `pthread-win32` implementation, which is provided as a git submodule.

   ```sh
   git submodule update --init --recursive
   ```

2. **Install required tools and ensure they are in your `PATH`:**

   - [Meson](https://github.com/mesonbuild/meson/releases)  
   - [Ninja](https://github.com/ninja-build/ninja/releases)  
     (required for the bundled pthread-win32 implementation)
   - [CMake](https://cmake.org/download/)
   - [Gvim](https://github.com/vim/vim-win32-installer/releases)  
     (provides the `xxd` utility needed when building the built‑in models)

### Compilation

3. **Use a Visual Studio environment with compiler variables pre‑configured**,  
   such as the **"x64 Native Tools Command Prompt"**.

4. **Configure and build:**

   ```cmd
   cd <vmaf project root>
   mkdir C:/vmaf-install

   meson setup libvmaf libvmaf/build --buildtype release --default-library static --prefix C:/vmaf-install
   meson install -C libvmaf/build
   ```

This produces a native MSVC build of `libvmaf` and installs it under `C:/vmaf-install`.

---

## Building with MSYS2 (MinGW)

1. **Install [MSYS2](https://www.msys2.org/)**

2. **From an MSYS2 MinGW64 shell, install the required packages:**

    ```sh
    pacman -S --noconfirm --needed \
        mingw-w64-x86_64-nasm \
        mingw-w64-x86_64-gcc \
        mingw-w64-x86_64-meson \
        mingw-w64-x86_64-ninja
    ```

3. **Configure and build:**

```sh
cd <vmaf project root>
mkdir C:/vmaf-install

meson setup libvmaf libvmaf/build --buildtype release --default-library static --prefix C:/vmaf-install
meson install -C libvmaf/build
```

This produces a MinGW‑compiled version of `libvmaf` compatible with MSYS2 environments.
