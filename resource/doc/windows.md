# Building libvmaf on Windows

This guide describes how to build `libvmaf` natively on Windows.
The steps mirror the configuration used in the official GitHub Actions workflow and have been tested on Windows 11.
They work from both `cmd` and PowerShell.

MSVC is a first-class toolchain. MSYS2/MinGW remains supported.

**Note:** This guide covers only the C/C++ library (`libvmaf`).
The Python components are platform-independent and follow the same setup process as on Linux or macOS.

---

## Building with Visual Studio (MSVC)

### Prerequisites

1. **Install required tools and ensure they are in your `PATH`:**

   - [Meson](https://github.com/mesonbuild/meson/releases)
   - [Ninja](https://github.com/ninja-build/ninja/releases)
   - [NASM](https://www.nasm.us/) (x86 builds)
   - [CMake](https://cmake.org/download/) (only required when using the bundled pthreads)
   - `xxd` on `PATH` (Vim, Chocolatey `xxd`, or pass `-Dbuilt_in_models=false`)

2. **Choose a pthreads implementation**

   By default, libvmaf uses a **bundled** `pthread-win32` implementation (provided as a git submodule).

   - **Bundled (default)**
     Initialize the submodules:

     ```sh
     git submodule update --init --recursive
     ```

   - **External pthreads**
     You can use an external Windows pthreads library instead (for example from vcpkg, or a pre-built pthread-win32/pthreads4w package).
     In this case **you do not need the git submodule**.

### Compilation

3. **Use a Visual Studio environment with compiler variables pre-configured**,
   such as the **"x64 Native Tools Command Prompt"**.
   `meson` must be able to find `cl`.

4. **Configure and build:**

   **Using the bundled pthreads (default):**

   ```cmd
   cd <vmaf project root>
   mkdir C:\vmaf-install

   meson setup libvmaf libvmaf\build --buildtype release --default-library static --prefix C:/vmaf-install
   meson install -C libvmaf\build
   ```

   **Using an external pthreads library:**

   ```cmd
   cd <vmaf project root>
   mkdir C:\vmaf-install

   meson setup libvmaf libvmaf\build --buildtype release --default-library static --prefix C:/vmaf-install -Dbundled_winpthreads=false
   meson install -C libvmaf\build
   ```

   Make sure the external library and its headers are discoverable (via `PATH`, `INCLUDE`, `LIB`, or pkg-config).

This produces a native MSVC build and installs it under `C:/vmaf-install`.

With `--default-library static` (recommended, and what CI uses) the layout is:

```
C:/vmaf-install/bin/vmaf.exe
C:/vmaf-install/include/libvmaf/...
C:/vmaf-install/lib/vmaf.lib
C:/vmaf-install/lib/pkgconfig/libvmaf.pc
```

`--default-library shared` installs `vmaf.dll` plus an import library `vmaf.lib`.
`--default-library both` keeps the import library as `vmaf.lib` and names the static archive `vmaf-static.lib` so the two do not clash.

Link consumers against `vmaf.lib` (or set `INCLUDE` / `LIB` / `PKG_CONFIG_PATH` to the prefix). FFmpeg on MSVC expects `vmaf.lib` on `LIB`.

Run the C test suite from the same VS prompt:

```cmd
meson test -C libvmaf\build
```

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

This produces a MinGW-compiled version of `libvmaf` compatible with MSYS2 environments.
