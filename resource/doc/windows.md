# Building vmaf on Windows

We are going to build vmaf in Windows. These steps are in accordance with the corresponding github action for building on Windows and have been tested successfully using a Windows10 machine. They work with either `cmd` or `PowerShell`.

**Note:** This guide is just to build libvmaf on Windows and not involves the python part of project as it is the same across all platforms (settings up virtual environment, ...).

## Steps
  1. Install [msys2](https://www.msys2.org/)
  
  2. Install required msys2 packages using its shell:
  
    pacman -S --noconfirm --needed mingw-w64-x86_64-nasm mingw-w64-x86_64-gcc mingw-w64-x86_64-meson mingw-w64-x86_64-ninja

  3. It is assumed the final results will be in `C:/vmaf-install` (You can change it to any directory if you want). Finally setup the meson and build:
        
        
    cd <Vmaf project root directory>
    mkdir C:/vmaf-install
    meson setup libvmaf libvmaf/build --buildtype release --default-library static --prefix C:/vmaf-install
    meson install -C libvmaf/build
